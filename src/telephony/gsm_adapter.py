"""
gsm_adapter.py — SIM7600EI AT-command control over a serial port.

Skeleton for the telephony layer: handles call *signalling* (answer, hang up,
dial, ring/caller-ID detection, link status) for the SIM7600EI HAT via AT
commands over pyserial. The call *audio* path (routing PCM to/from the module)
is a separate concern wired in once the Pi hardware is available.

All hardware access goes through pyserial; every method is unit-tested with a
mocked serial port, so this module is fully exercisable on WSL2 without the
HAT attached. Real-hardware validation happens on the Pi.

AT reference (SIM7600 series):
    AT              link check                      -> OK
    ATE0            disable command echo            -> OK
    AT+CLIP=1       enable caller-ID on RING        -> OK
    AT+CSQ          signal quality                  -> +CSQ: <rssi>,<ber>
    AT+CREG?        network registration            -> +CREG: <n>,<stat>
    AT+CPIN?        SIM status                       -> +CPIN: READY
    AT+CLCC         list current calls               -> +CLCC: ... (one per call)
    ATA             answer incoming call             -> OK
    AT+CHUP         hang up all calls                -> OK
    ATD<number>;    dial a voice call (note the ';') -> OK
    RING            unsolicited: incoming call
    +CLIP: "<num>"  unsolicited: caller ID (after CLIP=1)

Typical usage::

    with GSMAdapter() as gsm:
        event = gsm.wait_for_ring(timeout=30)
        if event:
            gsm.answer_call()
            ...                      # run the voice pipeline
            gsm.hangup()
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from types import TracebackType
from typing import Optional

import serial
import yaml

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "configs" / "dev_config.yaml"

# Defaults; the real port lives in pi_config.yaml on the Pi. On a SIM7600 HAT
# the AT command interface is typically /dev/ttyUSB2.
_DEFAULT_PORT = "/dev/ttyUSB2"
_DEFAULT_BAUDRATE = 115200
_DEFAULT_TIMEOUT_S = 2.0
_DEFAULT_RING_POLL_TIMEOUT_S = 30.0

# AT line terminator.
_TERMINATOR = "\r\n"

# Final result codes that terminate an AT response.
_OK = "OK"
_ERROR_CODES = ("ERROR", "NO CARRIER", "NO DIALTONE", "BUSY")


def _load_telephony_config(config_path: Optional[str] = None) -> dict:
    """Load the ``telephony:`` config section, or {} if unavailable.

    Resolution: explicit path → VOICE_ASSISTANT_CONFIG env → dev_config.yaml.
    Returns an empty dict (rather than raising) when the file is missing, so
    the adapter still works on bare defaults.
    """
    if config_path is not None:
        path = Path(config_path)
    else:
        env = os.environ.get("VOICE_ASSISTANT_CONFIG")
        path = Path(env) if env else _DEFAULT_CONFIG_PATH
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data.get("telephony", {}) or {}


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class GSMError(Exception):
    """Base class for all GSM adapter errors."""


class GSMConnectionError(GSMError):
    """Serial port could not be opened or the module did not respond."""


class GSMTimeout(GSMError):
    """An AT command did not return a final result code in time."""


class GSMCommandError(GSMError):
    """The module returned ERROR / BUSY / NO CARRIER for an AT command."""


# --------------------------------------------------------------------------- #
# Adapter
# --------------------------------------------------------------------------- #


class GSMAdapter:
    """Control a SIM7600EI modem's call signalling over a serial AT interface.

    Attributes:
        port: Serial device path.
        baudrate: Serial baud rate.
        timeout_s: Per-readline serial timeout and default AT response timeout.
        ring_poll_timeout_s: Default timeout for :meth:`wait_for_ring`.
    """

    def __init__(
        self,
        port: Optional[str] = None,
        baudrate: Optional[int] = None,
        timeout_s: Optional[float] = None,
        ring_poll_timeout_s: Optional[float] = None,
        config_path: Optional[str] = None,
    ) -> None:
        """Initialise the adapter (does not open the port — call connect()).

        Args:
            port: Serial device; falls back to config then ``/dev/ttyUSB2``.
            baudrate: Falls back to config then 115200.
            timeout_s: Falls back to config then 2.0.
            ring_poll_timeout_s: Falls back to config then 30.0.
            config_path: Optional explicit config path.
        """
        cfg = _load_telephony_config(config_path)
        self.port: str = port or cfg.get("port", _DEFAULT_PORT)
        self.baudrate: int = int(baudrate or cfg.get("baudrate", _DEFAULT_BAUDRATE))
        self.timeout_s: float = float(
            timeout_s if timeout_s is not None else cfg.get("timeout_s", _DEFAULT_TIMEOUT_S)
        )
        self.ring_poll_timeout_s: float = float(
            ring_poll_timeout_s
            if ring_poll_timeout_s is not None
            else cfg.get("ring_poll_timeout_s", _DEFAULT_RING_POLL_TIMEOUT_S)
        )
        self._serial: Optional[serial.Serial] = None

    # ------------------------------------------------------------------ #
    # Connection lifecycle
    # ------------------------------------------------------------------ #

    @property
    def is_connected(self) -> bool:
        """True if the serial port is open."""
        return self._serial is not None and self._serial.is_open

    def connect(self) -> None:
        """Open the serial port and initialise the module.

        Verifies the link with ``AT``, disables command echo (``ATE0``) and
        enables caller-ID presentation (``AT+CLIP=1``).

        Raises:
            GSMConnectionError: If the port can't be opened or ``AT`` fails.
        """
        try:
            self._serial = serial.Serial(
                self.port, self.baudrate, timeout=self.timeout_s
            )
        except (serial.SerialException, OSError) as exc:
            self._serial = None
            raise GSMConnectionError(
                f"Could not open serial port {self.port}: {exc}"
            ) from exc

        try:
            self.send_at("AT")          # link check
            self.send_at("ATE0")        # disable echo for clean parsing
            self.send_at("AT+CLIP=1")   # caller ID on RING
        except GSMError as exc:
            self.disconnect()
            raise GSMConnectionError(f"Module init failed: {exc}") from exc

    def disconnect(self) -> None:
        """Close the serial port. Idempotent."""
        if self._serial is not None:
            try:
                if self._serial.is_open:
                    self._serial.close()
            finally:
                self._serial = None

    # ------------------------------------------------------------------ #
    # Core AT command I/O
    # ------------------------------------------------------------------ #

    def send_at(self, command: str, timeout: Optional[float] = None) -> list[str]:
        """Send one AT command and collect its response.

        Reads response lines until a final result code is seen: ``OK`` returns
        the intermediate lines; any error code raises.

        Args:
            command: AT command without the line terminator (e.g. ``"AT+CSQ"``).
            timeout: Override the response timeout in seconds.

        Returns:
            Intermediate response lines (everything before the terminating
            ``OK``), stripped, in order. Empty list for commands that only
            return ``OK``.

        Raises:
            GSMConnectionError: If not connected.
            GSMCommandError: If the module returns ERROR/BUSY/NO CARRIER/etc.
            GSMTimeout: If no final result code arrives within the timeout.
        """
        if not self.is_connected:
            raise GSMConnectionError("send_at called while not connected")

        assert self._serial is not None  # for type checkers
        self._serial.reset_input_buffer()
        self._serial.write((command + _TERMINATOR).encode("ascii"))

        deadline = time.time() + (timeout if timeout is not None else self.timeout_s)
        lines: list[str] = []
        while time.time() < deadline:
            raw = self._serial.readline()
            if not raw:
                continue
            line = raw.decode("ascii", errors="ignore").strip()
            if not line:
                continue
            if line == command:        # command echo (if ATE0 not yet applied)
                continue
            if line == _OK:
                return lines
            if line in _ERROR_CODES or line.startswith(("+CME ERROR", "+CMS ERROR")):
                raise GSMCommandError(f"{command!r} failed: {line}")
            lines.append(line)

        raise GSMTimeout(f"{command!r} timed out after {timeout or self.timeout_s}s")

    # ------------------------------------------------------------------ #
    # Status queries
    # ------------------------------------------------------------------ #

    def check_sim(self) -> bool:
        """True if the SIM is present and ready (``AT+CPIN?`` → READY)."""
        for line in self.send_at("AT+CPIN?"):
            if line.startswith("+CPIN:"):
                return "READY" in line
        return False

    def check_signal(self) -> int:
        """Return RSSI from ``AT+CSQ`` (0–31; 99 = unknown/no signal)."""
        for line in self.send_at("AT+CSQ"):
            if line.startswith("+CSQ:"):
                return _parse_csq(line)
        return 99

    def check_registration(self) -> bool:
        """True if registered on the network (``AT+CREG?`` stat 1 or 5)."""
        for line in self.send_at("AT+CREG?"):
            if line.startswith("+CREG:"):
                return _parse_creg(line) in (1, 5)
        return False

    def is_call_active(self) -> bool:
        """True if at least one call is in progress (``AT+CLCC``)."""
        for line in self.send_at("AT+CLCC"):
            if line.startswith("+CLCC:"):
                return True
        return False

    # ------------------------------------------------------------------ #
    # Call control
    # ------------------------------------------------------------------ #

    def answer_call(self) -> None:
        """Answer an incoming call (``ATA``)."""
        self.send_at("ATA")

    def hangup(self) -> None:
        """Hang up all active calls (``AT+CHUP``)."""
        self.send_at("AT+CHUP")

    def dial(self, number: str) -> None:
        """Place an outbound voice call to ``number`` (``ATD<number>;``).

        Args:
            number: Destination number. Must be non-empty.

        Raises:
            ValueError: If ``number`` is empty/whitespace.
        """
        if not number or not number.strip():
            raise ValueError("dial requires a non-empty number")
        self.send_at(f"ATD{number.strip()};")

    # ------------------------------------------------------------------ #
    # Incoming call detection
    # ------------------------------------------------------------------ #

    def wait_for_ring(self, timeout: Optional[float] = None) -> Optional[dict]:
        """Block until an incoming call rings, or the timeout elapses.

        Listens for the unsolicited ``RING`` code, capturing the caller number
        from any preceding ``+CLIP:`` line (requires ``AT+CLIP=1``, set in
        :meth:`connect`).

        Args:
            timeout: Seconds to wait; falls back to ``ring_poll_timeout_s``.

        Returns:
            ``{"event": "RING", "caller": <number or None>}`` on a ring, or
            ``None`` if the timeout elapsed with no incoming call.

        Raises:
            GSMConnectionError: If not connected.
        """
        if not self.is_connected:
            raise GSMConnectionError("wait_for_ring called while not connected")

        assert self._serial is not None
        deadline = time.time() + (
            timeout if timeout is not None else self.ring_poll_timeout_s
        )
        caller: Optional[str] = None
        while time.time() < deadline:
            raw = self._serial.readline()
            if not raw:
                continue
            line = raw.decode("ascii", errors="ignore").strip()
            if not line:
                continue
            if line.startswith("+CLIP:"):
                caller = _parse_clip(line)
            elif line == "RING":
                return {"event": "RING", "caller": caller}
        return None

    # ------------------------------------------------------------------ #
    # Context manager
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "GSMAdapter":
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.disconnect()


# --------------------------------------------------------------------------- #
# Response parsers
# --------------------------------------------------------------------------- #


def _parse_csq(line: str) -> int:
    """Parse RSSI from ``+CSQ: <rssi>,<ber>``. Returns 99 if unparseable."""
    try:
        payload = line.split(":", 1)[1].strip()
        return int(payload.split(",")[0])
    except (IndexError, ValueError):
        return 99


def _parse_creg(line: str) -> int:
    """Parse registration stat from ``+CREG: <n>,<stat>``.

    Returns the ``<stat>`` field, or -1 if unparseable.
    """
    try:
        payload = line.split(":", 1)[1].strip()
        parts = [p.strip() for p in payload.split(",")]
        # Format is "<n>,<stat>"; stat is the second field.
        return int(parts[1]) if len(parts) >= 2 else int(parts[0])
    except (IndexError, ValueError):
        return -1


def _parse_clip(line: str) -> Optional[str]:
    """Parse the caller number from ``+CLIP: "<number>",<type>,...``."""
    try:
        payload = line.split(":", 1)[1].strip()
        first = payload.split(",", 1)[0].strip()
        return first.strip('"') or None
    except IndexError:
        return None