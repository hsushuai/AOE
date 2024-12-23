import sys
import warnings
from typing import Optional, Type, TextIO

# Logging levels
DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40
DISABLED = 50

# Default logging level
min_level = INFO

# Initialize global log stream to stdout
log_stream: TextIO = sys.stdout

# Set up warnings
warnings.simplefilter("once", DeprecationWarning)


def set_level(level: int) -> None:
    """Set logging threshold on current logger."""
    global min_level
    min_level = level


def set_stream(output_stream: TextIO) -> None:
    """Set logging output stream."""
    global log_stream
    log_stream = output_stream


def is_tty(stream: TextIO) -> bool:
    """Check if the stream is a TTY (interactive terminal)."""
    return hasattr(stream, 'isatty') and stream.isatty()


def debug(msg: str, *args: object):
    output_stream = log_stream if is_stream_available(log_stream) else sys.stdout
    if min_level <= DEBUG:
        message = f"DEBUG: {msg % args}"
        print(colorize(message, "cyan") if is_tty(log_stream) else message, file=output_stream, flush=True)


def info(msg: str, *args: object):
    output_stream = log_stream if is_stream_available(log_stream) else sys.stdout
    if min_level <= INFO:
        message = f"{msg % args}"
        print(colorize(message, "green") if is_tty(log_stream) else message, file=output_stream, flush=True)


def warn(
    msg: str,
    *args: object,
    category: Optional[Type[Warning]] = None,
    stacklevel: int = 1,
):
    if min_level <= WARN:
        message = f"WARN: {msg % args}"
        warnings.warn(
            colorize(message, "yellow"),
            category=category,
            stacklevel=stacklevel + 1,
        )


def deprecation(msg: str, *args: object):
    warn(msg, *args, category=DeprecationWarning, stacklevel=2)


def error(msg: str, *args: object):
    output_stream = log_stream if is_stream_available(log_stream) else sys.stdout
    if min_level <= ERROR:
        message = f"ERROR: {msg % args}"
        print(colorize(message, "red") if is_tty(log_stream) else message, file=output_stream, flush=True)


def is_stream_available(stream: TextIO) -> bool:
    return not stream.closed and stream.writable()


# Color codes for terminal output
color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38,
)


def colorize(string: str, color: str, bold: bool = False, highlight: bool = False) -> str:
    """Return string surrounded by appropriate terminal color codes to
    print colorized text. Valid colors: gray, red, green, yellow,
    blue, magenta, cyan, white, crimson.
    """
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append("1")
    attrs = ";".join(attr)
    return f"\x1b[{attrs}m{string}\x1b[0m"
