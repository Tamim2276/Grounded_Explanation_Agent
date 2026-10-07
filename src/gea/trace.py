# The one-line trace every node prints, and the `events` entry it leaves behind.
from gea import config as cfg


def trace(node: str, msg: str) -> None:
    if cfg.TRACE:
        print(f"  [{node:5s}] {msg}")


def step(node: str, msg: str, short: str = None) -> list:
    # Print a trace line and return it as an `events` entry. Every query-time node
    # ends with "events": step(...). The printed line is for the reader; the entry
    # is what N10c turns into the DAG trace, and `short` is its label in the path.
    trace(node, msg)
    return [{"node": node, "msg": msg, "short": short}]
