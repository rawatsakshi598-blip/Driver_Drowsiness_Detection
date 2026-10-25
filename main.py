"""Top-level CLI dispatcher.

Usage::

    python main.py monitor --source 0      # run the live detector
    python main.py server                  # run the status & control API
    python main.py detect --source vid.mp4 # run for one video file
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="drowsiness", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_mon = sub.add_parser("monitor", help="Run the live drowsiness detector.")
    p_mon.add_argument("--source", default=None)
    p_mon.add_argument("--config", default=None)
    p_mon.add_argument("--no-window", action="store_true")
    p_mon.add_argument("--no-cnn", action="store_true")
    p_mon.add_argument("--run-for", type=float, default=0.0)
    p_mon.add_argument("--verbose", action="store_true")

    p_srv = sub.add_parser("server", help="Run the status & control Flask API.")
    p_srv.add_argument("--config", default=None)
    p_srv.add_argument("--host", default=None)
    p_srv.add_argument("--port", type=int, default=None)
    p_srv.add_argument("--source", default=None)
    p_srv.add_argument("--no-window", action="store_true")
    p_srv.add_argument("--no-cnn", action="store_true")

    p_det = sub.add_parser("detect", help="Process a single video file.")
    p_det.add_argument("--source", required=True, help="Video file path.")
    p_det.add_argument("--config", default=None)
    p_det.add_argument("--no-cnn", action="store_true")
    p_det.add_argument("--verbose", action="store_true")

    args = parser.parse_args(argv)
    if args.cmd == "monitor":
        from drowsiness.app import main as _app_main
        rest = []
        if args.source:
            rest += ["--source", args.source]
        if args.config:
            rest += ["--config", args.config]
        if args.no_window:
            rest.append("--no-window")
        if args.no_cnn:
            rest.append("--no-cnn")
        if args.run_for:
            rest += ["--run-for", str(args.run_for)]
        if args.verbose:
            rest.append("--verbose")
        return _app_main(rest)
    elif args.cmd == "server":
        from server import main as _srv_main
        rest = []
        if args.config:
            rest += ["--config", args.config]
        if args.host:
            rest += ["--host", args.host]
        if args.port:
            rest += ["--port", str(args.port)]
        if args.source:
            rest += ["--source", args.source]
        if args.no_window:
            rest.append("--no-window")
        if args.no_cnn:
            rest.append("--no-cnn")
        return _srv_main(rest)
    elif args.cmd == "detect":
        from drowsiness.app import main as _app_main
        rest = ["--source", args.source]
        if args.config:
            rest += ["--config", args.config]
        if args.no_cnn:
            rest.append("--no-cnn")
        if args.verbose:
            rest.append("--verbose")
        rest += ["--no-window", "--run-for", "0"]
        return _app_main(rest)
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
