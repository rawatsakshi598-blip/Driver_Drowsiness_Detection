"""Flask status & control server — lets us verify the system with curl.

Endpoints
---------

GET  /health           basic liveness probe
GET  /status           live detection status (FPS, EAR, MAR, alarms, ...)
GET  /config           dump the current configuration as JSON
POST /alarm/test       force-trigger the alarm (returns the AlarmEvent)
POST /alarm/reset      reset the alarm state
POST /stop             request the monitor loop to stop

Run::

    python server.py --config config.yaml
    curl http://127.0.0.1:8000/health
"""
from __future__ import annotations

import argparse
import logging
import threading
import time
from dataclasses import asdict
from typing import Optional

from flask import Flask, jsonify, request

from drowsiness.app import DrowsinessMonitor
from drowsiness.config import Config, load_config

log = logging.getLogger(__name__)


def create_app(config: Config,
               monitor: Optional[DrowsinessMonitor] = None,
               standalone: bool = True) -> Flask:
    """Create the Flask app.

    Args:
        config: Config object.
        monitor: Optional pre-built monitor.  If None, the app will create
            one only when ``/start`` is POSTed, so importing the module
            (e.g. for tests) doesn't try to open a webcam.
        standalone: if True, the app auto-starts the monitor on a
            background thread when run via ``app.run()``.
    """
    app = Flask(__name__)
    state = {
        "monitor": monitor,
        "monitor_thread": None,
        "started_at": None,
    }

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "version": "1.0.0"})

    @app.get("/status")
    def status():
        mon = state["monitor"]
        if mon is None:
            return jsonify({
                "running": False,
                "started_at": None,
                "uptime_seconds": 0,
                "message": "Monitor not started. POST /start to begin.",
            })
        st = mon.status()
        started = state["started_at"]
        st["started_at"] = started
        st["uptime_seconds"] = (time.time() - started) if started else 0
        return jsonify(st)

    @app.get("/config")
    def get_config():
        return jsonify(config.to_dict())

    @app.post("/alarm/test")
    def test_alarm():
        mon = state["monitor"]
        if mon is None:
            return jsonify({"error": "monitor not running"}), 400
        ev = mon._alarm.trigger("manual_test", 0.0)
        return jsonify({
            "triggered": True,
            "event_time": ev.timestamp.isoformat(),
            "reason": ev.reason,
        })

    @app.post("/alarm/reset")
    def reset_alarm():
        mon = state["monitor"]
        if mon is None:
            return jsonify({"error": "monitor not running"}), 400
        mon._alarm.reset()
        return jsonify({"reset": True})

    @app.post("/start")
    def start_monitor():
        if state["monitor"] is not None and state["monitor_thread"].is_alive():
            return jsonify({"error": "monitor already running"}), 409
        mon = DrowsinessMonitor(config)
        state["monitor"] = mon
        state["started_at"] = time.time()
        # Open the source now so start() returns even if the camera is busy.
        try:
            mon._source.open()
        except Exception as exc:
            state["monitor"] = None
            state["started_at"] = None
            return jsonify({"error": f"failed to open source: {exc}"}), 500
        t = threading.Thread(target=_run_monitor, args=(mon,), daemon=True)
        state["monitor_thread"] = t
        t.start()
        return jsonify({"started": True, "started_at": state["started_at"]})

    @app.post("/stop")
    def stop_monitor():
        mon = state["monitor"]
        if mon is None:
            return jsonify({"stopped": False, "message": "not running"})
        mon.stop()
        t = state["monitor_thread"]
        if t is not None:
            t.join(timeout=2.0)
        try:
            mon._source.close()
            mon._detector.close()
        except Exception:
            pass
        state["monitor"] = None
        state["monitor_thread"] = None
        state["started_at"] = None
        return jsonify({"stopped": True})

    return app


def _run_monitor(monitor: DrowsinessMonitor) -> None:
    try:
        monitor.run()
    except Exception as exc:  # pragma: no cover - defensive
        log.exception("Monitor crashed: %s", exc)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="drowsiness-server", description=__doc__)
    p.add_argument("--config", default=None, help="Path to config.yaml.")
    p.add_argument("--host", default=None, help="Bind host.")
    p.add_argument("--port", type=int, default=None, help="Bind port.")
    p.add_argument("--source", default=None,
                   help="Video source override (0 = webcam, or path).")
    p.add_argument("--no-window", action="store_true",
                   help="Headless mode (no cv2 window).")
    p.add_argument("--no-cnn", action="store_true",
                   help="Disable CNN verifier.")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    overrides = {}
    if args.source is not None:
        try:
            overrides["video"] = {"source": int(args.source)}
        except ValueError:
            overrides["video"] = {"source": args.source}
    if args.no_window:
        overrides["show_window"] = False
    if args.no_cnn:
        overrides["cnn"] = {"model_path": ""}
    if args.host:
        overrides["server"] = {"host": args.host}
    if args.port:
        overrides["server"] = {**overrides.get("server", {}), "port": args.port}

    cfg = load_config(args.config, overrides=overrides if overrides else None)
    app = create_app(cfg)
    log.info("Starting status server on http://%s:%d",
             cfg.server.host, cfg.server.port)
    # threaded=True so /stop can interrupt the monitor while /status is being polled.
    app.run(
        host=cfg.server.host, port=cfg.server.port,
        threaded=True, use_reloader=False,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
