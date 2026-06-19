from __future__ import annotations

import argparse
import asyncio

from .agent import run_agent, run_once
from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local autonomous DeepSeek agent experiment.")
    parser.add_argument("--once", action="store_true", help="Run one cognition cycle and exit.")
    parser.add_argument("--mock", action="store_true", help="Use synthetic sensors instead of real hardware.")
    parser.add_argument("--dashboard", action="store_true", help="Serve the live dashboard alongside the loop.")
    args = parser.parse_args()

    config = load_config()

    if args.dashboard and not args.once:
        from .dashboard import serve

        serve(config.data_dir, config.dashboard_port, background=True)
        print(f"Dashboard: http://127.0.0.1:{config.dashboard_port}")

    try:
        if args.once:
            asyncio.run(run_once(config, mock_sensors=args.mock))
        else:
            asyncio.run(run_agent(config, mock_sensors=args.mock))
    except KeyboardInterrupt:
        print("\nstopped")
