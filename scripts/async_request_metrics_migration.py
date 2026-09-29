#!/usr/bin/env python3
"""Explicit maintenance migration for AsyncRequest telemetry counters."""

from __future__ import annotations

import argparse
import os

from kairyu.async_requests.postgres_store import PostgresRequestStore


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill AsyncRequest telemetry while writers are drained, or "
            "remove the legacy counter trigger after every Pod is upgraded."
        )
    )
    parser.add_argument("--store-id", required=True)
    parser.add_argument(
        "--dsn-env",
        default="KAIRYU_ASYNC_REQUEST_POSTGRES_DSN",
        help="environment variable containing the PostgreSQL DSN",
    )
    parser.add_argument(
        "--mode",
        choices=("backfill", "finalize-legacy"),
        required=True,
    )
    parser.add_argument(
        "--maintenance-window-confirmed",
        action="store_true",
        help="required acknowledgement that AsyncRequest writers are drained",
    )
    args = parser.parse_args()
    if not args.maintenance_window_confirmed:
        parser.error("--maintenance-window-confirmed is required")
    dsn = os.environ.get(args.dsn_env)
    if not dsn:
        parser.error(f"{args.dsn_env} is not set")

    with PostgresRequestStore(
        dsn,
        store_id=args.store_id,
        allow_store_creation=False,
    ) as store:
        if args.mode == "backfill":
            store.migrate_metrics_during_maintenance()
        else:
            store.finalize_legacy_metric_trigger_after_rollout()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
