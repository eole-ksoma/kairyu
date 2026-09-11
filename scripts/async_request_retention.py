#!/usr/bin/env python3
"""Bounded scheduled retention for durable AsyncRequest records."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from kairyu.async_requests.postgres_store import PostgresRequestStore
from kairyu.deploy.spec import load_deployment_spec


def _bounded_positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0 or parsed > 10_000:
        raise argparse.ArgumentTypeError("must be between 1 and 10000")
    return parsed


def _json_default(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"cannot encode {type(value).__name__}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare or apply bounded AsyncRequest retention from a DeploymentSpec. "
            "Without --apply, purge mode is a read-only preview."
        )
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--mode", choices=("prepare", "purge"), required=True)
    parser.add_argument(
        "--max-batches",
        type=_bounded_positive_int,
        default=100,
        help="hard cap for committed purge batches in one invocation",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="required for index preparation or data deletion",
    )
    args = parser.parse_args(argv)

    spec = load_deployment_spec(args.config, resolve_credentials=False)
    policy = spec.async_requests
    if policy is None:
        parser.error("the DeploymentSpec does not enable async_requests")

    summary: dict[str, object] = {
        "mode": args.mode,
        "store_id": policy.store_id,
        "applied": bool(args.apply),
        "batches": 0,
        "terminal_requests_deleted": 0,
        "audit_events_archived": 0,
        "audit_events_deleted": 0,
        "owner_deferrals_deleted": 0,
        "has_more": False,
    }
    if args.mode == "prepare" and not args.apply:
        print(json.dumps(summary, sort_keys=True))
        return 0
    if (
        args.mode == "purge"
        and policy.request_retention_s is None
        and policy.audit_retention_s is None
    ):
        print(json.dumps(summary, sort_keys=True))
        return 0

    dsn = os.environ.get(policy.dsn_env)
    if not dsn:
        parser.error(f"{policy.dsn_env} is not set")

    try:
        with PostgresRequestStore(
            dsn,
            store_id=policy.store_id,
            allow_store_creation=False,
            initialize_schema=False,
        ) as store:
            if args.mode == "prepare":
                if args.apply:
                    store.prepare_retention()
                print(json.dumps(summary, sort_keys=True))
                return 0

            max_batches = args.max_batches if args.apply else 1
            for _ in range(max_batches):
                result = store.purge_retained_data(
                    request_retention_seconds=policy.request_retention_s,
                    audit_retention_seconds=policy.audit_retention_s,
                    batch_size=policy.retention_batch_size,
                    dry_run=not args.apply,
                )
                summary["batches"] = int(summary["batches"]) + 1
                for field in (
                    "terminal_requests_deleted",
                    "audit_events_archived",
                    "audit_events_deleted",
                    "owner_deferrals_deleted",
                ):
                    summary[field] = int(summary[field]) + int(getattr(result, field))
                summary["request_cutoff"] = result.request_cutoff
                summary["audit_cutoff"] = result.audit_cutoff
                summary["has_more"] = result.has_more
                if not result.has_more:
                    break
        print(json.dumps(summary, default=_json_default, sort_keys=True))
        return 0
    except Exception as error:
        summary["error_type"] = type(error).__name__
        print(json.dumps(summary, default=_json_default, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
