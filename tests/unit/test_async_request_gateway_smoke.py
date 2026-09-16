from pathlib import Path

import yaml

from verification.fleet.resilience.async_request_gateway_smoke import (
    GATEWAY_IDS,
    find_takeover,
    metric_value,
    rank_gateways,
    sessions_by_gateway,
    terminal_state_counts,
)
from verification.fleet.resilience.fleet_gateway_bench import (
    wall_clock_envelope_contains,
)


def test_cross_process_clock_envelope_has_a_narrow_explicit_tolerance():
    assert wall_clock_envelope_contains(1_000_000, 999_500, 2_000_500, 2_000_000)
    assert not wall_clock_envelope_contains(
        1_000_000,
        -4_000_001,
        2_000_000,
        2_000_000,
    )
    assert not wall_clock_envelope_contains(
        1_000_000,
        1_500_000,
        7_000_001,
        2_000_000,
    )
    assert not wall_clock_envelope_contains(
        10_000_000,
        5_000_000,
        5_000_000,
        0,
    )


def test_metric_value_ignores_prometheus_label_order():
    text = (
        "# HELP kairyu_async_request_transitions_total test\n"
        'kairyu_async_request_transitions_total{event="reclaim",store="shared"} 2.0\n'
    )
    assert (
        metric_value(
            text,
            "kairyu_async_request_transitions_total",
            store="shared",
            event="reclaim",
        )
        == 2
    )


def test_terminal_state_counts_reads_every_terminal_gauge():
    text = "\n".join(
        f'kairyu_async_request_state{{state="{state}",store="kairyu-f1c-async"}} {count}'
        for state, count in {
            "succeeded": 3,
            "failed": 0,
            "cancelled": 2,
            "expired": 1,
        }.items()
    )

    assert terminal_state_counts(text) == {
        "succeeded": 3,
        "failed": 0,
        "cancelled": 2,
        "expired": 1,
    }


def test_sessions_target_every_gateway_deterministically():
    sessions = sessions_by_gateway()

    assert set(sessions) == set(GATEWAY_IDS)
    assert len(set(sessions.values())) == len(GATEWAY_IDS)
    for gateway_id, session_id in sessions.items():
        assert rank_gateways(session_id)[0] == gateway_id


def test_takeover_requires_a_new_worker_and_higher_fence():
    events = [
        {"event": "reclaim", "worker_id": "old", "fencing_token": 2},
        {"event": "reclaim", "worker_id": "new", "fencing_token": 1},
        {"event": "claim", "worker_id": "new", "fencing_token": 3},
        {"event": "reclaim", "worker_id": "new", "fencing_token": 3},
    ]

    assert find_takeover(events, original_worker="old", original_fence=2) == events[3]


def test_f1c_fixture_enables_async_requests_on_all_gateways():
    documents = list(yaml.safe_load_all(Path("deploy/kind/f1c/gateways.yaml").read_text()))
    config_map = documents[0]
    config = yaml.safe_load(config_map["data"]["config.yaml"])

    assert config["engines"]["async-smoke"] == {
        "backend": "mock",
        "options": {"latency_s": 5.0},
    }
    assert config["engines"]["async-failover-smoke"] == {
        "backend": "mock",
        "options": {"latency_s": 30.0},
    }
    assert config["async_requests"] == {
        "dsn_env": "KAIRYU_ASYNC_REQUEST_POSTGRES_DSN",
        "store_id": "kairyu-f1c-async",
        "max_concurrency": 1,
        "max_body_bytes": 1048576,
        "max_records_per_tenant": 64,
        "poll_interval_s": 0.1,
        "lease_seconds": 3,
        "request_retention_s": 60,
        "audit_retention_s": 86400,
        "retention_batch_size": 2,
    }
    deployments = [
        document for document in documents if document and document.get("kind") == "Deployment"
    ]
    assert {deployment["metadata"]["name"] for deployment in deployments} == {
        "f1c-gateway-a",
        "f1c-gateway-b",
        "f1c-gateway-c",
    }
    for deployment in deployments:
        env = {
            item["name"]: item
            for item in deployment["spec"]["template"]["spec"]["containers"][0]["env"]
        }
        assert env["KAIRYU_ASYNC_REQUEST_POSTGRES_DSN"]["valueFrom"]["secretKeyRef"] == {
            "name": "f1c-postgres",
            "key": "dsn",
        }
        assert env["KAIRYU_ASYNC_REQUEST_WORKER_ID"]["valueFrom"]["fieldRef"] == {
            "fieldPath": "metadata.uid"
        }


def test_wrapper_preserves_preexisting_cluster_and_runs_foundation_first():
    script = Path("scripts/kind_async_request_gate.sh").read_text()
    foundation_script = Path("scripts/kind_gateway_gate.sh").read_text()

    refusal = "kind cluster ${CLUSTER_NAME} already exists; choose another name"
    foundation = '"$SCRIPT_DIR/kind_gateway_gate.sh" --keep-cluster'
    smoke = "verification/fleet/resilience/async_request_gateway_smoke.py"
    assert refusal in script
    assert script.index('mkdir -- "$GATE_LOCK_DIR"') < script.index(foundation)
    assert script.index("CLUSTER_OWNED=1") < script.index(foundation)
    assert "F1C_REFUSE_EXISTING_CLUSTER=1" in script
    assert "F1C_GATE_LOCK_HELD=1" in script
    assert script.index(foundation) < script.index(smoke)
    assert "another F1c/AsyncRequest kind gate owns" in foundation_script
    assert "if ((REFUSE_EXISTING_CLUSTER == 1))" in foundation_script
    assert "mapfile" not in foundation_script
    assert 'done < <("$KIND" get clusters)' in foundation_script
    assert 'done < <("$KIND" get nodes --name "$CLUSTER_NAME")' in foundation_script
    assert 'run_bounded 120s "$KIND" delete cluster' in script
    assert "claim-audit.jsonl" in script
