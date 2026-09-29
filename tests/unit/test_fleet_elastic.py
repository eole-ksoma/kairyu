"""m10a gates: dynamic membership, HRW remap property, drain, registry,
reconciler, tracing spans, helm render."""

import asyncio
import json
import os
import subprocess
from pathlib import Path, PurePosixPath

import pytest
import yaml

from kairyu.deploy import registry as registry_module
from kairyu.deploy.registry import (
    PoolReconciler,
    RegistryDiscovery,
    ReplicaRegistry,
    StaticDiscovery,
)
from kairyu.engine.backend import (
    CacheHint,
    GenerationRequest,
    GenerationResult,
    SamplingParams,
)
from kairyu.engine.openai_backend import OpenAICompatBackend
from kairyu.orchestration.replica import ReplicaPool


class MockBackend:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, request):
        self.calls += 1
        return GenerationResult(request_id="req", prompt="p", completions=(), finished=True)

    async def stream(self, request):
        yield GenerationResult(request_id="req", prompt="p", completions=(), finished=True)

    async def shutdown(self) -> None:
        return None


class ShutdownRecordingBackend(MockBackend):
    def __init__(self, name: str, *, fail_shutdown: bool = False) -> None:
        super().__init__()
        self.name = name
        self.shutdown_calls = 0
        self.fail_shutdown = fail_shutdown

    async def shutdown(self) -> None:
        self.shutdown_calls += 1
        if self.fail_shutdown:
            raise RuntimeError(f"{self.name} shutdown failed")


class BlockingBackend(ShutdownRecordingBackend):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def generate(self, request):
        self.calls += 1
        self.started.set()
        await self.release.wait()
        return GenerationResult(
            request_id="req", prompt="p", completions=(), finished=True
        )


class MutableDiscovery:
    def __init__(self, members) -> None:
        self.members = dict(members)

    def poll(self):
        return dict(self.members)


class RecordingFactory:
    def __init__(
        self,
        pool: ReplicaPool,
        replica_id: str,
        old_backend: ShutdownRecordingBackend,
        *,
        fail: bool = False,
    ) -> None:
        self.pool = pool
        self.replica_id = replica_id
        self.old_backend = old_backend
        self.fail = fail
        self.identities = []
        self.candidates = []

    def __call__(self, identity):
        self.identities.append(identity)
        assert self.pool._entries[self.replica_id].backend is self.old_backend
        if not self.candidates:
            assert self.pool.is_draining(self.replica_id) is False
        if self.fail:
            raise RuntimeError("candidate factory failed")
        candidate = ShutdownRecordingBackend(
            f"candidate-{self.replica_id}-{len(self.candidates) + 1}"
        )
        self.candidates.append(candidate)
        return candidate, f"{identity.address.removesuffix('/v1')}/readyz"


def _request(session: str | None = None) -> GenerationRequest:
    hint = CacheHint(session_id=session) if session else None
    return GenerationRequest(
        request_id="req", prompt="p", sampling_params=SamplingParams(), cache_hint=hint
    )


class TestDynamicMembership:
    async def test_add_drain_remove_lifecycle(self):
        pool = ReplicaPool({"a": MockBackend(), "b": MockBackend()})
        pool.add_replica("c", MockBackend(), health_url="http://c/health")
        assert pool.replica_ids == ("a", "b", "c")
        assert pool.health_url("c") == "http://c/health"

        pool.drain("a")
        assert pool.is_draining("a")
        # drained replicas take no NEW placements
        for _ in range(8):
            await pool.generate(_request())
        assert pool.outstanding_by_id()["a"] == 0

        await pool.remove_replica("a")
        assert pool.replica_ids == ("b", "c")
        with pytest.raises(ValueError, match="already"):
            pool.add_replica("b", MockBackend())

    async def test_cancel_drain_preserves_health_and_outstanding(self):
        backend = MockBackend()
        pool = ReplicaPool({"replica": backend}, unhealthy_after=2)
        entry = pool._entries["replica"]
        entry.consecutive_failures = 2
        entry.outstanding = 3
        health_before = pool.healthy_by_id()
        outstanding_before = pool.outstanding_by_id()

        pool.drain("replica")
        pool.cancel_drain("replica")

        assert pool._entries["replica"] is entry
        assert pool._entries["replica"].backend is backend
        assert pool.is_draining("replica") is False
        assert pool.healthy_by_id() == health_before
        assert pool.outstanding_by_id() == outstanding_before

    async def test_manual_cancel_does_not_release_active_drain_lease(self):
        pool = ReplicaPool({"replica": MockBackend()})
        lease = pool.acquire_drain("replica")
        pool.drain("replica")

        pool.cancel_drain("replica")

        assert pool.is_draining("replica") is True
        with pytest.raises(RuntimeError, match="eligible"):
            pool._select(_request())
        pool.release_drain("replica", lease)
        assert pool.is_draining("replica") is False

    async def test_manual_drain_query_ignores_active_lease(self):
        pool = ReplicaPool({"replica": MockBackend()})

        assert pool.is_manually_draining("replica") is False
        lease = pool.acquire_drain("replica")
        assert pool.is_draining("replica") is True
        assert pool.is_manually_draining("replica") is False
        pool.drain("replica")
        assert pool.is_manually_draining("replica") is True
        pool.cancel_drain("replica")
        assert pool.is_manually_draining("replica") is False
        assert pool.is_draining("replica") is True
        pool.release_drain("replica", lease)

    async def test_release_drain_lease_preserves_state_and_restores_eligibility(self):
        backend = MockBackend()
        pool = ReplicaPool({"replica": backend}, unhealthy_after=2)
        entry = pool._entries["replica"]
        entry.consecutive_failures = 1
        entry.outstanding = 3
        health_before = pool.healthy_by_id()
        outstanding_before = pool.outstanding_by_id()
        lease = pool.acquire_drain("replica")

        pool.release_drain("replica", lease)

        assert pool._entries["replica"] is entry
        assert pool._entries["replica"].backend is backend
        assert pool.is_draining("replica") is False
        assert pool.healthy_by_id() == health_before
        assert pool.outstanding_by_id() == outstanding_before
        assert pool._select(_request())[0] == "replica"

    async def test_remove_refuses_inflight_then_force(self):
        import asyncio

        class SlowBackend(MockBackend):
            def __init__(self):
                super().__init__()
                self.release = asyncio.Event()

            async def generate(self, request):
                await self.release.wait()
                return GenerationResult(request_id="req", prompt="p", completions=(), finished=True)

        slow = SlowBackend()
        pool = ReplicaPool({"s": slow})
        task = asyncio.create_task(pool.generate(_request()))
        await asyncio.sleep(0.01)
        with pytest.raises(RuntimeError, match="in-flight"):
            await pool.remove_replica("s")
        await pool.remove_replica("s", force=True)
        slow.release.set()
        await task  # late completion on removed id is a no-op (A2)

    async def test_all_draining_raises(self):
        pool = ReplicaPool([MockBackend()])
        pool.drain("0")
        with pytest.raises(RuntimeError, match="eligible"):
            await pool.generate(_request())

    async def test_probe_never_clears_draining(self):
        pool = ReplicaPool([MockBackend(), MockBackend()])
        pool.drain("1")
        await pool.probe(1)  # legacy ordinal accepted
        assert pool.is_draining("1")


class TestHrwRemapProperty:
    def _mapping(self, pool: ReplicaPool, sessions: list[str]) -> dict[str, str]:
        return {
            session: pool._select(_request(session))[0] for session in sessions
        }

    async def test_removal_remaps_only_departed_sessions(self):
        backends = {str(i): MockBackend() for i in range(8)}
        pool = ReplicaPool(backends)
        sessions = [f"s{i}" for i in range(400)]
        before = self._mapping(pool, sessions)
        await pool.remove_replica("3")
        after = self._mapping(pool, sessions)
        moved = [s for s in sessions if before[s] != after[s]]
        assert all(before[s] == "3" for s in moved)  # only its own sessions

    async def test_addition_remaps_about_one_over_n(self):
        pool = ReplicaPool({str(i): MockBackend() for i in range(8)})
        sessions = [f"s{i}" for i in range(400)]
        before = self._mapping(pool, sessions)
        pool.add_replica("8", MockBackend())
        after = self._mapping(pool, sessions)
        moved = sum(before[s] != after[s] for s in sessions)
        assert moved <= 400 / 9 * 1.6  # ~1/N with slack
        assert all(after[s] == "8" for s in sessions if before[s] != after[s])


class TestRegistryAndReconciler:
    def test_ttl_expiry(self):
        clock = {"t": 0.0}
        registry = ReplicaRegistry(now=lambda: clock["t"])
        registry.register("r1", "http://r1/v1", ttl_s=10)
        assert registry.alive() == {
            "r1": registry_module.ReplicaConfig(address="http://r1/v1")
        }
        clock["t"] = 9.0
        registry.heartbeat("r1")
        clock["t"] = 18.0
        assert registry.alive() == {
            "r1": registry_module.ReplicaConfig(address="http://r1/v1")
        }
        clock["t"] = 30.0
        assert registry.alive() == {}
        with pytest.raises(KeyError):
            registry.heartbeat("ghost")

    async def test_reconciler_adds_and_drain_removes(self):
        pool = ReplicaPool({"old": MockBackend()})
        members = {"old": "http://old/v1", "new": "http://new/v1"}
        source = StaticDiscovery(members)
        reconciler = PoolReconciler(
            pool,
            source,
            default_model="llama",
            factory=lambda identity: (
                MockBackend(),
                f"{identity.address}/health",
            ),
        )
        result = await reconciler.reconcile()
        assert result["added"] == ["new"]
        assert pool.replica_ids == ("old", "new")

        members.pop("old")
        source._members.pop("old")
        result = await reconciler.reconcile()
        assert result["removed"] == ["old"]
        assert pool.replica_ids == ("new",)

    async def test_reconciler_retries_inflight_removal(self):
        pool = ReplicaPool({"busy": MockBackend(), "idle": MockBackend()})
        pool._entries["busy"].outstanding = 1  # simulate in-flight
        source = StaticDiscovery({"idle": "http://idle/v1"})
        reconciler = PoolReconciler(
            pool,
            source,
            default_model="llama",
            factory=lambda identity: (MockBackend(), None),
        )
        result = await reconciler.reconcile()
        assert result["removed"] == []
        assert "busy" in result["draining"]
        assert pool.is_draining("busy")
        pool._entries["busy"].outstanding = 0
        result = await reconciler.reconcile()
        assert result["removed"] == ["busy"]

    def test_registry_discovery_bridges(self):
        clock = {"t": 0.0}
        registry = ReplicaRegistry(now=lambda: clock["t"])
        registry.register(
            "r1",
            "http://r1/v1",
            model="llama",
            api_key_env=None,
        )
        assert RegistryDiscovery(registry).poll() == {
            "r1": registry_module.ReplicaConfig(
                address="http://r1/v1",
                model="llama",
                api_key_env=None,
            )
        }

    def test_default_openai_factory_receives_model_and_auth_identity(self):
        identity = registry_module.ReplicaIdentity(
            address="http://replica/v1",
            model="llama",
            api_key_env=None,
        )

        backend, health_url = registry_module.openai_replica_factory(identity)

        assert isinstance(backend, OpenAICompatBackend)
        assert backend._base_url == "http://replica/v1"
        assert backend._model == "llama"
        assert backend._api_key_env is None
        assert backend.capabilities.upstream == "kairyu"
        assert backend._client is None
        assert backend._owns_client is True
        assert health_url == "http://replica/readyz"

    async def test_initial_scale_uses_reconciler_default_model(self):
        pool = ReplicaPool({"seed": MockBackend()})
        source = StaticDiscovery({"replica": "http://replica/v1"})
        reconciler = PoolReconciler(pool, source, default_model="llama")

        result = await reconciler.reconcile()

        assert result["added"] == ["replica"]
        backend = pool._entries["replica"].backend
        assert isinstance(backend, OpenAICompatBackend)
        assert backend._model == "llama"

    async def test_discovery_identity_overrides_reconciler_model_and_auth_default(self):
        identities = []
        source_identity = registry_module.ReplicaIdentity(
            address="http://replica/v1",
            model="source-model",
            api_key_env=None,
        )
        source = StaticDiscovery(
            {
                "replica": registry_module.ReplicaConfig(
                    address=source_identity.address,
                    model=source_identity.model,
                    api_key_env=source_identity.api_key_env,
                )
            }
        )
        pool = ReplicaPool({"seed": MockBackend()})

        def recording_factory(identity):
            identities.append(identity)
            return MockBackend(), None

        reconciler = PoolReconciler(
            pool,
            source,
            factory=recording_factory,
            default_model="fallback-model",
        )

        await reconciler.reconcile()

        assert identities == [source_identity]

    async def test_unchanged_identity_is_a_complete_noop_on_second_reconcile(self):
        old = ShutdownRecordingBackend("old")
        pool = ReplicaPool({"replica": old})
        source = MutableDiscovery(
            {
                "replica": registry_module.ReplicaConfig(
                    address="http://old/v1",
                    model="llama",
                    api_key_env="OLD_API_KEY",
                )
            }
        )
        factory = RecordingFactory(pool, "replica", old)
        reconciler = PoolReconciler(pool, source, factory=factory)

        assert await reconciler.reconcile() == {
            "added": [],
            "draining": [],
            "removed": [],
        }
        result = await reconciler.reconcile()

        assert result == {"added": [], "draining": [], "removed": []}
        assert factory.identities == []
        assert pool._entries["replica"].backend is old
        assert pool.is_draining("replica") is False
        assert old.shutdown_calls == 0

    async def test_external_disappearance_clears_tracking_before_same_id_reappears(
        self,
    ):
        old = ShutdownRecordingBackend("old")
        pool = ReplicaPool({"replica": old})
        old_config = registry_module.ReplicaConfig(
            address="http://old/v1", model="llama"
        )
        source = MutableDiscovery({"replica": old_config})
        factory_calls = []

        def recording_factory(identity):
            factory_calls.append(identity)
            return ShutdownRecordingBackend("unexpected-candidate"), None

        reconciler = PoolReconciler(pool, source, factory=recording_factory)
        await reconciler.reconcile()
        pool._entries["replica"].outstanding = 1
        source.members.clear()
        blocked = await reconciler.reconcile()
        pool._entries["replica"].outstanding = 0

        await pool.remove_replica("replica")
        disappeared = await reconciler.reconcile()

        assert blocked == {
            "added": [],
            "draining": ["replica"],
            "removed": [],
        }
        assert disappeared == {"added": [], "draining": [], "removed": []}
        assert reconciler._applied == {}
        assert reconciler._draining == {}
        assert old.shutdown_calls == 1

        new = ShutdownRecordingBackend("new")
        new_config = registry_module.ReplicaConfig(
            address="http://new/v1", model="llama"
        )
        pool.add_replica("replica", new)
        source.members["replica"] = new_config

        restored = await reconciler.reconcile()

        assert restored == {"added": [], "draining": [], "removed": []}
        assert factory_calls == []
        assert pool._entries["replica"].backend is new
        assert pool._select(_request())[0] == "replica"
        assert new.shutdown_calls == 0

    async def test_same_id_external_readd_acquires_fresh_removal_drain(self):
        old = BlockingBackend("old")
        pool = ReplicaPool({"replica": old})
        source = MutableDiscovery(
            {
                "replica": registry_module.ReplicaConfig(
                    address="http://replica/v1", model="llama"
                )
            }
        )
        reconciler = PoolReconciler(pool, source)
        await reconciler.reconcile()

        old_request = asyncio.create_task(pool.generate(_request()))
        await old.started.wait()
        source.members.clear()
        blocked = await reconciler.reconcile()
        assert blocked == {
            "added": [],
            "draining": ["replica"],
            "removed": [],
        }
        assert pool.is_draining("replica") is True

        old.release.set()
        await old_request
        await pool.remove_replica("replica")
        fresh = BlockingBackend("fresh")
        pool.add_replica("replica", fresh)

        fresh_request = asyncio.create_task(pool.generate(_request()))
        await fresh.started.wait()
        try:
            retried = await reconciler.reconcile()

            assert retried == {
                "added": [],
                "draining": ["replica"],
                "removed": [],
            }
            assert pool.is_draining("replica") is True
            with pytest.raises(RuntimeError, match="eligible"):
                await pool.generate(_request())
        finally:
            fresh.release.set()
            await fresh_request

        completed = await reconciler.reconcile()
        assert completed == {
            "added": [],
            "draining": [],
            "removed": ["replica"],
        }
        assert pool.replica_ids == ()
        assert fresh.shutdown_calls == 1

    async def test_same_id_external_readd_with_desire_baselines_fresh_entry(self):
        old = BlockingBackend("old")
        old_config = registry_module.ReplicaConfig(
            address="http://old/v1", model="llama"
        )
        new_config = registry_module.ReplicaConfig(
            address="http://new/v1", model="llama"
        )
        pool = ReplicaPool({"replica": old})
        source = MutableDiscovery({"replica": old_config})
        candidates = []

        def factory(identity):
            candidate = ShutdownRecordingBackend(
                f"candidate-{len(candidates) + 1}"
            )
            candidates.append(candidate)
            return candidate, None

        reconciler = PoolReconciler(pool, source, factory=factory)
        await reconciler.reconcile()

        old_request = asyncio.create_task(pool.generate(_request()))
        await old.started.wait()
        source.members["replica"] = new_config
        blocked = await reconciler.reconcile()
        assert blocked == {
            "added": [],
            "draining": ["replica"],
            "removed": [],
        }
        assert len(candidates) == 1
        assert candidates[0].shutdown_calls == 1

        old.release.set()
        await old_request
        await pool.remove_replica("replica")
        fresh = ShutdownRecordingBackend("fresh")
        pool.add_replica("replica", fresh)
        pool.drain("replica")

        restored = await reconciler.reconcile()

        assert restored == {"added": [], "draining": [], "removed": []}
        assert len(candidates) == 1
        assert fresh.shutdown_calls == 0
        assert pool.is_draining("replica") is True
        assert pool.is_manually_draining("replica") is True
        pool.cancel_drain("replica")
        await pool.generate(_request())
        assert fresh.calls == 1

    async def test_reconciler_does_not_cancel_unowned_drain(self):
        old = ShutdownRecordingBackend("old")
        pool = ReplicaPool({"replica": old})
        pool.drain("replica")
        source = MutableDiscovery(
            {
                "replica": registry_module.ReplicaConfig(
                    address="http://replica/v1", model="llama"
                )
            }
        )
        reconciler = PoolReconciler(pool, source)

        result = await reconciler.reconcile()

        assert result == {"added": [], "draining": [], "removed": []}
        assert pool.is_draining("replica") is True
        assert reconciler._draining == {}

    async def test_address_change_replacement_constructs_before_drain(self):
        old = ShutdownRecordingBackend("old")
        pool = ReplicaPool({"replica": old})
        old_config = registry_module.ReplicaConfig(
            address="http://old/v1", model="llama", api_key_env="API_KEY"
        )
        source = MutableDiscovery({"replica": old_config})
        factory = RecordingFactory(pool, "replica", old)
        reconciler = PoolReconciler(pool, source, factory=factory)
        await reconciler.reconcile()  # seed the pre-existing identity baseline
        old_generation = pool.entry_generation("replica")
        new_config = registry_module.ReplicaConfig(
            address="http://new/v1", model="llama", api_key_env="API_KEY"
        )
        source.members["replica"] = new_config

        result = await reconciler.reconcile()

        candidate = factory.candidates[0]
        assert factory.identities == [
            registry_module.ReplicaIdentity(
                address="http://new/v1", model="llama", api_key_env="API_KEY"
            )
        ]
        assert result == {
            "added": ["replica"],
            "draining": [],
            "removed": ["replica"],
        }
        assert pool._entries["replica"].backend is candidate
        assert pool.health_url("replica") == "http://new/readyz"
        assert pool.entry_generation("replica") != old_generation
        assert pool.validated_by_id() == {"replica": False}
        assert pool.healthy_by_id() == {"replica": False}
        assert old.shutdown_calls == 1
        assert candidate.shutdown_calls == 0

    async def test_successful_replacement_preserves_manual_drain_on_new_backend(self):
        old = ShutdownRecordingBackend("old")
        candidate = ShutdownRecordingBackend("candidate")
        pool = ReplicaPool({"replica": old})
        source = MutableDiscovery(
            {
                "replica": registry_module.ReplicaConfig(
                    address="http://old/v1", model="llama"
                )
            }
        )

        def factory(identity):
            assert pool._entries["replica"].backend is old
            assert pool.is_draining("replica") is True
            return candidate, None

        reconciler = PoolReconciler(pool, source, factory=factory)
        await reconciler.reconcile()
        pool.drain("replica")
        source.members["replica"] = registry_module.ReplicaConfig(
            address="http://new/v1", model="llama"
        )

        result = await reconciler.reconcile()

        assert result == {
            "added": ["replica"],
            "draining": [],
            "removed": ["replica"],
        }
        assert pool._entries["replica"].backend is candidate
        assert pool.is_draining("replica") is True
        assert pool.is_manually_draining("replica") is True
        with pytest.raises(RuntimeError, match="eligible"):
            pool._select(_request())
        assert pool.healthy_by_id() == {"replica": True}
        assert pool.outstanding_by_id() == {"replica": 0}
        assert pool._entries["replica"].drain_leases == set()
        assert reconciler._draining == {}
        assert old.shutdown_calls == 1
        assert candidate.shutdown_calls == 0

    async def test_model_change_triggers_replacement(self):
        old = ShutdownRecordingBackend("old")
        pool = ReplicaPool({"replica": old})
        source = MutableDiscovery(
            {
                "replica": registry_module.ReplicaConfig(
                    address="http://replica/v1", model="old-model"
                )
            }
        )
        factory = RecordingFactory(pool, "replica", old)
        reconciler = PoolReconciler(pool, source, factory=factory)
        await reconciler.reconcile()
        source.members["replica"] = registry_module.ReplicaConfig(
            address="http://replica/v1", model="new-model"
        )

        result = await reconciler.reconcile()

        assert result["removed"] == ["replica"]
        assert result["added"] == ["replica"]
        assert factory.identities[0].model == "new-model"
        assert pool._entries["replica"].backend is factory.candidates[0]
        assert old.shutdown_calls == 1

    async def test_auth_change_triggers_replacement(self):
        old = ShutdownRecordingBackend("old")
        pool = ReplicaPool({"replica": old})
        source = MutableDiscovery(
            {
                "replica": registry_module.ReplicaConfig(
                    address="http://replica/v1",
                    model="llama",
                    api_key_env="OLD_API_KEY",
                )
            }
        )
        factory = RecordingFactory(pool, "replica", old)
        reconciler = PoolReconciler(pool, source, factory=factory)
        await reconciler.reconcile()
        source.members["replica"] = registry_module.ReplicaConfig(
            address="http://replica/v1",
            model="llama",
            api_key_env="NEW_API_KEY",
        )

        result = await reconciler.reconcile()

        assert result["removed"] == ["replica"]
        assert result["added"] == ["replica"]
        assert factory.identities[0].api_key_env == "NEW_API_KEY"
        assert pool._entries["replica"].backend is factory.candidates[0]
        assert old.shutdown_calls == 1

    async def test_replacement_factory_failure_keeps_old_replica_eligible(self):
        old = ShutdownRecordingBackend("old")
        pool = ReplicaPool({"replica": old})
        source = MutableDiscovery(
            {
                "replica": registry_module.ReplicaConfig(
                    address="http://old/v1", model="llama"
                )
            }
        )
        factory = RecordingFactory(pool, "replica", old, fail=True)
        reconciler = PoolReconciler(pool, source, factory=factory)
        await reconciler.reconcile()
        source.members["replica"] = registry_module.ReplicaConfig(
            address="http://new/v1", model="llama"
        )

        result = await reconciler.reconcile()

        assert result == {
            "added": [],
            "draining": ["replica"],
            "removed": [],
        }
        assert pool._entries["replica"].backend is old
        assert pool.is_draining("replica") is False
        assert pool._select(_request())[0] == "replica"
        assert old.shutdown_calls == 0

    async def test_inflight_replacement_cleans_candidate_then_retries(self):
        old = ShutdownRecordingBackend("old")
        pool = ReplicaPool({"replica": old})
        source = MutableDiscovery(
            {
                "replica": registry_module.ReplicaConfig(
                    address="http://old/v1", model="llama"
                )
            }
        )
        factory = RecordingFactory(pool, "replica", old)
        reconciler = PoolReconciler(pool, source, factory=factory)
        await reconciler.reconcile()
        source.members["replica"] = registry_module.ReplicaConfig(
            address="http://new/v1", model="llama"
        )
        pool._entries["replica"].outstanding = 1

        blocked = await reconciler.reconcile()

        first_candidate = factory.candidates[0]
        assert blocked == {
            "added": [],
            "draining": ["replica"],
            "removed": [],
        }
        assert pool._entries["replica"].backend is old
        assert pool.is_draining("replica") is True
        assert set(reconciler._draining) == {"replica"}
        assert first_candidate.shutdown_calls == 1
        assert old.shutdown_calls == 0

        pool._entries["replica"].outstanding = 0
        completed = await reconciler.reconcile()

        second_candidate = factory.candidates[1]
        assert completed == {
            "added": ["replica"],
            "draining": [],
            "removed": ["replica"],
        }
        assert pool._entries["replica"].backend is second_candidate
        assert first_candidate.shutdown_calls == 1
        assert second_candidate.shutdown_calls == 0
        assert old.shutdown_calls == 1
        assert reconciler._draining == {}

    async def test_reverted_replacement_cancels_owned_drain(self):
        old = ShutdownRecordingBackend("old")
        pool = ReplicaPool({"replica": old})
        applied = registry_module.ReplicaConfig(
            address="http://old/v1", model="llama"
        )
        source = MutableDiscovery({"replica": applied})
        factory = RecordingFactory(pool, "replica", old)
        reconciler = PoolReconciler(pool, source, factory=factory)
        await reconciler.reconcile()
        source.members["replica"] = registry_module.ReplicaConfig(
            address="http://new/v1", model="llama"
        )
        pool._entries["replica"].outstanding = 1
        await reconciler.reconcile()
        source.members["replica"] = applied

        result = await reconciler.reconcile()

        assert result == {"added": [], "draining": [], "removed": []}
        assert pool._entries["replica"].backend is old
        assert pool.is_draining("replica") is False
        assert pool._select(_request())[0] == "replica"
        assert reconciler._draining == {}
        assert factory.candidates[0].shutdown_calls == 1
        assert old.shutdown_calls == 0

    async def test_removed_desire_reappears_with_same_identity_cancels_owned_drain(self):
        old = ShutdownRecordingBackend("old")
        pool = ReplicaPool({"replica": old})
        config = registry_module.ReplicaConfig(
            address="http://replica/v1", model="llama"
        )
        source = MutableDiscovery({"replica": config})
        reconciler = PoolReconciler(pool, source)
        await reconciler.reconcile()
        pool._entries["replica"].outstanding = 1
        source.members.clear()
        blocked = await reconciler.reconcile()
        source.members["replica"] = config

        restored = await reconciler.reconcile()

        assert blocked == {
            "added": [],
            "draining": ["replica"],
            "removed": [],
        }
        assert restored == {"added": [], "draining": [], "removed": []}
        assert pool._entries["replica"].backend is old
        assert pool.is_draining("replica") is False
        assert pool._select(_request())[0] == "replica"
        assert reconciler._draining == {}
        assert old.shutdown_calls == 0

    async def test_retry_factory_failure_cancels_owned_drain(self):
        old = ShutdownRecordingBackend("old")
        pool = ReplicaPool({"replica": old})
        source = MutableDiscovery(
            {
                "replica": registry_module.ReplicaConfig(
                    address="http://old/v1", model="llama"
                )
            }
        )
        factory = RecordingFactory(pool, "replica", old)
        reconciler = PoolReconciler(pool, source, factory=factory)
        await reconciler.reconcile()
        source.members["replica"] = registry_module.ReplicaConfig(
            address="http://new/v1", model="llama"
        )
        pool._entries["replica"].outstanding = 1
        await reconciler.reconcile()
        factory.fail = True

        result = await reconciler.reconcile()

        assert result == {
            "added": [],
            "draining": ["replica"],
            "removed": [],
        }
        assert pool._entries["replica"].backend is old
        assert pool.is_draining("replica") is False
        assert pool._select(_request())[0] == "replica"
        assert reconciler._draining == {}
        assert factory.candidates[0].shutdown_calls == 1
        assert old.shutdown_calls == 0

    async def test_reverted_replacement_preserves_later_manual_drain(self):
        old = ShutdownRecordingBackend("old")
        pool = ReplicaPool({"replica": old})
        applied = registry_module.ReplicaConfig(
            address="http://old/v1", model="llama"
        )
        source = MutableDiscovery({"replica": applied})
        factory = RecordingFactory(pool, "replica", old)
        reconciler = PoolReconciler(pool, source, factory=factory)
        await reconciler.reconcile()
        source.members["replica"] = registry_module.ReplicaConfig(
            address="http://new/v1", model="llama"
        )
        pool._entries["replica"].outstanding = 1
        await reconciler.reconcile()
        pool.drain("replica")
        source.members["replica"] = applied

        result = await reconciler.reconcile()

        assert result == {"added": [], "draining": [], "removed": []}
        assert pool._entries["replica"].backend is old
        assert pool.is_draining("replica") is True
        with pytest.raises(RuntimeError, match="eligible"):
            pool._select(_request())
        assert reconciler._draining == {}
        assert old.shutdown_calls == 0

    async def test_retry_factory_failure_preserves_later_manual_drain(self):
        old = ShutdownRecordingBackend("old")
        pool = ReplicaPool({"replica": old})
        source = MutableDiscovery(
            {
                "replica": registry_module.ReplicaConfig(
                    address="http://old/v1", model="llama"
                )
            }
        )
        factory = RecordingFactory(pool, "replica", old)
        reconciler = PoolReconciler(pool, source, factory=factory)
        await reconciler.reconcile()
        source.members["replica"] = registry_module.ReplicaConfig(
            address="http://new/v1", model="llama"
        )
        pool._entries["replica"].outstanding = 1
        await reconciler.reconcile()
        pool.drain("replica")
        factory.fail = True

        result = await reconciler.reconcile()

        assert result == {
            "added": [],
            "draining": ["replica"],
            "removed": [],
        }
        assert pool._entries["replica"].backend is old
        assert pool.is_draining("replica") is True
        with pytest.raises(RuntimeError, match="eligible"):
            pool._select(_request())
        assert reconciler._draining == {}
        assert old.shutdown_calls == 0

    async def test_replacement_add_failure_cleans_candidate_and_applied_identity(
        self, monkeypatch
    ):
        old = ShutdownRecordingBackend("old")
        pool = ReplicaPool({"replica": old})
        source = MutableDiscovery(
            {
                "replica": registry_module.ReplicaConfig(
                    address="http://old/v1", model="llama"
                )
            }
        )
        factory = RecordingFactory(pool, "replica", old)
        reconciler = PoolReconciler(pool, source, factory=factory)
        await reconciler.reconcile()
        source.members["replica"] = registry_module.ReplicaConfig(
            address="http://new/v1", model="llama"
        )

        def reject_add(replica_id, backend, health_url=None):
            raise ValueError("replacement add failed")

        monkeypatch.setattr(pool, "add_replica", reject_add)

        with pytest.raises(ValueError, match="replacement add failed"):
            await reconciler.reconcile()

        candidate = factory.candidates[0]
        assert pool.replica_ids == ()
        assert reconciler._applied == {}
        assert reconciler._draining == {}
        assert old.shutdown_calls == 1
        assert candidate.shutdown_calls == 1

    async def test_replacement_double_shutdown_failure_clears_applied_identity(self):
        old = ShutdownRecordingBackend("old", fail_shutdown=True)
        candidate = ShutdownRecordingBackend("candidate", fail_shutdown=True)
        pool = ReplicaPool({"replica": old})
        source = MutableDiscovery(
            {
                "replica": registry_module.ReplicaConfig(
                    address="http://old/v1", model="llama"
                )
            }
        )

        def failing_candidate_factory(identity):
            assert pool._entries["replica"].backend is old
            assert pool.is_draining("replica") is False
            return candidate, None

        reconciler = PoolReconciler(
            pool, source, factory=failing_candidate_factory
        )
        await reconciler.reconcile()
        source.members["replica"] = registry_module.ReplicaConfig(
            address="http://new/v1", model="llama"
        )

        with pytest.raises(ExceptionGroup) as raised:
            await reconciler.reconcile()

        assert "replacement candidate 'replica' shutdown failed" in str(raised.value)
        assert str(raised.value.exceptions[0]) == "candidate shutdown failed"
        old_failure = raised.value.__context__
        assert isinstance(old_failure, ExceptionGroup)
        assert "replica 'replica' shutdown failed" in str(old_failure)
        assert str(old_failure.exceptions[0]) == "old shutdown failed"
        assert old.shutdown_calls == 1
        assert candidate.shutdown_calls == 1
        assert pool.replica_ids == ()
        assert reconciler._applied == {}
        assert reconciler._draining == {}

    async def test_reconciliation_result_lists_preserve_phase_order(self):
        remove_z = ShutdownRecordingBackend("remove-z")
        replace_b = ShutdownRecordingBackend("replace-b")
        remove_a = ShutdownRecordingBackend("remove-a")
        replace_a = ShutdownRecordingBackend("replace-a")
        keep = ShutdownRecordingBackend("keep")
        pool = ReplicaPool(
            {
                "remove-z": remove_z,
                "replace-b": replace_b,
                "remove-a": remove_a,
                "replace-a": replace_a,
                "keep": keep,
            }
        )

        def config(address):
            return registry_module.ReplicaConfig(address=address, model="llama")

        source = MutableDiscovery(
            {
                "remove-z": config("http://remove-z/v1"),
                "replace-b": config("http://replace-b-old/v1"),
                "remove-a": config("http://remove-a/v1"),
                "replace-a": config("http://replace-a-old/v1"),
                "keep": config("http://keep/v1"),
            }
        )
        factory_calls = []
        candidates = {}

        def ordered_factory(identity):
            factory_calls.append(identity.address)
            if identity.address == "http://replace-a-new/v1":
                assert pool._entries["replace-a"].backend is replace_a
                assert pool.is_draining("replace-a") is False
            if identity.address == "http://replace-b-new/v1":
                assert pool._entries["replace-b"].backend is replace_b
                assert pool.is_draining("replace-b") is False
            candidate = ShutdownRecordingBackend(identity.address)
            candidates[identity.address] = candidate
            return candidate, None

        reconciler = PoolReconciler(pool, source, factory=ordered_factory)
        await reconciler.reconcile()
        source.members = {
            "replace-a": config("http://replace-a-new/v1"),
            "new-b": config("http://new-b/v1"),
            "replace-b": config("http://replace-b-new/v1"),
            "new-a": config("http://new-a/v1"),
            "keep": config("http://keep/v1"),
        }

        result = await reconciler.reconcile()

        assert factory_calls == [
            "http://new-b/v1",
            "http://new-a/v1",
            "http://replace-a-new/v1",
            "http://replace-b-new/v1",
        ]
        assert result == {
            "added": ["new-b", "new-a", "replace-a", "replace-b"],
            "draining": ["remove-z", "remove-a"],
            "removed": ["replace-a", "replace-b", "remove-z", "remove-a"],
        }
        assert remove_z.shutdown_calls == 1
        assert replace_b.shutdown_calls == 1
        assert remove_a.shutdown_calls == 1
        assert replace_a.shutdown_calls == 1
        assert keep.shutdown_calls == 0
        assert all(candidate.shutdown_calls == 0 for candidate in candidates.values())

    async def test_desired_removal_drains_and_shuts_down_backend_once(self):
        old = ShutdownRecordingBackend("old")
        pool = ReplicaPool({"replica": old})
        source = MutableDiscovery(
            {
                "replica": registry_module.ReplicaConfig(
                    address="http://replica/v1", model="llama"
                )
            }
        )
        reconciler = PoolReconciler(pool, source)
        await reconciler.reconcile()
        source.members.clear()

        result = await reconciler.reconcile()

        assert result == {
            "added": [],
            "draining": ["replica"],
            "removed": ["replica"],
        }
        assert "replica" not in pool.replica_ids
        assert reconciler._draining == {}
        assert old.shutdown_calls == 1

    async def test_missing_model_validation_precedes_all_pool_mutation(self):
        seed = ShutdownRecordingBackend("seed")
        pool = ReplicaPool({"seed": seed})
        source = MutableDiscovery(
            {
                "valid": registry_module.ReplicaConfig(
                    address="http://valid/v1", model="llama"
                ),
                "invalid": registry_module.ReplicaConfig(
                    address="http://invalid/v1", model=None
                ),
            }
        )
        factory_calls = []
        reconciler = PoolReconciler(
            pool,
            source,
            factory=lambda identity: (factory_calls.append(identity) or MockBackend(), None),
        )

        with pytest.raises(ValueError, match="'invalid' requires a model"):
            await reconciler.reconcile()

        assert pool.replica_ids == ("seed",)
        assert pool.is_draining("seed") is False
        assert factory_calls == []
        assert reconciler._applied == {}

    async def test_add_failure_cleans_unused_candidate_and_does_not_record_identity(
        self, monkeypatch
    ):
        seed = ShutdownRecordingBackend("seed")
        candidate = ShutdownRecordingBackend("candidate")
        pool = ReplicaPool({"seed": seed})
        source = MutableDiscovery(
            {
                "replica": registry_module.ReplicaConfig(
                    address="http://replica/v1", model="llama"
                )
            }
        )
        reconciler = PoolReconciler(
            pool, source, factory=lambda identity: (candidate, None)
        )

        def reject_add(replica_id, backend, health_url=None):
            raise ValueError("add failed")

        monkeypatch.setattr(pool, "add_replica", reject_add)

        with pytest.raises(ValueError, match="add failed"):
            await reconciler.reconcile()

        assert pool.replica_ids == ("seed",)
        assert reconciler._applied == {}
        assert candidate.shutdown_calls == 1

    async def test_removal_shutdown_failure_does_not_leave_stale_applied_identity(self):
        old = ShutdownRecordingBackend("old", fail_shutdown=True)
        pool = ReplicaPool({"replica": old})
        source = MutableDiscovery(
            {
                "replica": registry_module.ReplicaConfig(
                    address="http://replica/v1", model="llama"
                )
            }
        )
        reconciler = PoolReconciler(pool, source)
        await reconciler.reconcile()
        source.members.clear()

        with pytest.raises(ExceptionGroup, match="replica 'replica' shutdown failed"):
            await reconciler.reconcile()

        assert "replica" not in pool.replica_ids
        assert "replica" not in reconciler._applied
        assert reconciler._draining == {}
        assert old.shutdown_calls == 1


class TestTracing:
    async def test_spans_recorded_when_enabled(self):
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        from kairyu.telemetry import configure_tracing, traced_span

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        configure_tracing(True)
        try:
            with traced_span("kairyu.pool.place", {"replica_id": "3", "reason": "x"}):
                pass
        finally:
            configure_tracing(False)
        spans = exporter.get_finished_spans()
        assert [s.name for s in spans] == ["kairyu.pool.place"]
        assert spans[0].attributes["replica_id"] == "3"

    async def test_disabled_is_noop_without_otel_import(self):
        from kairyu.telemetry import traced_span, tracing_enabled

        assert not tracing_enabled()
        with traced_span("anything") as span:
            assert span is None


async def test_kind_smoke_gates_default_and_gpu_chart_before_cluster_creation():
    lines = [
        line.strip()
        for line in Path("scripts/kind_smoke.sh").read_text(encoding="utf-8").splitlines()
    ]
    commands = [
        "helm lint deploy/helm/kairyu",
        (
            "helm lint deploy/helm/kairyu "
            "-f deploy/helm/kairyu/values-gpu.yaml"
        ),
        "helm template kairyu deploy/helm/kairyu >/dev/null",
        (
            "helm template kairyu deploy/helm/kairyu "
            "-f deploy/helm/kairyu/values-gpu.yaml >/dev/null"
        ),
    ]

    assert "set -euo pipefail" in lines
    for command in commands:
        assert lines.count(command) == 1
    command_positions = [lines.index(command) for command in commands]
    gate_call = lines.index("helm_gate")
    kind_create = lines.index('kind create cluster --name "$CLUSTER" --wait 120s')
    assert command_positions == sorted(command_positions)
    assert max(command_positions) < gate_call
    assert gate_call < kind_create
    assert 'if [[ "${1:-}" == "--helm-check" ]]; then' in lines[gate_call:kind_create]


async def test_helm_check_exits_after_four_helm_commands_without_cluster_side_effects(
    tmp_path,
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    call_log = tmp_path / "calls.log"

    helm = bin_dir / "helm"
    helm.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "helm $*" >>"$CALL_LOG"\n',
        encoding="utf-8",
    )
    helm.chmod(0o755)

    forbidden_command = (
        "#!/usr/bin/env bash\n"
        'name=${0##*/}\n'
        'printf \'%s\\n\' "$name $*" >>"$CALL_LOG"\n'
        "exit 97\n"
    )
    for name in ("kind", "docker", "kubectl", "curl"):
        command = bin_dir / name
        command.write_text(forbidden_command, encoding="utf-8")
        command.chmod(0o755)

    env = os.environ.copy()
    env["CALL_LOG"] = str(call_log)
    env["PATH"] = os.pathsep.join((str(bin_dir), env["PATH"]))
    result = subprocess.run(
        ["bash", "scripts/kind_smoke.sh", "--helm-check"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert call_log.read_text(encoding="utf-8").splitlines() == [
        "helm lint deploy/helm/kairyu",
        "helm lint deploy/helm/kairyu -f deploy/helm/kairyu/values-gpu.yaml",
        "helm template kairyu deploy/helm/kairyu",
        (
            "helm template kairyu deploy/helm/kairyu "
            "-f deploy/helm/kairyu/values-gpu.yaml"
        ),
    ]


async def test_ci_has_explicit_single_source_helm_schema_and_gpu_template_gate():
    workflow_text = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    workflow = yaml.safe_load(workflow_text)
    steps = workflow["jobs"]["kind-smoke"]["steps"]
    run_commands = [step.get("run") for step in steps]

    assert run_commands.index(
        "bash scripts/kind_smoke.sh --helm-check"
    ) < run_commands.index("bash scripts/kind_smoke.sh")
    assert "helm lint" not in workflow_text
    assert "helm template" not in workflow_text


@pytest.mark.helm
def test_helm_chart_renders():
    rendered = subprocess.run(
        ["helm", "template", "kairyu", "deploy/helm/kairyu"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "kind: Deployment" in rendered
    assert "path: /readyz" in rendered
    assert "mountPath: /etc/kairyu" in rendered  # the Dockerfile CMD path (A11)
    deployment = next(
        document
        for document in yaml.safe_load_all(rendered)
        if document and document.get("kind") == "Deployment"
    )
    pod_spec = deployment["spec"]["template"]["spec"]
    assert deployment["spec"]["selector"]["matchLabels"] == {"app": "kairyu"}
    assert deployment["spec"]["template"]["metadata"]["labels"]["app"] == "kairyu"
    assert pod_spec["automountServiceAccountToken"] is True
    assert pod_spec["securityContext"]["seccompProfile"]["type"] == "RuntimeDefault"
    assert "nodeSelector" not in pod_spec
    assert "runtimeClassName" not in pod_spec
    assert "tolerations" not in pod_spec
    assert "affinity" not in pod_spec
    container = pod_spec["containers"][0]
    assert container["securityContext"]["allowPrivilegeEscalation"] is False
    assert container["securityContext"]["capabilities"]["drop"] == ["ALL"]
    assert container["startupProbe"]["httpGet"]["path"] == "/health"
    assert all(
        variable["name"] != "KAIRYU_ATTENTION_BACKEND"
        for variable in container.get("env", [])
    )
    assert all(mount["name"] != "model-storage" for mount in container["volumeMounts"])
    assert all(volume["name"] != "model-storage" for volume in pod_spec["volumes"])


@pytest.mark.helm
def test_helm_chart_renders_placement_and_runtime_controls(tmp_path):
    node_selector = {
        "kairyu.ai/accelerator": "nvidia",
        "kubernetes.io/arch": "amd64",
    }
    tolerations = [
        {
            "key": "nvidia.com/gpu",
            "operator": "Exists",
            "effect": "NoSchedule",
        }
    ]
    affinity = {
        "nodeAffinity": {
            "requiredDuringSchedulingIgnoredDuringExecution": {
                "nodeSelectorTerms": [
                    {
                        "matchExpressions": [
                            {
                                "key": "kairyu.ai/gpu-model",
                                "operator": "In",
                                "values": ["RTX-PRO-6000"],
                            }
                        ]
                    }
                ]
            }
        }
    }
    override = tmp_path / "placement.yaml"
    override.write_text(
        yaml.safe_dump(
            {
                "nodeSelector": node_selector,
                "runtimeClassName": "nvidia",
                "tolerations": tolerations,
                "affinity": affinity,
            }
        )
    )

    rendered = subprocess.run(
        [
            "helm",
            "template",
            "kairyu",
            "deploy/helm/kairyu",
            "-f",
            str(override),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    deployment = next(
        document
        for document in yaml.safe_load_all(rendered)
        if document and document.get("kind") == "Deployment"
    )
    pod_spec = deployment["spec"]["template"]["spec"]

    assert pod_spec["nodeSelector"] == node_selector
    assert pod_spec["runtimeClassName"] == "nvidia"
    assert pod_spec["tolerations"] == tolerations
    assert pod_spec["affinity"] == affinity


@pytest.mark.parametrize(
    "attention_backend",
    ["auto", "torch", "flashinfer", "flashattention3", "flashattention4"],
)
@pytest.mark.helm
async def test_helm_chart_renders_supported_attention_backend(
    tmp_path, attention_backend
):
    override = tmp_path / "attention-backend.yaml"
    override.write_text(
        yaml.safe_dump({"attentionBackend": attention_backend}),
        encoding="utf-8",
    )
    rendered = subprocess.run(
        [
            "helm",
            "template",
            "kairyu",
            "deploy/helm/kairyu",
            "-f",
            str(override),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    deployment = next(
        document
        for document in yaml.safe_load_all(rendered)
        if document and document.get("kind") == "Deployment"
    )
    container = deployment["spec"]["template"]["spec"]["containers"][0]

    assert container["env"] == [
        {
            "name": "KAIRYU_ATTENTION_BACKEND",
            "value": attention_backend,
        }
    ]


def test_helm_chart_config_is_a_valid_deployment_spec():
    """kind-smoke root cause (PR #16): the chart shipped 'models:' which is
    not a DeploymentSpec field — the pod crash-looped at validation. Pin the
    embedded config to the real schema, no helm binary needed."""
    from kairyu.deploy.spec import load_deployment_spec

    values = yaml.safe_load(open("deploy/helm/kairyu/values.yaml"))
    spec = load_deployment_spec(values["config"])
    assert spec.engines, "chart config must declare at least one engine"


def test_helm_gpu_values_define_real_engine_and_model_storage():
    from kairyu.deploy.spec import load_deployment_spec

    chart_dir = Path("deploy/helm/kairyu")
    defaults = yaml.safe_load((chart_dir / "values.yaml").read_text())
    gpu_values = yaml.safe_load((chart_dir / "values-gpu.yaml").read_text())

    assert defaults["modelStorage"] == {
        "enabled": False,
        "pvcName": "",
        "hostPath": "",
        "mountPath": "/models",
    }
    assert gpu_values["modelStorage"] == {
        "enabled": True,
        "pvcName": "",
        "hostPath": "/models",
        "mountPath": "/models",
    }

    spec = load_deployment_spec(gpu_values["config"])
    engine = spec.engines["default"]
    assert engine.backend == "kairyu"
    assert engine.backend != "mock"
    model_path = PurePosixPath(engine.options["model_path"])
    assert model_path.is_relative_to(PurePosixPath(gpu_values["modelStorage"]["mountPath"]))

    template = (chart_dir / "templates/deployment.yaml").read_text()
    assert template.count("{{- if .Values.modelStorage.enabled }}") == 2
    assert ".Values.modelStorage.pvcName" in template
    assert ".Values.modelStorage.hostPath" in template
    assert ".Values.modelStorage.mountPath" in template

    schema = json.loads((chart_dir / "values.schema.json").read_text())
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == set(defaults)
    assert set(schema["required"]) == set(defaults)
    assert schema["properties"]["replicaCount"]["minimum"] == 1
    gpu_schema = schema["definitions"]["resourceList"]["properties"]["nvidia.com/gpu"]
    assert gpu_schema == {"type": "integer", "minimum": 1}
    storage_schema = schema["properties"]["modelStorage"]
    assert storage_schema["additionalProperties"] is False
    assert set(storage_schema["required"]) == {
        "enabled",
        "pvcName",
        "hostPath",
        "mountPath",
    }
    enabled_rule = storage_schema["allOf"][0]
    assert enabled_rule["if"]["properties"]["enabled"]["const"] is True
    assert len(enabled_rule["then"]["oneOf"]) == 2
    assert storage_schema["properties"]["mountPath"]["pattern"] == "^/"
    assert storage_schema["properties"]["hostPath"]["anyOf"] == [
        {"const": ""},
        {"pattern": "^/"},
    ]


async def test_helm_attention_backend_values_schema_and_template_are_strict():
    chart_dir = Path("deploy/helm/kairyu")
    defaults = yaml.safe_load((chart_dir / "values.yaml").read_text())
    gpu_values = yaml.safe_load((chart_dir / "values-gpu.yaml").read_text())
    schema = json.loads((chart_dir / "values.schema.json").read_text())
    template = (chart_dir / "templates/deployment.yaml").read_text()

    assert defaults["attentionBackend"] == ""
    assert gpu_values["attentionBackend"] == "auto"
    assert schema["properties"]["attentionBackend"] == {
        "type": "string",
        "enum": [
            "",
            "auto",
            "torch",
            "flashinfer",
            "flashattention3",
            "flashattention4",
        ],
    }
    assert "attentionBackend" in schema["required"]
    assert "{{- with .Values.attentionBackend }}" in template
    assert "name: KAIRYU_ATTENTION_BACKEND" in template
    assert "value: {{ . | quote }}" in template


async def test_helm_shared_batch_store_uses_secret_dsn_and_immutable_pod_uid():
    chart_dir = Path("deploy/helm/kairyu")
    defaults = yaml.safe_load((chart_dir / "values.yaml").read_text())
    schema = json.loads((chart_dir / "values.schema.json").read_text())
    template = (chart_dir / "templates/deployment.yaml").read_text()

    assert defaults["batchPostgres"] == {
        "enabled": False,
        "secretName": "",
        "secretKey": "dsn",
        "dsnEnvName": "KAIRYU_BATCH_POSTGRES_DSN",
    }
    batch_schema = schema["properties"]["batchPostgres"]
    assert batch_schema["additionalProperties"] is False
    assert batch_schema["required"] == [
        "enabled",
        "secretName",
        "secretKey",
        "dsnEnvName",
    ]
    assert "batchPostgres" in schema["required"]
    assert ".Values.batchPostgres.secretName" in template
    assert ".Values.batchPostgres.secretKey" in template
    assert ".Values.batchPostgres.dsnEnvName" in template
    assert "name: KAIRYU_BATCH_WORKER_ID" in template
    assert "fieldPath: metadata.uid" in template


def test_helm_operability_defaults_and_schema_are_strict():
    chart_dir = Path("deploy/helm/kairyu")
    defaults = yaml.safe_load((chart_dir / "values.yaml").read_text())
    schema = json.loads((chart_dir / "values.schema.json").read_text())
    deployment = (chart_dir / "templates/deployment.yaml").read_text()

    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == set(defaults)
    assert set(schema["required"]) == set(defaults)
    assert defaults["image"]["digest"] == ""
    assert schema["properties"]["image"]["properties"]["digest"]["pattern"] == (
        "^$|^sha256:[a-f0-9]{64}$"
    )
    assert defaults["automountServiceAccountToken"] is True
    assert defaults["podSecurityContext"] == {
        "seccompProfile": {"type": "RuntimeDefault"}
    }
    seccomp_schema = schema["properties"]["podSecurityContext"]["properties"][
        "seccompProfile"
    ]
    assert seccomp_schema["allOf"][0]["then"]["required"] == ["localhostProfile"]
    assert seccomp_schema["allOf"][0]["else"] == {
        "not": {"required": ["localhostProfile"]}
    }
    assert defaults["securityContext"]["allowPrivilegeEscalation"] is False
    assert defaults["securityContext"]["capabilities"]["drop"] == ["ALL"]
    assert defaults["probes"]["startup"]["failureThreshold"] == 90
    assert defaults["gracefulDrain"] == {
        "enabled": False,
        "propagationSeconds": 10,
        "auth": {"secretName": "", "secretKey": "admin-key"},
    }
    assert 'include "kairyu.image"' in deployment
    assert "checksum/config" in deployment
    assert "startupProbe:" in deployment
    assert "http://127.0.0.1:8000/admin/drain" in deployment
    assert "KAIRYU_HELM_DRAIN_TOKEN" in deployment
    assert "urllib.request" in deployment
    assert 'or "").strip()' in deployment
    reserved_labels = schema["definitions"]["customLabelMap"]["propertyNames"][
        "not"
    ]["enum"]
    assert "app.kubernetes.io/instance" in reserved_labels
    assert "app" in reserved_labels
    reserved_annotations = schema["definitions"]["podAnnotationMap"][
        "propertyNames"
    ]["not"]["enum"]
    assert reserved_annotations == ["checksum/config"]


@pytest.mark.helm
def test_helm_chart_renders_digest_drain_and_service_monitor(tmp_path):
    digest = "sha256:" + "a" * 64
    override = tmp_path / "operability.yaml"
    override.write_text(
        yaml.safe_dump(
            {
                "image": {
                    "repository": "harbor.private-ai.internal/turnkey-ai/kairyu",
                    "digest": digest,
                },
                "workloadRole": "replica",
                "strategyType": "Recreate",
                "gracefulDrain": {"enabled": True},
                "serviceMonitor": {
                    "enabled": True,
                    "labels": {"release": "kube-prometheus-stack"},
                },
            }
        ),
        encoding="utf-8",
    )
    rendered = subprocess.run(
        [
            "helm",
            "template",
            "kairyu-replica",
            "deploy/helm/kairyu",
            "-f",
            str(override),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    documents = [document for document in yaml.safe_load_all(rendered) if document]
    deployment = next(document for document in documents if document["kind"] == "Deployment")
    monitor = next(document for document in documents if document["kind"] == "ServiceMonitor")

    assert deployment["spec"]["strategy"]["type"] == "Recreate"
    assert deployment["spec"]["template"]["metadata"]["annotations"]["checksum/config"]
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    assert container["image"] == f"harbor.private-ai.internal/turnkey-ai/kairyu@{digest}"
    assert container["lifecycle"]["preStop"]["exec"]["command"][0] == "/bin/sh"
    assert monitor["spec"]["endpoints"][0]["path"] == "/metrics"
    assert monitor["metadata"]["labels"]["release"] == "kube-prometheus-stack"


@pytest.mark.helm
def test_helm_chart_renders_authenticated_graceful_drain(tmp_path):
    override = tmp_path / "authenticated-drain.yaml"
    override.write_text(
        yaml.safe_dump(
            {
                "gracefulDrain": {
                    "enabled": True,
                    "auth": {
                        "secretName": "kairyu-admin",
                        "secretKey": "token",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    rendered = subprocess.run(
        [
            "helm",
            "template",
            "kairyu",
            "deploy/helm/kairyu",
            "-f",
            str(override),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    deployment = next(
        document
        for document in yaml.safe_load_all(rendered)
        if document and document.get("kind") == "Deployment"
    )
    drain_env = next(
        item
        for item in deployment["spec"]["template"]["spec"]["containers"][0]["env"]
        if item["name"] == "KAIRYU_HELM_DRAIN_TOKEN"
    )
    assert drain_env["valueFrom"]["secretKeyRef"] == {
        "name": "kairyu-admin",
        "key": "token",
    }


@pytest.mark.helm
def test_helm_chart_rejects_insufficient_drain_grace(tmp_path):
    invalid_values = tmp_path / "insufficient-drain-grace.yaml"
    invalid_values.write_text(
        yaml.safe_dump(
            {
                "terminationGracePeriodSeconds": 19,
                "gracefulDrain": {
                    "enabled": True,
                    "propagationSeconds": 10,
                },
            }
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            "helm",
            "template",
            "kairyu",
            "deploy/helm/kairyu",
            "-f",
            str(invalid_values),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "gracefulDrain.propagationSeconds + 10" in result.stderr


@pytest.mark.helm
@pytest.mark.parametrize(
    "override",
    [
        {"podLabels": {"app.kubernetes.io/instance": "wrong"}},
        {"podAnnotations": {"checksum/config": "disabled"}},
    ],
)
def test_helm_schema_rejects_reserved_pod_metadata(tmp_path, override):
    invalid_values = tmp_path / "reserved-pod-metadata.yaml"
    invalid_values.write_text(yaml.safe_dump(override), encoding="utf-8")
    result = subprocess.run(
        [
            "helm",
            "template",
            "kairyu",
            "deploy/helm/kairyu",
            "-f",
            str(invalid_values),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


@pytest.mark.helm
def test_helm_gpu_values_render_real_engine_and_model_storage():
    rendered = subprocess.run(
        [
            "helm",
            "template",
            "kairyu",
            "deploy/helm/kairyu",
            "-f",
            "deploy/helm/kairyu/values-gpu.yaml",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    documents = [document for document in yaml.safe_load_all(rendered) if document]
    deployment = next(document for document in documents if document.get("kind") == "Deployment")
    configmap = next(document for document in documents if document.get("kind") == "ConfigMap")

    pod_spec = deployment["spec"]["template"]["spec"]
    assert pod_spec["runtimeClassName"] == "nvidia"
    assert pod_spec["nodeSelector"] == {"kairyu.dev/gpu-profile": "pcie-gddr"}

    container = pod_spec["containers"][0]
    assert container["resources"]["limits"]["nvidia.com/gpu"] == 1
    assert container["env"] == [
        {
            "name": "KAIRYU_ATTENTION_BACKEND",
            "value": "auto",
        }
    ]
    model_mount = next(
        mount for mount in container["volumeMounts"] if mount["mountPath"] == "/models"
    )
    assert model_mount["readOnly"] is True
    model_volume = next(
        volume for volume in pod_spec["volumes"] if volume["name"] == model_mount["name"]
    )
    assert model_volume["hostPath"]["path"] == "/models"
    assert "persistentVolumeClaim" not in model_volume

    config = yaml.safe_load(configmap["data"]["config.yaml"])
    engine = config["engines"]["default"]
    assert engine["backend"] == "kairyu"
    assert engine["backend"] != "mock"
    model_path = PurePosixPath(engine["options"]["model_path"])
    assert model_path.is_relative_to(PurePosixPath(model_mount["mountPath"]))


@pytest.mark.helm
def test_helm_model_storage_can_render_an_existing_pvc(tmp_path):
    pvc_values = tmp_path / "pvc-model-storage.yaml"
    pvc_values.write_text(
        yaml.safe_dump(
            {
                "modelStorage": {
                    "pvcName": "kairyu-models",
                    "hostPath": "",
                }
            }
        )
    )
    rendered = subprocess.run(
        [
            "helm",
            "template",
            "kairyu",
            "deploy/helm/kairyu",
            "-f",
            "deploy/helm/kairyu/values-gpu.yaml",
            "-f",
            str(pvc_values),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    deployment = next(
        document
        for document in yaml.safe_load_all(rendered)
        if document and document.get("kind") == "Deployment"
    )
    pod_spec = deployment["spec"]["template"]["spec"]
    model_volume = next(
        volume for volume in pod_spec["volumes"] if volume["name"] == "model-storage"
    )
    assert model_volume["persistentVolumeClaim"]["claimName"] == "kairyu-models"
    assert "hostPath" not in model_volume
    model_mount = next(
        mount
        for mount in pod_spec["containers"][0]["volumeMounts"]
        if mount["name"] == "model-storage"
    )
    assert model_mount == {
        "name": "model-storage",
        "mountPath": "/models",
        "readOnly": True,
    }


@pytest.mark.parametrize(
    "model_storage",
    [
        {
            "enabled": True,
            "pvcName": "",
            "hostPath": "",
            "mountPath": "/models",
        },
        {
            "enabled": True,
            "pvcName": "kairyu-models",
            "hostPath": "/models",
            "mountPath": "/models",
        },
    ],
    ids=["without-source", "pvc-and-host-path"],
)
@pytest.mark.helm
def test_helm_schema_rejects_invalid_model_storage(tmp_path, model_storage):
    invalid_values = tmp_path / "invalid-model-storage.yaml"
    invalid_values.write_text(yaml.safe_dump({"modelStorage": model_storage}))
    result = subprocess.run(
        [
            "helm",
            "template",
            "kairyu",
            "deploy/helm/kairyu",
            "-f",
            "deploy/helm/kairyu/values-gpu.yaml",
            "-f",
            str(invalid_values),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
