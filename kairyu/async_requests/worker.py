"""Lease-fenced worker for durable asynchronous chat requests."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket
import uuid
from collections.abc import Mapping
from collections.abc import Set as AbstractSet

from pydantic import ValidationError

from kairyu.async_requests.models import (
    AsyncRequest,
    AsyncRequestError,
    RequestClaim,
)
from kairyu.async_requests.store import (
    RequestStoreProtocol,
    StaleRequestClaimError,
)
from kairyu.engine.backend import (
    EngineBackend,
    UpstreamClientError,
    backend_admission_upper_bound_async,
    prepare_backend_request,
)
from kairyu.entrypoints.chat_template import ChatTemplate
from kairyu.entrypoints.server.chat_service import (
    ChatRequestError,
    ExecutedChat,
    chat_error_from_upstream_client_error,
    execute_chat,
    validate_chat_policy,
    validate_chat_request_async,
)
from kairyu.entrypoints.server.metering import (
    TokenLimiterSink,
    UsageLedgerSink,
    record_tenant_usage,
)
from kairyu.entrypoints.server.protocol import ChatCompletionRequest
from kairyu.entrypoints.server.tenancy import TenantConfig

logger = logging.getLogger("kairyu.async_requests.worker")
_WORKER_ID_ENV = "KAIRYU_ASYNC_REQUEST_WORKER_ID"
_PRESSURE_POLL_S = 0.01
_TENANT_DEFER_S = 0.5
_MAX_SHUTDOWN_DRAIN_S = 10.0


class _TransientAdmission(RuntimeError):
    def __init__(self, reason: str, retry_after_s: float = _TENANT_DEFER_S) -> None:
        super().__init__(reason)
        self.reason = reason
        self.retry_after_s = max(_TENANT_DEFER_S, retry_after_s)


def _request_error(
    *,
    code: str,
    message: str,
    retryable: bool = False,
) -> AsyncRequestError:
    """Bound terminal details before persistence or public projection."""
    bounded_code = str(code)[:128] or "request_error"
    bounded_message = str(message)[:1024] or "request failed"
    return AsyncRequestError(
        code=bounded_code,
        message=bounded_message,
        retryable=retryable,
    )


class AsyncRequestWorker:
    """Poll a shared RequestStore and execute claimed Chat Completions work."""

    def __init__(
        self,
        store: RequestStoreProtocol,
        engines: Mapping[str, EngineBackend],
        *,
        max_concurrency: int = 4,
        poll_interval_s: float = 0.5,
        lease_seconds: float = 30.0,
        worker_id: str | None = None,
        metrics=None,
        chat_templates: Mapping[str, ChatTemplate] | None = None,
        legacy_chat_models: AbstractSet[str] | None = None,
        usage_ledger: UsageLedgerSink | None = None,
        tenant_limiter: TokenLimiterSink | None = None,
        tenant_config: TenantConfig | None = None,
        admission_controller=None,
    ) -> None:
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        if poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be positive")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        legacy_chat_models = frozenset(legacy_chat_models or ())
        validate_chat_policy(chat_templates, legacy_chat_models)
        self._store = store
        self._engines = dict(engines)
        self._max_concurrency = max_concurrency
        self._poll_interval_s = poll_interval_s
        self._lease_seconds = lease_seconds
        self._worker_id = (
            worker_id
            or os.environ.get(_WORKER_ID_ENV)
            or f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:12]}"
        )
        self._metrics = metrics
        self._chat_templates = chat_templates
        self._legacy_chat_models = legacy_chat_models
        self._usage_ledger = usage_ledger
        self._tenant_limiter = tenant_limiter
        self._tenant_config = tenant_config or TenantConfig()
        self._admission_controller = admission_controller
        self._wakeup = asyncio.Event()
        self._active_cancellations: dict[str, asyncio.Event] = {}

    def submit(self, request_id: str) -> None:
        """Wake local consumers; PostgreSQL remains the queue authority."""
        del request_id
        self._wakeup.set()

    def supports_model(self, model: str) -> bool:
        return model in self._engines

    def queue_priority(self, owner: str) -> int:
        """Return the trusted tenant policy used by the shared store queue."""
        return self._tenant_config.limits_for(owner).batch_priority

    def notify_cancel(self, request_id: str) -> None:
        """Abort a locally executing request; remote owners observe fencing."""
        cancellation = self._active_cancellations.get(request_id)
        if cancellation is not None:
            cancellation.set()
        self._wakeup.set()

    async def run(self) -> None:
        """Run a fixed consumer pool until the application lifespan cancels it."""
        shutdown = asyncio.Event()
        consumers = [
            asyncio.create_task(self._consume(shutdown))
            for _ in range(self._max_concurrency)
        ]
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            shutdown.set()
            self._wakeup.set()
            for cancellation in self._active_cancellations.values():
                cancellation.set()
            done, pending = await asyncio.wait(
                consumers,
                timeout=min(_MAX_SHUTDOWN_DRAIN_S, max(1.0, self._lease_seconds)),
            )
            await asyncio.gather(*done, return_exceptions=True)
            if pending:
                logger.warning(
                    "async request consumers exceeded shutdown drain timeout",
                    extra={"pending_consumers": len(pending)},
                )
                for consumer in pending:
                    consumer.cancel()
                done_after_cancel, _still_pending = await asyncio.wait(
                    pending,
                    timeout=0.5,
                )
                await asyncio.gather(*done_after_cancel, return_exceptions=True)
            raise

    async def _consume(self, shutdown: asyncio.Event) -> None:
        while not shutdown.is_set():
            self._wakeup.clear()
            try:
                processed = await self.process_next(shutdown=shutdown)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("async request consumer failed")
                processed = False
            if processed:
                continue
            wake = asyncio.create_task(self._wakeup.wait())
            poll = asyncio.create_task(asyncio.sleep(self._poll_interval_s))
            stop = asyncio.create_task(shutdown.wait())
            try:
                await asyncio.wait(
                    (wake, poll, stop),
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                wake.cancel()
                poll.cancel()
                stop.cancel()
                await asyncio.gather(wake, poll, stop, return_exceptions=True)

    async def process_next(self, *, shutdown: asyncio.Event | None = None) -> bool:
        claim = await asyncio.to_thread(
            self._store.claim_next,
            self._worker_id,
            lease_seconds=self._lease_seconds,
        )
        if claim is None:
            return False
        if shutdown is not None and shutdown.is_set():
            # The claim remains fenced and becomes reclaimable after its lease;
            # shutdown must not begin new inference or publish through it.
            return True
        await self.process(claim)
        return True

    async def process(self, claim: RequestClaim) -> None:
        """Execute one claim and publish only through its current fence."""
        locally_cancelled = asyncio.Event()
        self._active_cancellations[claim.request_id] = locally_cancelled
        try:
            try:
                running = await asyncio.to_thread(self._store.mark_running, claim)
            except StaleRequestClaimError:
                return
            updated_claim = await asyncio.to_thread(
                claim.model_copy,
                update={"request": running},
                deep=True,
            )
            claim_state = [updated_claim]
            claim_lost = asyncio.Event()
            heartbeat = asyncio.create_task(self._heartbeat(claim_state, claim_lost))
            try:
                if not await self._wait_for_capacity(claim_lost, locally_cancelled):
                    return
                dispatch = asyncio.create_task(self._dispatch(running))
                lost_wait = asyncio.create_task(claim_lost.wait())
                cancel_wait = asyncio.create_task(locally_cancelled.wait())
                try:
                    await asyncio.wait(
                        (dispatch, lost_wait, cancel_wait),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if claim_lost.is_set() or locally_cancelled.is_set():
                        dispatch.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await dispatch
                        return
                    try:
                        executed = await dispatch
                    except _TransientAdmission as deferred:
                        try:
                            await asyncio.to_thread(
                                self._store.defer,
                                claim_state[0],
                                delay_seconds=deferred.retry_after_s,
                            )
                        except StaleRequestClaimError:
                            pass
                    except ValidationError:
                        await self._publish_failure(
                            claim_state[0],
                            _request_error(
                                code="invalid_request",
                                message="request payload failed validation",
                            ),
                        )
                    except ChatRequestError as error:
                        await self._publish_failure(
                            claim_state[0],
                            _request_error(
                                code=error.code,
                                message=str(error),
                                retryable=error.status_code >= 500,
                            ),
                        )
                    except Exception as error:
                        logger.error(
                            "async request backend error (%s)",
                            type(error).__name__,
                            extra={"request_id": running.id},
                        )
                        await self._publish_failure(
                            claim_state[0],
                            _request_error(
                                code="backend_error",
                                message=f"upstream backend error ({type(error).__name__})",
                                retryable=True,
                            ),
                        )
                    else:
                        try:
                            await asyncio.to_thread(
                                self._store.succeed,
                                claim_state[0],
                                await asyncio.to_thread(
                                    executed.response.model_dump,
                                    mode="json",
                                ),
                            )
                        except StaleRequestClaimError:
                            pass
                finally:
                    if not dispatch.done():
                        dispatch.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await dispatch
                    for task in (lost_wait, cancel_wait):
                        task.cancel()
                    for task in (lost_wait, cancel_wait):
                        with contextlib.suppress(asyncio.CancelledError):
                            await task
            finally:
                heartbeat.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat
        finally:
            if self._active_cancellations.get(claim.request_id) is locally_cancelled:
                self._active_cancellations.pop(claim.request_id, None)

    async def _heartbeat(
        self,
        claim_state: list[RequestClaim],
        claim_lost: asyncio.Event,
    ) -> None:
        while True:
            remaining = max(
                0.0,
                (
                    claim_state[0].lease_until
                    - claim_state[0].request.updated_at
                ).total_seconds(),
            )
            interval = max(0.01, min(self._lease_seconds, remaining) / 3.0)
            await asyncio.sleep(interval)
            try:
                claim_state[0] = await asyncio.to_thread(
                    self._store.renew_claim,
                    claim_state[0],
                    lease_seconds=self._lease_seconds,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("async request lease lost", exc_info=True)
                claim_lost.set()
                return

    async def _wait_for_capacity(
        self,
        claim_lost: asyncio.Event,
        locally_cancelled: asyncio.Event,
    ) -> bool:
        controller = self._admission_controller
        while controller is not None and controller.batch_should_yield():
            if claim_lost.is_set() or locally_cancelled.is_set():
                return False
            await asyncio.sleep(_PRESSURE_POLL_S)
        return not claim_lost.is_set() and not locally_cancelled.is_set()

    async def _dispatch(self, request: AsyncRequest) -> ExecutedChat:
        if request.endpoint != "/v1/chat/completions":
            raise ChatRequestError(
                "async request endpoint is not supported",
                code="unsupported_endpoint",
            )
        chat_request = await asyncio.to_thread(
            ChatCompletionRequest.model_validate,
            request.body,
        )
        if chat_request.stream:
            raise ChatRequestError(
                "async chat completions do not support stream=true",
                code="stream_not_supported",
            )
        validated = await validate_chat_request_async(
            chat_request,
            self._engines,
            self._chat_templates,
            request_id=request.id,
            priority=self._tenant_config.limits_for(request.owner).batch_priority,
            scheduling_class="batch",
            legacy_chat_models=self._legacy_chat_models,
        )
        admission = None
        metric_admitted = False
        quota_accounted = False
        executed: ExecutedChat | None = None
        try:
            await prepare_backend_request(
                validated.engine,
                validated.generation_request,
            )
        except UpstreamClientError as error:
            raise chat_error_from_upstream_client_error(error) from error
        except ValueError as error:
            raise ChatRequestError(
                str(error),
                code=getattr(error, "code", "invalid_request"),
            ) from error
        try:
            bound = await backend_admission_upper_bound_async(
                validated.engine,
                validated.generation_request,
            )
        except ValueError as error:
            raise ChatRequestError(
                str(error),
                code=getattr(error, "code", "invalid_request"),
            ) from error
        try:
            record_admission = getattr(self._metrics, "record_tenant_admission", None)
            acquire_reserved = getattr(self._tenant_limiter, "acquire_reserved", None)
            if callable(acquire_reserved):
                admission = acquire_reserved(
                    request.owner,
                    bound.tokens,
                    refundable_on_exact_usage=bound.refundable_on_exact_usage,
                )
                if not admission.admitted:
                    if admission.reason == "token_request_too_large":
                        raise ChatRequestError(
                            "request token bound exceeds the tenant token burst",
                            status_code=400,
                            code="token_request_too_large",
                        )
                    if callable(record_admission):
                        record_admission(
                            request.owner,
                            source="async",
                            admitted=False,
                            reason=admission.reason,
                        )
                    raise _TransientAdmission(
                        admission.reason,
                        admission.retry_after_s or _TENANT_DEFER_S,
                    )
                quota_accounted = True
                if callable(record_admission):
                    record_admission(
                        request.owner,
                        source="async",
                        admitted=True,
                        reason=admission.reason,
                    )
                    metric_admitted = True
            else:
                acquire = getattr(self._tenant_limiter, "acquire", None)
                if callable(acquire):
                    admission = acquire(request.owner)
                if admission is not None and not admission.admitted:
                    raise _TransientAdmission(
                        admission.reason,
                        getattr(admission, "retry_after_s", None) or _TENANT_DEFER_S,
                    )
                reserve = (
                    getattr(admission, "reserve_tokens", None)
                    if admission is not None
                    else None
                )
                if callable(reserve):
                    quota_accounted = reserve(
                        bound.tokens,
                        refundable_on_exact_usage=bound.refundable_on_exact_usage,
                    )
                if admission is not None and callable(record_admission):
                    record_admission(
                        request.owner,
                        source="async",
                        admitted=quota_accounted or not callable(reserve),
                        reason=admission.reason,
                    )
                    metric_admitted = quota_accounted or not callable(reserve)
                if callable(reserve) and not quota_accounted:
                    raise _TransientAdmission(
                        admission.reason,
                        getattr(admission, "retry_after_s", None) or _TENANT_DEFER_S,
                    )
            if self._metrics is not None:
                self._metrics.record_priority("batch", source="async")
            if quota_accounted:
                admission.mark_dispatched()
            try:
                executed = await execute_chat(validated)
            except ChatRequestError as error:
                if error.execution is not None:
                    self._record_usage(
                        request.owner,
                        error.execution,
                        quota_accounted=quota_accounted,
                    )
                    if quota_accounted:
                        self._settle(admission, error.execution)
                raise
            self._record_usage(
                request.owner,
                executed,
                quota_accounted=quota_accounted,
            )
            if quota_accounted:
                self._settle(admission, executed)
            return executed
        finally:
            if admission is not None and admission.admitted:
                admission.release()
                record_release = getattr(self._metrics, "record_tenant_release", None)
                if callable(record_release) and metric_admitted:
                    record_release(request.owner, source="async")

    def _record_usage(
        self,
        tenant: str,
        executed: ExecutedChat,
        *,
        quota_accounted: bool,
    ) -> None:
        usage = executed.response.usage
        details = usage.prompt_tokens_details
        record_tenant_usage(
            tenant=tenant,
            model=executed.response.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cached_tokens=details.cached_tokens if details is not None else 0,
            ledger=self._usage_ledger,
            metrics=self._metrics,
            limiter=None if quota_accounted else self._tenant_limiter,
        )

    @staticmethod
    def _settle(admission, executed: ExecutedChat) -> None:
        usage = executed.response.usage
        admission.settle_tokens(
            usage.prompt_tokens + usage.completion_tokens,
            exact=executed.result.usage is not None,
        )

    async def _publish_failure(
        self,
        claim: RequestClaim,
        error: AsyncRequestError,
    ) -> None:
        try:
            await asyncio.to_thread(self._store.fail, claim, error)
        except StaleRequestClaimError:
            pass
