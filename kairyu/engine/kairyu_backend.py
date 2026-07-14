"""Kairyu engine core exposed as an EngineBackend (design m2 §1, backend name "kairyu").

Full-stack integration on CPU: OpenAI server / LLM API → this backend →
``EngineLoop`` (tokenizer + Scheduler + RadixKVCache + runner). The model
forward is a deterministic toy runner on CPU; the GPU phase swaps in the real
ModelRunner behind the same protocol — nothing above it changes.

Threading discipline (m8 D1): all scheduler mutations happen inside
``EngineLoop.step()`` on the step thread; the event loop only enqueues ops and
reads queues. The ZMQ process-split backend ("kairyu-proc", m8 D6) drives the
same ``EngineLoop`` from a child process.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping

from kairyu.engine.backend import GenerationRequest, GenerationResult, GenerationUsage
from kairyu.engine.core.comm import FakeCommunicator
from kairyu.engine.core.radix_kv import RadixKVCache
from kairyu.engine.core.sampling_types import SampledToken, mix_seed
from kairyu.engine.core.scheduler import ScheduledChunk, Scheduler
from kairyu.engine.core.spec_runner import SpeculativeRunner
from kairyu.engine.core.tp_runner import TPModelRunner, validate_tp_degree
from kairyu.engine.engine_loop import EngineLoop, StreamUpdate
from kairyu.engine.registry import register_backend
from kairyu.engine.tokenizer import Tokenizer, resolve_tokenizer
from kairyu.outputs import CompletionOutput

_VOCAB_SIZE = 50_000


class _ToyRunner:
    """Deterministic CPU stand-in for the GPU model forward (greedy only —
    sampling params take effect with a Sampler-equipped runner, m8 D2)."""

    def execute(
        self, scheduled: tuple[ScheduledChunk, ...], states: Mapping[str, object]
    ) -> dict[str, tuple[SampledToken, ...]]:
        sampled: dict[str, tuple[SampledToken, ...]] = {}
        for chunk in scheduled:
            state = states[chunk.request_id]
            if not chunk.is_prefill or state.prefill_done:
                seed = sum(state.request.prompt_token_ids) if state.request.prompt_token_ids else 0
                token_id = (seed + 31 * chunk.position) % _VOCAB_SIZE
                sampled[chunk.request_id] = (SampledToken(token_id),)
        return sampled


def build_engine_loop(
    *,
    num_pages: int = 4096,
    page_size: int = 16,
    max_num_batched_tokens: int = 2048,
    runner: object | None = None,
    tensor_parallel_size: int = 1,
    tokenizer: str | Tokenizer | None = None,
    speculative: str | None = None,
    speculative_tokens: int = 4,
    model_path: str | None = None,
) -> tuple[EngineLoop, RadixKVCache, Scheduler]:
    """Assemble the engine stack; shared by KairyuBackend and the ZMQ service.

    ``model_path`` loads a real checkpoint (m12 D5): DenseDecoder +
    PagedKVPool + PagedModelRunner + Sampler, tokenizer from the same dir
    unless overridden. Mutually exclusive with ``runner``. Real-model TP > 1
    spawns the ``DistTPLauncher`` group (rank 0 here, ranks 1.. as workers) and
    drives it via ``DistTPModelRunner``; the loop's ``.tp_launcher`` handle must
    be ``shutdown()`` on serve teardown.
    """
    if speculative is not None and speculative != "ngram":
        raise ValueError(f"unknown speculative mode {speculative!r} (only 'ngram')")
    if speculative is not None and tensor_parallel_size > 1:
        raise ValueError("speculative decoding with tensor_parallel_size > 1 is not supported")
    if model_path is not None and runner is not None:
        raise ValueError("model_path and runner are mutually exclusive")

    if model_path is not None and tensor_parallel_size > 1:
        return _build_dist_tp_loop(
            model_path=model_path,
            tensor_parallel_size=tensor_parallel_size,
            num_pages=num_pages,
            page_size=page_size,
            max_num_batched_tokens=max_num_batched_tokens,
            tokenizer=tokenizer,
        )

    default_eos: int | None = None
    default_stop_ids: tuple[int, ...] = ()
    num_kv_heads_for_tp = None
    if model_path is not None:
        import torch

        from kairyu.engine.core.attention import select_backend
        from kairyu.engine.core.hw_profile import probe
        from kairyu.engine.core.kv_pool import PagedKVPool
        from kairyu.engine.core.model_runner import PagedModelRunner
        from kairyu.engine.core.sampler import Sampler
        from kairyu.models.loader import load_model

        # deploy day is config-free: the probed profile picks the kernel AND the
        # compute placement. GPU runs bf16 on-device (what the FlashInfer / FA2
        # kernels require — fp32 has no such kernels); CPU keeps fp32 on host, so
        # every CPU test path is byte-for-byte unchanged.
        profile = probe()
        gpu = profile.arch == "cuda"
        compute_device = "cuda" if gpu else "cpu"
        compute_dtype = torch.bfloat16 if gpu else torch.float32
        model, model_config, generation = load_model(
            model_path, dtype=compute_dtype, attention_backend=select_backend(profile)
        )
        model = model.to(compute_device)
        default_eos = generation.eos_token_id
        default_stop_ids = generation.stop_token_ids
        num_kv_heads_for_tp = model_config.num_key_value_heads
        resolved = resolve_tokenizer(tokenizer if tokenizer is not None else model_path)
        vocab_size = len(resolved.vocab())
        if vocab_size > model_config.vocab_size:
            raise ValueError(
                f"tokenizer vocab ({vocab_size}) exceeds the model's vocab_size "
                f"({model_config.vocab_size})"
            )
    else:
        resolved = resolve_tokenizer(tokenizer if tokenizer is not None else "toy")

    validate_tp_degree(
        tensor_parallel_size,
        **({"num_kv_heads": num_kv_heads_for_tp} if num_kv_heads_for_tp else {}),
    )
    cache = RadixKVCache(num_pages=num_pages, page_size=page_size)
    scheduler = Scheduler(
        cache,
        max_num_batched_tokens=max_num_batched_tokens,
        page_size=page_size,
        speculative_tokens=speculative_tokens if speculative else 0,
    )
    if model_path is not None:
        pool = PagedKVPool.for_cache(
            cache, model_config, dtype=compute_dtype, device=compute_device
        )
        runner = PagedModelRunner(
            model, pool, sampler=Sampler(vocab_provider=resolved.vocab), cache=cache
        )
    if tensor_parallel_size > 1:
        # CPU-testable TP path (design m5 D1/D3): deterministic rank runners
        # over a FakeCommunicator group; outputs are identical to TP=1.
        active: object = TPModelRunner(
            rank_runners=tuple(
                (runner if runner is not None else _ToyRunner())
                for _ in range(tensor_parallel_size)
            ),
            comms=FakeCommunicator.create_group(tensor_parallel_size),
        )
    else:
        active = runner or _ToyRunner()
    if speculative == "ngram":
        active = SpeculativeRunner(active)
    loop = EngineLoop(
        resolved,
        scheduler,
        active,
        default_eos_token_id=default_eos,
        default_stop_token_ids=default_stop_ids,
    )
    loop.tp_launcher = None  # single-process: nothing to tear down
    return loop, cache, scheduler


def _build_dist_tp_loop(
    *,
    model_path: str,
    tensor_parallel_size: int,
    num_pages: int,
    page_size: int,
    max_num_batched_tokens: int,
    tokenizer: str | Tokenizer | None,
) -> tuple[EngineLoop, RadixKVCache, Scheduler]:
    """Real multi-process TP for `kairyu serve --tp N`: spawn the worker ranks,
    drive them through DistTPModelRunner, and expose the launcher on the loop so
    serve teardown can stop the workers cleanly."""
    from kairyu.engine.core.worker import DistTPLauncher
    from kairyu.models.loader import load_generation_defaults

    launcher = DistTPLauncher(model_path, tensor_parallel_size, num_pages, page_size)
    generation = load_generation_defaults(model_path)
    resolved = resolve_tokenizer(tokenizer if tokenizer is not None else model_path)
    vocab_size = len(resolved.vocab())
    if vocab_size > launcher.full_config.vocab_size:
        launcher.shutdown()
        raise ValueError(
            f"tokenizer vocab ({vocab_size}) exceeds the model's vocab_size "
            f"({launcher.full_config.vocab_size})"
        )
    cache = RadixKVCache(num_pages=num_pages, page_size=page_size)
    scheduler = Scheduler(
        cache, max_num_batched_tokens=max_num_batched_tokens, page_size=page_size
    )
    loop = EngineLoop(
        resolved,
        scheduler,
        launcher.runner,
        default_eos_token_id=generation.eos_token_id,
        default_stop_token_ids=generation.stop_token_ids,
    )
    loop.tp_launcher = launcher  # serve teardown must call launcher.shutdown()
    return loop, cache, scheduler


class KairyuBackend:
    def __init__(
        self,
        num_pages: int = 4096,
        page_size: int = 16,
        max_num_batched_tokens: int = 2048,
        runner: object | None = None,
        tensor_parallel_size: int = 1,
        tokenizer: str | Tokenizer | None = None,
        speculative: str | None = None,
        speculative_tokens: int = 4,
        model_path: str | None = None,
    ) -> None:
        self.tensor_parallel_size = tensor_parallel_size
        self._loop, self._cache, self._scheduler = build_engine_loop(
            num_pages=num_pages,
            page_size=page_size,
            max_num_batched_tokens=max_num_batched_tokens,
            runner=runner,
            tensor_parallel_size=tensor_parallel_size,
            tokenizer=tokenizer,
            speculative=speculative,
            speculative_tokens=speculative_tokens,
            model_path=model_path,
        )
        self._queues: dict[str, asyncio.Queue] = {}  # event-loop thread only
        self._pump_task: asyncio.Task | None = None

    async def _pump(self) -> None:
        try:
            while self._loop.has_work():
                updates = await asyncio.to_thread(self._loop.step)
                for request_id, update in updates:
                    queue = self._queues.get(request_id)
                    if queue is not None:
                        queue.put_nowait(update)
        except Exception as error:
            failure = StreamUpdate((), "", True, None, error)
            for queue in self._queues.values():
                queue.put_nowait(failure)
        finally:
            self._pump_task = None

    def _submit(self, request: GenerationRequest) -> asyncio.Queue:
        self._loop.submit(request.request_id, request.prompt, request.sampling_params)
        queue: asyncio.Queue = asyncio.Queue()
        self._queues[request.request_id] = queue
        if self._pump_task is None:
            self._pump_task = asyncio.get_running_loop().create_task(self._pump())
        return queue

    def _result(self, request: GenerationRequest, update: StreamUpdate) -> GenerationResult:
        completion = CompletionOutput(
            index=0,
            text=update.text,
            token_ids=update.outputs,
            cumulative_logprob=update.cumulative_logprob,
            logprobs=update.logprobs,
            finish_reason=update.finish_reason,
            logprob_content=update.logprob_content,
        )
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(completion,),
            finished=update.finished,
            usage=GenerationUsage(
                prompt_tokens=update.num_prompt_tokens,
                completion_tokens=len(update.outputs),
                cached_tokens=update.num_cached_tokens,
            ),
        )

    def _sub_requests(self, request: GenerationRequest) -> list[GenerationRequest]:
        """n>1 as n engine sub-requests (m9 D3). Completion 0 uses the user
        seed IDENTICALLY (reproducibility parity with n=1); i>0 derive via
        splitmix. Siblings prefill independently (documented: same-schedule
        admissions do not share radix pages; n x prompt page pressure)."""
        params = request.sampling_params
        subs = []
        for index in range(params.n):
            seed = params.seed
            if seed is not None and index > 0:
                seed = mix_seed(seed, index)
            subs.append(
                GenerationRequest(
                    request_id=f"{request.request_id}#c{index}",
                    prompt=request.prompt,
                    sampling_params=params.clone(n=1, seed=seed),
                    cache_hint=request.cache_hint,
                )
            )
        return subs

    def _merged(
        self,
        request: GenerationRequest,
        latest: dict[int, StreamUpdate],
        finished: bool,
    ) -> GenerationResult:
        completions = tuple(
            CompletionOutput(
                index=index,
                text=update.text,
                token_ids=update.outputs,
                cumulative_logprob=update.cumulative_logprob,
                logprobs=update.logprobs,
                finish_reason=update.finish_reason,
                logprob_content=update.logprob_content,
            )
            for index, update in sorted(latest.items())
        )
        first = latest.get(0)
        usage = None
        if first is not None:
            # prompt counted ONCE (m9 D1 aggregation rule); completions summed
            usage = GenerationUsage(
                prompt_tokens=first.num_prompt_tokens,
                completion_tokens=sum(len(u.outputs) for u in latest.values()),
                cached_tokens=first.num_cached_tokens,
            )
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=completions,
            finished=finished,
            usage=usage,
        )

    async def _generate_one(self, request: GenerationRequest) -> GenerationResult:
        queue = self._submit(request)
        try:
            while True:
                update: StreamUpdate = await queue.get()
                if update.error is not None:
                    raise update.error
                if update.finished:
                    return self._result(request, update)
        finally:
            self._queues.pop(request.request_id, None)

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        if request.sampling_params.n <= 1:
            return await self._generate_one(request)
        subs = self._sub_requests(request)
        try:
            results = await asyncio.gather(*(self._generate_one(sub) for sub in subs))
        except Exception:
            for sub in subs:  # abort surviving siblings on any failure
                self._loop.abort(sub.request_id)
            raise
        latest = {
            index: StreamUpdate(
                outputs=result.completions[0].token_ids,
                text=result.completions[0].text,
                finished=True,
                finish_reason=result.completions[0].finish_reason,
                logprobs=result.completions[0].logprobs,
                cumulative_logprob=result.completions[0].cumulative_logprob or 0.0,
                num_prompt_tokens=result.usage.prompt_tokens if result.usage else 0,
                num_cached_tokens=result.usage.cached_tokens if result.usage else 0,
                logprob_content=result.completions[0].logprob_content,
            )
            for index, result in enumerate(results)
        }
        return self._merged(request, latest, finished=True)

    async def _stream_one(self, request: GenerationRequest) -> AsyncIterator[GenerationResult]:
        queue = self._submit(request)
        emitted = -1
        try:
            while True:
                update: StreamUpdate = await queue.get()
                if update.error is not None:
                    raise update.error
                if len(update.outputs) > emitted or update.finished:
                    emitted = len(update.outputs)
                    yield self._result(request, update)
                if update.finished:
                    return
        finally:
            self._queues.pop(request.request_id, None)

    async def stream(self, request: GenerationRequest) -> AsyncIterator[GenerationResult]:
        if request.sampling_params.n <= 1:
            async for result in self._stream_one(request):
                yield result
            return
        # merged n>1 stream: every partial is the cumulative snapshot of ALL
        # completions seen so far (MockBackend semantics — the SSE layer emits
        # finish chunks from the LAST partial's completions)
        subs = self._sub_requests(request)
        queues = {index: self._submit(sub) for index, sub in enumerate(subs)}
        pending = {
            index: asyncio.ensure_future(queue.get()) for index, queue in queues.items()
        }
        latest: dict[int, StreamUpdate] = {}
        finished: set[int] = set()
        try:
            while len(finished) < len(subs):
                done, _ = await asyncio.wait(
                    pending.values(), return_when=asyncio.FIRST_COMPLETED
                )
                for index in list(pending):
                    task = pending[index]
                    if task not in done:
                        continue
                    update: StreamUpdate = task.result()
                    if update.error is not None:
                        raise update.error
                    latest[index] = update
                    if update.finished:
                        finished.add(index)
                        del pending[index]
                    else:
                        pending[index] = asyncio.ensure_future(queues[index].get())
                yield self._merged(request, latest, finished=len(finished) == len(subs))
        except BaseException:
            for sub in subs:  # abandonment/failure aborts siblings
                self._loop.abort(sub.request_id)
            raise
        finally:
            for task in pending.values():
                task.cancel()
            for index in range(len(subs)):
                self._queues.pop(f"{request.request_id}#c{index}", None)

    async def shutdown(self) -> None:
        if self._pump_task is not None:
            self._pump_task.cancel()
        # stop the spawned TP worker ranks (no-op unless --tp N launched them)
        launcher = getattr(self._loop, "tp_launcher", None)
        if launcher is not None:
            await asyncio.to_thread(launcher.shutdown)


register_backend("kairyu", KairyuBackend)
