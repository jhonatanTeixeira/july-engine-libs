import asyncio
import os
import re
import json
import logging
import time
import uuid
import threading
import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from .resource_calculator import estimate_vram_ram, ModelMetadata
from .context import request_id_var, acquired_instances_var

logger = logging.getLogger("JulyEngine.Models.GGUF")

# Phrases emitted by llama_cpp when the KV cache / context window is exhausted
_CTX_OVERFLOW_PHRASES = (
    "Failed completely even with batch size 1",
    "Context Shift is explicitly disabled",
    "Context shift failed",
    "Context Shift failed",
    # llama_decode returns -1 when the tokenized prompt exceeds n_ctx
    "exceeding capacity",
    "llama_decode failed (code -1): Invalid input batch",
    "Fatal Decode Error at Pos 0, Batch size",
)

def _is_ctx_overflow(exc: BaseException) -> bool:
    msg = str(exc)
    return any(phrase in msg for phrase in _CTX_OVERFLOW_PHRASES)

# Global locks for model loading to prevent concurrent loads of the same file across GGUF instances
_GGUF_LOAD_LOCKS = {}
_GGUF_LOAD_LOCKS_LOCK = threading.Lock()

def get_gguf_load_lock(model_path: str):
    with _GGUF_LOAD_LOCKS_LOCK:
        if model_path not in _GGUF_LOAD_LOCKS:
            _GGUF_LOAD_LOCKS[model_path] = threading.Lock()
        return _GGUF_LOAD_LOCKS[model_path]


def detect_model_capabilities(repo_id_or_filename: str) -> dict:
    """
    Uses regex to map the model to the specific handlers from the JamePeng fork.
    """
    name = repo_id_or_filename.lower()
    capabilities = {
        "vision_handler": None,
        "chat_format": "jinja"
    }

    # ==========================================
    # 1. CHAT FORMAT FALLBACK DETECTION
    # ==========================================
    if re.search(r"qwen[_\-\.]?(?:2\.5|3|4)", name):
        capabilities["chat_format"] = "chatml"
    elif re.search(r"gemma[_\-\.]?[2-4]", name):
        capabilities["chat_format"] = "gemma"
    elif re.search(r"llama[_\-\.]?3|hermes", name):
        capabilities["chat_format"] = "llama-3"
    elif re.search(r"mistral|mixtral|pixtral|ministral", name):
        capabilities["chat_format"] = "mistral-instruct"

    # ==========================================
    # 2. VISION DETECTION (JAMEPENG HANDLERS)
    # ==========================================
    if re.search(r"gemma[_\-\.]?4", name):
        capabilities["vision_handler"] = "gemma4"
    elif re.search(r"gemma[_\-\.]?3", name):
        capabilities["vision_handler"] = "gemma3"
    elif 'qwen' in name:
        if '2.5' in name and 'vl' in name:
            capabilities["vision_handler"] = "qwen25vl"
        if '3' in name and 'vl' in name:
            capabilities["vision_handler"] = "qwen3vl"
        if '3.5' in name:
            capabilities["vision_handler"] = "qwen35"
    elif re.search(r"pixtral|ministral", name):
        capabilities["vision_handler"] = "pixtral"
    elif re.search(r"moondream", name):
        capabilities["vision_handler"] = "moondream"
    elif re.search(r"llava[_\-\.]?v?1\.6", name):
        capabilities["vision_handler"] = "llava-v1.6"
    elif re.search(r"llava", name):
        capabilities["vision_handler"] = "llava"

    return capabilities


class SeqAllocator:
    """
    Allocates free seq_ids for requests, with KV cache affinity by conversation_id.

    Does NOT make any native call into llama.cpp — it's just bookkeeping of "which
    seq_id is free" and "which conversation_id each seq_id currently holds". Mutual
    exclusion over the native calls themselves is guaranteed by a different mechanism:
    `model.decode_gate` (from the llama-cpp-python fork itself) is the ONLY point that
    calls into the shared Llama instance, so there's nothing left here that needs a
    per-slot lock.
    """

    def __init__(self, n_seq_max: int):
        self.n_seq_max = n_seq_max
        self._available: "asyncio.Queue[int]" = asyncio.Queue()
        self._allocated: set = set()
        self._pool_lock = asyncio.Lock()
        # which conversation_id each seq_id is currently holding (KV cache)
        self.seq_conversation: Dict[int, Optional[str]] = {}
        for i in range(n_seq_max):
            self._available.put_nowait(i)
            self.seq_conversation[i] = None

    async def acquire(self, conversation_id: Optional[str] = None) -> "tuple[int, bool]":
        """Acquires a free seq_id, with support for re-entrancy by request_id.

        Returns (seq_id, is_new_conversation).
        - is_new_conversation=False → re-entrancy (same HTTP request) OR the seq_id
          already held the KV cache for this exact conversation_id — do NOT reset,
          let generate() itself reuse the KV cache prefix automatically.
        - is_new_conversation=True  → this seq_id is changing owners (a different
          conversation than the one it held before, or it never held any) — the
          caller must do an explicit reset before generating.
        """
        rid = request_id_var.get()

        async with self._pool_lock:
            if rid:
                acquired = acquired_instances_var.get()
                if self in acquired:
                    # Re-entrancy: this request has already reserved a seq from this pool
                    return acquired[self], False

            # Try to find the seq_id that already holds the KV cache for this conversation_id
            if conversation_id:
                preferred = next(
                    (s for s, c in self.seq_conversation.items() if c == conversation_id),
                    None,
                )
                if preferred is not None:
                    drained = []
                    found = None
                    try:
                        while True:
                            s = self._available.get_nowait()
                            if s == preferred:
                                found = s
                                break
                            drained.append(s)
                    except asyncio.QueueEmpty:
                        pass
                    for s in drained:
                        self._available.put_nowait(s)

                    if found is not None:
                        self._allocated.add(found)
                        if rid:
                            acquired = acquired_instances_var.get()
                            acquired[self] = found
                            acquired_instances_var.set(acquired)
                        return found, False
                    # Preferred slot taken now — fall through to the normal wait below

        # Wait for a free seq (outside _pool_lock so releases aren't blocked)
        seq = await self._available.get()

        async with self._pool_lock:
            self._allocated.add(seq)
            is_new_conversation = self.seq_conversation.get(seq) != conversation_id
            self.seq_conversation[seq] = conversation_id
            if rid:
                acquired = acquired_instances_var.get()
                acquired[self] = seq
                acquired_instances_var.set(acquired)

        return seq, is_new_conversation

    def release(self, seq_id: int):
        """Releases the seq, unless it's reserved for re-entrancy."""
        rid = request_id_var.get()
        if rid:
            # In tracked HTTP requests, we don't release immediately because the
            # second turn may need the same instance.
            # The actual release happens in the Middleware at the end of the request.
            return
        self._real_release(seq_id)

    def _real_release(self, seq_id: int):
        """Puts the seq back in the availability queue (keeps conversation affinity)."""
        if seq_id in self._allocated:
            self._allocated.remove(seq_id)
            self._available.put_nowait(seq_id)

    def _force_release(self, seq_id: int):
        """Forces the release, ignoring request_id (used by the middleware)."""
        self._real_release(seq_id)

    def stop(self):
        """Stops the allocator and clears references to allow garbage collection."""
        self._available = asyncio.Queue()
        self._allocated.clear()
        self.seq_conversation.clear()


_SENTINEL = object()


def _accumulate_tool_call_deltas(acc: List[dict], delta_tool_calls: List[dict]):
    """Accumulates tool_call deltas in OpenAI's streaming format (by index; arguments
    arrive fragmented and need to be concatenated)."""
    for tc in delta_tool_calls:
        idx = tc.get("index", 0)
        while len(acc) <= idx:
            acc.append({"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
        entry = acc[idx]
        if tc.get("id"):
            entry["id"] = tc["id"]
        if tc.get("type"):
            entry["type"] = tc["type"]
        fn = tc.get("function") or {}
        if fn.get("name") and not entry["function"]["name"]:
            entry["function"]["name"] = fn["name"]
        if fn.get("arguments"):
            entry["function"]["arguments"] += fn["arguments"]


_NAMED_FORMATTERS: Optional[Dict[str, Any]] = None


def _get_named_formatter(chat_format: Optional[str]):
    """Named formats (chatml/llama-2/llama-3/mistral-instruct/gemma) aren't kept in any
    public registry as a raw `ChatFormatter` — only as an already-packaged handler
    (`LlamaChatCompletionHandlerRegistry`). The pure functions are still accessible
    directly by name in the module, so we map them manually here."""
    global _NAMED_FORMATTERS
    if _NAMED_FORMATTERS is None:
        from llama_cpp import llama_chat_format
        _NAMED_FORMATTERS = {
            "chatml": llama_chat_format.format_chatml,
            "llama-2": llama_chat_format.format_llama2,
            "llama-3": llama_chat_format.format_llama3,
            "mistral-instruct": llama_chat_format.format_mistral_instruct,
            "gemma": llama_chat_format.format_gemma,
        }
    return _NAMED_FORMATTERS.get(chat_format) if chat_format else None


@dataclass
class PreparedGeneration:
    """Result of `_prepare_session`: everything needed to generate via `DecodeGate`
    without calling `create_chat_completion()` (which would generate in a non-batchable
    way). `post_handler`, when present, needs its `_parse_response(response)` called on
    the final assembled text (see `_run_batched_generation_collect`), preserving the
    tool_call/reasoning parsing that the Qwen/Phi/MTMD handlers already do today."""
    prompt_tokens: List[int]
    stop: List[str]
    stopping_criteria: Optional[Any]
    grammar_str: str
    post_handler: Optional[Any]
    messages_norm: List[Dict[str, Any]]
    # Only populated when the message has real media addressed to an MTMD handler
    # — see `_mtmd_prefill`. `mtmd_chunks`/`mtmd_bitmap_cleanup` stay alive (not freed
    # by `_prepare_session`) until the prefill finishes processing them.
    mtmd_handler: Optional[Any] = None
    mtmd_chunk_spans: Optional[List[Tuple[int, int, Any, int, Optional[int]]]] = None
    mtmd_chunks: Optional[Any] = None
    mtmd_bitmap_cleanup: Optional[List[Any]] = None


def _prepare_session(llm, messages: List[Dict[str, Any]], **kwargs) -> Optional[PreparedGeneration]:
    """
    Resolves the right `ChatFormatter` for the handler/format configured on `llm` (the
    `Llama` instance) and builds everything needed to generate via `DecodeGate` (tokens,
    stop, stopping_criteria, grammar) — without triggering any generation. Mirrors the
    logic of `chat_formatter_to_chat_completion_handler` (which does render+tokenize+
    grammar before actually generating), just stopping before `create_completion()`.

    Receives `**kwargs` (the request payload, which includes a `model` key holding the
    model ALIAS, not the instance — hence the parameter is named `llm`, not `model`, to
    avoid a collision).

    Returns `None` if it can't safely decompose this handler/message — the caller should
    fall back to the fallback path (`model.decode_gate.run_exclusive` +
    `create_chat_completion()` in full, same as the old round-robin dispatcher did):
    - any custom `chat_handler` without a known way to extract just the prompt (not
      Qwen/Phi nor MTMD);
    - a `chat_format` with no resolvable formatter (neither in `llm._chat_formatters`
      nor one of the known named formats) — the same case that would already fail today.

    Messages with real media addressed to an MTMD handler (Gemma4/Qwen3.5-vision/etc.)
    no longer fall back — `mtmd_chunk_spans`/`mtmd_chunks`/`mtmd_bitmap_cleanup` get
    populated, and the caller must process the prefill chunk by chunk via
    `_mtmd_prefill` (text batchable in `model.decode_gate`, media encoded under a
    separate lock — see that function for details).
    """
    from llama_cpp.llama_chat_format import MTMDChatHandler

    chat_handler = getattr(llm, "chat_handler", None)
    formatter = None
    post_handler = None
    is_mtmd = False

    if chat_handler is not None:
        if isinstance(chat_handler, MTMDChatHandler):
            is_mtmd = True
            post_handler = chat_handler
        elif callable(getattr(chat_handler, "formatter", None)):
            # Qwen/PhiChatHandler convention: self.formatter is the raw ChatFormatter
            formatter = chat_handler.formatter
            post_handler = chat_handler
        else:
            return None  # unknown custom handler: fallback
    else:
        chat_format = getattr(llm, "chat_format", None)
        formatter = llm._chat_formatters.get(chat_format) or _get_named_formatter(chat_format)
        if formatter is None:
            return None

    tools = kwargs.get("tools")
    tool_choice = kwargs.get("tool_choice")
    functions = kwargs.get("functions")
    function_call = kwargs.get("function_call")
    response_format = kwargs.get("response_format")
    stop = kwargs.get("stop") or []
    if isinstance(stop, str):
        stop = [stop]
    else:
        stop = list(stop)

    from .chat_handlers import _flatten_and_normalize_messages
    messages_norm = _flatten_and_normalize_messages(messages)
    stopping_criteria = None
    mtmd_chunk_spans = None
    mtmd_chunks = None
    mtmd_bitmap_cleanup = None

    if is_mtmd:
        chat_handler._init_mtmd_context(llm)
        full_prompt_ids, chunk_spans, chunks, bitmap_cleanup = chat_handler._process_mtmd_prompt(
            llama=llm,
            messages=messages_norm,
            functions=functions,
            function_call=function_call,
            tools=tools,
            tool_choice=tool_choice,
            add_generation_prompt=True,
        )
        has_media = any(
            chat_handler._is_image_chunk(chunk_type) or chat_handler._is_audio_chunk(chunk_type)
            for (_start, _end, _chunk_ptr, chunk_type, _media_id) in chunk_spans
        )
        if has_media:
            # Keeps the C resources alive — `_mtmd_prefill` processes each chunk (text
            # via DecodeGate, media via encode_lock) and frees them at the end.
            mtmd_chunk_spans = chunk_spans
            mtmd_chunks = chunks
            mtmd_bitmap_cleanup = bitmap_cleanup
        else:
            # No real media in this message — nothing to hold on to, free immediately
            # (same cleanup the handler's own __call__ does in its finally block).
            if chunks is not None:
                chat_handler._mtmd_cpp.mtmd_input_chunks_free(chunks)
            if bitmap_cleanup:
                for bitmap in bitmap_cleanup:
                    chat_handler._mtmd_cpp.mtmd_bitmap_free(bitmap)
        prompt_tokens = full_prompt_ids
    else:
        result = formatter(
            messages=messages_norm,
            functions=functions,
            function_call=function_call,
            tools=tools,
            tool_choice=tool_choice,
        )
        prompt_tokens = llm.tokenize(
            result.prompt.encode("utf-8"),
            add_bos=not result.added_special,
            special=True,
        )
        if result.stop is not None:
            rstop = result.stop if isinstance(result.stop, list) else [result.stop]
            stop = stop + rstop
        if result.stopping_criteria is not None:
            stopping_criteria = result.stopping_criteria

    grammar_str = ""
    if response_format is not None and response_format.get("type") == "json_object":
        from llama_cpp.llama_chat_format import _grammar_for_response_format
        g = _grammar_for_response_format(response_format, verbose=llm.verbose)
        if g is not None:
            grammar_str = g._grammar

    if tool_choice is not None and isinstance(tool_choice, dict) and tools is not None:
        from llama_cpp import llama_grammar
        name = tool_choice.get("function", {}).get("name")
        tool = next((t for t in tools if t["function"]["name"] == name), None)
        if tool is not None:
            schema = tool["function"]["parameters"]
            try:
                g = llama_grammar.LlamaGrammar.from_json_schema(json.dumps(schema), verbose=llm.verbose)
            except Exception:
                g = llama_grammar.LlamaGrammar.from_string(llama_grammar.JSON_GBNF, verbose=llm.verbose)
            grammar_str = g._grammar

    return PreparedGeneration(
        prompt_tokens=prompt_tokens,
        stop=stop,
        stopping_criteria=stopping_criteria,
        grammar_str=grammar_str,
        post_handler=post_handler,
        messages_norm=messages_norm,
        mtmd_handler=chat_handler if mtmd_chunk_spans is not None else None,
        mtmd_chunk_spans=mtmd_chunk_spans,
        mtmd_chunks=mtmd_chunks,
        mtmd_bitmap_cleanup=mtmd_bitmap_cleanup,
    )


def _build_sampling_params(model, kwargs: dict, grammar_str: str):
    from llama_cpp._internals import LlamaSamplingParams
    seed = kwargs.get("seed")
    return LlamaSamplingParams(
        top_k=kwargs.get("top_k", 40),
        top_p=kwargs.get("top_p", 0.95),
        min_p=kwargs.get("min_p", 0.05),
        typical_p=kwargs.get("typical_p", 1.0),
        temp=kwargs.get("temperature", 0.2),
        top_n_sigma=kwargs.get("top_n_sigma", -1.0),
        min_keep=kwargs.get("min_keep", 0),
        dynatemp_range=kwargs.get("dynatemp_range", 0.0),
        dynatemp_exponent=kwargs.get("dynatemp_exponent", 1.0),
        penalty_last_n=kwargs.get("penalty_last_n", 64),
        penalty_repeat=kwargs.get("repeat_penalty", 1.0),
        penalty_freq=kwargs.get("frequency_penalty", 0.0),
        penalty_present=kwargs.get("present_penalty", 0.0),
        mirostat=kwargs.get("mirostat_mode", 0),
        mirostat_tau=kwargs.get("mirostat_tau", 5.0),
        mirostat_eta=kwargs.get("mirostat_eta", 0.1),
        xtc_probability=kwargs.get("xtc_probability", 0.0),
        xtc_threshold=kwargs.get("xtc_threshold", 0.1),
        dry_multiplier=kwargs.get("dry_multiplier", 0.0),
        dry_base=kwargs.get("dry_base", 1.75),
        dry_allowed_length=kwargs.get("dry_allowed_length", 2),
        dry_penalty_last_n=kwargs.get("dry_penalty_last_n", 0),
        dry_sequence_breakers=kwargs.get("dry_seq_breakers", ["\n", ":", "\"", "*"]),
        adaptive_target=kwargs.get("adaptive_target", -1.0),
        adaptive_decay=kwargs.get("adaptive_decay", 0.9),
        logit_bias=model._convert_logit_bias(kwargs.get("logit_bias")),
        grammar=grammar_str,
        grammar_lazy=kwargs.get("grammar_lazy", False),
        seed=seed if seed is not None else model._seed,
        reasoning_budget=kwargs.get("reasoning_budget", -1),
        reasoning_start=kwargs.get("reasoning_start", "<think>"),
        reasoning_end=kwargs.get("reasoning_end", "</think>"),
        reasoning_budget_message=kwargs.get("reasoning_budget_message"),
        reasoning_start_in_prompt=kwargs.get("reasoning_start_in_prompt", False),
        reasoning_start_max_tokens=kwargs.get("reasoning_start_max_tokens", 32),
    )


async def _mtmd_prefill(model, seq_id: int, prepared: PreparedGeneration) -> Tuple[int, int]:
    """
    Processes the prefill of a message with real media, chunk by chunk — text goes to
    `model.decode_gate` normally (batchable with other sessions, and already keeps
    `input_ids`/`n_tokens` up to date on its own), image/audio is encoded via
    `chat_handler.encode_chunk_exclusive()` (a SEPARATE lock, owned by the
    `MTMDChatHandler` itself, only protecting `mtmd_encode_chunk`/`mtmd_ctx`), and only
    the final injection into the KV cache uses `decode_gate`. This lets OTHER sessions
    keep decoding text while an image is being encoded — only the short step of
    injecting the already-ready result into the KV cache needs exclusivity.

    Mirrors `MTMDChatHandler.__call__` (llama_multimodal.py:1044-1189), including prefix
    reuse across turns of the same conversation (doesn't re-encode an image that's
    already in the KV cache from a previous turn).

    Returns (idx, pos): the logits index to sample the first response token from, and
    the current KV cache position (to continue steady-state generation).
    """
    from llama_cpp.llama import active_seq_id
    from llama_cpp import llama_cpp as llama_cpp_lib
    import ctypes

    chat_handler = prepared.mtmd_handler
    chunk_spans = prepared.mtmd_chunk_spans
    chunks = prepared.mtmd_chunks
    bitmap_cleanup = prepared.mtmd_bitmap_cleanup
    gate = model.decode_gate

    active_seq_id.set(seq_id)
    try:
        # Prefix reuse across turns: compares the "virtual ledger" (a mix of real
        # tokens and negative media IDs) against what's already in this session's KV
        # cache — same logic as `MTMDChatHandler.__call__` (lines 1066-1097).
        current_history = model.input_ids[:model.n_tokens].tolist()
        longest_prefix = model.longest_token_prefix(current_history, prepared.prompt_tokens, model.verbose)
        if longest_prefix < model.n_tokens:
            model._ctx.memory_seq_rm(seq_id, longest_prefix, -1)
            model.n_tokens = longest_prefix

        n_past = model.n_tokens
        last_idx = -1

        for start_idx, end_idx, chunk_ptr, chunk_type, media_id in chunk_spans:
            if end_idx <= n_past:
                continue  # already reused from this conversation's prefix

            if chat_handler._is_text_chunk(chunk_type):
                unprocessed_start = max(start_idx, n_past) - start_idx
                n_tokens_out = ctypes.c_size_t()
                tokens_ptr = chat_handler._mtmd_cpp.mtmd_input_chunk_get_tokens_text(chunk_ptr, ctypes.byref(n_tokens_out))
                if tokens_ptr and n_tokens_out.value > 0:
                    all_tokens = [tokens_ptr[j] for j in range(n_tokens_out.value)]
                    tokens_to_eval = all_tokens[unprocessed_start:]
                    if tokens_to_eval:
                        last_idx = await gate.submit_tokens(seq_id, tokens_to_eval, n_past)
                        active_seq_id.set(seq_id)
                        n_past += len(tokens_to_eval)

            else:  # image or audio
                # 1. Encode — only contends for the handler's own encode_lock, NOT
                #    decode_gate. Other sessions keep decoding text freely while this
                #    runs.
                embd = await chat_handler.encode_chunk_exclusive(chunk_ptr)

                # 2. Injects the already-ready result into the KV cache — this part
                #    does need decode_gate (calls llama_decode under the hood).
                # ctypes requires a properly-typed CFUNCTYPE instance for the callback
                # parameter — a bare `None` isn't accepted when the declared type is
                # CFUNCTYPE; it needs an explicitly-typed NULL function pointer.
                null_callback = ctypes.cast(None, chat_handler._mtmd_cpp.mtmd_helper_post_decode_callback)

                def _decode_media(_n_past=n_past, _chunk_ptr=chunk_ptr, _embd=embd):
                    new_n_past = llama_cpp_lib.llama_pos(0)
                    result = chat_handler._mtmd_cpp.mtmd_helper_decode_image_chunk(
                        chat_handler.mtmd_ctx, model._ctx.ctx, _chunk_ptr, _embd,
                        llama_cpp_lib.llama_pos(_n_past), llama_cpp_lib.llama_seq_id(seq_id),
                        model.n_batch, ctypes.byref(new_n_past), null_callback, None,
                    )
                    if result != 0:
                        raise RuntimeError(f"mtmd_helper_decode_image_chunk falhou (code {result})")
                    return new_n_past.value

                new_n_past = await gate.run_exclusive(_decode_media)
                active_seq_id.set(seq_id)
                model.input_ids[n_past:new_n_past] = media_id
                n_past = new_n_past
                model.n_tokens = n_past

        return last_idx, n_past
    finally:
        if chunks is not None:
            chat_handler._mtmd_cpp.mtmd_input_chunks_free(chunks)
        if bitmap_cleanup:
            for bitmap in bitmap_cleanup:
                chat_handler._mtmd_cpp.mtmd_bitmap_free(bitmap)


async def _generate_via_gate(model, seq_id: int, prepared: PreparedGeneration, kwargs: dict):
    """
    Common generation core via `model.decode_gate` (the fork's queue+leader mechanism,
    real batching across sessions — see `DecodeGate`/`Llama.decode_gate` in llama.py):
    reuses the KV cache already existing for this conversation
    (`model._reuse_prefix_and_eval`, or `_mtmd_prefill` for messages with real media),
    builds a `LlamaSamplingContext` per session (grammar/reasoning-budget/dry/mirostat/
    etc. included), and drives a plain (non-async) Python generator that on each
    `next()` submits the next step to the gate and returns the sampled token — along
    with the incremental text and the stop reason.

    Yields (new_text: bytes, finish_reason: Optional[str]) — the caller decides what to
    do with each piece (emit it incrementally, or accumulate until the end).
    """
    from llama_cpp.llama import active_seq_id, StopStringMatcher
    from llama_cpp._internals import LlamaSamplingContext
    from llama_cpp import llama_cpp as llama_cpp_lib

    active_seq_id.set(seq_id)
    gate = model.decode_gate

    sampling_params = _build_sampling_params(model, kwargs, prepared.grammar_str)
    sampling_ctx = LlamaSamplingContext(sampling_params, model._model)
    has_grammar = bool(prepared.grammar_str)

    max_tokens = kwargs.get("max_tokens")
    if not max_tokens or max_tokens <= 0:
        max_tokens = model._n_ctx - len(prepared.prompt_tokens)

    if prepared.mtmd_chunk_spans is not None:
        # Message with real media: prefill chunk by chunk (text batchable in
        # decode_gate, media via chat_handler.encode_chunk_exclusive()) — see
        # `_mtmd_prefill`. The ledger (input_ids/n_tokens) already comes out updated
        # from there, no need to repeat it here.
        idx, pos = await _mtmd_prefill(model, seq_id, prepared)
    else:
        delta_tokens = model._reuse_prefix_and_eval(prepared.prompt_tokens, seq_id=seq_id, reset=True)
        pos = model.n_tokens

        # `submit_tokens` already keeps input_ids/n_tokens up to date on its own (part
        # of decode_gate itself, in the fork) — no need to repeat that bookkeeping here.
        idx = await gate.submit_tokens(seq_id, delta_tokens, pos)
        active_seq_id.set(seq_id)
        pos += len(delta_tokens)

    active_seq_id.set(seq_id)
    token = sampling_ctx.sample(model._ctx, idx=idx)
    sampling_ctx.accept(token, has_grammar)

    completion_tokens: List[int] = []
    stop_matcher = StopStringMatcher(model, prepared.prompt_tokens, prepared.stop)
    emitted_len = 0
    finish_reason = "length"

    while True:
        if llama_cpp_lib.llama_token_is_eog(model._model.vocab, token):
            finish_reason = "stop"
            break

        completion_tokens.append(token)
        suppress, stop_matched, text = stop_matcher.step(completion_tokens)

        if not suppress:
            if stop_matched is not None:
                new_text = text[emitted_len:]
                if new_text:
                    yield new_text, None
                finish_reason = "stop"
                break

            if prepared.stopping_criteria is not None:
                import numpy as np
                ids_arr = np.array(list(prepared.prompt_tokens) + completion_tokens, dtype=np.intc)
                if prepared.stopping_criteria(ids_arr, np.empty(0, dtype=np.single)):
                    finish_reason = "stop"
                    break

            new_text = text[emitted_len:]
            emitted_len = len(text)
            if new_text:
                yield new_text, None

        if len(completion_tokens) >= max_tokens:
            finish_reason = "length"
            break

        idx = await gate.submit_tokens(seq_id, [token], pos)
        active_seq_id.set(seq_id)
        pos += 1

        token = sampling_ctx.sample(model._ctx, idx=idx)
        sampling_ctx.accept(token, has_grammar)

    yield b"", finish_reason


def _chat_chunk(response_id: str, created: int, model_name: str, delta: Optional[dict] = None, finish_reason: Optional[str] = None) -> dict:
    return {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": delta or {},
            "logprobs": None,
            "finish_reason": finish_reason,
        }],
    }


async def _run_batched_generation_stream(model, seq_id, prepared, kwargs, response_id, created, model_name):
    """Real incremental streaming via `model.decode_gate` — only used when there's NO
    `post_handler` (named format or generic jinja, no extra tool_call parsing at the
    end), where each piece of text can be delivered as soon as it's generated."""
    yield _chat_chunk(response_id, created, model_name, delta={"role": "assistant"})
    async for new_text, finish_reason in _generate_via_gate(model, seq_id, prepared, kwargs):
        if new_text:
            yield _chat_chunk(response_id, created, model_name, delta={"content": new_text.decode("utf-8", errors="ignore")})
        if finish_reason is not None:
            yield _chat_chunk(response_id, created, model_name, finish_reason=finish_reason)


async def _run_batched_generation_then_parse(model, seq_id, prepared, kwargs, response_id, created, model_name):
    """
    Used when there IS a `post_handler` (Qwen/Phi/MTMD without media) — these handlers
    parse tool_call/reasoning via regex over the FULL TEXT, and their original
    `_stream_response` consume a SYNCHRONOUS generator (what `create_completion
    (stream=True)` produces) — incompatible with `decode_gate`'s async generator.
    Instead of reimplementing each handler's parsing asynchronously (a real risk of
    subtly diverging from today's production behavior), we run the batched generation
    to completion (the throughput gain from `decode_gate` still applies — it's the
    token-by-token generation that gets fast), assemble the full response in the format
    `_parse_response` already expects, call it (synchronously, no generator), and only
    then slice the corrected result into chunks for the client. Real streaming
    (token-by-token visible to the client) is only for formats without `post_handler`
    in this first version.
    """
    full_text = b""
    finish_reason = "length"
    async for new_text, fr in _generate_via_gate(model, seq_id, prepared, kwargs):
        full_text += new_text
        if fr is not None:
            finish_reason = fr

    response = {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": full_text.decode("utf-8", errors="ignore")},
            "finish_reason": finish_reason,
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

    try:
        response = prepared.post_handler._parse_response(response)
    except TypeError:
        pass  # handler with an unexpected signature — proceed with the raw response

    message = response["choices"][0].get("message", {})
    final_finish_reason = response["choices"][0].get("finish_reason", finish_reason)

    yield _chat_chunk(response_id, created, model_name, delta={"role": "assistant"})
    if message.get("content"):
        yield _chat_chunk(response_id, created, model_name, delta={"content": message["content"]})
    if message.get("reasoning_content"):
        yield _chat_chunk(response_id, created, model_name, delta={"reasoning_content": message["reasoning_content"]})
    if message.get("tool_calls"):
        tool_call_deltas = []
        for i, tc in enumerate(message["tool_calls"]):
            tool_call_deltas.append({**tc, "index": i})
        yield _chat_chunk(response_id, created, model_name, delta={"tool_calls": tool_call_deltas})
    yield _chat_chunk(response_id, created, model_name, finish_reason=final_finish_reason)


async def _fallback_stream(model, create_kwargs: dict):
    """
    Safety path: reuses `create_chat_completion()` in full, one step (`next()`) at a
    time — same mechanics as the old round-robin dispatcher — but each step acquires
    the SAME mutual exclusion as `model.decode_gate`, guaranteeing it never runs at the
    same time as a batched round (or another fallback) from a different session. Used
    when `_prepare_session` returns `None` (an unknown handler we can't safely
    decompose).
    """
    from llama_cpp.llama import active_seq_id

    seq_id = create_kwargs.get("seq_id")
    state: Dict[str, Any] = {}

    def _step():
        active_seq_id.set(seq_id)
        if "gen" not in state:
            state["gen"] = model.create_chat_completion(**create_kwargs)
        return next(state["gen"], _SENTINEL)

    while True:
        chunk = await model.decode_gate.run_exclusive(_step)
        if chunk is _SENTINEL:
            break
        yield chunk
        finish_reason = chunk.get("choices", [{}])[0].get("finish_reason")
        if finish_reason is not None:
            break


class GGUF:
    def __init__(self, backend, model):
        from huggingface_hub import hf_hub_download

        self.backend = backend
        self.meta = model
        self.cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))
        self.model = None
        self.model_path = hf_hub_download(repo_id=model["model_id"], filename=model["filename"])
        self.model_metadata = ModelMetadata(self.model_path)
        self.allocator: Optional[SeqAllocator] = None
        # The decode gate (queue+leader, real batching) lives in the fork itself —
        # `self.model.decode_gate`, created by `Llama.__init__` — no need for its own
        # reference here. The image/audio encode lock also lives in the fork, on
        # `chat_handler.encode_lock` (`MTMDChatHandler.__init__`) — same logic.
        self.instances = []
        self.n_seq_max = int(model.get("n_seq_max") or model.get("n_parallel") or 1)
        self.offload_kqv = model.get("offload_kqv") if model.get("offload_kqv") is not None else True
        self.kv_unified = model.get("kv_unified") if model.get("kv_unified") is not None else True
        self.logits_all = model.get("logits_all") if model.get("logits_all") is not None else False
        self.vision_on_cpu = model.get("vision_on_cpu", False)

    def max_layers(self):
        return self.model_metadata.block_count

    def decrement_layers(self) -> bool:
        curr_layers = self.meta.get("num_layers")

        # If it's -1, resolve the total before decrementing
        if curr_layers == -1:
            curr_layers = self.max_layers()

        if curr_layers <= 0:
            logger.warning(f"GGUF: Model {self.meta['model_alias']} already at 0 layers. Cannot decrement further.")
            self.meta["num_layers"] = 0
            return False

        self.meta["num_layers"] = curr_layers - 1
        logger.info(f"GGUF: Decrementing layers for {self.meta['model_alias']}. New value: {self.meta['num_layers']}")
        return True

    async def get_required_vram(self, payload: Dict[str, Any]) -> int:
        if self.backend == "cpu":
            return 0

        meta = self.meta

        headers = payload.get("headers", {})
        n_ctx_per_req = int(headers.get("x-context-window") or payload.get("n_ctx") or meta.get("context_window") or 4096)

        # The real context on GPU is multiplied by the number of parallel slots
        effective_n_ctx = n_ctx_per_req * self.n_seq_max

        # 2. Get layers config
        # Estimates the required VRAM using the unified resource calculator
        estimate = await estimate_vram_ram(
            model_path=self.model_path,
            context_window=n_ctx_per_req,
            kv_cache_quantization=self.meta.get('kv_cache_quantization', 'FP16'),
            gpu_layers=meta.get("num_layers", -1),
            n_seq_max=self.n_seq_max,
            offload_kqv=self.offload_kqv,
            flash_attention=self.meta.get('flash_attn', True),
            logits_all=self.logits_all,
            kv_unified=self.kv_unified,
            vision_on_cpu=self.vision_on_cpu
        )

        return estimate["total_vram_mb"]

    def load(self, n_ctx: Optional[int] = None, num_layers: Optional[int] = None):
        from huggingface_hub import hf_hub_download

        meta = self.meta

        # Raised the default to 4096 to support more complex agents
        n_ctx_per_req = n_ctx or int(meta.get("context_window") or os.environ.get("LLM_CTX_TOKENS", '4096'))
        effective_n_ctx = n_ctx_per_req * self.n_seq_max

        # Serializes loading of the same model file
        lock = get_gguf_load_lock(self.model_path)
        with lock:
            if self.backend == 'cpu':
                n_gpu_layers = 0
            else:
                n_gpu_layers = num_layers if num_layers else meta.get("num_layers", -1)

            if self.is_loaded():
                if self.model.n_ctx() == effective_n_ctx:
                    logger.debug(f"GGUF: Modelo {self.meta['model_alias']} já carregado. Reaproveitando!")
                    return
                else:
                    logger.info(f"GGUF: Reloading model {self.meta['model_alias']} because n_ctx changed ({self.model.n_ctx()} -> {effective_n_ctx})")
                    self.unload(self.meta['model_alias'])

            model_path = self.model_path

            try:
                from llama_cpp import Llama
                import llama_cpp

                if self.backend == 'gpu':
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except ImportError:
                        pass

                logger.info(f"GGUF: Loading model {self.meta['model_alias']} on {self.backend} (n_seq_max={self.n_seq_max}, n_ctx_total={effective_n_ctx})")
                n_threads = os.environ.get("MAX_GGUF_THREADS", None)

                base_params = {
                    "model_path": model_path,
                    "n_gpu_layers": n_gpu_layers,
                    "n_ctx": effective_n_ctx,
                    "n_seq_max": self.n_seq_max,
                    "offload_kqv": self.offload_kqv,
                    "kv_unified": self.kv_unified,
                    "logits_all": self.logits_all,
                    "use_mmap": os.environ.get("USE_MMAP", "true").lower() == "true",
                    "verbose": os.environ.get("LLM_VERBOSE", "false").lower() == "true",
                    "n_batch": max(int(os.environ.get("LLM_N_BATCH", "512")), 2048),
                    "n_threads": int(n_threads) if n_threads else None,
                    "n_threads_batch": int(n_threads) if n_threads else None,
                }

                # Flash Attention: metadata > env var > default True
                flash_attn = meta.get("flash_attn")
                if flash_attn is None:
                    flash_attn = os.environ.get("FLASH_ATTN", "true").lower() == "true"

                if flash_attn:
                    base_params["flash_attn"] = True
                    logger.info("GGUF: Flash Attention enabled")
                else:
                    base_params["flash_attn"] = False
                    logger.info("GGUF: Flash Attention disabled")

                # Use KV Cache Quantization from metadata (preferred) or env var
                kv_quant = meta.get("kv_cache_quantization") or os.environ.get('KV_CACHE_QUANTIZATION')

                if kv_quant:
                    kv_quant = str(kv_quant).upper()
                    if "8" in kv_quant or "Q8_0" in kv_quant:
                        base_params["type_k"] = 8
                        base_params["type_v"] = 8
                        logger.info("GGUF: Using Q8_0 for KV Cache")
                    elif "4" in kv_quant or "Q4_0" in kv_quant:
                        base_params["type_k"] = 2
                        base_params["type_v"] = 2
                        logger.info("GGUF: Using Q4_0 for KV Cache")
                    else:
                        logger.info("GGUF: Using default FP16 for KV Cache")

                # Capability extraction
                model_identifier = meta["model_id"] + meta["filename"]
                caps = detect_model_capabilities(model_identifier)

                if meta.get("template"):
                    base_params["chat_format"] = meta["template"]
                else:
                    _formats = getattr(llama_cpp.llama_chat_format, "CHAT_FORMATS", {})
                    base_params["chat_format"] = "jinja" if "jinja" in _formats else caps["chat_format"]

                # ==========================================
                # LOAD SINGLE MULTI-SEQUENCE INSTANCE
                # ==========================================
                logger.info(f"GGUF: Loading single multi-sequence instance on {self.backend} (n_ctx={effective_n_ctx}, n_seq_max={self.n_seq_max})...")

                # Each seq_id needs its own Chat Handler (independent state)
                # Qwen Tool Calling & Reasoning Support (non-VL)
                if 'qwen' in model_identifier.lower() and meta.get("model_type") != "vision":
                    from .chat_handlers import QwenChatHandler
                    template = self.model_metadata.tokenizer_template or meta.get("template")
                    if isinstance(template, str) and template.strip():
                        base_params["chat_handler"] = QwenChatHandler(
                            template=template,
                            eos_token="<|im_end|>",
                            bos_token="<|im_start|>"
                        )
                        if "chat_format" in base_params:
                            del base_params["chat_format"]

                # Phi Tool Calling Support (non-VL)
                if re.search(r'phi', model_identifier.lower()) and meta.get("model_type") != "vision":
                    from .chat_handlers import PhiChatHandler
                    base_params["chat_handler"] = PhiChatHandler()
                    base_params.pop("chat_format", None)

                if meta.get("model_type") == "vision":
                    mmproj_path = None
                    mmproj_id = meta.get("mmproj_id")
                    mmproj_filename = meta.get("mmproj_filename")
                    if mmproj_id and mmproj_filename:
                        mmproj_path = hf_hub_download(mmproj_id, mmproj_filename)

                    v_handler = caps["vision_handler"]
                    use_gpu_vision = not self.vision_on_cpu
                    handler_kwargs = {"verbose": base_params.get("verbose", False), "use_gpu": use_gpu_vision}

                    try:
                        if v_handler == "gemma4":
                            from .chat_handlers import Gemma4Handler
                            base_params["chat_handler"] = Gemma4Handler(clip_model_path=mmproj_path, **handler_kwargs) if mmproj_path else Gemma4Handler(**handler_kwargs)
                        elif v_handler == "gemma3":
                            from llama_cpp.llama_chat_format import Gemma3ChatHandler
                            base_params["chat_handler"] = Gemma3ChatHandler(clip_model_path=mmproj_path, **handler_kwargs) if mmproj_path else Gemma3ChatHandler(**handler_kwargs)
                        elif v_handler == "qwen3vl":
                            from llama_cpp.llama_chat_format import Qwen3VLChatHandler
                            base_params["chat_handler"] = Qwen3VLChatHandler(clip_model_path=mmproj_path, **handler_kwargs) if mmproj_path else Qwen3VLChatHandler(**handler_kwargs)
                        elif v_handler == "qwen25vl":
                            from llama_cpp.llama_chat_format import Qwen25VLChatHandler
                            base_params["chat_handler"] = Qwen25VLChatHandler(clip_model_path=mmproj_path, **handler_kwargs) if mmproj_path else Qwen25VLChatHandler(**handler_kwargs)
                        elif v_handler == "qwen35":
                            from .chat_handlers import Qwen35Handler
                            base_params["chat_handler"] = Qwen35Handler(clip_model_path=mmproj_path, preserve_thinking=True, **handler_kwargs) if mmproj_path else Qwen35Handler(preserve_thinking=True, **handler_kwargs)
                        elif v_handler == "moondream":
                            from llama_cpp.llama_chat_format import MoondreamChatHandler
                            base_params["chat_handler"] = MoondreamChatHandler(clip_model_path=mmproj_path, **handler_kwargs)
                        elif v_handler == "llava-v1.6" or v_handler == "pixtral":
                            from llama_cpp.llama_chat_format import Llava16ChatHandler
                            base_params["chat_handler"] = Llava16ChatHandler(clip_model_path=mmproj_path, **handler_kwargs)
                        elif v_handler == "llava":
                            from llama_cpp.llama_chat_format import Llava15ChatHandler
                            base_params["chat_handler"] = Llava15ChatHandler(clip_model_path=mmproj_path, **handler_kwargs)
                    except ImportError:
                        if mmproj_path:
                            from llama_cpp.llama_chat_format import Llava15ChatHandler
                            base_params["chat_handler"] = Llava15ChatHandler(clip_model_path=mmproj_path, **handler_kwargs)

                logger.info(f"GGUF: Final base params for Llama instances: {base_params}")

                self.model = Llama(**base_params)
                self.allocator = SeqAllocator(self.n_seq_max)
                # self.model.decode_gate and (if MTMD) chat_handler.encode_lock already
                # come ready from the fork's own __init__.

            except Exception as e:
                logger.error(f"GGUF: Failed to load {self.meta['model_alias']}: {e}")
                raise e

    async def run_chat(self, messages: List[Dict[str, Any]], stream: bool = False, **kwargs):
        headers = kwargs.pop("headers", {})
        conversation_id = (
            headers.get("x-session-id")
            or kwargs.pop("session_id", None)
            or kwargs.pop("conversation_id", None)
        )
        n_ctx = headers.get("x-context-window", None)

        if n_ctx:
            try:
                n_ctx = int(n_ctx)
            except ValueError:
                pass

        await asyncio.to_thread(self.load, n_ctx, kwargs.pop('num_layers', None))

        if "repetition_penalty" in kwargs:
            kwargs["repeat_penalty"] = kwargs.pop("repetition_penalty")

        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        kwargs.pop("reasoning_enabled", None)
        kwargs.pop("reasoning_effort", None)

        # FIX: If response_format is json_object, repeat_penalty MUST be 1.0
        # Otherwise llama.cpp penalizes the JSON grammar characters and loops on whitespace
        fmt = kwargs.get("response_format")
        if isinstance(fmt, dict) and fmt.get("type") == "json_object":
            kwargs["repeat_penalty"] = 1.0

        force_reasoning = self.meta.get("force_reasoning", False)

        if "max_completion_tokens" in kwargs:
            kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")

        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = -1

        # The Force Reasoning trick: the AI starts out already thinking
        if force_reasoning:
            messages.append({"role": "assistant", "content": "<think>\n"})

        # Acquires a seq_id from the allocator — uses conversation_id (x-session-id) for KV cache affinity
        seq_id, is_new_conversation = await self.allocator.acquire(conversation_id=conversation_id)

        try:
            # We only reset the KV cache when this seq_id is changing owners (a
            # different conversation than the one it held). If it's the same
            # conversation resuming (or a re-entrant call from the same request), we do
            # NOT reset — we let generate() itself reuse the KV cache prefix
            # automatically.
            if is_new_conversation:
                try:
                    from llama_cpp.llama import active_seq_id
                    token = active_seq_id.set(seq_id)
                    try:
                        self.model.reset(seq_id=seq_id)
                    finally:
                        active_seq_id.reset(token)
                    logger.debug(f"GGUF: seq_id={seq_id} trocou de conversa — KV cache resetado")
                except Exception as e:
                    logger.warning(f"GGUF: Could not reset seq_id={seq_id}: {e}")

            logger.debug(f"GGUF chat call, {({'stream': stream, 'seq_id': seq_id, **kwargs})}")

            create_kwargs = dict(kwargs)
            create_kwargs["messages"] = messages
            create_kwargs["stream"] = True
            create_kwargs["seq_id"] = seq_id

            # Resolves whether this request can take the batched path (model.decode_gate,
            # in the fork — 1 decode() per round covering multiple sessions; messages
            # with real media also go through here, via `_mtmd_prefill` — text
            # batchable, image/audio encoded via `chat_handler.encode_chunk_exclusive()`,
            # separate from decode_gate) or needs the fallback path
            # (create_chat_completion() in full, under the SAME mutual exclusion as
            # decode_gate) — only for a custom handler we can't safely decompose.
            # `force_round_robin` in the model preset forces the fallback
            # unconditionally (escape hatch, no code deploy needed).
            prepared = None
            if not self.meta.get("force_round_robin"):
                prepared = _prepare_session(self.model, messages, **kwargs)

            response_id = f"chatcmpl-{uuid.uuid4().hex}"
            created_ts = int(time.time())
            model_alias = self.meta.get("model_alias", "unknown")

            if prepared is None:
                raw_chunks_gen = _fallback_stream(self.model, create_kwargs)
            elif prepared.post_handler is None:
                raw_chunks_gen = _run_batched_generation_stream(
                    self.model, seq_id, prepared, kwargs,
                    response_id, created_ts, model_alias,
                )
            else:
                raw_chunks_gen = _run_batched_generation_then_parse(
                    self.model, seq_id, prepared, kwargs,
                    response_id, created_ts, model_alias,
                )

            async def _raw_chunks():
                """Consumes the generation chunks (batched or fallback) one at a time."""
                async for chunk in raw_chunks_gen:
                    yield chunk

            async def stream_adapter():
                try:
                    tag_opened = force_reasoning
                    buffer = ""
                    raw_response = ""
                    chunk = None

                    async for chunk in _raw_chunks():
                        delta = chunk["choices"][0].get("delta", {})

                        finish_reason = chunk["choices"][0].get("finish_reason")
                        if finish_reason in ["stop", "tool_calls"]:
                            prompt_data = ""

                            for message in messages:
                                prompt_data += "\n".join(c['text'] for c in message.get("content")) if isinstance(message.get("content"), list) else (message.get("content") or "")

                            prompt_tokens = len(self.model.tokenize(prompt_data.encode('utf-8')))
                            completion_tokens = len(self.model.tokenize(raw_response.encode('utf-8')))

                            chunk["choices"][0].setdefault("usage", {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "total_tokens": (prompt_tokens + completion_tokens)
                            })

                            # If we received a STOP and there's still buffer, send it before the stop chunk
                            if buffer:
                                target_key = 'content' if not tag_opened else 'reasoning_content'
                                chunk_flush = copy.deepcopy(chunk)
                                chunk_flush["choices"][0]["delta"] = {target_key: buffer}
                                chunk_flush["choices"][0]["finish_reason"] = None
                                buffer = ""
                                yield chunk_flush

                            yield chunk
                            continue

                        # Pass-through for data that isn't plain text
                        if "audio_url" in delta or "image_url" in delta or "tool_calls" in delta:
                            yield chunk
                            continue

                        content = delta.get("content", "")
                        if not content:
                            # Just keep alive chunks that have an empty structure or no text
                            if not finish_reason and not delta:
                                yield chunk
                            continue

                        buffer += content
                        raw_response += content

                        while True:
                            processed = False
                            for tag_start, tag_end in [("<think>", "</think>"), ("<thought>", "</thought>"), ("<|thought|>", "<|thought|>"), ("<|channel>thought", "<channel|>")]:
                                if not tag_opened and tag_start in buffer:
                                    parts = buffer.split(tag_start, 1)
                                    if parts[0]:
                                        chunk_out = copy.deepcopy(chunk)
                                        chunk_out["choices"][0]["delta"] = {"content": parts[0]}
                                        yield chunk_out
                                    buffer = parts[1]
                                    tag_opened = True
                                    processed = True
                                    break

                                if tag_opened and tag_end in buffer:
                                    parts = buffer.split(tag_end, 1)
                                    if parts[0]:
                                        chunk_out = copy.deepcopy(chunk)
                                        chunk_out["choices"][0]["delta"] = {"reasoning_content": parts[0]}
                                        yield chunk_out
                                    buffer = parts[1]
                                    tag_opened = False
                                    processed = True
                                    break

                            if not processed:
                                break

                        # Keep only the last 20 characters in the buffer to make sure we
                        # don't break a tag in the middle. Send everything else.
                        if len(buffer) > 20:
                            safe_yield = buffer[:-20]
                            buffer = buffer[-20:]
                            if safe_yield:
                                target_key = 'content' if not tag_opened else 'reasoning_content'
                                chunk_out = copy.deepcopy(chunk)
                                chunk_out["choices"][0]["delta"] = {target_key: safe_yield}
                                yield chunk_out

                    # Flush any leftover buffer residue
                    if buffer and chunk is not None:
                        target_key = 'content' if not tag_opened else 'reasoning_content'
                        chunk_out = copy.deepcopy(chunk)
                        chunk_out["choices"][0]["delta"] = {target_key: buffer}
                        yield chunk_out

                except RuntimeError as ctx_err:
                    if _is_ctx_overflow(ctx_err):
                        logger.warning(f"GGUF: Context window overflow (stream) — finish_reason='length': {ctx_err}")
                        yield {
                            "id": f"chatcmpl-ctx-{uuid.uuid4().hex[:8]}",
                            "object": "chat.completion.chunk",
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
                        }
                    else:
                        raise
                finally:
                    self.allocator.release(seq_id)
                    logger.debug("GGUF: seq_id liberado de volta ao alocador")

            if stream:
                return stream_adapter()

            # NON-STREAM MODE: consumes the same stream_adapter() (already separates
            # content/reasoning_content/tool_calls) and assembles the full response at
            # the end — generation itself runs exactly the same way (interleaved with
            # other sessions active in the dispatcher); the only difference is whether
            # the client sees the chunks as they come out, or only gets the finished
            # response at the end.
            content_parts: List[str] = []
            reasoning_parts: List[str] = []
            tool_calls_acc: List[dict] = []
            final_finish_reason = "stop"
            final_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            response_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())

            try:
                async for chunk in stream_adapter():
                    response_id = chunk.get("id", response_id)
                    created = chunk.get("created", created)
                    choice = chunk["choices"][0]
                    delta = choice.get("delta", {})

                    if delta.get("content"):
                        content_parts.append(delta["content"])
                    if delta.get("reasoning_content"):
                        reasoning_parts.append(delta["reasoning_content"])
                    if delta.get("tool_calls"):
                        _accumulate_tool_call_deltas(tool_calls_acc, delta["tool_calls"])

                    if choice.get("finish_reason"):
                        final_finish_reason = choice["finish_reason"]
                    if chunk.get("usage"):
                        final_usage = chunk["usage"]
            except Exception as e:
                if _is_ctx_overflow(e):
                    logger.warning(f"GGUF: Context window overflow (non-stream) — finish_reason='length': {e}")
                    return {
                        "id": response_id,
                        "object": "chat.completion",
                        "created": created,
                        "model": self.meta.get("model_alias", "unknown"),
                        "choices": [{
                            "index": 0,
                            "message": {"role": "assistant", "content": ""},
                            "finish_reason": "length",
                        }],
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    }
                raise

            message: Dict[str, Any] = {"role": "assistant"}
            content = "".join(content_parts)
            message["content"] = content if content else None
            if reasoning_parts:
                message["reasoning_content"] = "".join(reasoning_parts)
            if tool_calls_acc:
                message["tool_calls"] = tool_calls_acc

            return {
                "id": response_id,
                "object": "chat.completion",
                "created": created,
                "model": self.meta.get("model_alias", "unknown"),
                "choices": [{
                    "index": 0,
                    "message": message,
                    "finish_reason": final_finish_reason,
                }],
                "usage": final_usage,
            }
        except Exception:
            # If we failed even before submitting the session to the dispatcher (or
            # stream_adapter() itself won't run its finally), make sure the seq doesn't
            # stay stuck.
            self.allocator.release(seq_id)
            raise

    def unload(self, model_name: str):
        logger.info(f"GGUF: Explicitly unloading {model_name}...")

        try:
            acquired = acquired_instances_var.get()
            if self.allocator in acquired:
                del acquired[self.allocator]
                acquired_instances_var.set(acquired)
        except Exception as e:
            logger.debug(f"GGUF: Non-critical error clearing context vars: {e}")

        # decode_gate and (if MTMD) chat_handler.encode_lock live on `self.model` itself
        # (fork) — they die together with it below, no cleanup needed here on our own.

        if self.allocator:
            try:
                self.allocator.stop()
            except Exception as e:
                logger.warning(f"GGUF: Error stopping allocator: {e}")
            self.allocator = None

        if self.model:
            if hasattr(self.model, 'close'):
                try:
                    logger.debug(f"GGUF: Calling model.close() for {model_name}")
                    self.model.close()
                except Exception as e:
                    logger.warning(f"GGUF: Error calling close() on model: {e}")

            # Explicitly deleting calls the destructor (__del__) in C++
            del self.model
            self.model = None

        import gc
        gc.collect()

        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

        logger.info(f"GGUF: Finished unloading {model_name}")

    def is_loaded(self):
        return self.model is not None
