import asyncio
import os
import re
import json
import logging
import time
import uuid
import threading
import copy
from typing import Any, Dict, List, Optional
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
    Usa RegEx para mapear o modelo para os Handlers específicos do fork JamePeng.
    """
    name = repo_id_or_filename.lower()
    capabilities = {
        "vision_handler": None,
        "chat_format": "jinja"
    }

    # ==========================================
    # 1. DETECÇÃO DO CHAT FORMAT FALLBACK
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
    # 2. DETECÇÃO DE VISÃO (JAMEPENG HANDLERS)
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
    Aloca seq_ids livres para requests, com afinidade de KV cache por conversation_id.

    NÃO faz nenhuma chamada nativa ao llama.cpp — é só bookkeeping de "qual seq_id está
    livre" e "qual conversation_id cada seq_id guarda atualmente". A exclusão mútua sobre
    as chamadas nativas em si é garantida por outro mecanismo: o GGUFDispatcher é a ÚNICA
    coroutine que chama a Llama compartilhada, então não existe mais nada aqui que precise
    de lock por slot.
    """

    def __init__(self, n_seq_max: int):
        self.n_seq_max = n_seq_max
        self._available: "asyncio.Queue[int]" = asyncio.Queue()
        self._allocated: set = set()
        self._pool_lock = asyncio.Lock()
        # qual conversation_id cada seq_id está segurando (KV cache) no momento
        self.seq_conversation: Dict[int, Optional[str]] = {}
        for i in range(n_seq_max):
            self._available.put_nowait(i)
            self.seq_conversation[i] = None

    async def acquire(self, conversation_id: Optional[str] = None) -> "tuple[int, bool]":
        """Adquire um seq_id livre, com suporte a re-entrância por request_id.

        Retorna (seq_id, is_new_conversation).
        - is_new_conversation=False → reentrância (mesma request HTTP) OU o seq_id já
          guardava o KV cache dessa exata conversation_id — NÃO resetar, deixar o próprio
          generate() reaproveitar o prefixo do KV cache automaticamente.
        - is_new_conversation=True  → esse seq_id está trocando de dono (conversa
          diferente da que ele guardava antes, ou nunca guardou nenhuma) — o chamador deve
          fazer um reset explícito antes de gerar.
        """
        rid = request_id_var.get()

        async with self._pool_lock:
            if rid:
                acquired = acquired_instances_var.get()
                if self in acquired:
                    # Re-entrância: Esta request já reservou uma seq deste pool
                    return acquired[self], False

            # Tenta achar o seq_id que já guarda o KV cache dessa conversation_id
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
                    # Slot preferido ocupado agora — cai pra espera normal abaixo

        # Espera por uma seq livre (fora do _pool_lock para não bloquear releases)
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
        """Libera a seq, a menos que esteja reservada para re-entrância."""
        rid = request_id_var.get()
        if rid:
            # Em requests HTTP rastreadas, não liberamos imediatamente pois o
            # segundo turno pode precisar da mesma instância.
            # A liberação real ocorrerá no Middleware ao fim da request.
            return
        self._real_release(seq_id)

    def _real_release(self, seq_id: int):
        """Põe a seq de volta na fila de disponibilidade (mantém a afinidade de conversa)."""
        if seq_id in self._allocated:
            self._allocated.remove(seq_id)
            self._available.put_nowait(seq_id)

    def _force_release(self, seq_id: int):
        """Força a liberação ignorando o request_id (usado pelo middleware)."""
        self._real_release(seq_id)

    def stop(self):
        """Para o alocador e limpa referências para permitir coleta de lixo."""
        self._available = asyncio.Queue()
        self._allocated.clear()
        self.seq_conversation.clear()


_SENTINEL = object()


class GenerationSession:
    """Uma geração em andamento, servida pelo GGUFDispatcher."""

    def __init__(self, seq_id: int, create_kwargs: dict):
        self.seq_id = seq_id
        self.create_kwargs = create_kwargs
        self.gen = None  # generator vivo de create_chat_completion(stream=True), criado no 1º passo
        self.done = False
        self.error: Optional[BaseException] = None
        self.queue: "asyncio.Queue" = asyncio.Queue()


class GGUFDispatcher:
    """
    Única coroutine que chama métodos de geração da instância `Llama` compartilhada.

    Round-robin single-thread: a cada volta do loop, dá EXATAMENTE um passo (um `next()`
    no generator de `create_chat_completion(stream=True, seq_id=...)`) para cada sessão
    ativa, antes de passar pra próxima. Isso intercala N gerações de verdade — cada
    cliente recebe tokens em paralelo, não em blocos — sem nunca ter duas chamadas
    nativas ao llama.cpp simultâneas, porque só existe esta coroutine chamando o C.
    """

    def __init__(self, model):
        self.model = model
        self._sessions: Dict[int, GenerationSession] = {}
        self._wakeup = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    async def ensure_started(self):
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    def stop(self):
        if self._task is not None and not self._task.done():
            try:
                loop = asyncio.get_running_loop()
                loop.call_soon_threadsafe(self._task.cancel)
            except RuntimeError:
                # Sem loop rodando nesta thread (ex.: unload chamado via to_thread) —
                # melhor esforço, cancela direto.
                self._task.cancel()
        self._task = None
        self._sessions.clear()

    def submit(self, session: GenerationSession):
        self._sessions[session.seq_id] = session
        self._wakeup.set()

    async def _run(self):
        from llama_cpp.llama import active_seq_id

        while True:
            active = [s for s in self._sessions.values() if not s.done]
            if not active:
                self._wakeup.clear()
                await self._wakeup.wait()
                continue

            for session in active:
                if session.done:
                    continue

                active_seq_id.set(session.seq_id)
                try:
                    if session.gen is None:
                        session.gen = self.model.create_chat_completion(**session.create_kwargs)
                    chunk = next(session.gen, _SENTINEL)
                except StopIteration:
                    chunk = _SENTINEL
                except Exception as e:
                    session.error = e
                    session.done = True
                    self._sessions.pop(session.seq_id, None)
                    session.queue.put_nowait(None)
                    continue

                if chunk is _SENTINEL:
                    session.done = True
                    self._sessions.pop(session.seq_id, None)
                    session.queue.put_nowait(None)
                else:
                    session.queue.put_nowait(chunk)
                    finish_reason = chunk.get("choices", [{}])[0].get("finish_reason")
                    if finish_reason is not None:
                        session.done = True
                        self._sessions.pop(session.seq_id, None)
                        session.queue.put_nowait(None)

            # Cede o controle pro event loop do asyncio entre voltas (não pro llama.cpp —
            # só esta coroutine chama o C, nunca duas ao mesmo tempo). Isso deixa outras
            # rotas do FastAPI (/health, outros modelos, etc.) responderem no meio das
            # voltas do dispatcher.
            await asyncio.sleep(0)


def _accumulate_tool_call_deltas(acc: List[dict], delta_tool_calls: List[dict]):
    """Acumula deltas de tool_calls no formato de streaming da OpenAI (por índice,
    arguments chega fragmentado e precisa ser concatenado)."""
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
        self.dispatcher: Optional[GGUFDispatcher] = None
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

        # Se for -1, resolvemos o total antes de decrementar
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

        # O contexto real na GPU é multiplicado pelo número de slots paralelos
        effective_n_ctx = n_ctx_per_req * self.n_seq_max

        # 2. Get layers config
        # Estima a VRAM necessária usando o calculador de recursos unificado
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

        # Aumentamos o padrão para 4096 para suportar agentes mais complexos
        n_ctx_per_req = n_ctx or int(meta.get("context_window") or os.environ.get("LLM_CTX_TOKENS", '4096'))
        effective_n_ctx = n_ctx_per_req * self.n_seq_max

        # Serialização de carregamento do mesmo arquivo de modelo
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

                # Extração de Capacidades
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

                # Cada seq_id precisa do seu próprio Chat Handler (estado independente)
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
                self.dispatcher = GGUFDispatcher(self.model)

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
        await self.dispatcher.ensure_started()

        if "repetition_penalty" in kwargs:
            kwargs["repeat_penalty"] = kwargs.pop("repetition_penalty")

        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        kwargs.pop("reasoning_enabled", None)
        kwargs.pop("reasoning_effort", None)

        # FIX: Se response_format for json_object, repeat_penalty DEVE ser 1.0
        # Caso contrário o llama.cpp penaliza os caracteres da gramática JSON e entra em loop de espaços
        fmt = kwargs.get("response_format")
        if isinstance(fmt, dict) and fmt.get("type") == "json_object":
            kwargs["repeat_penalty"] = 1.0

        force_reasoning = self.meta.get("force_reasoning", False)

        if "max_completion_tokens" in kwargs:
            kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")

        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = -1

        # O Truque do Force Reasoning: A IA começa já pensando
        if force_reasoning:
            messages.append({"role": "assistant", "content": "<think>\n"})

        # Adquire um seq_id do alocador — usa conversation_id (x-session-id) para affinity de KV cache
        seq_id, is_new_conversation = await self.allocator.acquire(conversation_id=conversation_id)

        try:
            # Só resetamos o KV cache quando essa seq_id está trocando de dono (conversa
            # diferente da que ela guardava). Se é a mesma conversa retomando (ou uma
            # chamada re-entrante da mesma request), NÃO resetamos — deixamos o próprio
            # generate() reaproveitar o prefixo do KV cache automaticamente.
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

            session = GenerationSession(seq_id=seq_id, create_kwargs=create_kwargs)
            self.dispatcher.submit(session)

            async def _raw_chunks():
                """Consome os chunks do dispatcher um a um — substitui o antigo
                `await asyncio.to_thread(next, response, None)`."""
                while True:
                    chunk = await session.queue.get()
                    if chunk is None:
                        if session.error is not None:
                            raise session.error
                        break
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

                            # Se recebemos um STOP e ainda há buffer, enviamos antes do chunk de stop
                            if buffer:
                                target_key = 'content' if not tag_opened else 'reasoning_content'
                                chunk_flush = copy.deepcopy(chunk)
                                chunk_flush["choices"][0]["delta"] = {target_key: buffer}
                                chunk_flush["choices"][0]["finish_reason"] = None
                                buffer = ""
                                yield chunk_flush

                            yield chunk
                            continue

                        # Pass-through para outros dados que não sejam texto puro
                        if "audio_url" in delta or "image_url" in delta or "tool_calls" in delta:
                            yield chunk
                            continue

                        content = delta.get("content", "")
                        if not content:
                            # Apenas manter vivos chunks que tenham estrutura vazia ou sem texto
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

                        # Retém apenas os últimos 20 caracteres no buffer para garantir que não vamos
                        # quebrar uma tag no meio. Envia todo o resto.
                        if len(buffer) > 20:
                            safe_yield = buffer[:-20]
                            buffer = buffer[-20:]
                            if safe_yield:
                                target_key = 'content' if not tag_opened else 'reasoning_content'
                                chunk_out = copy.deepcopy(chunk)
                                chunk_out["choices"][0]["delta"] = {target_key: safe_yield}
                                yield chunk_out

                    # Flush residual do buffer se sobrar algo
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

            # MODO NÃO-STREAM: consome o mesmo stream_adapter() (já separa content/
            # reasoning_content/tool_calls) e monta a resposta completa no final —
            # a geração em si roda exatamente do mesmo jeito (intercalada com outras
            # sequências ativas no dispatcher); só muda se o cliente vê os chunks
            # conforme saem, ou só recebe a resposta pronta no fim.
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
            # Se falhamos antes mesmo de submeter a sessão ao dispatcher (ou o próprio
            # stream_adapter() já não vai rodar seu finally), garante que a seq não fica presa.
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

        if self.dispatcher:
            try:
                self.dispatcher.stop()
            except Exception as e:
                logger.warning(f"GGUF: Error stopping dispatcher: {e}")
            self.dispatcher = None

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

            # Deletar explicitamente chama o destrutor (__del__) no C++
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
