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

# Global locks for model loading to prevent concurrent loads of the same file across GGUF instances
_GGUF_LOAD_LOCKS = {}
_GGUF_LOAD_LOCKS_LOCK = threading.Lock()

def get_gguf_load_lock(model_path: str):
    with _GGUF_LOAD_LOCKS_LOCK:
        if model_path not in _GGUF_LOAD_LOCKS:
            _GGUF_LOAD_LOCKS[model_path] = threading.Lock()
        return _GGUF_LOAD_LOCKS[model_path]

import re


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

class ReentrantAsyncLock:
    def __init__(self, seq_id=0):
        self._lock = asyncio.Lock()
        self._owner = None
        self._count = 0
        self._seq_id = seq_id
        self._token = None

    async def acquire(self):
        rid = request_id_var.get()
        if rid and self._owner == rid:
            self._count += 1
            return
        await self._lock.acquire()
        self._owner = rid
        self._count = 1
        try:
            from llama_cpp.llama import active_seq_id
            self._token = active_seq_id.set(self._seq_id)
        except ImportError:
            pass

    def release(self):
        rid = request_id_var.get()
        if rid and self._owner == rid:
            self._count -= 1
            if self._count == 0:
                self._owner = None
                if self._token:
                    try:
                        from llama_cpp.llama import active_seq_id
                        # active_seq_id.reset(self._token)
                    except Exception:
                        pass
                    self._token = None
                self._lock.release()
        else:
            self._owner = None
            self._count = 0
            if self._token:
                try:
                    from llama_cpp.llama import active_seq_id
                    # active_seq_id.reset(self._token)
                except Exception:
                    pass
                self._token = None
            if self._lock.locked():
                self._lock.release()

    async def __aenter__(self):
        await self.acquire()

    async def __aexit__(self, exc_type, exc, tb):
        self.release()

class SequenceSlot:
    def __init__(self, model: Any, seq_id: int):
        self.model = model
        self.seq_id = seq_id
        self.lock = ReentrantAsyncLock(seq_id=seq_id) # Lock reentrante to ensure only one request uses this slot/seq_id at a time

    def __getattr__(self, name):
        return getattr(self.model, name)

    def create_chat_completion(self, *args, **kwargs):
        kwargs["seq_id"] = self.seq_id
        return self.model.create_chat_completion(*args, **kwargs)
    
    def create_completion(self, *args, **kwargs):
        kwargs["seq_id"] = self.seq_id
        return self.model.create_completion(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        kwargs["seq_id"] = self.seq_id
        return self.model.generate(*args, **kwargs)

    def reset(self):
        from llama_cpp.llama import active_seq_id
        token = active_seq_id.set(self.seq_id)
        try:
            self.model.reset()
        finally:
            # active_seq_id.reset(token)
            pass

class SequencePool:
    def __init__(self, slots: List[SequenceSlot]):
        self.slots = slots
        self._available = asyncio.Queue()
        self._allocated = set()
        self._pool_lock = asyncio.Lock()
        for slot in slots:
            self._available.put_nowait(slot)

    async def acquire(self) -> SequenceSlot:
        """Adquire um slot de sequência livre, com suporte a re-entrância por request_id."""
        rid = request_id_var.get()
        
        async with self._pool_lock:
            if rid:
                acquired = acquired_instances_var.get()
                if self in acquired:
                    # Re-entrância: Esta request já reservou uma instância deste pool
                    return acquired[self]
        
        # Caso contrário, espera por uma instância livre no pool
        # Fora do _pool_lock para permitir que outros chamem acquire enquanto um espera
        slot = await self._available.get()
        
        async with self._pool_lock:
            self._allocated.add(slot)
            if rid:
                # Reserva a instância para futuras chamadas nesta mesma request
                acquired = acquired_instances_var.get()
                acquired[self] = slot
                acquired_instances_var.set(acquired)
            
        return slot

    def release(self, slot: SequenceSlot):
        """Libera a instância, a menos que esteja reservada para re-entrância."""
        rid = request_id_var.get()
        if rid:
            # Em requests HTTP rastreadas, não liberamos imediatamente pois o
            # segundo turno pode precisar da mesma instância.
            # A liberação real ocorrerá no Middleware ao fim da request.
            return
            
        self._real_release(slot)

    def _real_release(self, slot: SequenceSlot):
        """Põe a instância de volta na fila de disponibilidade."""
        if slot in self._allocated:
            self._allocated.remove(slot)
            self._available.put_nowait(slot)

    def _force_release(self, slot: SequenceSlot):
        """Força a liberação ignorando o request_id (usado pelo middleware)."""
        self._real_release(slot)

    def stop(self):
        """Para o pool e limpa referências para permitir coleta de lixo."""
        self.slots = []
        while not self._available.empty():
            try:
                self._available.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._allocated.clear()

class GGUF:
    def __init__(self, backend, model):
        from huggingface_hub import hf_hub_download

        self.backend = backend
        self.meta = model
        self.cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))
        self.model = None
        self.model_path = hf_hub_download(repo_id=model["model_id"], filename=model["filename"])
        self.model_metadata = ModelMetadata(self.model_path)
        self.sequence_pool: SequencePool = None
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
                    base_params["chat_format"] = "jinja" if "jinja" in llama_cpp.llama_chat_format.CHAT_FORMATS else caps["chat_format"]

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
                            base_params["chat_handler"] = Qwen35Handler(clip_model_path=mmproj_path, **handler_kwargs) if mmproj_path else Qwen35Handler(**handler_kwargs)
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
                self.slots = [SequenceSlot(self.model, i) for i in range(self.n_seq_max)]
                self.sequence_pool = SequencePool(self.slots)
                
            except Exception as e:
                logger.error(f"GGUF: Failed to load {self.meta['model_alias']}: {e}")
                raise e

    async def run_chat(self, messages: List[Dict[str, Any]], stream: bool = False, **kwargs):
        headers = kwargs.pop("headers", {})
        session_id = headers.get("x-session-id") or kwargs.pop("session_id", None)
        n_ctx = headers.get("x-context-window", None)
        
        if n_ctx:
            try:
                n_ctx = int(n_ctx)
            except ValueError:
                pass

        self.load(n_ctx, kwargs.pop('num_layers', None))

        if "repetition_penalty" in kwargs:
            kwargs["repeat_penalty"] = kwargs.pop("repetition_penalty")
            
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        kwargs.pop("reasoning_enabled", None)
        kwargs.pop("reasoning_effort", None)
        
        force_reasoning = self.meta.get("force_reasoning", False)

        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = -1

        # O Truque do Force Reasoning: A IA começa já pensando
        if force_reasoning:
            messages.append({"role": "assistant", "content": "<think>\n"})

        # Adquire um slot do pool
        slot = await self.sequence_pool.acquire()
        
        # O lock do slot garante que apenas uma operação ocorra por seq_id
        # Se for uma chamada re-entrante da mesma request, ela esperará aqui
        # até que a operação anterior (se houver) termine.
        # Nota: Chamadas de ferramenta geralmente são sequenciais.
        async with slot.lock:
            # Sempre reseta para garantir que não há lixo no KV Cache
            try:
                pass
                # slot.reset()
                # logger.debug(f"GGUF: Sequence slot KV cache reset")
            except Exception as e:
                logger.warning(f"GGUF: Could not reset slot: {e}")

            try:
                response = slot.create_chat_completion(
                    messages,
                    stream=stream,
                    **kwargs
                )
            except Exception as e:
                self.sequence_pool.release(slot)
                raise e

            if stream:
                async def stream_adapter():
                    # Re-adquire o lock para o processamento do stream
                    async with slot.lock:
                        try:
                            tag_opened = force_reasoning
                            buffer = ""
                            raw_response = ""
                            
                            for chunk in response:
                                delta = chunk["choices"][0].get("delta", {})
                                
                                finish_reason = chunk["choices"][0].get("finish_reason")
                                if finish_reason in ["stop", "tool_calls"]:
                                    prompt_data = ""

                                    for message in messages:
                                        # prompt_data += message["content"] if isinstance(message["content"], str) else "".join([c.get("text", "") for c in message['content']])
                                        prompt_data += "\n".join(c['text'] for c in message.get("content")) if isinstance(message.get("content"), list) else (message.get("content") or "")

                                    prompt_tokens = len(slot.tokenize(prompt_data.encode('utf-8')))
                                    completion_tokens = len(slot.tokenize(raw_response.encode('utf-8')))

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

                                await asyncio.sleep(0)

                            # Flush residual do buffer se sobrar algo
                            if buffer:
                                target_key = 'content' if not tag_opened else 'reasoning_content'
                                chunk_out = copy.deepcopy(chunk)
                                chunk_out["choices"][0]["delta"] = {target_key: buffer}
                                yield chunk_out
                                
                        finally:
                            # Libera o slot ao fim do stream
                            self.sequence_pool.release(slot)
                            logger.debug(f"GGUF: Sequence slot released back to pool")

                return stream_adapter()
                
            else:
                try:
                    # MODO NÃO-STREAM: Limpa a tag e separa tudo na raiz do JSON
                    raw_content = response["choices"][0]["message"].get("content", "") or ""

                    if force_reasoning:
                        raw_content = "<think>" + raw_content
                    
                    # Parser robusto para tags <think>, <thought> ou <|thought|> mesmo não fechadas
                    think_pattern = re.compile(r"<(?:\|thought\||think|thought)>(.*?)(?:</(?:think|thought)>|(?=<\|)|$)", re.DOTALL)
                    match = think_pattern.search(raw_content)
                    
                    if match:
                        reasoning = match.group(1).strip()
                        # Remove o bloco de pensamento do conteúdo principal
                        # Usamos match.group(0) para remover a tag inteira
                        content = raw_content.replace(match.group(0), "").strip()
                        
                        # Cleanup de tags de fechamento remanescentes se necessário
                        for close_tag in ["</think>", "</thought>", "<channel|>"]:
                            content = content.replace(close_tag, "").strip()
                        
                        response["choices"][0]["message"]["reasoning_content"] = reasoning
                        response["choices"][0]["message"]["content"] = content if content else None
                    
                    return response
                finally:
                    # Libera o slot após o processamento não-stream
                    self.sequence_pool.release(slot)
                    logger.debug(f"GGUF: Sequence slot released back to pool")

    def unload(self, model_name: str):
        logger.info(f"GGUF: Explicitly unloading {model_name}...")
        
        try:
            acquired = acquired_instances_var.get()
            if self.sequence_pool in acquired:
                del acquired[self.sequence_pool]
                acquired_instances_var.set(acquired)
        except Exception as e:
            logger.debug(f"GGUF: Non-critical error clearing context vars: {e}")

        if self.sequence_pool:
            try:
                self.sequence_pool.stop()
            except Exception as e:
                logger.warning(f"GGUF: Error stopping sequence pool: {e}")
            self.sequence_pool = None
            
        if hasattr(self, 'slots') and self.slots:
            for slot in self.slots:
                slot.model = None # Remove referência circular
                # Se o lock reentrante ficou preso, forçamos a soltura
                if hasattr(slot, 'lock') and slot.lock._token:
                    try:
                        slot.lock.release()
                    except Exception:
                        pass
            self.slots = []

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