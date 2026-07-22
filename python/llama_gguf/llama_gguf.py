import asyncio
import os
import re
import json
import logging
import time
import uuid
import threading
import copy
from dataclasses import dataclass, field
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
    as chamadas nativas em si é garantida por outro mecanismo: o `DecodeGate` é o ÚNICO
    ponto que chama a Llama compartilhada, então não existe mais nada aqui que precise
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


@dataclass
class DecodeRequest:
    """Um pedido de decode: adicionar `tokens` ao KV cache do `seq_id`, a partir da
    posição `pos`, e devolver o índice de logits do ÚLTIMO token (pra amostrar o próximo
    token dessa sessão). Prefill (muitos tokens) e geração em regime estacionário (1
    token) usam a MESMA classe — só varia o tamanho de `tokens`; é a mesma operação."""
    seq_id: int
    tokens: List[int]
    pos: int
    result_idx: int = -1
    error: Optional[BaseException] = None
    event: "asyncio.Event" = field(default_factory=asyncio.Event)


class DecodeGate:
    """
    Único ponto de exclusão mútua por instância `GGUF` sobre chamadas nativas de
    `decode()`. Substitui o antigo `GGUFDispatcher` (round-robin, 1 `decode()` por sessão
    por rodada) por fila + eleição de líder: quem chega e acha a fila livre vira líder,
    drena TUDO que estiver esperando naquele instante (não só quem chegou primeiro),
    monta um `LlamaBatch` com as contribuições de todos, decodifica UMA VEZ, distribui os
    resultados, então recheca a fila antes de largar a liderança. Sem espera artificial —
    o líder processa imediatamente quem já estiver esperando, seja 1 ou N pedidos.

    Prefill e geração em regime estacionário passam pela MESMA fila/mecanismo — não são
    dois caminhos diferentes. Isso evita por construção o bug (encontrado num protótipo
    anterior) de reamostrar o último token do prompt numa posição nova pra "encaixar" num
    modelo de rodada: aqui não existe rodada, só "submeta tokens, peça logits do último".

    `run_exclusive_step()` adquire a MESMA exclusão mútua pra rodar uma função síncrona
    arbitrária (usada pelo caminho de fallback — mensagens com mídia de verdade, ou
    handlers que não sabemos decompor com segurança — que reaproveita
    `create_chat_completion()` inteiro, um passo por vez, exatamente como o dispatcher
    round-robin antigo fazia). Garante que essas chamadas NUNCA corram ao mesmo tempo que
    uma rodada batchada de outra sessão — só existe UM lock por instância de modelo.
    """

    def __init__(self, model):
        self.model = model
        self._pending: List[DecodeRequest] = []
        self._lock = asyncio.Lock()

    async def submit(self, req: DecodeRequest) -> int:
        self._pending.append(req)
        async with self._lock:
            # Pode ser que, enquanto esperávamos o lock, um líder anterior já tenha
            # drenado a fila e processado este pedido — não reprocessar.
            if not req.event.is_set():
                await self._drain_and_decode()
        if req.error is not None:
            raise req.error
        return req.result_idx

    async def run_exclusive_step(self, fn):
        """Roda `fn()` (síncrono, sem argumentos) com exclusividade total sobre o
        modelo — nenhuma outra sessão (batchada ou não) decodifica enquanto isso."""
        async with self._lock:
            return fn()

    async def _drain_and_decode(self):
        if not self._pending:
            return
        batch_reqs, self._pending = self._pending, []

        from llama_cpp.llama import active_seq_id

        try:
            batch = self.model._batch
            batch.reset()
            batch_pos = 0  # posição BRUTA no batch — get_logits_ith() espera a posição
                            # real, não um contador compactado de quantos pedidos passaram
            for r in batch_reqs:
                active_seq_id.set(r.seq_id)
                pos = r.pos
                last_i = len(r.tokens) - 1
                for i, tok in enumerate(r.tokens):
                    is_last = (i == last_i)
                    batch.add_token(tok, pos, [r.seq_id], is_last)
                    pos += 1
                    if is_last:
                        r.result_idx = batch_pos
                    batch_pos += 1

            ret = self.model._ctx.decode(batch)
            if ret != 0:
                raise RuntimeError(f"llama_decode retornou {ret} (sem slot de KV cache disponível)")
        except Exception as e:
            for r in batch_reqs:
                r.error = e

        for r in batch_reqs:
            r.event.set()

        # Cede o controle pro event loop mesmo sem contenção no lock — um `asyncio.Lock`
        # descontendido não suspende sozinho, então sem isso uma sessão gerando muitos
        # tokens em sequência (sem nenhuma outra sessão concorrente) nunca devolveria o
        # controle pro event loop, travando outras rotas do FastAPI até ela terminar.
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


_NAMED_FORMATTERS: Optional[Dict[str, Any]] = None


def _get_named_formatter(chat_format: Optional[str]):
    """Formatos nomeados (chatml/llama-2/llama-3/mistral-instruct/gemma) não ficam
    guardados em nenhum registro público como `ChatFormatter` cru — só como handler já
    empacotado (`LlamaChatCompletionHandlerRegistry`). As funções puras continuam
    acessíveis diretamente pelo nome no módulo, então mapeamos manualmente aqui."""
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
    """Resultado de `_prepare_session`: tudo que é necessário pra gerar via `DecodeGate`
    sem precisar chamar `create_chat_completion()` (que geraria de forma não-batchável).
    `post_handler`, quando presente, precisa ter seu `_parse_response(response)` chamado
    no texto final montado (ver `_run_batched_generation_collect`), preservando o parsing
    de tool_call/reasoning que os handlers Qwen/Phi/MTMD já fazem hoje."""
    prompt_tokens: List[int]
    stop: List[str]
    stopping_criteria: Optional[Any]
    grammar_str: str
    post_handler: Optional[Any]
    messages_norm: List[Dict[str, Any]]


def _prepare_session(llm, messages: List[Dict[str, Any]], **kwargs) -> Optional[PreparedGeneration]:
    """
    Resolve o `ChatFormatter` certo pro handler/formato configurado em `llm` (a instância
    `Llama`) e monta tudo que é necessário pra gerar via `DecodeGate` (tokens, stop,
    stopping_criteria, grammar) — sem chamar geração nenhuma. Espelha a lógica de
    `chat_formatter_to_chat_completion_handler` (que faz render+tokenize+grammar antes de
    gerar de verdade), só que parando antes do `create_completion()`.

    Recebe `**kwargs` (o payload da request, que inclui uma chave `model` com o ALIAS do
    modelo, não a instância — daí o parâmetro se chamar `llm`, não `model`, evitando
    colisão).

    Devolve `None` se não souber decompor esse handler/mensagem com segurança — quem
    chama deve cair no caminho de fallback (`DecodeGate.run_exclusive_step` +
    `create_chat_completion()` inteiro, igual o dispatcher round-robin antigo fazia):
    - mensagem com mídia de verdade endereçada a um handler MTMD (Gemma4/Qwen3.5-vision/
      etc.) — o `__call__` desses handlers sempre muta o KV cache inline como parte do
      preparo do prompt, não é separável;
    - qualquer `chat_handler` customizado sem um jeito conhecido de extrair só o prompt
      (não é Qwen/Phi nem MTMD);
    - `chat_format` sem formatter resolvível (nem em `llm._chat_formatters`, nem um dos
      formatos nomeados conhecidos) — o mesmo caso que já falharia hoje.
    """
    from llama_cpp.llama_chat_format import MTMDChatHandler

    chat_handler = getattr(llm, "chat_handler", None)
    formatter = None
    post_handler = None
    is_mtmd_no_media = False

    if chat_handler is not None:
        if isinstance(chat_handler, MTMDChatHandler):
            media = chat_handler._get_media_items(messages)
            if media:
                return None  # mídia de verdade: precisa do __call__ completo (fallback)
            is_mtmd_no_media = True
            post_handler = chat_handler
        elif callable(getattr(chat_handler, "formatter", None)):
            # Convenção Qwen/PhiChatHandler: self.formatter é o ChatFormatter puro
            formatter = chat_handler.formatter
            post_handler = chat_handler
        else:
            return None  # handler customizado desconhecido: fallback
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

    if is_mtmd_no_media:
        chat_handler._init_mtmd_context(llm)
        full_prompt_ids, _spans, chunks, bitmap_cleanup = chat_handler._process_mtmd_prompt(
            llama=llm,
            messages=messages_norm,
            functions=functions,
            function_call=function_call,
            tools=tools,
            tool_choice=tool_choice,
            add_generation_prompt=True,
        )
        # Sem mídia, não há nada pra reter — libera os recursos C imediatamente (mesma
        # limpeza que o __call__ do handler faz no seu próprio bloco finally).
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


async def _generate_via_gate(model, gate: DecodeGate, seq_id: int, prepared: PreparedGeneration, kwargs: dict):
    """
    Núcleo comum de geração via `DecodeGate`: reaproveita o KV cache já existente pra
    essa conversa (`model._reuse_prefix_and_eval`), monta um `LlamaSamplingContext` por
    sessão (grammar/reasoning-budget/dry/mirostat/etc. inclusos), e faz um generator
    Python simples (não-async) que a cada `next()` submete o próximo passo ao gate e
    devolve o token amostrado — junto com o texto incremental e o motivo de parada.

    Yields (new_text: bytes, finish_reason: Optional[str]) — o chamador decide o que
    fazer com cada pedaço (emitir incrementalmente, ou acumular até o fim).
    """
    from llama_cpp.llama import active_seq_id, StopStringMatcher
    from llama_cpp._internals import LlamaSamplingContext
    from llama_cpp import llama_cpp as llama_cpp_lib

    active_seq_id.set(seq_id)
    delta_tokens = model._reuse_prefix_and_eval(prepared.prompt_tokens, seq_id=seq_id, reset=True)
    pos = model.n_tokens

    sampling_params = _build_sampling_params(model, kwargs, prepared.grammar_str)
    sampling_ctx = LlamaSamplingContext(sampling_params, model._model)
    has_grammar = bool(prepared.grammar_str)

    max_tokens = kwargs.get("max_tokens")
    if not max_tokens or max_tokens <= 0:
        max_tokens = model._n_ctx - len(prepared.prompt_tokens)

    req = DecodeRequest(seq_id=seq_id, tokens=delta_tokens, pos=pos)
    idx = await gate.submit(req)
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

        req = DecodeRequest(seq_id=seq_id, tokens=[token], pos=pos)
        idx = await gate.submit(req)
        pos += 1

        active_seq_id.set(seq_id)
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


async def _run_batched_generation_stream(model, gate, seq_id, prepared, kwargs, response_id, created, model_name):
    """Streaming incremental de verdade via `DecodeGate` — só usado quando NÃO há
    `post_handler` (formato nomeado ou jinja genérico, sem parsing extra de tool_call no
    final), onde cada pedaço de texto pode ser entregue assim que é gerado."""
    yield _chat_chunk(response_id, created, model_name, delta={"role": "assistant"})
    async for new_text, finish_reason in _generate_via_gate(model, gate, seq_id, prepared, kwargs):
        if new_text:
            yield _chat_chunk(response_id, created, model_name, delta={"content": new_text.decode("utf-8", errors="ignore")})
        if finish_reason is not None:
            yield _chat_chunk(response_id, created, model_name, finish_reason=finish_reason)


async def _run_batched_generation_then_parse(model, gate, seq_id, prepared, kwargs, response_id, created, model_name):
    """
    Usado quando HÁ `post_handler` (Qwen/Phi/MTMD sem mídia) — esses handlers fazem
    parsing de tool_call/reasoning via regex sobre o TEXTO COMPLETO, e seus
    `_stream_response` originais consomem um generator SÍNCRONO (o que `create_completion
    (stream=True)` produz) — incompatível com o generator assíncrono do `DecodeGate`.
    Em vez de reimplementar o parsing de cada handler de forma assíncrona (risco real de
    divergir sutilmente do comportamento hoje em produção), rodamos a geração batchada
    até o fim (o ganho de throughput do `DecodeGate` continua valendo — é a geração de
    token a token que fica rápida), montamos a resposta completa no formato que
    `_parse_response` já espera, chamamos ele (síncrono, sem generator), e só então
    fatiamos o resultado corrigido em chunks pro cliente. Streaming real (token a token
    visível pro cliente) fica só pros formatos sem `post_handler` nesta primeira versão.
    """
    full_text = b""
    finish_reason = "length"
    async for new_text, fr in _generate_via_gate(model, gate, seq_id, prepared, kwargs):
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
        pass  # handler com assinatura inesperada — segue com a resposta crua

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


async def _fallback_stream(model, gate: DecodeGate, create_kwargs: dict):
    """
    Caminho de segurança: reaproveita `create_chat_completion()` inteiro, um passo
    (`next()`) por vez — mesma mecânica do dispatcher round-robin antigo — mas cada passo
    adquire a MESMA exclusão mútua do `DecodeGate`, garantindo que nunca corra ao mesmo
    tempo que uma rodada batchada (ou outro fallback) de outra sessão. Usado quando
    `_prepare_session` devolve `None` (mensagem com mídia de verdade, ou handler
    desconhecido que não sabemos decompor com segurança).
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
        chunk = await gate.run_exclusive_step(_step)
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
        self.gate: Optional[DecodeGate] = None
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
                self.gate = DecodeGate(self.model)

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

            # Resolve se essa request pode ir pelo caminho batchado (DecodeGate — 1
            # decode() por rodada cobrindo múltiplas sessões) ou precisa do caminho de
            # fallback (create_chat_completion() inteiro, sob a MESMA exclusão mútua do
            # gate) — mensagem com mídia de verdade, ou handler que não sabemos decompor
            # com segurança. `force_round_robin` no preset do modelo força o fallback
            # incondicionalmente (escape-hatch, sem precisar de deploy de código).
            prepared = None
            if not self.meta.get("force_round_robin"):
                prepared = _prepare_session(self.model, messages, **kwargs)

            response_id = f"chatcmpl-{uuid.uuid4().hex}"
            created_ts = int(time.time())
            model_alias = self.meta.get("model_alias", "unknown")

            if prepared is None:
                raw_chunks_gen = _fallback_stream(self.model, self.gate, create_kwargs)
            elif prepared.post_handler is None:
                raw_chunks_gen = _run_batched_generation_stream(
                    self.model, self.gate, seq_id, prepared, kwargs,
                    response_id, created_ts, model_alias,
                )
            else:
                raw_chunks_gen = _run_batched_generation_then_parse(
                    self.model, self.gate, seq_id, prepared, kwargs,
                    response_id, created_ts, model_alias,
                )

            async def _raw_chunks():
                """Consome os chunks da geração (batchada ou fallback) um a um."""
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

        # DecodeGate não tem task de fundo nem estado persistente além da fila (vazia
        # entre requests) e do lock — basta soltar a referência.
        self.gate = None

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
