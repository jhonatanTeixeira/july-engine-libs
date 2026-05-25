import re
import uuid
import logging
import time
import datetime

import json
from contextlib import ExitStack
from copy import deepcopy
from typing import Any, Dict, List, Union, Iterator, Optional
from jinja2.sandbox import ImmutableSandboxedEnvironment
from llama_cpp.llama_chat_format import (
    Gemma4ChatHandler,
    Jinja2ChatFormatter,
    LlamaChatCompletionHandler,
    MTMDChatHandler,
    _convert_text_completion_to_chat,
    _convert_text_completion_chunks_to_chat,
)

logger = logging.getLogger(__name__)


class Gemma4Handler(Gemma4ChatHandler):
    def __call__(self, **kwargs):
        # Garante que os argumentos de tool_call sejam dicionários para o template Jinja
        messages = kwargs.get("messages", [])
        for message in messages:
            # Flatten content if it's a list (common for non-vision models receiving multimodal data)
            content = message.get("content")
            if isinstance(content, list):
                # Only flatten if there are no images. Vision models need the list format.
                has_image = any(isinstance(part, dict) and part.get("type") in ["image_url", "image"] for part in content)
                if not has_image:
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif isinstance(part, str):
                            text_parts.append(part)
                    message["content"] = "\n".join(text_parts)

            if "tool_calls" in message and message["tool_calls"]:
                for tool_call in message["tool_calls"]:
                    f = tool_call.get("function")
                    if f and isinstance(f.get("arguments"), str):
                        try:
                            f["arguments"] = json.loads(f["arguments"])
                        except Exception:
                            pass

        response = super().__call__(**kwargs)

        if kwargs.get("stream"):
            return self._stream_response(response)
        else:
            return self._parse_response(response)

    def _parse_response(self, response):
        message = response.get("choices", [{}])[0].get("message", {})
        content = message.get("content", "") or ""

        if not isinstance(content, str):
            return response

        # 1. Extract thinking blocks: <|channel>thought\n...<channel|> (or end of string)
        thinking_pattern = re.compile(
            r'<\|channel>thought([\s\S]*?)(?:<channel\|>|$)', re.DOTALL
        )
        think_match = thinking_pattern.search(content)
        reasoning = None

        if think_match:
            reasoning = think_match.group(1).strip()
            content = content.replace(think_match.group(0), "")

        # 2. Extract tool calls: <|tool_call>call:name{args}<tool_call|>
        tool_call_pattern = re.compile(
            r'<\|tool_call>(.*?)<tool_call\|>', re.DOTALL
        )
        parsed_tools = []

        for tc_match in tool_call_pattern.finditer(content):
            raw_tc = tc_match.group(1).replace('<|"|>', '"')

            for call_block in raw_tc.split('call:'):
                if not call_block.strip():
                    continue

                fn_match = re.match(r'(\w+)(\{.*\})', call_block, re.DOTALL)
                if not fn_match:
                    continue

                name = fn_match.group(1)
                args = fn_match.group(2)
                
                # Fix missing quotes around keys (relaxed JSON)
                args = re.sub(r'([{,])\s*([a-zA-Z0-9_-]+)\s*:', r'\1"\2":', args)

                unique_id = uuid.uuid4().hex
                parsed_tools.append({
                    "id": f"call_{unique_id}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": args,
                    }
                })

        # Remove tool call blocks from content
        content = tool_call_pattern.sub('', content)

        # 3. Unescape remaining Gemma4 quote tokens
        content = content.replace('<|"|>', '"')
        content = content.strip() or None

        # 4. Build the final response
        message["content"] = content

        if reasoning:
            message["reasoning_content"] = reasoning

        if parsed_tools:
            message["tool_calls"] = parsed_tools
            response["choices"][0]["finish_reason"] = "tool_calls"

        return response
        
    def _stream_response(self, response):
        channel = None
        tools_calls = ""
        called_tools = False

        for chunk in response:
            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content")

            if content == '<|channel>':
                channel = "enter"
                continue

            if channel == 'enter':
                if content == 'thought':
                    channel = "think"
                    copy = deepcopy(chunk)
                    copy["choices"][0]["delta"]["content"] = "<think>"
                    yield copy
                    continue
            
            if channel == "think" and content != "<channel|>":
                yield chunk
                continue
                
            if content == "<channel|>":
                if channel == "think":
                    copy = deepcopy(chunk)
                    copy["choices"][0]["delta"]["content"] = "</think>"
                    channel = None

                    yield copy
                    continue
            
            if content == "<|tool_call>":
                channel = "tool_call"
                called_tools = True
                continue

            if content == "<tool_call|>":
                channel = None

                for idx, tool_call in enumerate(tools_calls.replace('<|"|>', '"').split('call:')):
                    if not tool_call.strip():
                        continue

                    matches = re.match(r'(\w+)(\{.*\})', tool_call, re.DOTALL)

                    if matches:
                        name = matches.group(1)
                        args = matches.group(2)
                        
                        # Fix missing quotes around keys (relaxed JSON)
                        args = re.sub(r'([{,])\s*([a-zA-Z0-9_-]+)\s*:', r'\1"\2":', args)
                        
                        tool_id = f"call_{uuid.uuid4().hex[:10]}"
                        yield self.parse_tool_calls(self.parse_tool_call(idx, id=tool_id, name=name))
                        yield self.parse_tool_calls(self.parse_tool_call(idx, id=tool_id, args=args))
                    else:
                        logger.warning(f"Gemma4Handler: Failed to match tool call pattern in: {tool_call}")

                tools_calls = ""

                continue

            if chunk.get("choices", [{}])[0].get("finish_reason") == "stop" and called_tools:
                copy = deepcopy(chunk)
                copy["choices"][0]["delta"]["content"] = ""
                copy["choices"][0]["finish_reason"] = "tool_calls"
                
                yield copy
                
                continue

            if channel == "tool_call":
                tools_calls += content
                continue

            yield chunk


    def parse_tool_calls(self, tool_call):
        return {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [tool_call]
                    }
                }
            ]
        }

    def parse_tool_call(self, idx, id=None, name=None, args=None):
        tool_call = {
            "index": idx,
            "id": id or f"call_{idx}",
        }

        if name:
            tool_call["type"] = 'function'
            tool_call.setdefault("function", {})["name"] = name
        
        if args:
            tool_call.setdefault("function", {})["arguments"] = args

        return tool_call

class QwenChatHandler(LlamaChatCompletionHandler):
    """
    Handler para modelos Qwen (2.5+) que suporta Tool Calling via tags <|tool_call|>
    e pensamento via <|thought|>.
    """
    def __init__(self, template: str, eos_token: str, bos_token: str):
        self.formatter = Jinja2ChatFormatter(
            template=template,
            eos_token=eos_token,
            bos_token=bos_token
        )
        self.handler = self.formatter.to_chat_handler()

    def __call__(self, **kwargs):
        # Garante que os argumentos de tool_call sejam dicionários para o template Jinja
        messages = kwargs.get("messages", [])
        for message in messages:
            # Flatten content if it's a list (common for non-vision models receiving multimodal data)
            content = message.get("content")
            if isinstance(content, list):
                # Only flatten if there are no images. Vision models need the list format.
                has_image = any(isinstance(part, dict) and part.get("type") in ["image_url", "image"] for part in content)
                if not has_image:
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif isinstance(part, str):
                            text_parts.append(part)
                    message["content"] = "\n".join(text_parts)

            if "tool_calls" in message and message["tool_calls"]:
                for tool_call in message["tool_calls"]:
                    f = tool_call.get("function")
                    if f and isinstance(f.get("arguments"), str):
                        try:
                            f["arguments"] = json.loads(f["arguments"])
                        except Exception:
                            pass

        # O handler interno do Jinja2ChatFormatter faz o trabalho pesado de renderização
        response = self.handler(**kwargs)

        if kwargs.get("stream"):
            return self._stream_response(response, messages=messages)
        else:
            return self._parse_response(response)

    def _parse_response(self, response):
        message = response.get("choices", [{}])[0].get("message", {})
        content = message.get("content", "") or ""

        if not isinstance(content, str):
            return response

        # 1. Extração de Pensamento: <|thought|>, <think> ou <thought>
        thinking_pattern = re.compile(r'<(?:\|thought\||think|thought)>([\s\S]+?)(?:</(?:think|thought)>|(?=<\|)|$)', re.DOTALL)
        think_match = thinking_pattern.search(content)
        reasoning = None

        if think_match:
            reasoning = think_match.group(1).strip()
            content = content.replace(think_match.group(0), "")

        # 2. Extração de Tool Calls: <|tool_call|> ou <tool_call>
        tool_call_pattern = re.compile(r'<(?:\|tool_call\||tool_call)>([\s\S]+?)(?:</tool_call>|(?=<\|)|$)', re.DOTALL)
        parsed_tools = []

        for tc_match in tool_call_pattern.finditer(content):
            raw_json = tc_match.group(1).strip()
            try:
                # Qwen geralmente entrega um JSON direto ou lista de JSONs
                data = json.loads(raw_json)
                calls = data if isinstance(data, list) else [data]
                
                for call in calls:
                    unique_id = uuid.uuid4().hex
                    parsed_tools.append({
                        "id": f"call_{unique_id}",
                        "type": "function",
                        "function": {
                            "name": call.get("name"),
                            "arguments": json.dumps(call.get("arguments", {})) if isinstance(call.get("arguments"), dict) else call.get("arguments", "{}"),
                        }
                    })
            except Exception as e:
                logger.warning(f"QwenChatHandler: Falha ao parsear JSON de tool_call: {e}")

        # Limpa o conteúdo de tags
        content = tool_call_pattern.sub('', content)
        content = content.replace('<|im_start|>assistant', '').replace('<|im_end|>', '')
        content = content.strip() or None

        # 3. Fallback: modelo gerou a tool como tag XML direta, ex: <filesystem__list_directory>{"path":"/"}</filesystem__list_directory>
        # Isso ocorre em modelos menores (Qwen3.5-4B) que misturam o formato XML com o formato padrão
        if not parsed_tools and content:
            # Corresponde a: <nome_tool>args_json</nome_tool> ou <nome_tool> (sem args)
            # nome_tool: letras, dígitos, _ e __ (padrão MCP filesystem__list_directory)
            fallback_pattern = re.compile(
                r'<([a-zA-Z][a-zA-Z0-9_]*)(?:>([\s\S]*?)</\1>|(?:\s*/>|\s*$))',
                re.DOTALL
            )
            for fb in fallback_pattern.finditer(content):
                tool_name = fb.group(1)
                raw_args  = (fb.group(2) or "").strip()
                try:
                    args = json.loads(raw_args) if raw_args else {}
                    if not isinstance(args, dict):
                        args = {}
                except Exception:
                    args = {}
                parsed_tools.append({
                    "id":       f"call_{uuid.uuid4().hex}",
                    "type":     "function",
                    "function": {
                        "name":      tool_name,
                        "arguments": json.dumps(args),
                    }
                })
                logger.info(f"QwenChatHandler: fallback XML tool call detectado: {tool_name}({args})")
            if parsed_tools:
                content = fallback_pattern.sub('', content).strip() or None

        message["content"] = content
        if reasoning:
            message["reasoning_content"] = reasoning
        if parsed_tools:
            message["tool_calls"] = parsed_tools
            response["choices"][0]["finish_reason"] = "tool_calls"

        return response

    def _stream_response(self, response: Iterator, messages: List[Dict[str, Any]] = None):
        current_tag = None
        buffer = ""
        called_tools = False
        first_chunk = True

        # Detecção de pensamento pré-semeado (prompt prefill)
        is_preseeded_think = False
        if messages:
            last_msg = messages[-1]
            if last_msg.get("role") == "assistant" and last_msg.get("content"):
                content = last_msg["content"]
                if isinstance(content, str) and (content.strip() == "<think>" or content.strip() == "<thought>" or content.strip() == "<|thought|>"):
                    is_preseeded_think = True

        for chunk in response:
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content", "")

            if not content:
                yield chunk
                continue

            if first_chunk:
                first_chunk = False
                if is_preseeded_think and not content.startswith("<think>") and not content.startswith("<thought>") and not content.startswith("<|thought|>"):
                    import copy
                    chunk_start = copy.deepcopy(chunk)
                    chunk_start["choices"][0]["delta"]["content"] = "<think>"
                    yield chunk_start
                    current_tag = 'thought'

            buffer += content

            # Detecção de início de tags (Pensamento)
            thought_start = None
            for ts in ['<|thought|>', '<think>', '<thought>']:
                if ts in buffer:
                    thought_start = ts
                    break

            if thought_start and current_tag != 'thought':
                current_tag = 'thought'
                before, after = buffer.split(thought_start, 1)
                if before.strip():
                    delta["content"] = before
                    yield chunk
                
                # Inicia bloco de pensamento
                delta["content"] = "<think>"
                yield chunk
                buffer = after
                continue

            # Detecção de início de tags (Tool Call)
            tool_start = None
            for ts in ['<|tool_call|>', '<tool_call>']:
                if ts in buffer:
                    tool_start = ts
                    break

            if tool_start and current_tag != 'tool_call':
                current_tag = 'tool_call'
                called_tools = True
                before, after = buffer.split(tool_start, 1)
                
                # Se estávamos em pensamento, fechamos antes de abrir tool_call
                if before.strip() and current_tag == 'thought':
                    delta["content"] = "</think>"
                    yield chunk
                if before.strip() and not current_tag:
                     delta["content"] = before
                     yield chunk

                buffer = after
                continue

            # Limpeza preventiva
            delta.pop("content", None)
            delta.pop("reasoning_content", None)
            delta.pop("tool_calls", None)

            if current_tag == 'thought':
                thought_end = None
                for te in ['</think>', '</thought>', '<|tool_call|>', '<tool_call>', '<|im_end|>', '<|']:
                    if te in buffer:
                        thought_end = te
                        break
                
                if thought_end:
                    content_inside, rest = buffer.split(thought_end, 1)
                    if content_inside.strip():
                        # O GGUF.stream_adapter espera <think>...</think> para converter em reasoning_content
                        delta["content"] = content_inside
                        yield chunk
                    
                    # Fecha a tag explicitamente para o GGUF.stream_adapter detectar
                    delta["content"] = "</think>"
                    yield chunk

                    buffer = rest
                    current_tag = None
                    continue
                
                # Enquanto está em pensamento, encaminha como conteúdo normal. 
                # O GGUF.stream_adapter cuidará de converter para reasoning_content se estiver entre <think>
                delta["content"] = buffer
                yield chunk
                buffer = ""
                continue

            # Se estamos em tool_call, bufferizamos até o fim
            if current_tag == 'tool_call':
                tool_end = None
                for te in ['</tool_call>', '<|im_end|>', '<|']:
                    if te in buffer:
                        tool_end = te
                        break

                if tool_end:
                    raw_json, rest = buffer.split(tool_end, 1)
                    try:
                        clean_json = re.sub(r'```json\s*|\s*```', '', raw_json).strip()
                        data = json.loads(clean_json)
                        calls = data if isinstance(data, list) else [data]
                        for call in calls:
                            uid = uuid.uuid4().hex[:10]
                            # Emite o cabeçalho
                            chunk_head = json.loads(json.dumps(chunk))
                            chunk_head["choices"][0]["delta"]["tool_calls"] = [{
                                "index": 0, "id": f"call_{uid}", "type": "function", "function": {"name": call.get("name")}
                            }]
                            yield chunk_head
                            # Emite os argumentos
                            chunk_args = json.loads(json.dumps(chunk))
                            chunk_args["choices"][0]["delta"]["tool_calls"] = [{
                                "index": 0, "id": f"call_{uid}", "function": {"arguments": json.dumps(call.get("arguments", {}))}
                            }]
                            yield chunk_args
                    except:
                        # Fallback se falhar o JSON
                        delta["content"] = raw_json
                        yield chunk
                    
                    buffer = rest
                    current_tag = None
                    continue
                
                # Enquanto captura a tool, não emite nada
                continue

            # Fallback: Se não há tag ativa, verifica se inicia uma
            if not current_tag:
                # Se detectarmos início de pensamento, emitimos a tag de abertura
                for ts in ['<|thought|>', '<think>', '<thought>']:
                    if ts in buffer:
                        before, after = buffer.split(ts, 1)
                        if before.strip():
                            delta["content"] = before
                            yield chunk
                        
                        current_tag = 'thought'
                        delta["content"] = "<think>"
                        yield chunk
                        buffer = after
                        break
                
                if current_tag: continue

                # Se detectarmos início de tool call
                for ts in ['<|tool_call|>', '<tool_call>']:
                    if ts in buffer:
                        before, after = buffer.split(ts, 1)
                        if before.strip():
                            delta["content"] = before
                            yield chunk
                        
                        current_tag = 'tool_call'
                        buffer = after
                        break
                
                if current_tag: continue

            # Se não há tag ativa, envia normal
            delta["content"] = buffer
            yield chunk
            buffer = ""

        # Finish reason fix
        if called_tools:
             # O último chunk deve ter o finish_reason correto
             pass 

    def _build_stream_tool_call(self, idx, name=None, args=None):
        tool_call = {"index": 0, "id": f"call_{idx}"}
        if name:
            tool_call["type"] = "function"
            tool_call["function"] = {"name": name}
        if args:
            tool_call.setdefault("function", {})["arguments"] = args
            
        return {
            "choices": [{"delta": {"tool_calls": [tool_call]}}]
        }
            

class Qwen35Handler(MTMDChatHandler):
    """
    Handler para Qwen3 / Qwen3.5+ com o template oficial correto (sem prefill de <think>).
    Estende MTMDChatHandler diretamente para controle total sobre template e parsing.
    Suporta modelos de texto puro (sem clip_model_path) e modelos de visão.

    Diferença chave vs Qwen35ChatHandler do vendor:
    - Remove o prefill `<think>\\n` do add_generation_prompt (o modelo gera naturalmente).
    - O `enable_thinking=False` ainda injeta `<think>\\n\\n</think>\\n\\n` para suprimir reasoning.
    - Nenhuma gambiarra de injeção de <think> no stream — o modelo gera o token real.
    """

    CHAT_FORMAT = (
        "{%- set image_count = namespace(value=0) -%}"
        "{%- set video_count = namespace(value=0) -%}"
        "{%- macro render_content(content, do_vision_count, is_system_content=false) -%}"
        "    {%- if content is string -%}"
        "        {{- content -}}"
        "    {%- elif content is iterable and content is not mapping -%}"
        "        {%- for item in content -%}"
        "            {%- if 'image_url' in item or item.type == 'image_url' -%}"
        "                {%- if is_system_content -%}"
        "                    {{- raise_exception('System message cannot contain images.') -}}"
        "                {%- endif -%}"
        "                {%- if do_vision_count -%}"
        "                    {%- set image_count.value = image_count.value + 1 -%}"
        "                {%- endif -%}"
        "                {%- if add_vision_id -%}"
        "                    {{- 'Picture ' -}}"
        "                    {{- image_count.value | string -}}"
        "                    {{- ': ' -}}"
        "                {%- endif -%}"
        "                {{- '<|vision_start|>' -}}"
        "                {%- if item.image_url is string -%}"
        "                    {{- item.image_url -}}"
        "                {%- else -%}"
        "                    {{- item.image_url.url -}}"
        "                {%- endif -%}"
        "                {{- '<|vision_end|>' -}}"
        "            {%- elif 'text' in item -%}"
        "                {{- item.text -}}"
        "            {%- endif -%}"
        "        {%- endfor -%}"
        "    {%- elif content is none or content is undefined -%}"
        "        {{- '' -}}"
        "    {%- endif -%}"
        "{%- endmacro -%}"
        "{%- if not messages -%}"
        "    {{- raise_exception('No messages provided.') -}}"
        "{%- endif -%}"
        "{%- if tools and tools is iterable and tools is not mapping -%}"
        "    {{- '<|im_start|>system\n' -}}"
        "    {{- '# Tools\n\nYou have access to the following functions:\n\n<tools>' -}}"
        "    {%- for tool in tools -%}"
        "        {{- '\n' -}}"
        "        {{- tool | tojson -}}"
        "    {%- endfor -%}"
        "    {{- '\n</tools>' -}}"
        "    {{- '\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>' -}}"
        "    {%- if messages[0].role == 'system' -%}"
        "        {%- set content = render_content(messages[0].content, false, true) | trim -%}"
        "        {%- if content -%}"
        "            {{- '\n\n' + content -}}"
        "        {%- endif -%}"
        "    {%- endif -%}"
        "    {{- '<|im_end|>\n' -}}"
        "{%- elif messages[0].role == 'system' -%}"
        "    {%- set content = render_content(messages[0].content, false, true) -%}"
        "    {{- '<|im_start|>system\n' + content + '<|im_end|>\n' -}}"
        "{%- endif -%}"
        "{%- set ns = namespace(multi_step_tool=true, last_query_index=messages | length - 1) -%}"
        "{%- for message in messages[::-1] -%}"
        "    {%- set index = messages | length - 1 - loop.index0 -%}"
        "    {%- if ns.multi_step_tool and message.role == 'user' -%}"
        "        {%- set content = render_content(message.content, false) | trim -%}"
        "        {%- if not (content.startswith('<tool_response>') and content.endswith('</tool_response>')) -%}"
        "            {%- set ns.multi_step_tool = false -%}"
        "            {%- set ns.last_query_index = index -%}"
        "        {%- endif -%}"
        "    {%- endif -%}"
        "{%- endfor -%}"
        "{%- if ns.multi_step_tool -%}"
        "    {{- raise_exception('No user query found in messages.') -}}"
        "{%- endif -%}"
        "{%- for message in messages -%}"
        "    {%- set content = render_content(message.content, true) | trim -%}"
        "    {%- if message.role == 'system' -%}"
        "        {%- if not loop.first -%}"
        "            {{- raise_exception('System message must be at the beginning.') -}}"
        "        {%- endif -%}"
        "    {%- elif message.role == 'user' -%}"
        "        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>\n' -}}"
        "    {%- elif message.role == 'assistant' -%}"
        "        {%- set reasoning_content = '' -%}"
        "        {%- if message.reasoning_content is string -%}"
        "            {%- set reasoning_content = message.reasoning_content -%}"
        "        {%- elif '</think>' in content -%}"
        "            {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') -%}"
        "            {%- set content = content.split('</think>')[-1].lstrip('\n') -%}"
        "        {%- endif -%}"
        "        {%- set reasoning_content = reasoning_content | trim -%}"
        "        {%- if (preserve_thinking is defined and preserve_thinking is true) or (loop.index0 > ns.last_query_index) -%}"
        "            {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content -}}"
        "        {%- else -%}"
        "            {{- '<|im_start|>' + message.role + '\n' + content -}}"
        "        {%- endif -%}"
        "        {%- if message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping -%}"
        "            {%- for tool_call in message.tool_calls -%}"
        "                {%- if tool_call.function is defined -%}"
        "                    {%- set tool_call = tool_call.function -%}"
        "                {%- endif -%}"
        "                {%- if loop.first -%}"
        "                    {%- if content | trim -%}"
        "                        {{- '\n\n<tool_call>\n<function=' + tool_call.name + '>\n' -}}"
        "                    {%- else -%}"
        "                        {{- '<tool_call>\n<function=' + tool_call.name + '>\n' -}}"
        "                    {%- endif -%}"
        "                {%- else -%}"
        "                    {{- '\n<tool_call>\n<function=' + tool_call.name + '>\n' -}}"
        "                {%- endif -%}"
        "                {%- if tool_call.arguments is defined -%}"
        "                    {%- for (args_name, args_value) in tool_call.arguments | items -%}"
        "                        {{- '<parameter=' + args_name + '>\n' -}}"
        "                        {%- set args_value = args_value | string if args_value is string else args_value | tojson | safe %}"
        "                        {{- args_value -}}"
        "                        {{- '\n</parameter>' -}}"
        "                    {%- endfor -%}"
        "                {%- endif -%}"
        "                {{- '</function>\n</tool_call>' -}}"
        "            {%- endfor -%}"
        "        {%- endif -%}"
        "        {{- '<|im_end|>\n' -}}"
        "    {%- elif message.role == 'tool' -%}"
        "        {%- if loop.previtem and loop.previtem.role != 'tool' -%}"
        "            {{- '<|im_start|>user' -}}"
        "        {%- endif -%}"
        "        {{- '\n<tool_response>\n' -}}"
        "        {{- content -}}"
        "        {{- '\n</tool_response>' -}}"
        "        {%- if not loop.last and loop.nextitem.role != 'tool' -%}"
        "            {{- '<|im_end|>\n' -}}"
        "        {%- elif loop.last -%}"
        "            {{- '<|im_end|>\n' -}}"
        "        {%- endif -%}"
        "    {%- else -%}"
        "        {{- raise_exception('Unexpected message role.') -}}"
        "    {%- endif -%}"
        "{%- endfor -%}"
        # add_generation_prompt: official Qwen3 behavior — NO <think> prefill when enable_thinking=True.
        # The model generates <think> naturally as its first token.
        # Only inject empty <think></think> when enable_thinking=False to suppress reasoning.
        "{%- if add_generation_prompt -%}"
        "    {{- '<|im_start|>assistant\n' -}}"
        "    {%- if enable_thinking is defined and enable_thinking is false -%}"
        "        {{- '<think>\n\n</think>\n\n' -}}"
        "    {%- endif -%}"
        "{%- endif -%}"
    )

    def __init__(
        self,
        enable_thinking: bool = True,
        preserve_thinking: bool = False,
        add_vision_id: bool = True,
        clip_model_path: Optional[str] = None,
        verbose: bool = False,
        use_gpu: bool = True,
        image_min_tokens: int = -1,
        image_max_tokens: int = -1,
        **kwargs,
    ):
        if clip_model_path:
            super().__init__(
                clip_model_path=clip_model_path,
                verbose=verbose,
                use_gpu=use_gpu,
                image_min_tokens=image_min_tokens,
                image_max_tokens=image_max_tokens,
            )
        else:
            # Modo texto puro: configura atributos sem carregar clip model
            self.log_prefix = self.__class__.__name__
            self.clip_model_path = None
            self.image_min_tokens = image_min_tokens
            self.image_max_tokens = image_max_tokens
            self.use_gpu = use_gpu
            self.verbose = verbose
            self._mtmd_cpp = None
            self.mtmd_ctx = None
            self.extra_template_arguments = {}
            self.chat_template = ImmutableSandboxedEnvironment(
                trim_blocks=True,
                lstrip_blocks=True,
            ).from_string(self.CHAT_FORMAT)
            self._exit_stack = ExitStack()

        self.enable_thinking = enable_thinking
        self.preserve_thinking = preserve_thinking
        self.extra_template_arguments["enable_thinking"] = enable_thinking
        self.extra_template_arguments["preserve_thinking"] = preserve_thinking
        self.extra_template_arguments["add_vision_id"] = add_vision_id

    def __call__(self, **kwargs):
        messages = kwargs.get("messages", [])

        # Normaliza mensagens: flatten de content em lista, arguments de tool_call como dict
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                has_image = any(
                    isinstance(p, dict) and p.get("type") in ["image_url", "image"]
                    for p in content
                )
                if not has_image:
                    text_parts = []
                    for p in content:
                        if isinstance(p, dict) and p.get("type") == "text":
                            text_parts.append(p.get("text", ""))
                        elif isinstance(p, str):
                            text_parts.append(p)
                    message["content"] = "\n".join(text_parts)

            if message.get("tool_calls"):
                for tc in message["tool_calls"]:
                    f = tc.get("function")
                    if f and isinstance(f.get("arguments"), str):
                        try:
                            f["arguments"] = json.loads(f["arguments"])
                        except Exception:
                            pass

        if self.clip_model_path:
            llama = kwargs.get("llama")
            if llama and hasattr(llama, "input_ids"):
                llama.input_ids.fill(0)
            response = super().__call__(**kwargs)
        else:
            response = self._call_text_only(**kwargs)

        if kwargs.get("stream"):
            return self._stream_response(response)
        else:
            return self._parse_response(response)

    def _call_text_only(self, **kwargs):
        """Modo texto puro: renderiza o template Jinja2 e chama llama.create_completion."""
        llama = kwargs["llama"]
        messages = kwargs.get("messages", [])
        tools = kwargs.get("tools")
        stream = kwargs.get("stream", False)
        stop = kwargs.get("stop") or []
        if isinstance(stop, str):
            stop = [stop]

        eos = llama.detokenize([llama.token_eos()]).decode("utf-8", errors="ignore")
        bos_id = llama.token_bos()
        bos = llama.detokenize([bos_id]).decode("utf-8", errors="ignore") if bos_id >= 0 else ""

        prompt = self.chat_template.render(
            messages=messages,
            tools=tools,
            add_generation_prompt=True,
            eos_token=eos,
            bos_token=bos,
            raise_exception=lambda msg: (_ for _ in ()).throw(ValueError(msg)),
            strftime_now=lambda fmt="%Y-%m-%d %H:%M:%S": datetime.datetime.now().strftime(fmt),
            **self.extra_template_arguments,
        )

        tokens = llama.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)
        stop_strs = list(stop) + [eos, "<|im_end|>"]

        raw = llama.create_completion(
            prompt=tokens,
            stream=stream,
            max_tokens=kwargs.get("max_tokens"),
            temperature=kwargs.get("temperature", 0.2),
            top_p=kwargs.get("top_p", 0.95),
            top_k=kwargs.get("top_k", 40),
            min_p=kwargs.get("min_p", 0.05),
            repeat_penalty=kwargs.get("repeat_penalty", 1.1),
            stop=stop_strs,
        )

        if stream:
            return _convert_text_completion_chunks_to_chat(raw)
        return _convert_text_completion_to_chat(raw)

    def _parse_response(self, response):
        message = response.get("choices", [{}])[0].get("message", {})
        content = message.get("content", "") or ""

        if not isinstance(content, str):
            return response

        # Extração de Tool Calls: formato Qwen3.5 XML
        # <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
        tool_call_pattern = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
        function_pattern = re.compile(r'<function=(.*?)>(.*?)</function>', re.DOTALL)
        parameter_pattern = re.compile(r'<parameter=(.*?)>(.*?)</parameter>', re.DOTALL)

        parsed_tools = []
        for tc_match in tool_call_pattern.finditer(content):
            raw_tc = tc_match.group(1)
            for fn_match in function_pattern.finditer(raw_tc):
                name = fn_match.group(1).strip()
                args = {}
                for pm in parameter_pattern.finditer(fn_match.group(2)):
                    k = pm.group(1).strip()
                    v = pm.group(2).strip()
                    if v.lower() == "true":   v = True
                    elif v.lower() == "false": v = False
                    elif v.isdigit():          v = int(v)
                    else:
                        try: v = float(v)
                        except: pass
                    args[k] = v
                parsed_tools.append({
                    "id": f"call_{uuid.uuid4().hex}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                })

        content = tool_call_pattern.sub("", content).strip() or None

        # Fallback: bare XML com __ no nome (modelos menores)
        if not parsed_tools and content:
            fallback_pattern = re.compile(
                r'<([a-zA-Z][a-zA-Z0-9_]*__[a-zA-Z0-9_]+)(?:>([\s\S]*?)</\1>|/>|>)',
                re.DOTALL
            )
            for fb in fallback_pattern.finditer(content):
                tool_name = fb.group(1)
                raw_args = (fb.group(2) or "").strip()
                try:
                    args = json.loads(raw_args) if raw_args else {}
                    if not isinstance(args, dict):
                        args = {}
                except Exception:
                    args = {}
                parsed_tools.append({
                    "id": f"call_{uuid.uuid4().hex}",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": json.dumps(args)},
                })
                logger.info(f"Qwen35Handler: fallback XML tool call: {tool_name}({args})")
            if parsed_tools:
                content = fallback_pattern.sub("", content).strip() or None

        message["content"] = content
        if parsed_tools:
            message["tool_calls"] = parsed_tools
            response["choices"][0]["finish_reason"] = "tool_calls"

        return response

    def _stream_response(self, response: Iterator):
        """
        Processa stream do Qwen3: passa <think>...</think> transparentemente para o
        stream_adapter e parseia <tool_call>...</tool_call> em deltas de tool_calls.
        Sem injeção de <think> — o modelo gera o token naturalmente.
        """
        current_tag = None
        buffer = ""
        called_tools = False

        uid = None
        current_tool_name = None
        current_param_name = None
        first_param = True

        for chunk in response:
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content", "")

            if not content:
                yield chunk
                continue

            buffer += content

            # Processamento de Tool Call ativa
            if current_tag == 'tool_call':
                if '</tool_call>' in buffer:
                    if current_tool_name:
                        chunk_end = json.loads(json.dumps(chunk))
                        chunk_end["choices"][0]["delta"]["tool_calls"] = [{
                            "index": 0, "id": f"call_{uid}", "function": {"arguments": "}"}
                        }]
                        yield chunk_end
                        current_tool_name = None

                    _, rest = buffer.split('</tool_call>', 1)
                    buffer = rest
                    current_tag = None
                    continue

                if not current_tool_name:
                    fn_match = re.search(r'<function=(.*?)>', buffer)
                    if fn_match:
                        current_tool_name = fn_match.group(1).strip().replace('"', '').replace("'", "")
                        uid = uuid.uuid4().hex[:10]
                        first_param = True

                        chunk_head = json.loads(json.dumps(chunk))
                        chunk_head["choices"][0]["delta"]["tool_calls"] = [{
                            "index": 0, "id": f"call_{uid}", "type": "function",
                            "function": {"name": current_tool_name}
                        }]
                        yield chunk_head

                        chunk_start_args = json.loads(json.dumps(chunk))
                        chunk_start_args["choices"][0]["delta"]["tool_calls"] = [{
                            "index": 0, "id": f"call_{uid}", "function": {"arguments": "{"}
                        }]
                        yield chunk_start_args

                        buffer = buffer.split(fn_match.group(0), 1)[1]
                        continue

                if current_tool_name and '</function>' in buffer:
                    chunk_end = json.loads(json.dumps(chunk))
                    chunk_end["choices"][0]["delta"]["tool_calls"] = [{
                        "index": 0, "id": f"call_{uid}", "function": {"arguments": "}"}
                    }]
                    yield chunk_end
                    current_tool_name = None
                    _, rest = buffer.split('</function>', 1)
                    buffer = rest
                    continue

                if current_tool_name:
                    if not current_param_name:
                        param_start_match = re.search(r'<parameter=(.*?)>', buffer)
                        if param_start_match:
                            current_param_name = param_start_match.group(1).strip().replace('"', '').replace("'", "")
                            buffer = buffer.split(param_start_match.group(0), 1)[1]

                    if current_param_name and '</parameter>' in buffer:
                        val, rest = buffer.split('</parameter>', 1)
                        val = val.strip()
                        v_lower = val.lower()
                        if v_lower == "true":    val_parsed = True
                        elif v_lower == "false": val_parsed = False
                        elif val.isdigit():      val_parsed = int(val)
                        else:
                            try: val_parsed = float(val)
                            except: val_parsed = val

                        arg_delta = f"{'' if first_param else ', '}{json.dumps(current_param_name)}: {json.dumps(val_parsed)}"
                        first_param = False

                        chunk_args = json.loads(json.dumps(chunk))
                        chunk_args["choices"][0]["delta"]["tool_calls"] = [{
                            "index": 0, "id": f"call_{uid}", "function": {"arguments": arg_delta}
                        }]
                        yield chunk_args
                        buffer = rest
                        current_param_name = None
                        continue

                if len(buffer) > 1000 and '<' not in buffer:
                    buffer = ""
                continue

            # Sem tag ativa: detecta início de <tool_call> ou passa conteúdo adiante
            if '<tool_call>' in buffer:
                current_tag = 'tool_call'
                called_tools = True
                before, after = buffer.split('<tool_call>', 1)
                if before:
                    chunk_copy = json.loads(json.dumps(chunk))
                    chunk_copy["choices"][0]["delta"]["content"] = before
                    yield chunk_copy
                buffer = after
                continue

            # Sem tag: emite conteúdo preservando possíveis prefixos de <tool_call>
            while buffer:
                idx = buffer.find('<')
                if idx > 0:
                    chunk_copy = json.loads(json.dumps(chunk))
                    chunk_copy["choices"][0]["delta"]["content"] = buffer[:idx]
                    yield chunk_copy
                    buffer = buffer[idx:]
                elif idx == 0:
                    if "<tool_call>".startswith(buffer):
                        break  # prefixo incompleto, aguarda próximo chunk
                    else:
                        chunk_copy = json.loads(json.dumps(chunk))
                        chunk_copy["choices"][0]["delta"]["content"] = "<"
                        yield chunk_copy
                        buffer = buffer[1:]
                else:
                    chunk_copy = json.loads(json.dumps(chunk))
                    chunk_copy["choices"][0]["delta"]["content"] = buffer
                    yield chunk_copy
                    buffer = ""
                    break

        if called_tools:
            yield {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "qwen",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            }
