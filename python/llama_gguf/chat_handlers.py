import re
import uuid
import logging
import time

import json
from copy import deepcopy
from typing import Any, Dict, List, Union, Iterator, Optional
from llama_cpp.llama_chat_format import Gemma4ChatHandler, Jinja2ChatFormatter, LlamaChatCompletionHandler, Qwen35ChatHandler

logger = logging.getLogger(__name__)


class Gemma4Handler(Gemma4ChatHandler):
    def __call__(self, **kwargs):
        # Garante que os argumentos de tool_call sejam dicionários para o template Jinja
        messages = kwargs.get("messages", [])
        for message in messages:
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
            return self._stream_response(response)
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

        message["content"] = content
        if reasoning:
            message["reasoning_content"] = reasoning
        if parsed_tools:
            message["tool_calls"] = parsed_tools
            response["choices"][0]["finish_reason"] = "tool_calls"

        return response

    def _stream_response(self, response: Iterator):
        current_tag = None
        buffer = ""
        called_tools = False

        for chunk in response:
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content", "")

            if not content:
                yield chunk
                continue

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
            

class Qwen35Handler(Qwen35ChatHandler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, **kwargs):
        # Garante que os argumentos de tool_call sejam dicionários para o template Jinja
        # (Alguns templates como o do Qwen 3.5 usam | items e quebram se for string JSON)
        messages = kwargs.get("messages", [])
        for message in messages:
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

        # Extração de Tool Calls (Formato Qwen 3.5 XML-like)
        # Ex: <tool_call> <function=search_web> <parameter=query>... </parameter> </function> </tool_call>
        tool_call_pattern = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
        function_pattern = re.compile(r'<function=(.*?)>(.*?)</function>', re.DOTALL)
        parameter_pattern = re.compile(r'<parameter=(.*?)>(.*?)</parameter>', re.DOTALL)
        
        parsed_tools = []
        for tc_match in tool_call_pattern.finditer(content):
            raw_tc = tc_match.group(1)
            for fn_match in function_pattern.finditer(raw_tc):
                name = fn_match.group(1).strip()
                params_raw = fn_match.group(2)
                args = {}
                for param_match in parameter_pattern.finditer(params_raw):
                    p_name = param_match.group(1).strip()
                    p_val = param_match.group(2).strip()
                    
                    # Conversão básica de tipos
                    if p_val.lower() == "true": p_val = True
                    elif p_val.lower() == "false": p_val = False
                    elif p_val.isdigit(): p_val = int(p_val)
                    else:
                        try: p_val = float(p_val)
                        except: pass
                    args[p_name] = p_val
                
                unique_id = uuid.uuid4().hex
                parsed_tools.append({
                    "id": f"call_{unique_id}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args),
                    }
                })

        # Extração de Thinking Block
        thinking_pattern = re.compile(r'<(?:\|thought\||think|thought|\|channel>thought)>([\s\S]+?)(?:</(?:think|thought)>|<channel\|>|(?=<\|)|$)', re.DOTALL)
        think_match = thinking_pattern.search(content)
        if think_match:
            thinking = think_match.group(1).strip()
            message["reasoning_content"] = thinking
            content = content.replace(think_match.group(0), "").strip()

        # Remove as tags de tool call do conteúdo, mas preserva o resto
        content = tool_call_pattern.sub('', content).strip() or None
        
        message["content"] = content
        if parsed_tools:
            message["tool_calls"] = parsed_tools
            response["choices"][0]["finish_reason"] = "tool_calls"

        return response

    def _stream_response(self, response: Iterator):
        current_tag = None
        buffer = ""
        called_tools = False
        
        # Estado para parsing de tools em stream
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

            # Detecção de Início de Tool Call
            if not current_tag:
                if '<tool_call>' in buffer:
                    current_tag = 'tool_call'
                    called_tools = True
                    before, after = buffer.split('<tool_call>', 1)
                    if before:
                        chunk_copy = json.loads(json.dumps(chunk))
                        chunk_copy["choices"][0]["delta"]["content"] = before
                        yield chunk_copy
                    buffer = after
                    # Não retornamos continue aqui para processar o 'after' imediatamente
                
                # Se não entrou em tag de tool_call, verifica se pode emitir parte do buffer
                if not current_tag:
                    while buffer:
                        idx = buffer.find('<')
                        if idx > 0:
                            chunk_copy = json.loads(json.dumps(chunk))
                            chunk_copy["choices"][0]["delta"]["content"] = buffer[:idx]
                            yield chunk_copy
                            buffer = buffer[idx:]
                        elif idx == 0:
                            if "<tool_call>".startswith(buffer):
                                # Prefixo de tool_call (pode ser incompleto), para o while e espera próximo chunk
                                break
                            else:
                                # Não é prefixo, emite o '<' e continua o while
                                chunk_copy = json.loads(json.dumps(chunk))
                                chunk_copy["choices"][0]["delta"]["content"] = "<"
                                yield chunk_copy
                                buffer = buffer[1:]
                        else:
                            # Não tem '<' no buffer
                            chunk_copy = json.loads(json.dumps(chunk))
                            chunk_copy["choices"][0]["delta"]["content"] = buffer
                            yield chunk_copy
                            buffer = ""
                            break
                    continue

            # Processamento de Tool Call Ativa
            if current_tag == 'tool_call':
                # Fim da tool call
                if '</tool_call>' in buffer:
                    # Se uma função ainda estava aberta, fecha-a
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

                # Início da função: <function=name>
                if not current_tool_name:
                    fn_match = re.search(r'<function=(.*?)>', buffer)
                    if fn_match:
                        current_tool_name = fn_match.group(1).strip().replace('"', '').replace("'", "")
                        uid = uuid.uuid4().hex[:10]
                        first_param = True
                        
                        # 1. Emite o início da tool call
                        chunk_head = json.loads(json.dumps(chunk))
                        chunk_head["choices"][0]["delta"]["tool_calls"] = [{
                            "index": 0, "id": f"call_{uid}", "type": "function", "function": {"name": current_tool_name}
                        }]
                        yield chunk_head
                        
                        # 2. Inicia o JSON dos argumentos
                        chunk_start_args = json.loads(json.dumps(chunk))
                        chunk_start_args["choices"][0]["delta"]["tool_calls"] = [{
                            "index": 0, "id": f"call_{uid}", "function": {"arguments": "{"}
                        }]
                        yield chunk_start_args
                        
                        buffer = buffer.split(fn_match.group(0), 1)[1]
                        continue
                
                # Fim da função: </function>
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

                # Processamento de Parâmetros dentro da função ativa
                if current_tool_name:
                    # Se não estamos capturando um parâmetro, busca o próximo <parameter=name>
                    if not current_param_name:
                        param_start_match = re.search(r'<parameter=(.*?)>', buffer)
                        if param_start_match:
                            current_param_name = param_start_match.group(1).strip().replace('"', '').replace("'", "")
                            buffer = buffer.split(param_start_match.group(0), 1)[1]
                            # Continua para processar o valor se estiver no buffer
                    
                    # Se estamos capturando um parâmetro, busca o fim </parameter>
                    if current_param_name:
                        if '</parameter>' in buffer:
                            val, rest = buffer.split('</parameter>', 1)
                            val = val.strip()
                            
                            # Conversão de tipos
                            v_lower = val.lower()
                            if v_lower == "true": val_parsed = True
                            elif v_lower == "false": val_parsed = False
                            elif val.isdigit(): val_parsed = int(val)
                            else:
                                try: val_parsed = float(val)
                                except: val_parsed = val
                            
                            arg_key = json.dumps(current_param_name)
                            arg_val = json.dumps(val_parsed)
                            
                            arg_delta = f"{'' if first_param else ', '}{arg_key}: {arg_val}"
                            first_param = False
                            
                            chunk_args = json.loads(json.dumps(chunk))
                            chunk_args["choices"][0]["delta"]["tool_calls"] = [{
                                "index": 0, "id": f"call_{uid}", "function": {"arguments": arg_delta}
                            }]
                            yield chunk_args
                            
                            buffer = rest
                            current_param_name = None
                            continue
                
                # Enquanto estamos dentro de <tool_call>, nunca emitimos conteúdo para o usuário
                # Se o buffer crescer demais sem tags, algo está errado, mas mantemos o buffer
                if len(buffer) > 1000 and '<' not in buffer:
                    # Fallback preventivo: se parecer texto perdido, limpa
                    buffer = ""
                continue

        if called_tools:
            try:
                yield {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "qwen",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "tool_calls"
                        }
                    ]
                }
            except:
                pass
