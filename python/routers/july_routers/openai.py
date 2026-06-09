import json
import time
import base64
import os
from typing import List, Optional, Dict, Any, Union, AsyncGenerator

from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from .bridge_interface import BridgeInterface
from july_telemetry.metrics import (
    llm_time_to_first_token_seconds,
    llm_tokens_per_second,
    llm_tokens_total,
    tts_time_to_first_chunk_seconds,
)

bridge: BridgeInterface = None


def set_bridge(b: BridgeInterface):
    global bridge
    bridge = b


router = APIRouter(tags=["OpenAI"])

class ToolFunctionProperty(BaseModel):
    type: str
    description: Optional[str] = None
    enum: Optional[List[str]] = None

class ToolFunctionParameters(BaseModel):
    type: str = "object"
    properties: Optional[Dict[str, Any]] = None
    required: Optional[List[str]] = None

class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[ToolFunctionParameters] = None

class ToolDefinition(BaseModel):
    type: str = "function"
    function: ToolFunction
    fire_and_forget: Optional[bool] = Field(None, alias="fire-and-forget")
    
    # Allow extra fields without breaking validation
    model_config = {
        "populate_by_name": True,
        "extra": "allow"
    }

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    stream: Optional[bool] = False
    tools: Optional[List[ToolDefinition]] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    response_format: Optional[Dict[str, Any]] = None
    
    # Allow extra fields for num_ctx or others (from extra_body)
    model_config = {"extra": "allow"}

class EmbeddingRequest(BaseModel):
    model: Optional[str] = None
    input: Union[str, List[str]]

class SpeechRequest(BaseModel):
    model: Optional[str] = None
    input: str
    voice: Optional[str] = None
    language: Optional[str] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    semitones: Optional[float] = None

class ImageGenerationRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    n: Optional[int] = 1
    size: Optional[str] = None
    response_format: Optional[str] = "b64_json"

# --- Response DTOs for Swagger Documentation ---

class ChatCompletionResponse(BaseModel):
    id: str = Field(..., examples=["chatcmpl-123"])
    object: str = "chat.completion"
    created: int = Field(..., examples=[1677652288])
    model: str = Field(..., examples=["qwen3-0.6b.gguf"])
    choices: List[Dict[str, Any]] = Field(..., examples=[{
        "index": 0,
        "message": {"role": "assistant", "content": "Hello! How can I help you?"},
        "finish_reason": "stop"
    }])
    usage: Dict[str, Any] = Field(..., examples=[{"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21}])

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[Dict[str, Any]] = Field(..., examples=[{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}])
    model: str = Field(..., examples=["bge-micro"])
    usage: Dict[str, Any] = Field(..., examples=[{"prompt_tokens": 8, "total_tokens": 8}])

class ImageResponse(BaseModel):
    created: int = Field(..., examples=[1677652288])
    data: List[Dict[str, str]] = Field(..., examples=[{"b64_json": "iVBORw0KGgoAAAANSUhEUgAA..."}])

@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest, http_request: Request):
    req_start = time.monotonic()
    payload = request.model_dump()
    headers = dict(http_request.headers)
    stream = payload.get('stream', False)
    model = payload.get('model', 'unknown')
    response = await bridge.process_openai_chat(payload, headers)

    if stream:
        async def sse_formatter(generator):
            first_chunk = True
            first_token_time = None
            completion_tokens = 0
            usage_counted = False
            try:
                async for chunk_dict in generator:
                    now = time.monotonic()
                    if first_chunk:
                        llm_time_to_first_token_seconds.labels(model=model).observe(now - req_start)
                        first_token_time = now
                        first_chunk = False

                    for choice in chunk_dict.get('choices', []):
                        content = choice.get('delta', {}).get('content')
                        if content:
                            completion_tokens += 1

                    usage = chunk_dict.get('usage')
                    if usage and not usage_counted:
                        llm_tokens_total.labels(model=model, token_type='prompt').inc(
                            usage.get('prompt_tokens', 0)
                        )
                        llm_tokens_total.labels(model=model, token_type='completion').inc(
                            usage.get('completion_tokens', completion_tokens)
                        )
                        completion_tokens = 0
                        usage_counted = True

                    chunk_str = json.dumps(chunk_dict)
                    yield f"data: {chunk_str}\n\n"
            finally:
                if first_token_time and completion_tokens > 0:
                    elapsed = time.monotonic() - first_token_time
                    if elapsed > 0:
                        llm_tokens_per_second.labels(model=model).observe(completion_tokens / elapsed)
                    llm_tokens_total.labels(model=model, token_type='completion').inc(completion_tokens)
                yield "data: [DONE]\n\n"

        return StreamingResponse(sse_formatter(response), media_type="text/event-stream")
    else:
        if isinstance(response, dict):
            usage = response.get('usage', {})
            if usage:
                llm_tokens_total.labels(model=model, token_type='prompt').inc(
                    usage.get('prompt_tokens', 0)
                )
                llm_tokens_total.labels(model=model, token_type='completion').inc(
                    usage.get('completion_tokens', 0)
                )
        return response

@router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest, http_request: Request):
    headers = dict(http_request.headers)
    payload = request.model_dump()
    embeddings = await bridge.process_embeddings(payload, headers)
    data = [{"object": "embedding", "index": i, "embedding": emb} for i, emb in enumerate(embeddings)]
    return {
        "object": "list",
        "data": data,
        "model": request.model,
        "usage": {"prompt_tokens": 0, "total_tokens": 0}
    }

@router.post("/audio/speech")
async def create_speech(request: SpeechRequest, http_request: Request):
    req_start = time.monotonic()
    headers = dict(http_request.headers)
    payload = request.model_dump()
    stream = payload.get('stream', False)
    model = payload.get('model', 'unknown')

    result = await bridge.process_tts(payload, headers)

    if stream and hasattr(result, '__aiter__'):
        async def sse_formatter(generator):
            first_chunk = True
            try:
                async for chunk_bytes in generator:
                    if first_chunk:
                        tts_time_to_first_chunk_seconds.labels(model=model).observe(
                            time.monotonic() - req_start
                        )
                        first_chunk = False
                    chunk_base64 = base64.b64encode(chunk_bytes).decode('utf-8')
                    json_data = json.dumps({"audio": chunk_base64})
                    yield f"data: {json_data}\n\n"
            finally:
                yield "data: [DONE]\n\n"

        return StreamingResponse(sse_formatter(result), media_type="text/event-stream")
    
    if result:
        return Response(content=result, media_type="audio/wav")
    
    return Response(status_code=500, content="TTS failed to generate audio")

@router.post("/audio/transcriptions")
async def create_transcription(
    http_request: Request,
    file: UploadFile = File(...),
    model: str = Form(...),
    language: Optional[str] = Form(None),
):
    headers = dict(http_request.headers)
    audio_bytes = await file.read()
    payload = {
        "audio": audio_bytes,
        "model": model,
        "language": language
    }
    transcription = await bridge.process_stt(payload, headers)
    return {"text": transcription}

@router.post("/images/edits", response_model=ImageResponse)
async def create_image_edit(
    http_request: Request,
    image: UploadFile = File(...),
    prompt: str = Form(...),
    model: Optional[str] = Form(None),
    size: Optional[str] = Form(None),
    n: Optional[int] = Form(1),
):
    headers = dict(http_request.headers)
    image_bytes = await image.read()
    image_data = base64.b64encode(image_bytes).decode()
    payload = {
        "image": image_data,
        "prompt": prompt,
        "model": model,
        "size": size,
        "n": n
    }
    result = await bridge.process_image_edit(payload, headers)

    return {
        "created": int(time.time()),
        "data": [{"b64_json": result}]
    }


@router.post("/images/generations", response_model=ImageResponse)
async def create_image_generation(request: ImageGenerationRequest, http_request: Request):
    headers = dict(http_request.headers)
    payload = request.model_dump()
    result = await bridge.process_image_generation(payload, headers)
    
    return {
        "created": int(time.time()),
        "data": [{"b64_json": result}]
    }

@router.post("/images/resize")
async def create_image_resize(payload: dict, http_request: Request):
    headers = dict(http_request.headers)
    result = await bridge.process_image_resize(payload, headers)
    return {"image": result}
