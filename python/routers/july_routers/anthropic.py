import time
import base64
import json
from typing import List, Optional, Dict, Any, Union, AsyncGenerator

from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from .bridge_interface import BridgeInterface
from july_telemetry.metrics import (
    llm_time_to_first_token_seconds,
    llm_tokens_per_second,
    llm_tokens_total,
)

bridge: BridgeInterface = None


def set_bridge(b: BridgeInterface):
    global bridge
    bridge = b


router = APIRouter(tags=["Anthropic"])

class MessageRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    max_tokens: int
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    
    # Allow extra fields for parity
    model_config = {"extra": "allow"}

class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]

class SpeechRequest(BaseModel):
    model: str
    input: str
    voice: str

class ImageGenerationRequest(BaseModel):
    prompt: str
    model: Optional[str] = "pix2pix"
    n: Optional[int] = 1
    size: Optional[str] = "1024x1024"
    response_format: Optional[str] = "b64_json"

@router.post("/messages")
async def create_message(request: MessageRequest, http_request: Request):
    req_start = time.monotonic()
    payload = request.model_dump()
    headers = dict(http_request.headers)
    model = payload.get('model', 'unknown')

    response = await bridge.process_anthropic_message(payload, headers)

    if isinstance(response, AsyncGenerator):
        async def sse_formatter(generator):
            first_chunk = True
            first_token_time = None
            completion_tokens = 0
            try:
                yield f"data: {json.dumps({'type': 'message_start', 'message': {'role': 'assistant', 'content': []}})}\n\n"
                yield f"data: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

                async for chunk in generator:
                    now = time.monotonic()
                    if first_chunk:
                        llm_time_to_first_token_seconds.labels(model=model).observe(now - req_start)
                        first_token_time = now
                        first_chunk = False

                    if isinstance(chunk, dict):
                        if "choices" in chunk:
                            delta = chunk["choices"][0].get("delta", {})
                            text = delta.get("content") or ""
                            reasoning = delta.get("reasoning_content") or ""

                            usage = chunk.get("usage")
                            if usage:
                                llm_tokens_total.labels(model=model, token_type='prompt').inc(
                                    usage.get('prompt_tokens', 0)
                                )
                                llm_tokens_total.labels(model=model, token_type='completion').inc(
                                    usage.get('completion_tokens', completion_tokens)
                                )
                                completion_tokens = 0

                            if reasoning:
                                completion_tokens += 1
                                yield f"data: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': reasoning}})}\n\n"
                            if text:
                                completion_tokens += 1
                                yield f"data: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': text}})}\n\n"
                        else:
                            yield f"data: {json.dumps(chunk)}\n\n"
                    else:
                        completion_tokens += 1
                        yield f"data: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': str(chunk)}})}\n\n"
            finally:
                if first_token_time and completion_tokens > 0:
                    elapsed = time.monotonic() - first_token_time
                    if elapsed > 0:
                        llm_tokens_per_second.labels(model=model).observe(completion_tokens / elapsed)
                    llm_tokens_total.labels(model=model, token_type='completion').inc(completion_tokens)
                yield f"data: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                yield f"data: {json.dumps({'type': 'message_stop'})}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(sse_formatter(response), media_type="text/event-stream")

    # Non-streaming: count tokens from usage block
    if isinstance(response, dict):
        usage = response.get('usage', {})
        if usage:
            llm_tokens_total.labels(model=model, token_type='prompt').inc(
                usage.get('input_tokens', 0)
            )
            llm_tokens_total.labels(model=model, token_type='completion').inc(
                usage.get('output_tokens', 0)
            )
    return response

@router.post("/embeddings")
async def create_embeddings(request: EmbeddingRequest, http_request: Request):
    headers = dict(http_request.headers)
    payload = request.model_dump()
    embeddings = await bridge.process_embeddings(payload, headers)
    data = [{"index": i, "embedding": emb} for i, emb in enumerate(embeddings)]
    return {
        "object": "list",
        "data": data,
        "model": request.model,
        "usage": {"input_tokens": 0, "output_tokens": 0}
    }

@router.post("/audio/speech")
async def create_speech(request: SpeechRequest, http_request: Request):
    headers = dict(http_request.headers)
    payload = request.model_dump()
    # Note: Anthropic doesn't have a standard speech endpoint, 
    # but we provide parity with OpenAI one here.
    output_path = await bridge.process_tts(payload, headers)
    
    import os
    if output_path and os.path.exists(output_path):
        with open(output_path, "rb") as f:
            audio_bytes = f.read()
        return Response(content=audio_bytes, media_type="audio/wav")
    
    return Response(status_code=500, content="TTS failed")

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

@router.post("/images/generations")
async def create_image_generation(request: ImageGenerationRequest, http_request: Request):
    headers = dict(http_request.headers)
    payload = request.model_dump()
    image_base64 = await bridge.process_image_generation(payload, headers)
    return {
        "created": int(time.time()),
        "data": [{"b64_json": image_base64}]
    }

@router.post("/images/edits")
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
    edited_image_base64 = await bridge.process_image_edit(payload, headers)
    return {
        "created": int(time.time()),
        "data": [{"b64_json": edited_image_base64}]
    }
