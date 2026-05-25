from fastapi import APIRouter
from typing import Dict, Any, List

router = APIRouter(prefix="/v1/models", tags=["Models Services"])

# Dados extraídos do Studio e organizados por serviço
SERVICES_METADATA = {
    "brain": [
        {"id": "gpt-4o", "name": "GPT-4o (OpenAI)", "description": "Modelo ultra-avançado da OpenAI."},
        {"id": "gpt-4o-mini", "name": "GPT-4o Mini (OpenAI)", "description": "Versão leve e rápida do GPT-4o."},
        {"id": "claude-3-5-sonnet-latest", "name": "Claude 3.5 Sonnet (Anthropic)", "description": "Modelo de alto desempenho da Anthropic."},
        {"id": "gemini/gemini-2.0-flash-exp", "name": "Gemini 2.0 Flash (Google)", "description": "Modelo multimodal ultra-rápido do Google."},
        {"id": "deepseek/deepseek-chat", "name": "DeepSeek V3", "description": "Modelo de texto otimizado e econômico."}
    ],
    "tts": [
        {"id": "xtts", "name": "XTTS v2", "description": "Clonagem de voz realista e zero-shot."},
        {"id": "piper", "name": "Piper", "description": "Síntese de voz neural otimizada para CPU."},
        {"id": "kokoro", "name": "Kokoro", "description": "Vozes sintéticas expressivas e de alta qualidade."},
        {"id": "chatterbox", "name": "Chatterbox", "description": "Clonagem de voz rápida via Resemble."},
        {"id": "qwen3-tts", "name": "Faster Qwen3-TTS", "description": "Síntese de voz ultra-rápida."}
    ],
    "stt": [
        {"id": "faster-whisper", "name": "Faster Whisper", "description": "Transcrição de áudio ultra-rápida."}
    ],
    "embeddings": [
        {"id": "bge-micro", "name": "BGE Micro", "description": "Embeddings leves e rápidos."},
        {"id": "multilingual-e5", "name": "Multilingual E5", "description": "Embeddings de alta qualidade multi-idioma."}
    ],
    "vision": [
        {"id": "emotion", "name": "Emotion Detection", "description": "Detecta emoções faciais."},
        {"id": "fastvlm", "name": "FastVLM", "description": "Visão computacional leve."},
        {"id": "tagger", "name": "Danbooru Tagger", "description": "Extrai tags de imagens."},
        {"id": "moondream", "name": "Moondream", "description": "VLM minúsculo e poderoso."}
    ],
    "image_edit": [
        {"id": "pix2pix", "name": "InstructPix2Pix", "description": "Edita imagens via texto."},
        {"id": "flux-klein", "name": "Flux Klein (Edit)", "description": "Edição de imagens via FLUX.2."}
    ],
    "image_create": [
        {"id": "lcm", "name": "Stable Diffusion LCM", "description": "Criação de imagens ultra-rápida."},
        {"id": "flux-klein", "name": "Flux Klein (Create)", "description": "Geração de alta fidelidade."}
    ],
    "resize": [
        {"id": "pillow", "name": "Pillow", "description": "Redimensionamento básico."},
        {"id": "opencv", "name": "OpenCV", "description": "Processamento avançado."},
        {"id": "gfpgan", "name": "GFPGAN", "description": "Restauração de rostos e upscale."},
        {"id": "codeformer", "name": "CodeFormer", "description": "Restauração robusta."},
        {"id": "realesrgan", "name": "Real-ESRGAN", "description": "Upscale generativo."}
    ],
    "web_search": [
        {"id": "google", "name": "Google Search", "description": "Busca na web via Google."},
        {"id": "tavily", "name": "Tavily Search", "description": "Busca otimizada para agentes."}
    ],
    "repository_search": [
        {"id": "github", "name": "GitHub Search", "description": "Busca código no GitHub."}
    ]
}

GGUF_TEMPLATES = [
    "llama-2", "llama-3", "alpaca", "vicuna", "mistral-7b", "mixtral-moe", "chatml",
    "open-orca", "deepseek", "qwen", "chatml-function-calling", "oasst_llama",
    "baichuan2", "baichuan", "openbuddy", "redpajama-incite", "snoozy", "phind",
    "intel", "mistrallite", "zephyr", "pygmalion", "mistral-instruct", "chatglm3",
    "openchat", "saiga", "gemma", "jinja"
]

TTS_VOICES = {
    "piper": {
        "pt-BR": ["faber-medium", "edresson-low", "boris-low"],
        "en-US": ["lessac-medium", "ryan-medium", "amy-medium", "kathleen-low"]
    },
    "kokoro": {
        "p": ["pf_dora", "pm_alex", "pm_santa"],
        "a": ["af_heart", "af_bella", "af_nicole"]
    }
}

@router.get("/services")
async def get_services():
    """Retorna todos os serviços, modelos e templates suportados."""
    # Flatten categories for Studio compatibility if needed, but grouped is better
    # We return a structured object that covers everything in constants.ts
    return {
        "grouped": SERVICES_METADATA,
        "templates": GGUF_TEMPLATES,
        "voices": TTS_VOICES,
        # Legacy list for compatibility with Studio's HARDCODED_MODELS
        "all_models": [
            {**m, "category": cat.upper()} 
            for cat, models in SERVICES_METADATA.items() 
            for m in models
        ]
    }
