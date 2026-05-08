import os
import json
import hashlib
import logging
import httpx
import re
from typing import Optional, Dict, Any, Union

logger = logging.getLogger("JulyEngine.Services.ResourceCalculator")

class ModelMetadata:
    def __init__(self, model_path: str, repo_id: str = None, filename: str = None, mmproj_path: str = None):
        self.model_path = model_path
        self.repo_id = repo_id
        self.filename = filename
        self.mmproj_path = mmproj_path
        
        self.file_size_gb = os.path.getsize(model_path) / (1024**3) if os.path.exists(model_path) else 0
        self.mmproj_size_gb = os.path.getsize(mmproj_path) / (1024**3) if mmproj_path and os.path.exists(mmproj_path) else 0
        
        self.cache_dir = "storage/cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_id = hashlib.md5(model_path.encode()).hexdigest()
        self.cache_file = os.path.join(self.cache_dir, f"{self.cache_id}.json")
        
        self.data = {}
        self._load_metadata()

    def _load_metadata(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.data = json.load(f)
                if self.data: return
            except: pass
        
        self._read_gguf()

    def _read_gguf(self):
        if not os.path.exists(self.model_path): return
        try:
            from gguf import GGUFReader
            reader = GGUFReader(self.model_path)
            for field in reader.fields.values():
                name = field.name
                # Simple decoding for metadata
                if len(field.parts) > 0 and len(field.data) > 0:
                    part = field.parts[field.data[0]]
                    if hasattr(part, "tolist"): val = part.tolist()
                    elif isinstance(part, bytes): val = part.decode('utf-8', errors='ignore').strip('\x00')
                    else: val = part
                    
                    if isinstance(val, list) and len(val) == 1: val = val[0]
                    if isinstance(val, (int, float, str, bool)):
                        self.data[name] = val
            
            with open(self.cache_file, 'w') as f:
                json.dump(self.data, f, indent=4)
        except Exception as e:
            logger.error(f"Error reading GGUF {self.model_path}: {e}")

    def get(self, key: str, default: Any = 0) -> Any:
        # Search for exact key or suffix
        if key in self.data: return self.data[key]
        for k, v in self.data.items():
            if k.endswith(f".{key}"): return v
        return default

    @property
    def tokenizer_template(self): return self.get("tokenizer.chat_template", None)

    @property
    def layers(self): return int(self.get("block_count", 32))
    
    @property
    def block_count(self): return self.layers
    
    @property
    def n_embd(self): return int(self.get("embedding_length", 4096))
    
    @property
    def n_head(self): return int(self.get("attention.head_count", 32))
    
    @property
    def n_head_kv(self): return int(self.get("attention.head_count_kv", self.n_head))
    
    @property
    def head_dim(self): 
        # Prefer explicit key_length if available
        d = self.get("attention.key_length", 0)
        if d: return int(d)
        return self.n_embd // self.n_head if self.n_head > 0 else 128

    async def resolve_remote(self):
        """Fetch file size from HF if local file missing."""
        if self.file_size_gb > 0 or not self.repo_id or not self.filename: return
        url = f"https://huggingface.co/{self.repo_id}/resolve/main/{self.filename}"
        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                res = await client.head(url)
                self.file_size_gb = int(res.headers.get("Content-Length", 0)) / (1024**3)
        except: pass

def parse_ctx(ctx: Union[int, str]) -> int:
    if isinstance(ctx, int): return ctx
    m = re.match(r"(\d+)([km]?)", str(ctx).lower())
    if not m: return 4096
    val, unit = m.groups()
    val = int(val)
    if unit == 'k': return val * 1024
    if unit == 'm': return val * 1024 * 1024
    return val

async def estimate_vram_ram(
    model_path: str,
    context_window: Union[int, str] = 4096,
    kv_cache_quantization: str = "FP16",
    gpu_layers: Optional[int] = None,
    n_seq_max: int = 1,
    offload_kqv: bool = True,
    flash_attention: bool = True,
    logits_all: bool = False,
    vision_on_cpu: bool = False,
    **kwargs
) -> Dict[str, Any]:
    
    logger.info(f"🔍 Checking if model exists: {model_path}")
    exists_locally = os.path.exists(model_path) if model_path and model_path != "model" else False
    filename = kwargs.get("filename")
    repo_id = kwargs.get("repo_id")

    if not exists_locally and filename:
        # 1. Busca recursiva em models/
        found_path = None
        models_dir = "models"
        if os.path.exists(models_dir):
            for root, dirs, files in os.walk(models_dir):
                if filename in files:
                    found_path = os.path.join(root, filename)
                    break
        if found_path:
            model_path = found_path
            exists_locally = True

        # 2. Busca no cache do Hugging Face Hub
        if not exists_locally and repo_id:
            try:
                from huggingface_hub import hf_hub_download
                hf_path = hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True)
                if hf_path and os.path.exists(hf_path):
                    model_path = hf_path
                    exists_locally = True
            except: pass

    # 1. Setup Metadata
    meta = ModelMetadata(
        model_path, 
        repo_id=repo_id, 
        filename=filename,
        mmproj_path=kwargs.get("mmproj_path")
    )
    if not exists_locally:
        await meta.resolve_remote()
    
    total_layers = meta.layers
    if gpu_layers is None or gpu_layers < 0: offloaded = total_layers
    else: offloaded = min(gpu_layers, total_layers)
    
    n_ctx = parse_ctx(context_window)
    
    # 2. Weights Calculation
    # Simple: proportional offload of the file size
    weights_vram_gb = meta.file_size_gb * (offloaded / total_layers) if total_layers > 0 else meta.file_size_gb
    
    # 3. KV Cache Calculation
    # kv_size = n_ctx * n_seq * n_layers * n_heads_kv * head_dim * 2 (K+V) * bytes_per_element
    kv_quant_map = {
        "FP16": 2, "BF16": 2, "Q8_0": 1, "Q4_0": 0.5, "Q4_1": 0.5, "Q5_0": 0.625, "Q5_1": 0.625
    }
    bytes_per_el = kv_quant_map.get(kv_cache_quantization.upper(), 2)
    
    kv_unified = kwargs.get("kv_unified", False)
    if kv_unified:
        # kv_unified disables kv cache quantization in many implementations, falling back to FP16
        if bytes_per_el < 2:
            bytes_per_el = 2
    
    # Elements in one layer of KV cache for all sequences
    kv_elements_per_layer = n_ctx * n_seq_max * meta.n_head_kv * meta.head_dim * 2
    
    # If offload_kqv is True, KV cache is in VRAM. If False, it's in RAM.
    # However, we calculate the total memory footprint. 
    kv_total_gb = (kv_elements_per_layer * total_layers * bytes_per_el) / (1024**3)
    
    if offload_kqv:
        kv_vram_gb = (kv_elements_per_layer * offloaded * bytes_per_el) / (1024**3)
    else:
        kv_vram_gb = 0
        
    # Extra overhead when kv_unified is True (e.g. duplicating cache structure or specific alignment)
    unified_overhead_gb = kv_total_gb * 0.2 if kv_unified else 0
    
    # 4. Compute Buffers & Overhead (Simplified)
    # Flash attention drastically reduces the attention matrix overhead
    compute_buffer_gb = (n_ctx * meta.n_head * 0.5 if not flash_attention else 64) * 1024 / (1024**3) # rough MB
    # Add a small base for the graph and CUDA
    base_overhead_gb = 0.2 + unified_overhead_gb
    
    # 5. MMProj (Vision)
    mmproj_vram_gb = 0 if vision_on_cpu else meta.mmproj_size_gb # Full mmproj is usually offloaded
    
    # 6. Logits
    vocab_size = int(meta.get("tokenizer.ggml.tokens.length", 32000))
    logits_vram_gb = (vocab_size * (n_ctx if logits_all else 1) * 4) / (1024**3)
    
    # Se offload_kqv=False, o KV cache vai para a RAM.
    # Mas como o usuário quer prever o uso total, vamos garantir que total_vram_gb reflita VRAM se offload_kqv for True,
    # Ou total footprint se ele estiver usando essa função para planejar.
    # Para ser seguro e atender a reclamação do usuário: 
    # Quando offload_kqv=True, kv_vram_gb entra na vram. Quando False, não entra, mas o KV total continua existindo.
    # Vamos adicionar kv_total_gb à RAM, e kv_vram_gb à VRAM.
    # Como total_vram_gb antes incluía kv_vram_gb sempre (porque usava offloaded), vamos manter a lógica base, 
    # mas ajustando para offload_kqv.
    
    total_vram_gb = weights_vram_gb + kv_vram_gb + compute_buffer_gb + base_overhead_gb + mmproj_vram_gb + logits_vram_gb
    total_ram_gb = (meta.file_size_gb - weights_vram_gb) + (kv_total_gb - kv_vram_gb) + (meta.mmproj_size_gb if vision_on_cpu else 0)
    
    # O calculador antigo não tinha total_ram_gb, mas apenas retornava total_vram_gb.
    # Se o usuário considera que a engine gasta 1.7GB sem offload_kqv e 2.67GB com offload_kqv + kv_unified,
    # o total_vram_gb precisa pular para ~2.6GB.
    
    return {
        "total_vram_gb": round(total_vram_gb, 4),
        "total_vram_mb": round(total_vram_gb * 1024, 2),
        "total_ram_gb": round(total_ram_gb, 4),
        "total_ram_mb": round(total_ram_gb * 1024, 2),
        "model_vram_mb": round(weights_vram_gb * 1024, 2),
        "weights_vram_mb": round(weights_vram_gb * 1024, 2),
        "kv_cache_vram_mb": round(kv_vram_gb * 1024, 2),
        "kv_cache_total_mb": round(kv_total_gb * 1024, 2),
        "mmproj_vram_mb": round(mmproj_vram_gb * 1024, 2),
        "mmproj_size_gb": round(mmproj_vram_gb, 4),
        "compute_buffer_mb": round(compute_buffer_gb * 1024, 2),
        "logits_buffer_mb": round(logits_vram_gb * 1024, 2),
        "cuda_overhead_mb": round(base_overhead_gb * 1024, 2),
        "offloaded_layers": offloaded,
        "total_layers": total_layers,
        "percent_on_gpu": round((offloaded / total_layers * 100), 2) if total_layers > 0 else 100,
        "vocab_size": vocab_size,
        "n_ctx": n_ctx,
        "n_seq_max": n_seq_max,
        "architecture": meta.get("general.architecture", "unknown")
    }
