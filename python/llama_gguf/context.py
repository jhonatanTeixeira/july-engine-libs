import contextvars
from typing import Dict, Any, Optional

# ContextVar para rastrear o ID da requisição atual (UUID único por request HTTP)
request_id_var = contextvars.ContextVar("request_id", default=None)

# ContextVar para rastrear as instâncias de modelo já adquiridas por esta requisição.
# Mapeia o objeto SequencePool para a instância de modelo reservada.
acquired_instances_var = contextvars.ContextVar("acquired_instances", default={})
