"""Prometheus metrics for July Engine.

Grafana integration:
  1. Start Prometheus + Grafana: docker compose -f docker-compose.grafana.yml up
  2. Prometheus scrapes GET /metrics every 15s
  3. In Grafana → Add Prometheus data source (http://prometheus:9090)
  4. Use PromQL:
       - Req/min:             rate(july_http_requests_total[1m]) * 60
       - Tokens/min:          rate(july_llm_tokens_total[1m]) * 60
       - Tokens/hour:         increase(july_llm_tokens_total[1h])
       - Tokens/day:          increase(july_llm_tokens_total[24h])
       - p95 TTFT:            histogram_quantile(0.95, rate(july_llm_time_to_first_token_seconds_bucket[5m]))
       - Median TPS:          histogram_quantile(0.5, rate(july_llm_tokens_per_second_bucket[5m]))
"""
from prometheus_client import Counter, Histogram

# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------
http_requests_total = Counter(
    'july_http_requests_total',
    'Total HTTP requests received',
    ['method', 'path', 'status_code']
)

http_request_duration_seconds = Histogram(
    'july_http_request_duration_seconds',
    'HTTP request duration in seconds (non-streaming: full round-trip; streaming: time-to-headers)',
    ['method', 'path'],
    buckets=[.005, .01, .025, .05, .1, .25, .5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
)

# ---------------------------------------------------------------------------
# LLM / Chat
# ---------------------------------------------------------------------------
llm_time_to_first_token_seconds = Histogram(
    'july_llm_time_to_first_token_seconds',
    'Latency from request arrival to first generated token (includes model load if cold)',
    ['model'],
    buckets=[.05, .1, .25, .5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0]
)

llm_tokens_per_second = Histogram(
    'july_llm_tokens_per_second',
    'Token generation throughput after first token',
    ['model'],
    buckets=[1, 2, 5, 10, 20, 30, 50, 80, 120, 200]
)

llm_tokens_total = Counter(
    'july_llm_tokens_total',
    'Cumulative tokens processed — use rate() or increase() in Grafana for per-minute/hour/day',
    ['model', 'token_type']   # token_type: prompt | completion
)

# ---------------------------------------------------------------------------
# TTS / Audio
# ---------------------------------------------------------------------------
tts_time_to_first_chunk_seconds = Histogram(
    'july_tts_time_to_first_chunk_seconds',
    'Latency from TTS request arrival to first audio chunk',
    ['model'],
    buckets=[.05, .1, .25, .5, 1.0, 2.0, 3.0, 5.0, 10.0]
)
