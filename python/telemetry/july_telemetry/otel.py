import logging
import os
import socket

logger = logging.getLogger(__name__)

# Silence the OTEL SDK's own retry/export error messages.
# These appear as "Transient error" and "Failed to export span batch"
# whenever the collector (port 4318) is unreachable — they pollute logs
# and carry no actionable information when monitoring is simply not running.
logging.getLogger("opentelemetry.sdk.trace.export").setLevel(logging.CRITICAL)
logging.getLogger("opentelemetry.exporter.otlp").setLevel(logging.CRITICAL)
logging.getLogger("opentelemetry").setLevel(logging.WARNING)


def _is_reachable(endpoint: str, timeout: float = 1.0) -> bool:
    """Quick TCP probe — returns False without raising on any error."""
    try:
        from urllib.parse import urlparse
        url = urlparse(endpoint)
        host = url.hostname or "localhost"
        port = url.port or (443 if url.scheme == "https" else 80)
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def setup_otel(service_name: str) -> None:
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")

    if not _is_reachable(endpoint):
        logger.info(f"OpenTelemetry endpoint {endpoint} not reachable — tracing disabled for '{service_name}'")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME
        from opentelemetry.propagate import set_global_textmap
        from opentelemetry.propagators.composite import CompositePropagator
        from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
        from opentelemetry.baggage.propagation import W3CBaggagePropagator

        resource = Resource.create({SERVICE_NAME: service_name})
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces"))
        )
        trace.set_tracer_provider(provider)
        set_global_textmap(CompositePropagator([
            TraceContextTextMapPropagator(),
            W3CBaggagePropagator(),
        ]))
        logger.info(f"OpenTelemetry tracing enabled for '{service_name}' → {endpoint}")
    except ImportError:
        logger.warning("opentelemetry packages not installed — tracing disabled")
    except Exception as e:
        logger.warning(f"Failed to setup OpenTelemetry for '{service_name}': {e}")
