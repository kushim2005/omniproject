import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@asynccontextmanager
async def trace_span(name: str, **meta) -> Dict[str, Any]:
    """Async context manager for lightweight tracing around async operations.

    Usage:
        async with trace_span("step_name", foo=bar) as span:
            result = await do_work()
            span["result"] = summarize(result)

    On exit the span logs duration and any metadata added to the dict.
    """
    span: Dict[str, Any] = {"name": name, "meta": dict(meta), "start": time.perf_counter()}
    try:
        logger.debug("trace.start %s %s", name, meta)
        yield span
    except Exception as exc:  # re-raise after logging
        span["error"] = str(exc)
        raise
    finally:
        span_end = time.perf_counter()
        duration = span_end - span["start"]
        out = {"name": name, "duration_sec": duration, **span}
        # Avoid leaking large objects: only show keys and small summary values
        try:
            logger.info("trace.end %s dur=%.4fs meta=%s extras=%s", name, duration, span.get("meta"), {k: v for k, v in span.items() if k not in ("name", "meta", "start")})
        except Exception:
            logger.info("trace.end %s dur=%.4fs", name, duration)
