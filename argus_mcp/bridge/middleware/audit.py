"""Audit middleware — structured request/response audit logging.

Logs structured NIST SP 800-53 aligned records for every request that
flows through the middleware chain.  When an :class:`AuditLogger` is
provided, events are written as JSON lines to a dedicated
audit file.  Otherwise, the middleware falls back to standard Python
logging.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from argus_mcp.audit.models import AuditEvent, AuditOutcome, AuditSource, AuditTarget
from argus_mcp.bridge.middleware.chain import RequestContext

logger = logging.getLogger("argus_mcp.audit")


class AuditMiddleware:
    """Log audit events for every forwarded request.

    Parameters
    ----------
    audit_logger:
        Optional :class:`AuditLogger` instance for JSON-line file output.
        When ``None``, events are emitted via standard ``logging`` only.
    """

    def __init__(self, audit_logger: Optional[Any] = None) -> None:
        self._audit_logger = audit_logger

    async def __call__(self, ctx: RequestContext, next_handler: Any) -> Any:
        logger.info(
            "AUDIT REQUEST  id=%s method=%s capability=%s args_keys=%s",
            ctx.request_id,
            ctx.mcp_method,
            ctx.capability_name,
            list((ctx.arguments or {}).keys()),
        )

        try:
            result = await next_handler(ctx)
        except Exception as exc:
            logger.info(
                "AUDIT RESPONSE id=%s method=%s capability=%s backend=%s outcome=error elapsed_ms=%.1f error=%s",
                ctx.request_id,
                ctx.mcp_method,
                ctx.capability_name,
                ctx.server_name or "unknown",
                ctx.elapsed_ms,
                exc,
            )
            if self._audit_logger is not None:
                event = AuditEvent(
                    event_id=ctx.request_id,
                    event_type="mcp_operation",
                    source=AuditSource(
                        session_id=ctx.metadata.get("session_id"),
                        client_ip=ctx.metadata.get("client_ip"),
                        user_id=ctx.metadata.get("user_subject"),
                    ),
                    target=AuditTarget(
                        backend=ctx.server_name,
                        method=ctx.mcp_method,
                        capability_name=ctx.capability_name,
                        original_name=ctx.original_name,
                    ),
                    outcome=AuditOutcome(
                        status="error",
                        latency_ms=round(ctx.elapsed_ms, 2),
                        error=str(exc),
                        error_type=type(exc).__name__,
                    ),
                )
                self._audit_logger.emit(event)
            raise

        outcome_status = "error" if ctx.error else "success"
        logger.info(
            "AUDIT RESPONSE id=%s method=%s capability=%s backend=%s outcome=%s elapsed_ms=%.1f",
            ctx.request_id,
            ctx.mcp_method,
            ctx.capability_name,
            ctx.server_name or "unknown",
            outcome_status,
            ctx.elapsed_ms,
        )

        # Emit structured audit event when logger is available
        if self._audit_logger is not None:
            event = AuditEvent(
                event_id=ctx.request_id,
                event_type="mcp_operation",
                source=AuditSource(
                    session_id=ctx.metadata.get("session_id"),
                    client_ip=ctx.metadata.get("client_ip"),
                    user_id=ctx.metadata.get("user_subject"),
                ),
                target=AuditTarget(
                    backend=ctx.server_name,
                    method=ctx.mcp_method,
                    capability_name=ctx.capability_name,
                    original_name=ctx.original_name,
                ),
                outcome=AuditOutcome(
                    status=outcome_status,
                    latency_ms=round(ctx.elapsed_ms, 2),
                    error=str(ctx.error) if ctx.error else None,
                    error_type=(type(ctx.error).__name__ if ctx.error else None),
                ),
            )
            self._audit_logger.emit(event)

        return result
