"""Optimizer meta-tool middleware — resolves ``find_tool`` locally.

The optimizer exposes ``find_tool`` and ``call_tool`` as meta-tools that
do not belong to any backend server.  ``call_tool`` is unwrapped by the
MCP handler into a real capability call, but ``find_tool`` must be
resolved against the local :class:`ToolIndex`.

Routing ``find_tool`` through the middleware chain (SEC-02) requires a
layer that resolves it *before* :class:`RoutingMiddleware` attempts to
map it to a backend session — otherwise the capability lookup fails with
"Capability 'find_tool' does not exist".
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Optional

from mcp import types as mcp_types

from argus_mcp.bridge.middleware.chain import RequestContext
from argus_mcp.bridge.optimizer.meta_tools import FIND_TOOL_NAME
from argus_mcp.constants import OPTIMIZER_SEARCH_LIMIT
from argus_mcp.errors import BackendServerError

logger = logging.getLogger(__name__)


class OptimizerMiddleware:
    """Handle optimizer meta-tools inside the middleware chain.

    Intercepts ``call_tool`` requests for ``find_tool`` and executes the
    search against the optimizer index.  Every other capability is
    forwarded unchanged to the next handler (typically routing).

    Placed immediately before ``RoutingMiddleware`` so auth, recovery,
    telemetry and audit all apply to meta-tool calls (SEC-02).

    Parameters
    ----------
    optimizer_provider:
        Zero-argument callable returning the current optimizer
        :class:`ToolIndex` (or ``None`` when the optimizer is disabled).
        Resolved lazily per request because the index is populated after
        the middleware chain is built.
    """

    def __init__(self, optimizer_provider: Callable[[], Optional[Any]]) -> None:
        self._get_optimizer = optimizer_provider

    async def __call__(self, ctx: RequestContext, next_handler: Any) -> Any:
        if ctx.mcp_method != "call_tool" or ctx.capability_name != FIND_TOOL_NAME:
            return await next_handler(ctx)

        optimizer = self._get_optimizer()
        if optimizer is None:
            raise BackendServerError("Optimizer index is not available.")

        args = ctx.arguments or {}
        query = args.get("query", "")
        try:
            limit = int(args.get("limit", OPTIMIZER_SEARCH_LIMIT))
        except (TypeError, ValueError):
            limit = OPTIMIZER_SEARCH_LIMIT

        results = optimizer.search(query, limit)
        logger.info(
            "find_tool resolved locally: query=%r -> %d result(s)",
            query,
            len(results),
        )
        return mcp_types.CallToolResult(
            content=[
                mcp_types.TextContent(
                    type="text",
                    text=json.dumps(results, indent=2, ensure_ascii=False),
                )
            ],
            isError=False,
        )
