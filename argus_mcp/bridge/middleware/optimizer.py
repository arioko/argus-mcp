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
import re
from typing import Any, Callable, Dict, List, Optional

from mcp import types as mcp_types

from argus_mcp.bridge.middleware.chain import RequestContext
from argus_mcp.bridge.optimizer.meta_tools import FIND_TOOL_NAME
from argus_mcp.constants import OPTIMIZER_SEARCH_LIMIT

logger = logging.getLogger(__name__)


def _search_composites(composites: List[Any], query: str) -> List[Dict[str, Any]]:
    """Score composite workflow tools by token overlap with *query*.

    Composite tools are not part of the backend capability registry, so
    they are absent from the optimizer index.  Searching them here keeps
    them discoverable through ``find_tool``.
    """
    tokens = [t for t in re.split(r"[^0-9A-Za-z_]+", query.lower()) if t]
    if not tokens:
        return []

    scored: List[tuple[float, Dict[str, Any]]] = []
    for ct in composites:
        try:
            info = ct.to_tool_info()
        except Exception:  # noqa: BLE001
            continue
        name = info.get("name", "")
        description = info.get("description", "")
        text = f"{name} {description}".lower()
        hits = sum(1 for token in tokens if token in text)
        if hits:
            scored.append(
                (
                    hits / len(tokens),
                    {
                        "name": name,
                        "description": description,
                        "input_schema": info.get("inputSchema", {}),
                        "backend": "composite",
                    },
                )
            )
    scored.sort(key=lambda item: item[0], reverse=True)
    return [{**entry, "score": round(score, 4)} for score, entry in scored]


class OptimizerMiddleware:
    """Handle optimizer meta-tools inside the middleware chain.

    Intercepts ``call_tool`` requests for ``find_tool`` and executes the
    search against the optimizer index (plus composite workflow tools).
    When the optimizer is disabled — or the index is not built yet — the
    request is forwarded so a backend may still provide a real
    ``find_tool`` capability.

    Placed immediately before ``RoutingMiddleware`` so auth, recovery,
    telemetry and audit all apply to meta-tool calls (SEC-02).

    Parameters
    ----------
    state_provider:
        Zero-argument callable returning the current server state
        (:class:`ServerState`).  Resolved lazily per request because the
        optimizer index is populated after the chain is built.
    """

    def __init__(self, state_provider: Callable[[], Optional[Any]]) -> None:
        self._get_state = state_provider

    async def __call__(self, ctx: RequestContext, next_handler: Any) -> Any:
        if ctx.mcp_method != "call_tool" or ctx.capability_name != FIND_TOOL_NAME:
            return await next_handler(ctx)

        state = self._get_state()
        optimizer = getattr(state, "optimizer_index", None)
        if optimizer is None:
            return await next_handler(ctx)

        args = ctx.arguments or {}
        query = args.get("query", "")
        try:
            limit = int(args.get("limit", OPTIMIZER_SEARCH_LIMIT))
        except (TypeError, ValueError):
            limit = OPTIMIZER_SEARCH_LIMIT
        if limit < 0:
            limit = OPTIMIZER_SEARCH_LIMIT

        results = optimizer.search(query, limit)
        composite_hits = _search_composites(
            getattr(state, "composite_tools", None) or [], query
        )
        if composite_hits:
            results = sorted(
                [*results, *composite_hits],
                key=lambda r: r.get("score", 0.0),
                reverse=True,
            )[:limit]

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
