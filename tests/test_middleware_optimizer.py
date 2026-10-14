"""Tests for argus_mcp.bridge.middleware.optimizer — OptimizerMiddleware.

Covers:
- find_tool interception: returns CallToolResult built from optimizer.search()
- Pass-through: non-meta capabilities and non-call_tool methods reach next_handler
- Missing optimizer index → BackendServerError
- ``limit`` argument parsing (valid, invalid, default)
- End-to-end chain: find_tool resolves before RoutingMiddleware even when
  the capability registry cannot resolve it (SEC-02 regression test)
"""

from __future__ import annotations

import json
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.types import CallToolResult

from argus_mcp.bridge.middleware import build_chain
from argus_mcp.bridge.middleware.chain import RequestContext
from argus_mcp.bridge.middleware.optimizer import OptimizerMiddleware
from argus_mcp.bridge.middleware.routing import RoutingMiddleware
from argus_mcp.bridge.optimizer.meta_tools import FIND_TOOL_NAME
from argus_mcp.constants import OPTIMIZER_SEARCH_LIMIT
from argus_mcp.errors import BackendServerError


def _make_optimizer(results: Optional[list] = None) -> MagicMock:
    optimizer = MagicMock()
    optimizer.search = MagicMock(
        return_value=results
        if results is not None
        else [{"name": "sql_run", "description": "Run SQL", "score": 0.9}]
    )
    return optimizer


def _ctx(
    capability_name: str = FIND_TOOL_NAME,
    mcp_method: str = "call_tool",
    arguments: Optional[dict] = None,
) -> RequestContext:
    return RequestContext(
        capability_name=capability_name,
        mcp_method=mcp_method,
        arguments=arguments,
    )


class TestFindToolInterception:
    async def test_find_tool_returns_search_results(self) -> None:
        optimizer = _make_optimizer()
        mw = OptimizerMiddleware(lambda: optimizer)
        next_handler = AsyncMock()

        result = await mw(_ctx(arguments={"query": "sql"}), next_handler)

        assert isinstance(result, CallToolResult)
        assert result.isError is False
        payload = json.loads(result.content[0].text)
        assert payload[0]["name"] == "sql_run"
        optimizer.search.assert_called_once_with("sql", OPTIMIZER_SEARCH_LIMIT)
        next_handler.assert_not_awaited()

    async def test_find_tool_forwards_limit(self) -> None:
        optimizer = _make_optimizer()
        mw = OptimizerMiddleware(lambda: optimizer)

        await mw(_ctx(arguments={"query": "sql", "limit": 10}), AsyncMock())

        optimizer.search.assert_called_once_with("sql", 10)

    async def test_find_tool_invalid_limit_uses_default(self) -> None:
        optimizer = _make_optimizer()
        mw = OptimizerMiddleware(lambda: optimizer)

        await mw(
            _ctx(arguments={"query": "sql", "limit": "not-a-number"}),
            AsyncMock(),
        )

        optimizer.search.assert_called_once_with("sql", OPTIMIZER_SEARCH_LIMIT)

    async def test_find_tool_missing_arguments_defaults(self) -> None:
        optimizer = _make_optimizer(results=[])
        mw = OptimizerMiddleware(lambda: optimizer)

        result = await mw(_ctx(arguments=None), AsyncMock())

        optimizer.search.assert_called_once_with("", OPTIMIZER_SEARCH_LIMIT)
        assert json.loads(result.content[0].text) == []

    async def test_find_tool_missing_index_raises(self) -> None:
        mw = OptimizerMiddleware(lambda: None)

        with pytest.raises(BackendServerError, match="not available"):
            await mw(_ctx(arguments={"query": "sql"}), AsyncMock())


class TestPassThrough:
    async def test_other_capability_forwarded(self) -> None:
        mw = OptimizerMiddleware(lambda: _make_optimizer())
        next_handler = AsyncMock(return_value="backend_result")

        result = await mw(_ctx(capability_name="jira_search"), next_handler)

        assert result == "backend_result"
        next_handler.assert_awaited_once()

    async def test_find_tool_non_call_tool_method_forwarded(self) -> None:
        """Only call_tool requests for find_tool are intercepted."""
        mw = OptimizerMiddleware(lambda: _make_optimizer())
        next_handler = AsyncMock(return_value="forwarded")

        result = await mw(
            _ctx(mcp_method="read_resource"), next_handler
        )

        assert result == "forwarded"
        next_handler.assert_awaited_once()


class TestChainIntegration:
    async def test_find_tool_resolves_through_full_chain(self) -> None:
        """SEC-02 regression: find_tool dispatched through the middleware
        chain must resolve locally instead of failing in routing.

        The capability registry cannot resolve meta-tools; without the
        optimizer layer the call dies in RoutingMiddleware with
        "Capability 'find_tool' does not exist".
        """
        registry = MagicMock()
        registry.resolve_capability = MagicMock(return_value=None)
        manager = MagicMock()

        chain = build_chain(
            middlewares=[OptimizerMiddleware(lambda: _make_optimizer())],
            handler=RoutingMiddleware(registry, manager),
        )

        result = await chain(_ctx(arguments={"query": "oracle"}))

        assert isinstance(result, CallToolResult)
        assert result.isError is False
        registry.resolve_capability.assert_not_called()

    async def test_unknown_capability_still_fails_in_routing(self) -> None:
        """Non-meta capabilities keep flowing to RoutingMiddleware."""
        registry = MagicMock()
        registry.resolve_capability = MagicMock(return_value=None)
        manager = MagicMock()

        chain = build_chain(
            middlewares=[OptimizerMiddleware(lambda: _make_optimizer())],
            handler=RoutingMiddleware(registry, manager),
        )

        with pytest.raises(ValueError, match="does not exist"):
            await chain(_ctx(capability_name="not_a_tool"))
