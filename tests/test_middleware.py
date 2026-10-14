"""Tests for argus_mcp.bridge.middleware — chain, audit, auth, authz, telemetry."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from argus_mcp.bridge.middleware.chain import RequestContext, build_chain

# RequestContext


class TestRequestContext:
    def test_defaults(self):
        ctx = RequestContext(capability_name="search", mcp_method="call_tool")
        assert ctx.capability_name == "search"
        assert ctx.mcp_method == "call_tool"
        assert ctx.arguments is None
        assert len(ctx.request_id) == 12  # uuid hex[:12]
        assert ctx.server_name is None
        assert ctx.original_name is None
        assert ctx.start_time > 0
        assert ctx.metadata == {}
        assert ctx.error is None

    def test_request_id_unique(self):
        ids = {RequestContext(capability_name="x", mcp_method="y").request_id for _ in range(50)}
        assert len(ids) == 50

    def test_elapsed_ms(self):
        ctx = RequestContext(capability_name="x", mcp_method="y")
        # elapsed_ms should be non-negative and small
        assert ctx.elapsed_ms >= 0
        assert ctx.elapsed_ms < 5000  # within 5 seconds at worst

    def test_explicit_arguments(self):
        ctx = RequestContext(
            capability_name="read",
            mcp_method="read_resource",
            arguments={"uri": "file:///etc/hosts"},
        )
        assert ctx.arguments["uri"] == "file:///etc/hosts"

    def test_metadata_mutable(self):
        ctx = RequestContext(capability_name="x", mcp_method="y")
        ctx.metadata["key"] = "value"
        assert ctx.metadata["key"] == "value"

    def test_error_field(self):
        ctx = RequestContext(capability_name="x", mcp_method="y")
        ctx.error = RuntimeError("boom")
        assert isinstance(ctx.error, RuntimeError)


# build_chain


class TestBuildChain:
    @pytest.mark.asyncio
    async def test_no_middleware(self):
        """No middleware → handler called directly."""
        handler = AsyncMock(return_value="result")
        chain = build_chain([], handler)
        ctx = RequestContext(capability_name="t", mcp_method="call_tool")
        result = await chain(ctx)
        assert result == "result"
        handler.assert_awaited_once_with(ctx)

    @pytest.mark.asyncio
    async def test_single_middleware(self):
        """Single middleware wraps the handler."""
        call_order: list[str] = []

        async def mw(ctx, nxt):
            call_order.append("mw-before")
            res = await nxt(ctx)
            call_order.append("mw-after")
            return res

        async def handler(ctx):
            call_order.append("handler")
            return "ok"

        chain = build_chain([mw], handler)
        ctx = RequestContext(capability_name="t", mcp_method="call_tool")
        result = await chain(ctx)
        assert result == "ok"
        assert call_order == ["mw-before", "handler", "mw-after"]

    @pytest.mark.asyncio
    async def test_middleware_order(self):
        """First middleware in list is outermost (executed first)."""
        call_order: list[str] = []

        async def outer(ctx, nxt):
            call_order.append("outer-in")
            res = await nxt(ctx)
            call_order.append("outer-out")
            return res

        async def inner(ctx, nxt):
            call_order.append("inner-in")
            res = await nxt(ctx)
            call_order.append("inner-out")
            return res

        async def handler(ctx):
            call_order.append("handler")
            return "done"

        chain = build_chain([outer, inner], handler)
        ctx = RequestContext(capability_name="t", mcp_method="call_tool")
        await chain(ctx)
        assert call_order == ["outer-in", "inner-in", "handler", "inner-out", "outer-out"]

    @pytest.mark.asyncio
    async def test_middleware_can_modify_context(self):
        """Middleware can add metadata before passing to handler."""

        async def injector(ctx, nxt):
            ctx.metadata["injected"] = True
            return await nxt(ctx)

        async def handler(ctx):
            return ctx.metadata.get("injected")

        chain = build_chain([injector], handler)
        ctx = RequestContext(capability_name="t", mcp_method="call_tool")
        result = await chain(ctx)
        assert result is True

    @pytest.mark.asyncio
    async def test_middleware_exception_propagation(self):
        """Exceptions from handler propagate through middleware."""

        async def mw(ctx, nxt):
            return await nxt(ctx)

        async def handler(ctx):
            raise ValueError("boom")

        chain = build_chain([mw], handler)
        ctx = RequestContext(capability_name="t", mcp_method="call_tool")
        with pytest.raises(ValueError, match="boom"):
            await chain(ctx)


# AuditMiddleware


class TestAuditMiddleware:
    @pytest.mark.asyncio
    async def test_calls_next_handler(self):
        from argus_mcp.bridge.middleware.audit import AuditMiddleware

        handler = AsyncMock(return_value="result")
        mw = AuditMiddleware(audit_logger=None)
        ctx = RequestContext(capability_name="search", mcp_method="call_tool")
        result = await mw(ctx, handler)
        assert result == "result"
        handler.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_emits_audit_event(self):
        from argus_mcp.bridge.middleware.audit import AuditMiddleware

        mock_logger = MagicMock()
        handler = AsyncMock(return_value="ok")
        mw = AuditMiddleware(audit_logger=mock_logger)
        ctx = RequestContext(
            capability_name="read",
            mcp_method="read_resource",
            server_name="github",
        )
        await mw(ctx, handler)
        mock_logger.emit.assert_called_once()
        event = mock_logger.emit.call_args[0][0]
        assert event.event_type == "mcp_operation"
        assert event.outcome.status == "success"

    @pytest.mark.asyncio
    async def test_records_error_outcome(self):
        from argus_mcp.bridge.middleware.audit import AuditMiddleware

        mock_logger = MagicMock()
        handler = AsyncMock(return_value="ok")
        mw = AuditMiddleware(audit_logger=mock_logger)
        ctx = RequestContext(capability_name="x", mcp_method="call_tool")
        ctx.error = RuntimeError("fail")
        await mw(ctx, handler)
        event = mock_logger.emit.call_args[0][0]
        assert event.outcome.status == "error"
        assert event.outcome.error_type == "RuntimeError"

    @pytest.mark.asyncio
    async def test_no_logger_no_emit(self):
        """When audit_logger is None, no emit is called (only logging)."""
        from argus_mcp.bridge.middleware.audit import AuditMiddleware

        mw = AuditMiddleware(audit_logger=None)
        handler = AsyncMock(return_value="ok")
        ctx = RequestContext(capability_name="x", mcp_method="call_tool")
        result = await mw(ctx, handler)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_exception_emits_error_event_and_reraises(self):
        """An exception from downstream is audited as an error outcome
        before propagating to outer middleware (e.g. recovery)."""
        from argus_mcp.bridge.middleware.audit import AuditMiddleware

        mock_logger = MagicMock()
        handler = AsyncMock(side_effect=ValueError("downstream boom"))
        mw = AuditMiddleware(audit_logger=mock_logger)
        ctx = RequestContext(capability_name="find_tool", mcp_method="call_tool")

        with pytest.raises(ValueError, match="downstream boom"):
            await mw(ctx, handler)

        mock_logger.emit.assert_called_once()
        event = mock_logger.emit.call_args[0][0]
        assert event.outcome.status == "error"
        assert event.outcome.error_type == "ValueError"
        assert "downstream boom" in event.outcome.error


# AuthMiddleware


class TestAuthMiddleware:
    @pytest.mark.asyncio
    async def test_success_injects_user(self):
        from argus_mcp.bridge.middleware.auth import AuthMiddleware

        # Mock user identity
        mock_user = MagicMock()
        mock_user.subject = "testuser"

        # Mock registry
        mock_registry = AsyncMock()
        mock_registry.authenticate = AsyncMock(return_value=mock_user)

        mw = AuthMiddleware(provider_registry=mock_registry)
        handler = AsyncMock(return_value="result")
        ctx = RequestContext(capability_name="x", mcp_method="call_tool")
        ctx.metadata["auth_token"] = "bearer-tok"

        result = await mw(ctx, handler)
        assert result == "result"
        assert ctx.metadata["user"] is mock_user
        assert ctx.metadata["user_subject"] == "testuser"
        mock_registry.authenticate.assert_awaited_once_with("bearer-tok")

    @pytest.mark.asyncio
    async def test_auth_failure_raises(self):
        from argus_mcp.bridge.middleware.auth import AuthMiddleware
        from argus_mcp.server.auth.providers import AuthenticationError

        mock_registry = AsyncMock()
        mock_registry.authenticate = AsyncMock(side_effect=AuthenticationError("bad token"))

        mw = AuthMiddleware(provider_registry=mock_registry)
        handler = AsyncMock()
        ctx = RequestContext(capability_name="x", mcp_method="call_tool")
        ctx.metadata["auth_token"] = "invalid"

        with pytest.raises(AuthenticationError, match="bad token"):
            await mw(ctx, handler)
        handler.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_token(self):
        """When no auth_token in metadata, passes None to authenticate."""
        from argus_mcp.bridge.middleware.auth import AuthMiddleware

        mock_user = MagicMock()
        mock_user.subject = ""

        mock_registry = AsyncMock()
        mock_registry.authenticate = AsyncMock(return_value=mock_user)

        mw = AuthMiddleware(provider_registry=mock_registry)
        handler = AsyncMock(return_value="ok")
        ctx = RequestContext(capability_name="x", mcp_method="call_tool")

        await mw(ctx, handler)
        mock_registry.authenticate.assert_awaited_once_with(None)
        # With empty subject, user_subject not set
        assert "user_subject" not in ctx.metadata


# AuthzMiddleware


class TestAuthzMiddleware:
    @pytest.mark.asyncio
    async def test_allow(self):
        from argus_mcp.bridge.middleware.authz import AuthzMiddleware
        from argus_mcp.server.authz.policies import PolicyDecision

        mock_engine = MagicMock()
        mock_engine.evaluate.return_value = PolicyDecision.ALLOW

        mw = AuthzMiddleware(engine=mock_engine)
        handler = AsyncMock(return_value="ok")
        ctx = RequestContext(capability_name="search", mcp_method="call_tool")

        # Mock user in metadata
        mock_user = MagicMock()
        mock_user.roles = ["admin"]
        mock_user.subject = "admin_user"
        ctx.metadata["user"] = mock_user

        result = await mw(ctx, handler)
        assert result == "ok"
        mock_engine.evaluate.assert_called_once_with(["admin"], "tool:search")
        handler.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_deny_raises(self):
        from argus_mcp.bridge.middleware.authz import AuthorizationError, AuthzMiddleware
        from argus_mcp.server.authz.policies import PolicyDecision

        mock_engine = MagicMock()
        mock_engine.evaluate.return_value = PolicyDecision.DENY

        mw = AuthzMiddleware(engine=mock_engine)
        handler = AsyncMock()
        ctx = RequestContext(capability_name="secret_tool", mcp_method="call_tool")
        mock_user = MagicMock()
        mock_user.roles = ["viewer"]
        mock_user.subject = "viewer_user"
        ctx.metadata["user"] = mock_user

        with pytest.raises(AuthorizationError, match="Access denied"):
            await mw(ctx, handler)
        handler.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_user_uses_empty_roles(self):
        """When no user in metadata, uses empty roles list."""
        from argus_mcp.bridge.middleware.authz import AuthzMiddleware
        from argus_mcp.server.authz.policies import PolicyDecision

        mock_engine = MagicMock()
        mock_engine.evaluate.return_value = PolicyDecision.NO_MATCH

        mw = AuthzMiddleware(engine=mock_engine)
        handler = AsyncMock(return_value="ok")
        ctx = RequestContext(capability_name="public", mcp_method="call_tool")

        result = await mw(ctx, handler)
        assert result == "ok"
        mock_engine.evaluate.assert_called_once_with([], "tool:public")

    @pytest.mark.asyncio
    async def test_error_message_includes_capability_and_roles(self):
        from argus_mcp.bridge.middleware.authz import AuthorizationError, AuthzMiddleware
        from argus_mcp.server.authz.policies import PolicyDecision

        mock_engine = MagicMock()
        mock_engine.evaluate.return_value = PolicyDecision.DENY

        mw = AuthzMiddleware(engine=mock_engine)
        handler = AsyncMock()
        ctx = RequestContext(capability_name="admin_tool", mcp_method="call_tool")
        mock_user = MagicMock()
        mock_user.roles = ["reader"]
        mock_user.subject = "bob"
        ctx.metadata["user"] = mock_user

        with pytest.raises(AuthorizationError) as exc_info:
            await mw(ctx, handler)
        msg = str(exc_info.value)
        assert "admin_tool" in msg
        assert "reader" in msg


# TelemetryMiddleware


class TestTelemetryMiddleware:
    @pytest.mark.asyncio
    async def test_success_records_metrics(self):
        from argus_mcp.bridge.middleware.telemetry import TelemetryMiddleware

        handler = AsyncMock(return_value="result")
        mw = TelemetryMiddleware()
        ctx = RequestContext(
            capability_name="search",
            mcp_method="call_tool",
            server_name="github",
        )

        with (
            patch("argus_mcp.bridge.middleware.telemetry.start_span") as mock_span,
            patch("argus_mcp.bridge.middleware.telemetry.record_request") as _mock_record,
        ):
            # Mock the span context manager
            span_obj = MagicMock()
            mock_span.return_value.__enter__ = MagicMock(return_value=span_obj)
            mock_span.return_value.__exit__ = MagicMock(return_value=False)

            result = await mw(ctx, handler)

        assert result == "result"
        handler.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_exception_records_and_reraises(self):
        from argus_mcp.bridge.middleware.telemetry import TelemetryMiddleware

        handler = AsyncMock(side_effect=RuntimeError("boom"))
        mw = TelemetryMiddleware()
        ctx = RequestContext(capability_name="x", mcp_method="call_tool")

        with (
            patch("argus_mcp.bridge.middleware.telemetry.start_span") as mock_span,
            patch("argus_mcp.bridge.middleware.telemetry.record_request") as _mock_record,
        ):
            span_obj = MagicMock()
            mock_span.return_value.__enter__ = MagicMock(return_value=span_obj)
            mock_span.return_value.__exit__ = MagicMock(return_value=False)

            with pytest.raises(RuntimeError, match="boom"):
                await mw(ctx, handler)

            span_obj.record_exception.assert_called_once()
