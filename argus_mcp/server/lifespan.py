"""Application lifespan management - startup and shutdown sequences.

This module provides the Starlette ``lifespan`` async context manager that
delegates lifecycle management to :class:`~argus_mcp.runtime.ArgusService`.

The display/console status callbacks are kept here so that the runtime service
layer (``runtime/service.py``) remains presentation-agnostic and can be reused
by the management API.

**Signal Handling Note (Ctrl+C)**

Uvicorn's ``Server.capture_signals()`` context manager replaces *all*
SIGINT/SIGTERM handlers with its own ``handle_exit()`` before the lifespan
runs.  That means any signal handlers registered in ``cli.py`` are silently
overwritten and never fire.  Worse, ``handle_exit()`` merely sets boolean
flags (``should_exit`` / ``force_exit``) that are only polled in the main
serve-loop — they are *not* checked during the blocking
``await startup() → lifespan.startup()`` call.

The fix is the **Temporary Signal Override** pattern: inside ``app_lifespan``
we save uvicorn's current handler, install our own handler that (a) cancels
in-flight startup tasks and (b) raises ``SystemExit`` via ``os._exit`` on a
second press, then restore uvicorn's original handler before ``yield`` so
that normal graceful shutdown continues to work.

**Thread-safety**: Per the Python asyncio documentation
(https://docs.python.org/3/library/asyncio-dev.html#signals), calling
``task.cancel()`` from a POSIX signal handler installed via
``signal.signal()`` is **unsafe** because POSIX handlers execute in an
arbitrary thread context.  We therefore use ``loop.add_signal_handler()``
which schedules the callback within the event loop's thread.  On platforms
where ``loop.add_signal_handler()`` is unavailable (Windows), we fall back
to ``signal.signal()`` — acceptable since Windows SIGINT handling is
fundamentally different (Ctrl+C raises ``KeyboardInterrupt``).
"""

import asyncio
import logging
import os
import signal
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, AsyncIterator, Optional

import yaml
from starlette.applications import Starlette

from argus_mcp.constants import EXIT_STACK_CLOSE_TIMEOUT, SERVER_NAME, SERVER_VERSION
from argus_mcp.display.console import (
    disp_console_status,
    gen_status_info,
    log_file_status,
)
from argus_mcp.display.installer import InstallerDisplay
from argus_mcp.errors import BackendServerError, ConfigurationError
from argus_mcp.runtime.service import ArgusService

logger = logging.getLogger(__name__)

DEFAULT_LOG_FPATH = "unknown_argus.log"
DEFAULT_LOG_LVL = "INFO"

# Directories to scan for workflow YAML files (relative to cwd or project root).
_WORKFLOW_YAML_DIRS = ("workflows", "examples/workflows")
_YAML_EXTS = (".yaml", ".yml")


def _discover_workflow_yamls(extra_dirs: tuple[str, ...] = ()) -> list[dict]:
    """Scan known directories for workflow YAML files and return parsed dicts.

    Parameters
    ----------
    extra_dirs:
        Additional directories to scan (e.g. from config ``workflows.directory``).
    """
    from pathlib import Path

    results: list[dict] = []
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        logger.debug("pyyaml not installed — skipping YAML workflow discovery.")
        return results

    # Build de-duplicated scan list: extra_dirs first, then defaults.
    seen: set[str] = set()
    scan_dirs: list[str] = []
    for d in (*extra_dirs, *_WORKFLOW_YAML_DIRS):
        if d not in seen:
            seen.add(d)
            scan_dirs.append(d)

    for rel_dir in scan_dirs:
        d = Path(rel_dir)
        if not d.is_dir():
            d = Path(__file__).resolve().parents[2] / rel_dir
        if not d.is_dir():
            continue
        for fpath in sorted(d.iterdir()):
            if fpath.suffix in _YAML_EXTS and fpath.is_file():
                try:
                    data = yaml.safe_load(fpath.read_text(encoding="utf-8"))
                    if isinstance(data, dict) and data.get("name"):
                        data.setdefault("_source", str(fpath))
                        results.append(data)
                except (OSError, yaml.YAMLError):
                    logger.debug("Failed to parse workflow YAML: %s", fpath, exc_info=True)
    return results


def _load_composite_workflows(
    mcp_svr_instance: Any, chain: Any, extra_dirs: tuple[str, ...] = ()
) -> None:
    """Discover workflow YAML files and register them as composite tools.

    The ``invoke_tool`` callback delegates to the middleware chain so that
    composite tool steps benefit from audit, recovery, and routing middleware.
    """
    from argus_mcp.bridge.middleware.chain import RequestContext
    from argus_mcp.workflows.composite_tool import load_composite_tools

    wf_defs = _discover_workflow_yamls(extra_dirs=extra_dirs)
    if not wf_defs:
        mcp_svr_instance.composite_tools = []
        logger.debug("No composite workflow definitions found.")
        return

    async def _invoke_via_chain(tool_name: str, arguments: dict) -> Any:
        """Route a tool call through the middleware chain."""
        ctx = RequestContext(
            capability_name=tool_name,
            mcp_method="call_tool",
            arguments=arguments,
        )
        result = await chain(ctx)
        if ctx.error is not None:
            raise ctx.error
        return result

    tools = load_composite_tools(wf_defs, _invoke_via_chain)
    mcp_svr_instance.composite_tools = tools
    logger.info(
        "Loaded %d composite workflow tool(s): %s",
        len(tools),
        [t.name for t in tools],
    )


def _setup_incoming_auth(full_cfg: Any) -> tuple[Any, str]:
    """Configure incoming authentication from full config.

    Returns ``(auth_registry, auth_mode)``.
    """
    from argus_mcp.server.auth.providers import AuthProviderRegistry

    auth_registry: AuthProviderRegistry | None = None
    auth_mode: str = "strict"
    if full_cfg is None:
        raise ConfigurationError(
            "Auth configuration is required — cannot start gateway without auth config. "
            "Set incoming_auth.type in config.yaml (use 'anonymous' explicitly for dev mode)."
        )

    incoming_auth_type = full_cfg.incoming_auth.type
    auth_mode = full_cfg.incoming_auth.auth_mode
    if incoming_auth_type != "anonymous":
        auth_registry = AuthProviderRegistry.from_config(full_cfg.incoming_auth.model_dump())
        import argus_mcp.server.transport as _transport_mod

        _transport_mod._incoming_auth_provider = auth_registry
        _transport_mod._auth_mode = auth_mode
        _transport_mod._auth_issuer = full_cfg.incoming_auth.issuer
        logger.info(
            "Incoming auth enabled: type=%s, mode=%s (ASGI gate active on transports).",
            incoming_auth_type,
            auth_mode,
        )
    elif auth_mode == "permissive":
        auth_registry = AuthProviderRegistry.from_config(full_cfg.incoming_auth.model_dump())
        import argus_mcp.server.transport as _transport_mod

        _transport_mod._auth_mode = auth_mode
        logger.info("Incoming auth: anonymous with permissive mode (tracking enabled).")
    else:
        logger.info("Incoming auth: anonymous (no ASGI gate).")
    return auth_registry, auth_mode


def _setup_telemetry(full_cfg: Any) -> bool:
    """Initialize OpenTelemetry if enabled in config. Returns *True* if active."""
    if full_cfg is None or not full_cfg.telemetry.enabled:
        return False
    try:
        from argus_mcp.telemetry.config import TelemetryConfig

        tel_config = TelemetryConfig(
            enabled=True,
            otlp_endpoint=full_cfg.telemetry.otlp_endpoint,
            service_name=full_cfg.telemetry.service_name,
        )
        tel_config.initialize()
        logger.info(
            "Telemetry initialized: endpoint=%s, service=%s",
            full_cfg.telemetry.otlp_endpoint,
            full_cfg.telemetry.service_name,
        )
        return True
    except Exception:  # noqa: BLE001
        logger.debug("Telemetry init failed; continuing without OTel.", exc_info=True)
        return False


async def _build_middleware_stack(
    service: ArgusService,
    audit_logger: Any,
    auth_registry: Any,
    auth_mode: str,
    full_cfg: Any,
    telemetry_enabled: bool,
    mcp_server: Any = None,
) -> tuple[Any, Any]:
    """Build the middleware chain. Returns ``(chain, plugin_manager)``."""
    from argus_mcp.bridge.middleware import (
        AuditMiddleware,
        OptimizerMiddleware,
        RecoveryMiddleware,
        RoutingMiddleware,
        build_chain,
    )
    from argus_mcp.bridge.middleware.telemetry import TelemetryMiddleware
    from argus_mcp.server.state import get_state

    middlewares: list = []
    if auth_registry is not None:
        from argus_mcp.bridge.middleware.auth import AuthMiddleware

        middlewares.append(AuthMiddleware(auth_registry, auth_mode=auth_mode))
    middlewares.append(RecoveryMiddleware())

    plugin_manager: Any = None
    if full_cfg is not None and full_cfg.plugins.enabled and full_cfg.plugins.entries:
        try:
            from argus_mcp.plugins import PluginManager, PluginMiddleware, PluginRegistry

            plugin_registry = PluginRegistry()
            plugin_registry.load_from_config(full_cfg.plugins.entries)
            await plugin_registry.load_all()
            plugin_manager = PluginManager(plugin_registry)
            middlewares.append(PluginMiddleware(plugin_manager))
            logger.info(
                "Plugin framework enabled: %d plugin(s) loaded.",
                plugin_registry.count,
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "Plugin framework init failed; continuing without plugins.", exc_info=True
            )

    if telemetry_enabled:
        middlewares.append(TelemetryMiddleware())
    middlewares.append(AuditMiddleware(audit_logger=audit_logger))

    # Resolve optimizer meta-tools (find_tool) locally, just before routing.
    # The index is attached after the chain is built, so resolve it lazily
    # per request via the server state.
    middlewares.append(
        OptimizerMiddleware(lambda: get_state(mcp_server).optimizer_index)
    )

    routing = RoutingMiddleware(service.registry, service.manager)
    chain = build_chain(middlewares=middlewares, handler=routing)
    logger.info(
        "Middleware chain attached (telemetry=%s).",
        "enabled" if telemetry_enabled else "disabled",
    )
    return chain, plugin_manager


async def _setup_optimizer(
    service: ArgusService,
    full_cfg: Any,
) -> tuple[Any, bool, list[str]]:
    """Initialize the ToolIndex optimizer.

    Returns ``(index_or_None, enabled, keep_list)``.
    """
    from argus_mcp.bridge.optimizer import ToolIndex

    optimizer_enabled = full_cfg.optimizer.enabled if full_cfg else False
    keep_list: list[str] = list(full_cfg.optimizer.keep_tools) if full_cfg else []

    if not optimizer_enabled:
        logger.debug("Optimizer disabled.")
        return None, False, keep_list

    tool_index = ToolIndex()
    tools = service.registry.get_aggregated_tools()
    route_map = service.registry.get_route_map()
    await tool_index.store(tools, route_map)

    if keep_list:
        known = set(tool_index.tool_names)
        missing = [t for t in keep_list if t not in known]
        if missing:
            logger.warning(
                "Optimizer keep_tools references unknown tool(s): %s. "
                "These may be misspelled or not yet registered.",
                missing,
            )

    logger.info(
        "Optimizer enabled: indexed %d tool(s), keep-list=%s.",
        tool_index.tool_count,
        keep_list or "(none)",
    )
    return tool_index, True, keep_list


async def _attach_to_mcp_server(
    mcp_svr_instance: Any,
    service: ArgusService,
    app_state: Any = None,
) -> None:
    """Attach bridge components from the service to the MCP server instance.

    This preserves the existing monkey-patch pattern (mcp_server.manager /
    mcp_server.registry) until it is properly replaced in a later phase.
    Also builds and attaches the middleware chain and optimizer index.
    """
    from argus_mcp.audit import AuditLogger
    from argus_mcp.config.loader import load_argus_config
    from argus_mcp.config.schema import ArgusConfig

    mcp_svr_instance.manager = service.manager
    mcp_svr_instance.registry = service.registry

    config_path = getattr(service, "_config_path", None)
    full_cfg: ArgusConfig | None = None
    if config_path:
        try:
            full_cfg = load_argus_config(config_path)
        except (OSError, yaml.YAMLError, AttributeError, KeyError, ValueError):
            logger.debug(
                "Could not load full config; sub-features will use defaults.", exc_info=True
            )

    audit_logger = AuditLogger()
    mcp_svr_instance.audit_logger = audit_logger

    # At app-factory time only the env var is checked.  Now that the
    # full config is loaded, apply the config-file token if the env var
    # was unset.
    if full_cfg is not None and app_state is not None:
        mgmt_app = getattr(app_state, "mgmt_app", None)
        if mgmt_app is not None and hasattr(mgmt_app, "set_token"):
            mgmt_cfg = getattr(getattr(full_cfg, "server", None), "management", None)
            cfg_token = getattr(mgmt_cfg, "token", None)
            if not mgmt_app.auth_enabled and cfg_token:
                mgmt_app.set_token(cfg_token)

    auth_registry, auth_mode = _setup_incoming_auth(full_cfg)
    telemetry_enabled = _setup_telemetry(full_cfg)
    mcp_svr_instance.telemetry_enabled = telemetry_enabled

    chain, plugin_manager = await _build_middleware_stack(
        service,
        audit_logger,
        auth_registry,
        auth_mode,
        full_cfg,
        telemetry_enabled,
        mcp_server=mcp_svr_instance,
    )
    mcp_svr_instance.middleware_chain = chain

    optimizer_index, optimizer_enabled, keep_list = await _setup_optimizer(service, full_cfg)
    mcp_svr_instance.optimizer_enabled = optimizer_enabled
    mcp_svr_instance.optimizer_keep_list = keep_list
    mcp_svr_instance.optimizer_index = optimizer_index

    from argus_mcp.server.session import SessionManager

    session_manager = SessionManager()
    session_manager.start()
    mcp_svr_instance.session_manager = session_manager
    logger.info("SessionManager attached to mcp_server instance.")

    from argus_mcp.config.flags import FeatureFlags

    ff_overrides = dict(full_cfg.feature_flags) if full_cfg else {}
    mcp_svr_instance.feature_flags = FeatureFlags(ff_overrides)
    logger.info("Feature flags: %s", mcp_svr_instance.feature_flags)

    # Propagate container_isolation flag to env so the container
    # wrapper module can read it without direct FeatureFlags access.
    if not mcp_svr_instance.feature_flags.is_enabled("container_isolation"):
        os.environ.setdefault("ARGUS_CONTAINER_ISOLATION", "false")
        logger.info("Container isolation disabled via feature flag.")

    # Propagate build_on_startup flag — when False (the default),
    # server startup will NOT attempt to build missing container images;
    # backends gracefully fall back to bare subprocess.  Users should
    # run ``argus-mcp build`` first to pre-build images.
    if not mcp_svr_instance.feature_flags.is_enabled("build_on_startup"):
        os.environ.setdefault("ARGUS_BUILD_ON_STARTUP", "false")
        logger.info(
            "Container image builds during startup disabled "
            "(build_on_startup=false). Run 'argus-mcp build' to pre-build images."
        )

    from argus_mcp.bridge.version_checker import VersionChecker

    mcp_svr_instance.version_checker = VersionChecker()
    logger.info("VersionChecker attached (registry_client=None — drift available on demand).")

    from argus_mcp.skills.manager import SkillManager

    skills_dir = "skills"
    if full_cfg is not None and hasattr(full_cfg, "skills"):
        skills_dir = full_cfg.skills.directory
    skill_manager = SkillManager(skills_dir=skills_dir)
    skill_manager.discover()
    mcp_svr_instance.skill_manager = skill_manager
    logger.info(
        "SkillManager attached (dir=%s): %d skill(s) discovered.",
        skills_dir,
        len(skill_manager.list_skills()),
    )

    wf_extra_dirs: tuple[str, ...] = ()
    if full_cfg is not None and hasattr(full_cfg, "workflows"):
        cfg_wf_dir = full_cfg.workflows.directory
        if cfg_wf_dir not in _WORKFLOW_YAML_DIRS:
            wf_extra_dirs = (cfg_wf_dir,)
    _load_composite_workflows(mcp_svr_instance, chain, extra_dirs=wf_extra_dirs)

    # Attach a single typed object so consumers can access state with
    # proper typing instead of ad-hoc getattr(mcp_server, ...) calls.
    from argus_mcp.server.state import ServerState

    mcp_svr_instance._argus_state = ServerState(
        manager=service.manager,
        registry=service.registry,
        audit_logger=audit_logger,
        middleware_chain=chain,
        session_manager=session_manager,
        feature_flags=mcp_svr_instance.feature_flags,
        skill_manager=skill_manager,
        version_checker=mcp_svr_instance.version_checker,
        optimizer_index=optimizer_index,
        optimizer_enabled=optimizer_enabled,
        optimizer_keep_list=keep_list,
        telemetry_enabled=telemetry_enabled,
        plugin_manager=plugin_manager,
        composite_tools=getattr(mcp_svr_instance, "composite_tools", []),
    )


# Startup-Phase Signal Override


def _install_startup_signal_override(
    service: "ArgusService",
) -> tuple[Any, Any]:
    """Replace SIGINT/SIGTERM with handlers that cancel in-flight startup.

    Returns ``(original_sigint_handler, original_sigterm_handler)`` so the
    caller can restore them after startup completes.

    **Why this exists**: Uvicorn installs its own signal handlers *before*
    calling ``lifespan.startup()``.  Those handlers only set a flag that the
    main serve-loop polls — they do nothing while the lifespan is blocked
    on ``await service.start()``.  By temporarily overriding the handlers
    inside the lifespan we regain the ability to react to Ctrl+C during
    the (potentially very long) backend-connection phase.

    Behaviour:
    * **First Ctrl+C** — calls ``service._manager.cancel_startup()`` to
      cooperatively cancel all pending tasks, sets ``should_exit`` on the
      uvicorn server so it will exit after the lifespan returns, and prints
      a user-visible message.
    * **Second Ctrl+C** — calls ``os._exit(1)`` for an immediate hard exit,
      just like the user would expect from double Ctrl+C.

    **Thread-safety**: Uses ``loop.add_signal_handler()`` per Python asyncio
    docs (https://docs.python.org/3/library/asyncio-dev.html#signals) so
    that ``cancel_startup()`` / ``task.cancel()`` are called within the
    event-loop thread rather than from a POSIX signal context.  Falls back
    to ``signal.signal()`` on platforms without ``add_signal_handler``
    support (e.g. Windows / ProactorEventLoop).
    """
    _force_count = 0

    def _cancel_startup() -> None:
        """Cancel all pending startup tasks (runs in event-loop thread)."""
        mgr = getattr(service, "_manager", None)
        if mgr is not None and hasattr(mgr, "cancel_startup"):
            mgr.cancel_startup()

        # Tell uvicorn to exit once the lifespan returns
        import argus_mcp.cli as _cli_mod

        uvicorn_svr = getattr(_cli_mod, "uvicorn_svr_inst", None)
        if uvicorn_svr is not None:
            uvicorn_svr.should_exit = True

    def _handle_sigint() -> None:
        """SIGINT handler — first press cancels, second force-exits."""
        nonlocal _force_count
        _force_count += 1

        if _force_count >= 2:
            logger.warning("Force-exit requested (double Ctrl+C during startup).")
            print("\n[Ctrl+C] Force exit.")
            os._exit(1)

        logger.info("Ctrl+C received during startup — cancelling backend connections…")
        print("\n[Ctrl+C] Cancelling startup… (press again to force-quit)")
        _cancel_startup()

    def _handle_sigterm() -> None:
        """SIGTERM handler — cancel startup gracefully."""
        logger.info("SIGTERM received during startup — cancelling backend connections…")
        _cancel_startup()

    # Legacy-style signal handlers (fallback for Windows / non-Unix loops).
    def _startup_sigint_legacy(_signum: int, _frame: Any) -> None:
        _handle_sigint()

    def _startup_sigterm_legacy(_signum: int, _frame: Any) -> None:
        _handle_sigterm()

    # Save originals so we can restore them later.
    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)

    # Prefer loop.add_signal_handler (thread-safe, asyncio-native).
    # Per Python docs: "loop.add_signal_handler() is the preferred way
    # to register signal handlers in asyncio programs."
    try:
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, _handle_sigint)
        loop.add_signal_handler(signal.SIGTERM, _handle_sigterm)
        logger.debug(
            "Startup signal override installed via loop.add_signal_handler() "
            "(thread-safe, replaces uvicorn handlers temporarily)."
        )
    except (NotImplementedError, AttributeError, RuntimeError):
        # NotImplementedError / AttributeError → Windows ProactorEventLoop
        #   does not support add_signal_handler.
        # RuntimeError → No running event loop (e.g. sync test context).
        # Fall back to signal.signal() — acceptable on Windows where
        # SIGINT handling is fundamentally different (raises KeyboardInterrupt).
        signal.signal(signal.SIGINT, _startup_sigint_legacy)
        signal.signal(signal.SIGTERM, _startup_sigterm_legacy)
        logger.debug(
            "Startup signal override installed via signal.signal() "
            "(fallback — loop.add_signal_handler not available)."
        )

    return original_sigint, original_sigterm


def _restore_signal_handlers(
    original_sigint: Any,
    original_sigterm: Any,
) -> None:
    """Restore the signal handlers that were active before the override.

    Removes any ``loop.add_signal_handler`` registrations first, then
    restores the POSIX-level handlers saved during ``_install_…()``.
    """
    # Remove loop-level handlers (safe even if they were never installed —
    # remove_signal_handler returns False in that case).
    try:
        loop = asyncio.get_running_loop()
        loop.remove_signal_handler(signal.SIGINT)
        loop.remove_signal_handler(signal.SIGTERM)
    except (NotImplementedError, AttributeError, RuntimeError):
        pass

    # Restore the POSIX-level handlers to whatever uvicorn had installed.
    signal.signal(signal.SIGINT, original_sigint)
    signal.signal(signal.SIGTERM, original_sigterm)
    logger.debug("Signal handlers restored to uvicorn defaults.")


def _setup_installer_display(
    config_path: str, verbosity: int, *, parallel: bool = False
) -> tuple["InstallerDisplay | None", "Any"]:
    """Create the verbose installer display if conditions are met."""
    # In parallel mode the display is useful even at default verbosity (0).
    min_verbosity = 0 if parallel else 1
    if verbosity < min_verbosity or not config_path:
        return None, None
    try:
        from argus_mcp.config.loader import load_and_validate_config

        raw_config = load_and_validate_config(config_path)
        installer_display = InstallerDisplay(raw_config, parallel=parallel, verbosity=verbosity)
        installer_display.render_initial()
        return installer_display, installer_display.make_callback()
    except (OSError, yaml.YAMLError, AttributeError, KeyError, ValueError):
        logger.debug(
            "Could not initialise installer display; falling back to standard output.",
            exc_info=True,
        )
        return None, None


def _propagate_to_mgmt_app(app_state: Any, service: "ArgusService") -> None:
    """Forward service + host/port/transport to the management sub-app."""
    mgmt_app = getattr(app_state, "mgmt_app", None)
    if mgmt_app is None:
        return
    setattr(mgmt_app.state, "argus_service", service)
    setattr(mgmt_app.state, "host", getattr(app_state, "host", "127.0.0.1"))
    setattr(mgmt_app.state, "port", getattr(app_state, "port", 0))
    setattr(
        mgmt_app.state,
        "transport_type",
        getattr(app_state, "transport_type", "streamable-http"),
    )


async def _shutdown_streamable_http(_exit_stack: "AsyncExitStack | None", mcp_server: Any) -> None:
    """Shut down the StreamableHTTPSessionManager and Argus session manager."""
    if _exit_stack is not None:
        try:
            await asyncio.wait_for(_exit_stack.aclose(), timeout=EXIT_STACK_CLOSE_TIMEOUT)
            logger.info("SDK StreamableHTTPSessionManager stopped.")
        except asyncio.TimeoutError:
            logger.warning(
                "StreamableHTTPSessionManager aclose() timed out after %ss.",
                EXIT_STACK_CLOSE_TIMEOUT,
            )
        except RuntimeError as exc:
            logger.debug(
                "Cancel scope error during session manager shutdown: %s",
                exc,
            )
        except Exception:  # noqa: BLE001
            logger.debug(
                "Error stopping StreamableHTTPSessionManager",
                exc_info=True,
            )
        try:
            import argus_mcp.server.app as app_module

            app_module.streamable_session_manager = None
        except Exception:  # noqa: BLE001
            logger.debug("Module reference cleanup failed", exc_info=True)

    from argus_mcp.server.state import get_state

    sm = get_state(mcp_server).session_manager
    if sm is not None:
        await sm.stop()


@asynccontextmanager
async def app_lifespan(app: Starlette) -> AsyncIterator[None]:
    """Application lifespan management: startup and shutdown.

    Creates a :class:`ArgusService`, drives its lifecycle, and decorates
    each phase with display/console status updates for the TUI / ``--no-tui``
    console.
    """
    from argus_mcp.server.app import mcp_server

    app_state = app.state
    logger.info(
        "Server '%s' v%s startup sequence started...",
        SERVER_NAME,
        SERVER_VERSION,
    )
    logger.debug(
        "Lifespan received host='%s', port=%s",
        getattr(app_state, "host", "N/A"),
        getattr(app_state, "port", 0),
    )
    logger.info(
        "Configured file log level: %s",
        getattr(app_state, "file_log_level_configured", DEFAULT_LOG_LVL),
    )
    logger.info(
        "Actual log file: %s",
        getattr(app_state, "actual_log_file", DEFAULT_LOG_FPATH),
    )

    config_path: str = getattr(app_state, "config_file_path", "")
    if not config_path:
        # Fallback auto-detect (should rarely hit — CLI sets this)
        from argus_mcp.config.loader import find_config_file

        config_path = find_config_file()
    logger.info("Configuration file in use: %s", config_path)

    service = ArgusService()
    # Propagate CLI --auto-reauth flag to the runtime service.
    service._auto_reauth = getattr(app_state, "auto_reauth", False)
    # Propagate CLI --parallel flag for concurrent container builds.
    service._parallel = getattr(app_state, "parallel", False)
    # Store service on app.state so management API can access it later (0.2).
    setattr(app_state, "argus_service", service)

    # Also propagate to the management sub-app so its request handlers see
    # argus_service on *their* request.app.state (the sub-app's state).
    _propagate_to_mgmt_app(app_state, service)

    startup_ok = False
    err_detail_msg: Optional[str] = None

    try:
        status_info_init = gen_status_info(app_state, "Server is starting...")
        disp_console_status("Initialization", status_info_init)
        log_file_status(status_info_init)

        verbosity: int = getattr(app_state, "verbosity", 0)
        parallel: bool = getattr(app_state, "parallel", False)
        installer_display, progress_callback = _setup_installer_display(
            config_path, verbosity, parallel=parallel
        )

        # Install a temporary signal override so Ctrl+C works during
        # the (potentially very long) backend-connection phase.
        # See module docstring for the full rationale.
        orig_sigint, orig_sigterm = _install_startup_signal_override(service)
        try:
            await service.start(config_path, progress_callback=progress_callback)
        finally:
            # Always restore uvicorn's signal handlers — even if startup
            # was cancelled — so that normal graceful shutdown works for
            # the running server.
            _restore_signal_handlers(orig_sigint, orig_sigterm)

        # Finalize the installer display (print summary line)
        if installer_display is not None:
            installer_display.finalize()

        await _attach_to_mcp_server(mcp_server, service, app_state)

        # This must happen *after* handlers are attached so that
        # mcp_server.run() (called internally per session) can serve
        # tools/resources/prompts.
        # Uses AsyncExitStack for reliable cleanup
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

        import argus_mcp.server.app as app_module

        sm_http = StreamableHTTPSessionManager(app=mcp_server)
        app_module.streamable_session_manager = sm_http
        _exit_stack = AsyncExitStack()
        await _exit_stack.enter_async_context(sm_http.run())
        logger.info("SDK StreamableHTTPSessionManager started.")

        logger.info("Lifespan startup phase completed successfully.")
        startup_ok = True

        # During the (potentially very long) startup phase, a stale
        # should_exit flag may have been set (e.g. by signal handler
        # overlap, or by uvicorn's own LifespanOn error detection).
        # Clear it so that uvicorn enters main_loop() instead of
        # returning immediately from _serve().
        import argus_mcp.cli as _cli_mod

        _uv = getattr(_cli_mod, "uvicorn_svr_inst", None)
        if _uv is not None and getattr(_uv, "should_exit", False):
            logger.warning(
                "Clearing stale should_exit flag on uvicorn server "
                "(was True after successful startup)."
            )
            _uv.should_exit = False
        # Also clear the lifespan-level flag that uvicorn checks in
        # Server.startup() right after our lifespan returns.
        _lifespan = getattr(_uv, "lifespan", None) if _uv is not None else None
        if _lifespan is not None and getattr(_lifespan, "should_exit", False):
            logger.warning("Clearing stale should_exit flag on uvicorn lifespan.")
            _lifespan.should_exit = False

        status_info_ready = gen_status_info(
            app_state,
            "Server started successfully and is ready.",
            tools=service.tools,
            resources=service.resources,
            prompts=service.prompts,
            conn_svrs_num=service.backends_connected,
            total_svrs_num=service.backends_total,
            route_map=service.registry.get_route_map(),
        )
        disp_console_status("✅ Service Ready", status_info_ready)
        log_file_status(status_info_ready)
        yield

    except ConfigurationError as exc:
        logger.exception("Configuration error: %s", exc)
        err_detail_msg = f"Configuration error: {exc}"
        status_info_fail = gen_status_info(
            app_state,
            "Server startup failed.",
            err_msg=err_detail_msg,
            total_svrs_num=service.backends_total,
        )
        disp_console_status("❌ Startup Failed", status_info_fail)
        log_file_status(status_info_fail, log_lvl=logging.ERROR)
        raise
    except BackendServerError as exc:
        logger.exception("Backend error: %s", exc)
        err_detail_msg = f"Backend error: {exc}"
        status_info_fail = gen_status_info(
            app_state,
            "Server startup failed.",
            err_msg=err_detail_msg,
            conn_svrs_num=service.backends_connected,
            total_svrs_num=service.backends_total,
        )
        disp_console_status("❌ Startup Failed", status_info_fail)
        log_file_status(status_info_fail, log_lvl=logging.ERROR)
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "Unexpected error during lifespan startup: %s",
            exc,
        )
        err_detail_msg = f"Unexpected error: {type(exc).__name__} - {exc}"
        status_info_fail = gen_status_info(
            app_state,
            "Server startup failed.",
            err_msg=err_detail_msg,
            conn_svrs_num=service.backends_connected,
            total_svrs_num=service.backends_total,
        )
        disp_console_status("❌ Startup Failed", status_info_fail)
        log_file_status(status_info_fail, log_lvl=logging.ERROR)
        raise
    finally:
        logger.info(
            "Server '%s' shutdown sequence started...",
            SERVER_NAME,
        )
        status_info_shutdown = gen_status_info(
            app_state,
            "Server is shutting down...",
            tools=service.tools,
            resources=service.resources,
            prompts=service.prompts,
            conn_svrs_num=service.backends_connected,
            total_svrs_num=service.backends_total,
        )
        disp_console_status("🛑 Shutting Down", status_info_shutdown, is_final=False)
        log_file_status(status_info_shutdown, log_lvl=logging.WARNING)

        # Stop SDK streamable-HTTP session manager first (closes all
        # active MCP sessions and their task groups).
        _es = _exit_stack if "_exit_stack" in dir() else None
        await _shutdown_streamable_http(_es, mcp_server)
        await service.stop()

        final_msg_short = (
            "Server shut down normally."
            if startup_ok
            else (
                f"Server exited abnormally"
                f"{(f' - Error: {err_detail_msg}' if err_detail_msg else '')}"
            )
        )
        final_icon = "✅" if startup_ok else "❌"
        final_log_lvl = logging.INFO if startup_ok else logging.ERROR

        status_info_final = gen_status_info(
            app_state,
            final_msg_short,
            err_msg=err_detail_msg if not startup_ok else None,
        )
        disp_console_status(f"{final_icon} Final Status", status_info_final, is_final=True)
        log_file_status(status_info_final, log_lvl=final_log_lvl)
        logger.info(
            "Server '%s' shutdown sequence completed.",
            SERVER_NAME,
        )
