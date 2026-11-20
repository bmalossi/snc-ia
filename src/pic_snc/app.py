"""API FastAPI do PIC-SNC v0.1 — Motor Cognitivo Ativo desacoplado do agente principal.

Endpoints:
- GET  /health                  -> Estado da API, protocolo, avaliador e política de autoridade
- GET  /health/ready            -> Prontidão operacional do banco SQLite e dependências
- POST /events                  -> Recebe evento observado, valida tenant autenticado e enfileira (202 Accepted)
- GET  /impulses                -> Consome impulsos emitidos espontaneamente (paginado, filtrado por estado)
- POST /impulses/{id}/ack       -> Confirma recebimento de impulso (`acknowledged`)
- POST /impulses/{id}/apply     -> Confirma aplicação ou respeito à solicitação de pausa A3 (`applied`)
"""

from typing import Annotated

from fastapi import Body, Depends, FastAPI, Header, HTTPException, Query, status
from fastapi.responses import JSONResponse

from pic_snc import __version__
from pic_snc.bus import SQLiteImpulseBus
from pic_snc.config import EnvironmentMode, SNCConfig
from pic_snc.logging_utils import get_structured_logger, log_cognitive_step
from pic_snc.pipeline import CognitivePipeline
from pic_snc.schemas import (
    AuthContext,
    Event,
    EventIngestResponse,
    HealthResponse,
    Impulse,
    ImpulseAckRequest,
    ImpulseApplyRequest,
    ImpulsePageResponse,
    ImpulseStatus,
    Outcome,
    OutcomeResult,
    ReadinessResponse,
)
from pic_snc.storage import StorageError, TenantIsolationError


def _resolve_auth_context(
    config: SNCConfig,
    authorization: str | None,
    x_dev_tenant_id: str | None,
    x_dev_principal_id: str | None,
) -> AuthContext:
    """Resolve identidade autenticada para isolamento multi-tenant.

    Em produção (`EnvironmentMode.PRODUCTION`), cabeçalhos de desenvolvimento são bloqueados.
    """
    if authorization is not None and authorization.strip():
        parts = authorization.strip().split(" ", 1)
        if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Cabeçalho Authorization deve usar o formato 'Bearer <token>'.",
            )
        token = parts[1].strip()
        mapped = config.static_api_tokens.get(token)
        if not mapped:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token Bearer inválido ou não autorizado.",
            )
        if ":" in mapped:
            tenant_id, principal_id = mapped.split(":", 1)
        else:
            tenant_id, principal_id = mapped, f"token:{mapped}"
        return AuthContext(
            principal_id=principal_id.strip(),
            tenant_id=tenant_id.strip(),
            authenticated=True,
            auth_scheme="bearer_token",
        )

    dev_headers_allowed = (
        config.allow_dev_identity_headers
        and config.environment != EnvironmentMode.PRODUCTION
    )
    if dev_headers_allowed and x_dev_tenant_id and x_dev_tenant_id.strip():
        principal = (x_dev_principal_id or "").strip() or "dev_principal"
        return AuthContext(
            principal_id=principal,
            tenant_id=x_dev_tenant_id.strip(),
            authenticated=True,
            auth_scheme="dev_header",
        )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Identidade autenticada obrigatória (Bearer token ou cabeçalho de desenvolvimento em ambiente local).",
    )


def create_app(
    config: SNCConfig | None = None,
    pipeline: CognitivePipeline | None = None,
) -> FastAPI:
    """Fábrica da aplicação FastAPI para o PIC-SNC v0.1."""
    active_config = config or (pipeline.config if pipeline is not None else SNCConfig.from_env())
    active_pipeline = pipeline or CognitivePipeline(active_config)
    storage = active_pipeline.storage
    logger = get_structured_logger("pic_snc.api")

    api = FastAPI(
        title="PIC-SNC v0.1 — Sistema Nervoso Cognitivo para Agentes de IA",
        version=__version__,
        description=(
            "API independente de memória ativa: observa eventos e publica impulsos "
            "cognitivos espontaneamente sem consultas de busca do agente principal."
        ),
    )
    api.state.config = active_config
    api.state.pipeline = active_pipeline

    def get_auth_context(
        authorization: Annotated[str | None, Header(alias="Authorization")] = None,
        x_dev_tenant_id: Annotated[str | None, Header(alias="X-Dev-Tenant-Id")] = None,
        x_dev_principal_id: Annotated[str | None, Header(alias="X-Dev-Principal-Id")] = None,
    ) -> AuthContext:
        return _resolve_auth_context(
            active_config,
            authorization=authorization,
            x_dev_tenant_id=x_dev_tenant_id,
            x_dev_principal_id=x_dev_principal_id,
        )

    dev_identity_active = (
        active_config.allow_dev_identity_headers
        and active_config.environment != EnvironmentMode.PRODUCTION
    )

    @api.get("/health", response_model=HealthResponse)
    def get_health() -> HealthResponse:
        db_ok = storage.check_health()
        return HealthResponse(
            status="ok",
            version=__version__,
            evaluator_mode=active_config.evaluator_mode.value,
            a4_enabled=active_config.enable_a4,
            external_action_policy=active_config.external_action_policy.value,
            environment=active_config.environment.value,
            dev_identity_enabled=dev_identity_active,
            database_ready=db_ok,
        )

    @api.get("/health/ready", response_model=ReadinessResponse)
    def get_readiness() -> JSONResponse:
        db_ok = storage.check_health()
        payload = ReadinessResponse(
            status="ready" if db_ok else "not_ready",
            database_ok=db_ok,
            evaluator_mode=active_config.evaluator_mode.value,
            a4_enabled=active_config.enable_a4,
            environment=active_config.environment.value,
            dev_identity_enabled=dev_identity_active,
        )
        return JSONResponse(
            status_code=status.HTTP_200_OK if db_ok else status.HTTP_503_SERVICE_UNAVAILABLE,
            content=payload.model_dump(mode="json"),
        )

    @api.post(
        "/events",
        status_code=status.HTTP_202_ACCEPTED,
        response_model=EventIngestResponse,
    )
    def post_events(
        event: Event,
        auth_context: Annotated[AuthContext, Depends(get_auth_context)],
    ) -> EventIngestResponse:
        """Recebe evento observado, valida isolamento multi-tenant e enfileira para o worker."""
        if event.tenant_id != auth_context.tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="tenant_id do evento não corresponde à identidade autenticada.",
            )

        try:
            ingest_res = active_pipeline.submit_event(event, auth_context=auth_context)
        except TenantIsolationError as exc:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc

        log_cognitive_step(
            logger,
            stage="api.post_events",
            action="event_accepted_for_worker",
            tenant_id=auth_context.tenant_id,
            event_id=ingest_res.event.event_id,
            details={
                "deduplicated": ingest_res.deduplicated,
                "processing_status": ingest_res.event.processing_status.value,
            },
        )
        return EventIngestResponse(
            event_id=ingest_res.event.event_id,
            tenant_id=ingest_res.event.tenant_id,
            status="duplicate_ignored" if ingest_res.deduplicated else "queued",
            deduplicated=ingest_res.deduplicated,
            processing_status=ingest_res.event.processing_status,
            outbox_id=ingest_res.outbox_entry.outbox_id if ingest_res.outbox_entry else None,
        )

    @api.get("/impulses", response_model=ImpulsePageResponse)
    def get_impulses(
        auth_context: Annotated[AuthContext, Depends(get_auth_context)],
        tenant_id: str | None = Query(default=None),
        impulse_status: ImpulseStatus | None = Query(default=None, alias="status"),
        limit: int = Query(default=20, ge=1, le=100),
        cursor: str | None = Query(default=None),
        mark_delivered: bool = Query(default=False),
    ) -> ImpulsePageResponse:
        """Lista impulsos gerados espontaneamente para o tenant autenticado."""
        if tenant_id is not None and tenant_id.strip() != auth_context.tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Acesso negado: parâmetro tenant_id diverge da identidade autenticada.",
            )

        try:
            items, next_cursor = storage.list_impulses_paginated(
                auth_context.tenant_id,
                status=impulse_status,
                limit=limit,
                cursor=cursor,
                auth_context=auth_context,
            )
        except StorageError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except TenantIsolationError as exc:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc

        if mark_delivered and items:
            updated_items: list[Impulse] = []
            for imp in items:
                if imp.status in {ImpulseStatus.PUBLISHED, ImpulseStatus.DOWNGRADED}:
                    if isinstance(active_pipeline.impulse_bus, SQLiteImpulseBus):
                        delivered_imp = active_pipeline.impulse_bus.mark_delivered(
                            auth_context.tenant_id,
                            imp.impulse_id,
                            auth_context=auth_context,
                        )
                    else:
                        delivered_imp = storage.update_impulse_status(
                            auth_context.tenant_id,
                            imp.impulse_id,
                            ImpulseStatus.DELIVERED,
                            auth_context=auth_context,
                        )
                    updated_items.append(delivered_imp)
                else:
                    updated_items.append(imp)
            items = updated_items

        return ImpulsePageResponse(
            tenant_id=auth_context.tenant_id,
            items=items,
            count=len(items),
            next_cursor=next_cursor,
        )

    @api.post("/impulses/{impulse_id}/ack", response_model=Impulse)
    def ack_impulse(
        impulse_id: str,
        auth_context: Annotated[AuthContext, Depends(get_auth_context)],
        _body: ImpulseAckRequest | None = Body(default=None),
    ) -> Impulse:
        """Confirma o recebimento do impulso pelo agente consumidor (`acknowledged`)."""
        existing = storage.get_impulse(
            auth_context.tenant_id,
            impulse_id,
            auth_context=auth_context,
        )
        if existing is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Impulso {impulse_id!r} não encontrado para o tenant autenticado.",
            )

        return active_pipeline.impulse_bus.acknowledge(
            auth_context.tenant_id,
            impulse_id,
            auth_context=auth_context,
        )

    @api.post("/impulses/{impulse_id}/apply", response_model=Impulse)
    def apply_impulse(
        impulse_id: str,
        payload: ImpulseApplyRequest,
        auth_context: Annotated[AuthContext, Depends(get_auth_context)],
    ) -> Impulse:
        """Confirma aplicação de um impulso ou acatamento de solicitação de pausa A3 (`applied`)."""
        existing = storage.get_impulse(
            auth_context.tenant_id,
            impulse_id,
            auth_context=auth_context,
        )
        if existing is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Impulso {impulse_id!r} não encontrado para o tenant autenticado.",
            )

        if isinstance(active_pipeline.impulse_bus, SQLiteImpulseBus):
            applied = active_pipeline.impulse_bus.apply_impulse(
                auth_context.tenant_id,
                impulse_id,
                pause_respected=payload.pause_respected,
                auth_context=auth_context,
            )
        else:
            applied = storage.mark_impulse_applied(
                auth_context.tenant_id,
                impulse_id,
                intervention_applied=payload.pause_respected,
                auth_context=auth_context,
            )

        if payload.record_outcome:
            active_pipeline.record_outcome(
                Outcome(
                    impulse_id=impulse_id,
                    tenant_id=auth_context.tenant_id,
                    result=(
                        OutcomeResult.ACCEPTED
                        if payload.pause_respected
                        else OutcomeResult.IGNORED
                    ),
                    notes=payload.notes,
                    context={
                        "action_taken": payload.action_taken,
                        "pause_respected": payload.pause_respected,
                    },
                ),
                auth_context=auth_context,
            )

        return applied

    return api


def __getattr__(name: str) -> object:
    """Permite `uvicorn pic_snc.app:app` sem criar banco local ao importar o módulo em testes."""
    if name == "app":
        return create_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
