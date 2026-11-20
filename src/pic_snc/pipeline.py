"""Orquestrador do Pipeline Cognitivo Ativo do PIC-SNC v0.1.

Fluxo autônomo (sem solicitação de busca do agente principal):
`Event → Observe → Activate → Evaluate → Govern → Publish (→ Outcome)`

Suporta:
1. Processamento imediato (`ingest_and_process`)
2. Enfileiramento desacoplado + trabalhador independente (`submit_event` + `process_pending_events`)
3. Recuperação segura após falhas (`include_failed=True` / `retry_event`) sem duplicar candidatos ou impulsos.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pic_snc.activation import AssociativeActivationEngine
from pic_snc.bus import SQLiteImpulseBus
from pic_snc.config import SNCConfig
from pic_snc.evaluator import create_evaluator
from pic_snc.governor import CognitiveGovernor
from pic_snc.interfaces import (
    ActivationEngine,
    Evaluator,
    Governor,
    ImpulseBus,
    Observer,
    OutcomeTracker,
)
from pic_snc.logging_utils import get_structured_logger, log_cognitive_step
from pic_snc.observer import ObserverPolicy, ProgressiveCognitiveObserver
from pic_snc.outcome import SQLiteOutcomeTracker
from pic_snc.schemas import (
    ActivatedMemory,
    AuthContext,
    Candidate,
    Event,
    EventProcessingStatus,
    Impulse,
    ObservationDecision,
    OutboxStatus,
    OutboxTopic,
    Outcome,
)
from pic_snc.storage import EventIngestResult, SQLiteStorage


@dataclass(frozen=True, slots=True)
class PipelineExecutionResult:
    """Resultado auditável do processamento de um evento pelo pipeline cognitivo."""

    event: Event
    deduplicated: bool
    observation: ObservationDecision | None
    activated_memories: list[ActivatedMemory] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    governed_impulses: list[Impulse] = field(default_factory=list)
    published_impulses: list[Impulse] = field(default_factory=list)


class CognitivePipeline:
    """Motor cognitivo ativo desacoplado do agente principal."""

    def __init__(
        self,
        config: SNCConfig,
        storage: SQLiteStorage | None = None,
        *,
        observer: Observer | None = None,
        activation_engine: ActivationEngine | None = None,
        evaluator: Evaluator | None = None,
        governor: Governor | None = None,
        impulse_bus: ImpulseBus | None = None,
        outcome_tracker: OutcomeTracker | None = None,
    ) -> None:
        self.config = config
        self.storage = storage or SQLiteStorage(
            config.db_path,
            require_authenticated_context=config.require_authenticated_context,
        )
        self.observer: Observer = observer or ProgressiveCognitiveObserver(
            ObserverPolicy(
                min_single_event_density=config.progressive_eval_min_density,
            ),
            storage=self.storage,
        )
        self.activation_engine: ActivationEngine = (
            activation_engine
            or AssociativeActivationEngine(
                self.storage,
                min_salience=config.min_activation_salience,
            )
        )
        self.evaluator: Evaluator = evaluator or create_evaluator(config)
        self.governor: Governor = governor or CognitiveGovernor(
            config,
            storage=self.storage,
        )
        self.impulse_bus: ImpulseBus = impulse_bus or SQLiteImpulseBus(self.storage)
        self.outcome_tracker: OutcomeTracker = (
            outcome_tracker or SQLiteOutcomeTracker(self.storage)
        )
        self._cached_decisions: dict[tuple[str, str], ObservationDecision] = {}
        self._logger = get_structured_logger("pic_snc.pipeline")

    def submit_event(
        self,
        event: Event,
        *,
        auth_context: AuthContext | None = None,
    ) -> EventIngestResult:
        """Persiste um evento de forma transacional e idempotente para processamento independente."""
        return self.storage.ingest_event(
            event,
            auth_context=auth_context,
            enqueue_outbox=True,
            min_eval_density=0.0,
        )

    def ingest_and_process(
        self,
        event: Event,
        *,
        auth_context: AuthContext | None = None,
    ) -> PipelineExecutionResult:
        """Ingere o evento de forma idempotente e executa o ciclo cognitivo completo."""
        ingest_res = self.submit_event(event, auth_context=auth_context)
        if ingest_res.deduplicated:
            log_cognitive_step(
                self._logger,
                stage="pipeline.ingest_and_process",
                action="duplicate_event_short_circuited",
                tenant_id=event.tenant_id,
                event_id=ingest_res.event.event_id,
            )
            return PipelineExecutionResult(
                event=ingest_res.event,
                deduplicated=True,
                observation=None,
            )

        return self._execute_for_persisted_event(
            ingest_res.event,
            auth_context=auth_context,
        )

    def process_pending_events(
        self,
        tenant_id: str,
        *,
        include_failed: bool = False,
        limit: int = 50,
        auth_context: AuthContext | None = None,
    ) -> list[PipelineExecutionResult]:
        """Trabalhador independente: processa eventos pendentes (e falhos quando `include_failed=True`)."""
        pending = self.storage.list_events(
            tenant_id,
            status=EventProcessingStatus.PENDING,
            limit=limit,
            auth_context=auth_context,
        )
        targets = list(pending)
        if include_failed:
            failed = self.storage.list_events(
                tenant_id,
                status=EventProcessingStatus.FAILED,
                limit=limit,
                auth_context=auth_context,
            )
            targets.extend(failed)

        results: list[PipelineExecutionResult] = []
        for event in targets:
            res = self._execute_for_persisted_event(
                event,
                auth_context=auth_context,
            )
            results.append(res)
        return results

    def retry_event(
        self,
        tenant_id: str,
        event_id: str,
        *,
        auth_context: AuthContext | None = None,
    ) -> PipelineExecutionResult:
        """Reprocessa com segurança um evento pendente ou que falhou anteriormente."""
        event = self.storage.get_event(tenant_id, event_id, auth_context=auth_context)
        if event is None:
            raise ValueError(
                f"Evento {event_id!r} não encontrado para o tenant {tenant_id!r}."
            )
        if event.processing_status == EventProcessingStatus.PROCESSED:
            existing_impulses = [
                imp
                for imp in self.storage.list_impulses(tenant_id, auth_context=auth_context)
                if imp.event_id == event_id
            ]
            return PipelineExecutionResult(
                event=event,
                deduplicated=True,
                observation=None,
                published_impulses=existing_impulses,
            )
        return self._execute_for_persisted_event(event, auth_context=auth_context)

    def record_outcome(
        self,
        outcome: Outcome,
        *,
        auth_context: AuthContext | None = None,
    ) -> Outcome:
        """Delega o registro verificável de desfecho ao OutcomeTracker."""
        return self.outcome_tracker.record(outcome, auth_context=auth_context)

    def _execute_for_persisted_event(
        self,
        event: Event,
        *,
        auth_context: AuthContext | None = None,
    ) -> PipelineExecutionResult:
        tenant_id = event.tenant_id
        cache_key = (tenant_id, event.event_id)

        self.storage.update_event_status(
            tenant_id,
            event.event_id,
            EventProcessingStatus.PROCESSING,
            auth_context=auth_context,
        )

        try:
            # 1. OBSERVE (reutiliza decisão em cache se for retry após falha em estágio posterior)
            decision = self._cached_decisions.get(cache_key)
            if decision is None:
                decision = self.observer.observe(event, auth_context=auth_context)
                self._cached_decisions[cache_key] = decision

            if not decision.should_activate:
                new_status = (
                    EventProcessingStatus.BUFFERED
                    if not decision.stream_finalized
                    else EventProcessingStatus.SKIPPED
                )
                self.storage.update_event_status(
                    tenant_id,
                    event.event_id,
                    new_status,
                    auth_context=auth_context,
                )
                self._mark_event_outbox_done(tenant_id, event.event_id, auth_context)
                self._cached_decisions.pop(cache_key, None)
                updated_event = event.model_copy(update={"processing_status": new_status})
                return PipelineExecutionResult(
                    event=updated_event,
                    deduplicated=False,
                    observation=decision,
                )

            # 2. ACTIVATE
            activated_memories = self.activation_engine.activate(
                decision,
                auth_context=auth_context,
            )
            if not activated_memories:
                self._finalize_accumulated_events(
                    tenant_id,
                    decision.accumulated_event_ids or [event.event_id],
                    EventProcessingStatus.PROCESSED,
                    auth_context,
                )
                self._cached_decisions.pop(cache_key, None)
                updated_event = event.model_copy(
                    update={"processing_status": EventProcessingStatus.PROCESSED}
                )
                return PipelineExecutionResult(
                    event=updated_event,
                    deduplicated=False,
                    observation=decision,
                )

            # 3. EVALUATE
            candidates = self.evaluator.evaluate(decision, activated_memories)
            for candidate in candidates:
                self.storage.save_candidate(candidate, auth_context=auth_context)

            # 4. GOVERN
            governed_impulses = self.governor.govern(
                decision,
                candidates,
                auth_context=auth_context,
            )

            # 5. PUBLISH
            published_impulses = self.impulse_bus.publish(
                governed_impulses,
                auth_context=auth_context,
            )

            # Marca todos os eventos acumulados na decisão como PROCESSED
            target_event_ids = list(
                dict.fromkeys([*decision.accumulated_event_ids, event.event_id])
            )
            self._finalize_accumulated_events(
                tenant_id,
                target_event_ids,
                EventProcessingStatus.PROCESSED,
                auth_context,
            )
            self._cached_decisions.pop(cache_key, None)

            updated_event = event.model_copy(
                update={"processing_status": EventProcessingStatus.PROCESSED}
            )
            log_cognitive_step(
                self._logger,
                stage="pipeline.complete",
                action="cognitive_cycle_finished",
                tenant_id=tenant_id,
                event_id=event.event_id,
                details={
                    "activated_count": len(activated_memories),
                    "candidates_count": len(candidates),
                    "published_impulses_count": len(published_impulses),
                },
            )
            return PipelineExecutionResult(
                event=updated_event,
                deduplicated=False,
                observation=decision,
                activated_memories=activated_memories,
                candidates=candidates,
                governed_impulses=governed_impulses,
                published_impulses=published_impulses,
            )

        except Exception as exc:
            self.storage.update_event_status(
                tenant_id,
                event.event_id,
                EventProcessingStatus.FAILED,
                auth_context=auth_context,
            )
            self._mark_event_outbox_failed(
                tenant_id,
                event.event_id,
                str(exc),
                auth_context,
            )
            log_cognitive_step(
                self._logger,
                stage="pipeline.error",
                action="cognitive_cycle_failed",
                tenant_id=tenant_id,
                event_id=event.event_id,
                details={"error_type": type(exc).__name__, "error": str(exc)},
            )
            raise

    def _finalize_accumulated_events(
        self,
        tenant_id: str,
        event_ids: list[str],
        status: EventProcessingStatus,
        auth_context: AuthContext | None,
    ) -> None:
        for ev_id in event_ids:
            existing = self.storage.get_event(tenant_id, ev_id, auth_context=auth_context)
            if existing is not None:
                self.storage.update_event_status(
                    tenant_id,
                    ev_id,
                    status,
                    auth_context=auth_context,
                )
                self._mark_event_outbox_done(tenant_id, ev_id, auth_context)

    def _mark_event_outbox_done(
        self,
        tenant_id: str,
        event_id: str,
        auth_context: AuthContext | None,
    ) -> None:
        entries = self.storage.list_outbox_entries(
            tenant_id,
            topic=OutboxTopic.EVENT_INGESTED,
            auth_context=auth_context,
        )
        for entry in entries:
            if entry.aggregate_id == event_id and entry.status != OutboxStatus.PUBLISHED:
                self.storage.mark_outbox_processed(
                    tenant_id,
                    entry.outbox_id,
                    auth_context=auth_context,
                )

    def _mark_event_outbox_failed(
        self,
        tenant_id: str,
        event_id: str,
        error_msg: str,
        auth_context: AuthContext | None,
    ) -> None:
        entries = self.storage.list_outbox_entries(
            tenant_id,
            topic=OutboxTopic.EVENT_INGESTED,
            auth_context=auth_context,
        )
        for entry in entries:
            if entry.aggregate_id == event_id and entry.status != OutboxStatus.PUBLISHED:
                self.storage.mark_outbox_failed(
                    tenant_id,
                    entry.outbox_id,
                    error_msg,
                    auth_context=auth_context,
                )
