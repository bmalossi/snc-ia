"""Barramento de Impulsos (ImpulseBus) apoiado no SQLite Outbox do PIC-SNC v0.1.

Responsabilidades:
- Publicar impulsos espontaneamente sem qualquer solicitação de busca do agente principal.
- Garantir idempotência na publicação (`(tenant_id, impulse_id)` e `(tenant_id, topic, aggregate_id)`).
- Separar o ciclo de vida: `created`/`pending` -> `published` -> `delivered` -> `acknowledged` (ou `failed`).
- Diferenciar o tópico de eventos recebidos (`event.ingested`) dos tópicos de impulsos
  resultantes (`impulse.emitted` e `impulse.protective_block`).
- Registrar falhas explicitamente no Outbox e nos logs estruturados, sem perda silenciosa.
"""

from __future__ import annotations

from collections.abc import Callable

from pic_snc.logging_utils import get_structured_logger, log_cognitive_step
from pic_snc.schemas import (
    AuthContext,
    Impulse,
    ImpulseStatus,
    OutboxStatus,
    OutboxTopic,
)
from pic_snc.storage import SQLiteStorage

ImpulseSubscriber = Callable[[Impulse], None]


class SQLiteImpulseBus:
    """Barramento transacional de impulsos com suporte a polling, outbox e assinantes."""

    def __init__(self, storage: SQLiteStorage) -> None:
        self.storage = storage
        self._subscribers: list[ImpulseSubscriber] = []
        self._logger = get_structured_logger("pic_snc.bus")

    def subscribe(self, subscriber: ImpulseSubscriber) -> None:
        """Registra um callback assinante para entrega proativa de impulsos."""
        if subscriber not in self._subscribers:
            self._subscribers.append(subscriber)

    def publish(
        self,
        impulses: list[Impulse],
        *,
        auth_context: AuthContext | None = None,
    ) -> list[Impulse]:
        """Persiste impulsos governados e enfileira os publicáveis no outbox de impulsos.

        Retorna apenas os impulsos efetivamente publicados (`PUBLISHED` ou `DOWNGRADED`),
        mantendo os `SUPPRESSED` persistidos para auditoria.
        """
        published_impulses: list[Impulse] = []

        for impulse in impulses:
            should_enqueue = impulse.status in {
                ImpulseStatus.PUBLISHED,
                ImpulseStatus.DOWNGRADED,
            }
            persisted, outbox_entry = self.storage.save_impulse(
                impulse,
                auth_context=auth_context,
                enqueue_outbox=should_enqueue,
            )

            if persisted.status in {
                ImpulseStatus.PUBLISHED,
                ImpulseStatus.DOWNGRADED,
                ImpulseStatus.DELIVERED,
                ImpulseStatus.ACKNOWLEDGED,
            }:
                published_impulses.append(persisted)
                log_cognitive_step(
                    self._logger,
                    stage="bus.publish",
                    action="impulse_published_to_bus",
                    tenant_id=persisted.tenant_id,
                    event_id=persisted.event_id,
                    candidate_id=persisted.candidate_id,
                    impulse_id=persisted.impulse_id,
                    details={
                        "autonomy_level": persisted.autonomy_level.value,
                        "status": persisted.status.value,
                        "outbox_id": outbox_entry.outbox_id if outbox_entry else None,
                        "outbox_topic": outbox_entry.topic.value if outbox_entry else None,
                    },
                )
            else:
                log_cognitive_step(
                    self._logger,
                    stage="bus.publish",
                    action="impulse_suppressed_recorded",
                    tenant_id=persisted.tenant_id,
                    event_id=persisted.event_id,
                    candidate_id=persisted.candidate_id,
                    impulse_id=persisted.impulse_id,
                    details={
                        "status": persisted.status.value,
                        "suppressed_reason": persisted.suppressed_reason,
                    },
                )

        return published_impulses

    def deliver_pending(
        self,
        tenant_id: str,
        *,
        limit: int = 50,
        auth_context: AuthContext | None = None,
    ) -> list[Impulse]:
        """Consome registros pendentes de impulsos no Outbox, notifica assinantes e marca como `delivered`.

        Ignora registros de `event.ingested` (diferenciação estrita entre fila de eventos e fila de impulsos).
        """
        pending_outbox = [
            entry
            for entry in self.storage.list_outbox_entries(
                tenant_id,
                status=OutboxStatus.PENDING,
                limit=limit,
                auth_context=auth_context,
            )
            if entry.topic
            in {OutboxTopic.IMPULSE_EMITTED, OutboxTopic.PROTECTIVE_BLOCK_TRIGGERED}
        ]

        delivered: list[Impulse] = []
        for entry in pending_outbox:
            impulse = self.storage.get_impulse(
                tenant_id,
                entry.aggregate_id,
                auth_context=auth_context,
            )
            if impulse is None:
                self.storage.mark_outbox_failed(
                    tenant_id,
                    entry.outbox_id,
                    f"Impulso {entry.aggregate_id!r} não encontrado.",
                    auth_context=auth_context,
                )
                continue

            try:
                for subscriber in self._subscribers:
                    subscriber(impulse)

                updated = self.storage.update_impulse_status(
                    tenant_id,
                    impulse.impulse_id,
                    ImpulseStatus.DELIVERED,
                    auth_context=auth_context,
                )
                self.storage.mark_outbox_processed(
                    tenant_id,
                    entry.outbox_id,
                    auth_context=auth_context,
                )
                delivered.append(updated)
                log_cognitive_step(
                    self._logger,
                    stage="bus.deliver_pending",
                    action="impulse_delivered",
                    tenant_id=tenant_id,
                    event_id=updated.event_id,
                    impulse_id=updated.impulse_id,
                    details={"outbox_id": entry.outbox_id},
                )
            except Exception as exc:
                self.storage.mark_outbox_failed(
                    tenant_id,
                    entry.outbox_id,
                    str(exc),
                    auth_context=auth_context,
                )
                self.storage.update_impulse_status(
                    tenant_id,
                    impulse.impulse_id,
                    ImpulseStatus.FAILED,
                    suppressed_reason=f"delivery_error: {exc}",
                    auth_context=auth_context,
                )
                log_cognitive_step(
                    self._logger,
                    stage="bus.deliver_pending",
                    action="impulse_delivery_failed",
                    tenant_id=tenant_id,
                    event_id=impulse.event_id,
                    impulse_id=impulse.impulse_id,
                    details={"outbox_id": entry.outbox_id, "error": str(exc)},
                )

        return delivered

    def mark_delivered(
        self,
        tenant_id: str,
        impulse_id: str,
        *,
        auth_context: AuthContext | None = None,
    ) -> Impulse:
        """Marca um impulso específico como entregue (`delivered`) e conclui seu outbox."""
        updated = self.storage.update_impulse_status(
            tenant_id,
            impulse_id,
            ImpulseStatus.DELIVERED,
            auth_context=auth_context,
        )
        self._complete_impulse_outbox(tenant_id, impulse_id, auth_context=auth_context)
        log_cognitive_step(
            self._logger,
            stage="bus.mark_delivered",
            action="impulse_delivered",
            tenant_id=tenant_id,
            event_id=updated.event_id,
            impulse_id=updated.impulse_id,
        )
        return updated

    def acknowledge(
        self,
        tenant_id: str,
        impulse_id: str,
        *,
        auth_context: AuthContext | None = None,
    ) -> Impulse:
        """Marca um impulso como reconhecido (`acknowledged`) pelo consumidor/adaptador."""
        updated = self.storage.update_impulse_status(
            tenant_id,
            impulse_id,
            ImpulseStatus.ACKNOWLEDGED,
            auth_context=auth_context,
        )
        self._complete_impulse_outbox(tenant_id, impulse_id, auth_context=auth_context)
        log_cognitive_step(
            self._logger,
            stage="bus.acknowledge",
            action="impulse_acknowledged",
            tenant_id=tenant_id,
            event_id=updated.event_id,
            impulse_id=updated.impulse_id,
        )
        return updated

    def apply_impulse(
        self,
        tenant_id: str,
        impulse_id: str,
        *,
        pause_respected: bool = True,
        auth_context: AuthContext | None = None,
    ) -> Impulse:
        """Registra que o impulso foi efetivamente aplicado (ex.: pausa A3 respeitada)."""
        updated = self.storage.mark_impulse_applied(
            tenant_id,
            impulse_id,
            intervention_applied=pause_respected,
            auth_context=auth_context,
        )
        self._complete_impulse_outbox(tenant_id, impulse_id, auth_context=auth_context)
        log_cognitive_step(
            self._logger,
            stage="bus.apply_impulse",
            action="impulse_applied",
            tenant_id=tenant_id,
            event_id=updated.event_id,
            impulse_id=updated.impulse_id,
            details={
                "autonomy_level": updated.autonomy_level.value,
                "intervention_applied": updated.intervention_applied,
            },
        )
        return updated

    def _complete_impulse_outbox(
        self,
        tenant_id: str,
        impulse_id: str,
        *,
        auth_context: AuthContext | None = None,
    ) -> None:
        """Conclui eventual registro pendente no outbox associado ao impulso."""
        for topic in (OutboxTopic.IMPULSE_EMITTED, OutboxTopic.PROTECTIVE_BLOCK_TRIGGERED):
            entries = self.storage.list_outbox_entries(
                tenant_id,
                topic=topic,
                auth_context=auth_context,
            )
            for entry in entries:
                if entry.aggregate_id == impulse_id and entry.status != OutboxStatus.PUBLISHED:
                    self.storage.mark_outbox_processed(
                        tenant_id,
                        entry.outbox_id,
                        auth_context=auth_context,
                    )

    def list_active_impulses(
        self,
        tenant_id: str,
        *,
        limit: int = 50,
        auth_context: AuthContext | None = None,
    ) -> list[Impulse]:
        """Lista impulsos ativos emitidos ao barramento (`published`, `delivered`, `acknowledged`, `applied`, `downgraded`)."""
        all_impulses = self.storage.list_impulses(
            tenant_id,
            limit=limit,
            auth_context=auth_context,
        )
        return [
            imp
            for imp in all_impulses
            if imp.status
            in {
                ImpulseStatus.PUBLISHED,
                ImpulseStatus.DELIVERED,
                ImpulseStatus.ACKNOWLEDGED,
                ImpulseStatus.APPLIED,
                ImpulseStatus.DOWNGRADED,
            }
        ]
