"""Interfaces formais (Protocols) dos componentes cognitivos do PIC-SNC v0.1.

Separação estrita de responsabilidades:
- `Observer`: observa eventos parciais/completos, acumula contexto por `(tenant_id, stream_id)`
  e emite uma `ObservationDecision` (sinal de ativação) para o pipeline — nunca emite impulsos.
- `ActivationEngine`: recupera e pontua memórias associadas a partir da decisão de observação.
- `Evaluator`: avalia a relevância dos pares (contexto observado, memória ativada) produzindo `Candidate`s.
- `Governor`: aplica políticas de autoridade (A0–A3; A4 bloqueado por padrão), deduplicação e cooldown.
- `ImpulseBus`: publica e gerencia o ciclo de vida dos impulsos via SQLite Outbox sem busca do agente.
- `OutcomeTracker`: registra a decisão do agente e o desfecho observado sem confundir aceitação com correção.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pic_snc.schemas import (
    ActivatedMemory,
    AuthContext,
    Candidate,
    Event,
    Impulse,
    ObservationDecision,
    Outcome,
)


@runtime_checkable
class Observer(Protocol):
    """Observador cognitivo progressivo."""

    def observe(
        self,
        event: Event,
        *,
        auth_context: AuthContext | None = None,
    ) -> ObservationDecision:
        """Observa um evento, acumula contexto e decide se deve iniciar ativação associativa."""
        ...


@runtime_checkable
class ActivationEngine(Protocol):
    """Motor de ativação associativa."""

    def activate(
        self,
        observation: ObservationDecision,
        *,
        auth_context: AuthContext | None = None,
    ) -> list[ActivatedMemory]:
        """Ativa memórias associadas ao contexto observado para o tenant correspondente."""
        ...


@runtime_checkable
class Evaluator(Protocol):
    """Avaliador cognitivo de candidatos (MockEvaluator ou JevEvaluatorAdapter)."""

    def evaluate(
        self,
        observation: ObservationDecision,
        activated_memories: list[ActivatedMemory],
    ) -> list[Candidate]:
        """Avalia memórias ativadas frente ao contexto observado e produz candidatos."""
        ...


@runtime_checkable
class Governor(Protocol):
    """Governador de autoridade cognitiva (A0–A3; A4 desativado por padrão)."""

    def govern(
        self,
        observation: ObservationDecision,
        candidates: list[Candidate],
        *,
        auth_context: AuthContext | None = None,
    ) -> list[Impulse]:
        """Classifica autoridade, previne duplicatas/repetições e emite impulsos governados."""
        ...


@runtime_checkable
class ImpulseBus(Protocol):
    """Barramento de publicação espontânea de impulsos apoiado no Outbox."""

    def publish(
        self,
        impulses: list[Impulse],
        *,
        auth_context: AuthContext | None = None,
    ) -> list[Impulse]:
        """Persiste e publica impulsos de forma idempotente no barramento/outbox."""
        ...

    def deliver_pending(
        self,
        tenant_id: str,
        *,
        limit: int = 50,
        auth_context: AuthContext | None = None,
    ) -> list[Impulse]:
        """Despacha impulsos pendentes do outbox marcando-os como entregues (`delivered`)."""
        ...

    def acknowledge(
        self,
        tenant_id: str,
        impulse_id: str,
        *,
        auth_context: AuthContext | None = None,
    ) -> Impulse:
        """Registra confirmação de recebimento (`acknowledged`) de um impulso."""
        ...


@runtime_checkable
class OutcomeTracker(Protocol):
    """Rastreador de desfechos e evidências pós-impulso."""

    def record(
        self,
        outcome: Outcome,
        *,
        auth_context: AuthContext | None = None,
    ) -> Outcome:
        """Registra o desfecho observado de um impulso sem presumir correção por mera aceitação."""
        ...
