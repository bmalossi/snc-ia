"""Observador Cognitivo Progressivo do PIC-SNC v0.1.

Responsabilidades:
- Observar eventos parciais, completos e marcos semânticos (`partial`, `complete`, `milestone`).
- Agregar contexto incremental isolado por `(tenant_id, stream_id)`.
- Determinar deterministicamente (sem consultar LLM) quando há informação suficiente
  para iniciar ativação associativa e avaliação.
- Suportar políticas configuráveis de ativação, gatilhos de sinal antecipado,
  ordenação por `sequence_no`, limite de buffer, expiração por TTL e finalização de stream.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from pic_snc.logging_utils import get_structured_logger, log_cognitive_step
from pic_snc.schemas import (
    AuthContext,
    Event,
    EventCompleteness,
    EventProcessingStatus,
    EvidenceRef,
    ObservationDecision,
)
from pic_snc.storage import SQLiteStorage

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_.\-À-ÿ]{3,}")

DEFAULT_EARLY_TRIGGER_SIGNALS: frozenset[str] = frozenset(
    {
        "risk",
        "risco",
        "contradiction",
        "contradicao",
        "contradição",
        "overdue",
        "atraso",
        "critical",
        "critico",
        "crítico",
        "breach",
        "violation",
        "violacao",
        "violação",
        "anomaly",
        "anomalia",
        "unauthorized",
        "rollback",
        "outage",
        "quota_exceeded",
        "unlimited_liability",
        "suspended",
        "suspensao",
        "suspensão",
    }
)


def extract_semantic_signals(event_type: str, payload: dict[str, Any]) -> list[str]:
    """Extrai tokens e sinais semânticos determinísticos do tipo e payload do evento."""
    raw_tokens: list[str] = []

    def _collect(value: Any) -> None:
        if isinstance(value, str):
            raw_tokens.append(value.lower())
            for match in _TOKEN_RE.findall(value.lower()):
                raw_tokens.append(match)
                if "." in match or "_" in match or "-" in match:
                    for part in re.split(r"[._\-]+", match):
                        if len(part) >= 3:
                            raw_tokens.append(part)
        elif isinstance(value, dict):
            for k, v in value.items():
                _collect(str(k))
                _collect(v)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                _collect(item)
        elif isinstance(value, (int, float, bool)) and not isinstance(value, bool):
            raw_tokens.append(str(value))

    _collect(event_type)
    _collect(payload)
    return list(dict.fromkeys(t.strip() for t in raw_tokens if t and len(t.strip()) >= 3))


@dataclass(frozen=True, slots=True)
class ObserverPolicy:
    """Política configurável de observação progressiva e gatilho de ativação."""

    min_single_event_density: float = 0.40
    accumulated_density_threshold: float = 0.65
    max_buffer_events: int = 16
    buffer_ttl_seconds: int = 300
    early_trigger_signals: frozenset[str] = DEFAULT_EARLY_TRIGGER_SIGNALS
    activate_on_milestone: bool = True
    activate_on_buffer_full: bool = True


@dataclass(slots=True)
class _StreamBuffer:
    """Estado interno acumulado de uma conversa/stream de um tenant."""

    tenant_id: str
    stream_id: str
    events: list[Event] = field(default_factory=list)
    seen_event_ids: set[str] = field(default_factory=set)
    seen_sequence_nos: set[int] = field(default_factory=set)
    last_updated_at: datetime | None = None
    finalized: bool = False


class ProgressiveCognitiveObserver:
    """Observador que acumula contexto incremental e emite `ObservationDecision`."""

    def __init__(
        self,
        policy: ObserverPolicy | None = None,
        storage: SQLiteStorage | None = None,
    ) -> None:
        self.policy = policy or ObserverPolicy()
        self.storage = storage
        self._buffers: dict[tuple[str, str], _StreamBuffer] = {}
        self._logger = get_structured_logger("pic_snc.observer")

    def observe(
        self,
        event: Event,
        *,
        auth_context: AuthContext | None = None,
    ) -> ObservationDecision:
        """Processa um evento observado, atualiza o buffer da stream e decide sobre ativação."""
        if auth_context is not None:
            auth_context.assert_tenant_access(event.tenant_id)

        stream_id = event.stream_id or f"stream:{event.event_id}"
        buffer_key = (event.tenant_id, stream_id)

        buffer = self._buffers.get(buffer_key)
        if buffer is not None and self._is_expired(buffer, event.timestamp):
            self._mark_expired_events_skipped(buffer, auth_context=auth_context)
            del self._buffers[buffer_key]
            buffer = None

        if (buffer is None or buffer.finalized) and event.stream_id is not None and self.storage is not None:
            reconstructed = self._reconstruct_buffer_from_storage(
                tenant_id=event.tenant_id,
                stream_id=stream_id,
                current_event=event,
                auth_context=auth_context,
            )
            if reconstructed is not None:
                buffer = reconstructed
                self._buffers[buffer_key] = buffer

        if buffer is None or buffer.finalized:
            buffer = _StreamBuffer(tenant_id=event.tenant_id, stream_id=stream_id)
            self._buffers[buffer_key] = buffer

        # Proteção contra duplicata dentro da mesma stream
        if event.event_id in buffer.seen_event_ids or (
            event.sequence_no is not None and event.sequence_no in buffer.seen_sequence_nos
        ):
            return ObservationDecision(
                tenant_id=event.tenant_id,
                stream_id=stream_id,
                trigger_event_id=event.event_id,
                trigger_event_type=event.event_type,
                trigger_source=event.source,
                accumulated_event_ids=[e.event_id for e in buffer.events],
                should_activate=False,
                trigger_reason="duplicate_in_stream_ignored",
                accumulated_density=self._compute_accumulated_density(buffer.events),
                detected_signals=[],
                aggregated_payload={},
                evidence=list(event.evidence),
                stream_finalized=buffer.finalized,
                observed_at=event.timestamp,
            )

        buffer.events.append(event)
        buffer.seen_event_ids.add(event.event_id)
        if event.sequence_no is not None:
            buffer.seen_sequence_nos.add(event.sequence_no)
        buffer.last_updated_at = event.timestamp

        # Ordena eventos por sequence_no (quando presente) e timestamp
        buffer.events.sort(
            key=lambda e: (
                0 if e.sequence_no is not None else 1,
                e.sequence_no if e.sequence_no is not None else 0,
                e.timestamp,
            )
        )

        # Limite de buffer: se exceder max_buffer_events, mantém janela recente
        buffer_reached_limit = len(buffer.events) >= self.policy.max_buffer_events
        if len(buffer.events) > self.policy.max_buffer_events:
            evicted = buffer.events.pop(0)
            buffer.seen_event_ids.discard(evicted.event_id)
            if evicted.sequence_no is not None:
                buffer.seen_sequence_nos.discard(evicted.sequence_no)

        aggregated_payload = self._aggregate_payloads(buffer.events)
        accumulated_density = self._compute_accumulated_density(buffer.events)
        combined_evidence = self._aggregate_evidence(buffer.events)
        detected_signals = extract_semantic_signals(event.event_type, aggregated_payload)

        early_signal_matched = self._has_early_trigger_signal(detected_signals)
        explicit_finalize = bool(event.payload.get("stream_finalized", False))

        should_activate, reason, finalize_now = self._decide_activation(
            event=event,
            accumulated_density=accumulated_density,
            early_signal_matched=early_signal_matched,
            buffer_reached_limit=buffer_reached_limit,
            explicit_finalize=explicit_finalize,
        )

        accumulated_ids = [e.event_id for e in buffer.events]
        if finalize_now:
            buffer.finalized = True
            self._buffers.pop(buffer_key, None)
        elif should_activate:
            # Ao disparar avaliação intermediária em stream contínua, limpa fragmentos já avaliados
            buffer.events.clear()

        decision = ObservationDecision(
            tenant_id=event.tenant_id,
            stream_id=stream_id,
            trigger_event_id=event.event_id,
            trigger_event_type=event.event_type,
            trigger_source=event.source,
            accumulated_event_ids=accumulated_ids,
            should_activate=should_activate,
            trigger_reason=reason,
            accumulated_density=accumulated_density,
            detected_signals=detected_signals,
            aggregated_payload=aggregated_payload,
            evidence=combined_evidence,
            stream_finalized=finalize_now,
            observed_at=event.timestamp,
        )

        log_cognitive_step(
            self._logger,
            stage="observer.observe",
            action="observation_evaluated",
            tenant_id=event.tenant_id,
            event_id=event.event_id,
            details={
                "stream_id": stream_id,
                "completeness": event.completeness.value,
                "should_activate": should_activate,
                "trigger_reason": reason,
                "accumulated_density": round(accumulated_density, 4),
                "buffered_events_count": len(accumulated_ids),
            },
        )
        return decision

    def get_active_stream_count(self, tenant_id: str | None = None) -> int:
        """Retorna a quantidade de streams ativas em buffer (filtrada opcionalmente por tenant)."""
        if tenant_id is None:
            return len(self._buffers)
        return sum(1 for (t_id, _) in self._buffers if t_id == tenant_id)

    def _is_expired(self, buffer: _StreamBuffer, current_time: datetime) -> bool:
        if buffer.last_updated_at is None:
            return False
        ttl = timedelta(seconds=self.policy.buffer_ttl_seconds)
        return (current_time - buffer.last_updated_at) > ttl

    def _mark_expired_events_skipped(
        self,
        buffer: _StreamBuffer,
        *,
        auth_context: AuthContext | None = None,
    ) -> None:
        if self.storage is None:
            return
        for buffered_ev in buffer.events:
            existing = self.storage.get_event(
                buffer.tenant_id,
                buffered_ev.event_id,
                auth_context=auth_context,
            )
            if existing is not None and existing.processing_status == EventProcessingStatus.BUFFERED:
                self.storage.update_event_status(
                    buffer.tenant_id,
                    buffered_ev.event_id,
                    EventProcessingStatus.SKIPPED,
                    auth_context=auth_context,
                )

    def _reconstruct_buffer_from_storage(
        self,
        *,
        tenant_id: str,
        stream_id: str,
        current_event: Event,
        auth_context: AuthContext | None = None,
    ) -> _StreamBuffer | None:
        """Reconstrói contexto conversacional de eventos `buffered` após reinicialização.

        Descarta e marca como `skipped` fragmentos expirados pelo TTL para evitar associações falsas.
        """
        assert self.storage is not None
        persisted = self.storage.list_buffered_stream_events(
            tenant_id,
            stream_id,
            auth_context=auth_context,
        )
        prior_events = [e for e in persisted if e.event_id != current_event.event_id]
        if not prior_events:
            return None

        ttl = timedelta(seconds=self.policy.buffer_ttl_seconds)
        valid_events: list[Event] = []
        expired_count = 0

        for ev in prior_events:
            if (current_event.timestamp - ev.timestamp) > ttl:
                self.storage.update_event_status(
                    tenant_id,
                    ev.event_id,
                    EventProcessingStatus.SKIPPED,
                    auth_context=auth_context,
                )
                expired_count += 1
            else:
                valid_events.append(ev)

        if expired_count > 0:
            log_cognitive_step(
                self._logger,
                stage="observer.reconstruct",
                action="expired_buffered_events_skipped",
                tenant_id=tenant_id,
                event_id=current_event.event_id,
                details={
                    "stream_id": stream_id,
                    "expired_count": expired_count,
                    "valid_reconstructed_count": len(valid_events),
                },
            )

        if not valid_events:
            return None

        reconstructed = _StreamBuffer(tenant_id=tenant_id, stream_id=stream_id)
        for ev in valid_events[-self.policy.max_buffer_events :]:
            reconstructed.events.append(ev)
            reconstructed.seen_event_ids.add(ev.event_id)
            if ev.sequence_no is not None:
                reconstructed.seen_sequence_nos.add(ev.sequence_no)
        reconstructed.last_updated_at = valid_events[-1].timestamp

        log_cognitive_step(
            self._logger,
            stage="observer.reconstruct",
            action="stream_buffer_reconstructed",
            tenant_id=tenant_id,
            event_id=current_event.event_id,
            details={
                "stream_id": stream_id,
                "reconstructed_event_ids": [e.event_id for e in reconstructed.events],
            },
        )
        return reconstructed

    def _has_early_trigger_signal(self, signals: list[str]) -> bool:
        signal_set = set(signals)
        return bool(signal_set & self.policy.early_trigger_signals)

    @staticmethod
    def _compute_accumulated_density(events: list[Event]) -> float:
        """Combina densidades semânticas dos eventos no buffer com saturação em 1.0."""
        if not events:
            return 0.0
        # Soma complementar probabilística: 1 - prod(1 - d_i * 0.75) ou soma limitada a 1.0
        total = sum(float(e.semantic_density) for e in events)
        return min(1.0, round(total, 4))

    @staticmethod
    def _aggregate_payloads(events: list[Event]) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        fragments: list[str] = []
        event_types: list[str] = []

        for ev in events:
            event_types.append(ev.event_type)
            for k, v in ev.payload.items():
                if k in {"fragment", "chunk", "text_chunk"} and isinstance(v, str):
                    fragments.append(v)
                elif isinstance(v, list) and isinstance(merged.get(k), list):
                    merged[k] = list(dict.fromkeys([*merged[k], *v]))
                else:
                    merged[k] = v

        if fragments:
            merged["assembled_text"] = " ".join(f.strip() for f in fragments if f.strip())
        if len(event_types) > 1:
            merged["observed_event_types"] = list(dict.fromkeys(event_types))
        return merged

    @staticmethod
    def _aggregate_evidence(events: list[Event]) -> list[EvidenceRef]:
        seen_ids: set[str] = set()
        collected: list[EvidenceRef] = []
        for ev in events:
            for item in ev.evidence:
                if item.evidence_id not in seen_ids:
                    seen_ids.add(item.evidence_id)
                    collected.append(item)
        if not collected and events:
            latest = events[-1]
            collected.append(
                EvidenceRef(
                    evidence_id=f"evd_obs_{latest.event_id}",
                    source_type="observed_event",
                    reference=f"event://{latest.tenant_id}/{latest.event_id}",
                    excerpt=f"{latest.event_type} from {latest.source}",
                    weight=max(0.5, float(latest.semantic_density)),
                    observed_at=latest.timestamp,
                )
            )
        return collected

    def _decide_activation(
        self,
        *,
        event: Event,
        accumulated_density: float,
        early_signal_matched: bool,
        buffer_reached_limit: bool,
        explicit_finalize: bool,
    ) -> tuple[bool, str, bool]:
        if not event.requires_evaluation:
            return False, "evaluation_disabled_on_event", explicit_finalize

        if early_signal_matched:
            return (
                True,
                "early_signal_detected",
                event.completeness == EventCompleteness.COMPLETE or explicit_finalize,
            )

        if event.completeness == EventCompleteness.MILESTONE and self.policy.activate_on_milestone:
            return True, "semantic_milestone_reached", explicit_finalize

        if event.completeness == EventCompleteness.COMPLETE:
            if accumulated_density >= self.policy.min_single_event_density:
                return True, "complete_semantic_event", True
            return False, "complete_event_below_min_density", True

        # Evento parcial: avalia se o contexto acumulado já atingiu o limiar
        if accumulated_density >= self.policy.accumulated_density_threshold:
            return True, "accumulated_context_threshold_reached", explicit_finalize

        if buffer_reached_limit and self.policy.activate_on_buffer_full:
            return True, "buffer_limit_reached", explicit_finalize

        if explicit_finalize:
            return (
                accumulated_density >= self.policy.min_single_event_density,
                "stream_finalized",
                True,
            )

        return False, "insufficient_signal_buffered", False
