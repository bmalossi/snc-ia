"""Governador Cognitivo (Governor) do PIC-SNC v0.1.

Responsabilidades:
- Classificar e limitar autoridade nos níveis A0, A1, A2 e A3.
- Manter A4 desativado por padrão (`enable_a4=False`), reservando-o exclusivamente
  a bloqueios protetivos explicitamente autorizados.
- Garantir que A3 emita uma **solicitação de pausa/revisão** (`intervention_requested=True`,
  `protective_block=False`), sem jamais alegar que a ação principal foi interrompida
  sem confirmação do agente ou de seu adaptador.
- Prevenir impulsos duplicados e controlar repetição por janela de espera (`cooldown_seconds`).
"""

from __future__ import annotations

import hashlib
from datetime import timedelta

from pic_snc.config import ExternalActionPolicy, SNCConfig
from pic_snc.logging_utils import get_structured_logger, log_cognitive_step
from pic_snc.schemas import (
    AuthContext,
    AutonomyLevel,
    Candidate,
    CognitiveKind,
    Impulse,
    ImpulseStatus,
    ObservationDecision,
)
from pic_snc.storage import SQLiteStorage


def _build_impulse_id(tenant_id: str, event_id: str, candidate_id: str) -> str:
    digest = hashlib.sha256(
        f"{tenant_id}:{event_id}:{candidate_id}".encode("utf-8")
    ).hexdigest()[:12]
    return f"imp_{digest}"


def _build_dedupe_fingerprint(
    tenant_id: str,
    memory_id: str,
    kind: CognitiveKind,
    autonomy_level: AutonomyLevel,
) -> str:
    digest = hashlib.sha256(
        f"{tenant_id}:{memory_id}:{kind.value}:{autonomy_level.value}".encode("utf-8")
    ).hexdigest()[:16]
    return f"fp_{digest}"


class CognitiveGovernor:
    """Governador determinístico de autoridade, intervenção e controle de repetição."""

    def __init__(
        self,
        config: SNCConfig,
        storage: SQLiteStorage | None = None,
        *,
        cooldown_seconds: int = 120,
    ) -> None:
        self.config = config
        self.storage = storage
        self.cooldown_seconds = cooldown_seconds
        self._logger = get_structured_logger("pic_snc.governor")

    def govern(
        self,
        observation: ObservationDecision,
        candidates: list[Candidate],
        *,
        auth_context: AuthContext | None = None,
    ) -> list[Impulse]:
        """Governa candidatos avaliados, produzindo impulsos prontos para o barramento."""
        if not candidates:
            return []

        impulses: list[Impulse] = []
        batch_fingerprints: set[str] = set()

        for candidate in candidates:
            confidence = self._compute_confidence(candidate)
            has_evidence = len(candidate.evidence) > 0
            trace = dict(candidate.source_trace)

            # 1. Verificação de evidência e confiança mínima
            if not has_evidence or confidence < self.config.min_impulse_confidence:
                suppressed_imp = self._build_suppressed_impulse(
                    observation=observation,
                    candidate=candidate,
                    confidence=confidence,
                    reason=(
                        "insufficient_evidence"
                        if not has_evidence
                        else "below_min_impulse_confidence"
                    ),
                    trace=trace,
                )
                impulses.append(suppressed_imp)
                continue

            # 2. Resolução de nível de autoridade (A0-A3; A4 desativado por padrão)
            resolved_level, downgraded_from_a4 = self._resolve_autonomy_level(candidate)
            if downgraded_from_a4:
                trace["downgraded_from"] = AutonomyLevel.A4.value
                trace["a4_blocked_by_policy"] = True
                trace["governance_note"] = (
                    "Nível A4 desativado por padrão; rebaixado para solicitação de pausa A3."
                )

            if resolved_level == AutonomyLevel.A0:
                suppressed_a0 = self._build_suppressed_impulse(
                    observation=observation,
                    candidate=candidate,
                    confidence=confidence,
                    reason="a0_silent_observation",
                    trace=trace,
                    autonomy_level=AutonomyLevel.A0,
                )
                impulses.append(suppressed_a0)
                continue

            # 3. Prevenção de duplicatas no lote e controle de repetição (cooldown)
            fingerprint = _build_dedupe_fingerprint(
                candidate.tenant_id,
                candidate.memory_id,
                candidate.kind,
                resolved_level,
            )
            if self._is_in_cooldown(
                tenant_id=candidate.tenant_id,
                fingerprint=fingerprint,
                observed_at=observation.observed_at,
                batch_fingerprints=batch_fingerprints,
                auth_context=auth_context,
            ):
                trace["cooldown_seconds"] = self.cooldown_seconds
                suppressed_dup = self._build_suppressed_impulse(
                    observation=observation,
                    candidate=candidate,
                    confidence=confidence,
                    reason="cooldown_active_or_duplicate_impulse",
                    trace=trace,
                    autonomy_level=resolved_level,
                    fingerprint=fingerprint,
                )
                impulses.append(suppressed_dup)
                continue

            batch_fingerprints.add(fingerprint)

            # 4. Política de intervenção (A3 = solicitação de pausa; A4 = bloqueio protetivo autorizado)
            intervention_requested = resolved_level in {
                AutonomyLevel.A3,
                AutonomyLevel.A4,
            }
            protective_block = (
                resolved_level == AutonomyLevel.A4 and self.config.enable_a4
            )
            # Nenhuma ação externa autônoma é executada
            external_action_allowed = (
                self.config.external_action_policy
                == ExternalActionPolicy.REQUIRE_EXPLICIT_GRANT
                and bool(observation.aggregated_payload.get("explicit_external_action_grant"))
                and False  # Sempre False na Etapa 2 por restrição arquitetural
            )

            if resolved_level == AutonomyLevel.A3:
                trace["intervention_mode"] = "pause_request_pending_agent_confirmation"
                trace["execution_halted"] = False

            message = self._compose_impulse_message(
                candidate=candidate,
                autonomy_level=resolved_level,
                downgraded_from_a4=downgraded_from_a4,
            )

            memory_ids = list(
                dict.fromkeys([candidate.memory_id, *candidate.related_memory_ids])
            )

            impulse = Impulse(
                impulse_id=_build_impulse_id(
                    candidate.tenant_id,
                    candidate.event_id,
                    candidate.candidate_id,
                ),
                tenant_id=candidate.tenant_id,
                event_id=candidate.event_id,
                candidate_id=candidate.candidate_id,
                memory_ids=memory_ids,
                kind=candidate.kind,
                autonomy_level=resolved_level,
                intervention_requested=intervention_requested,
                protective_block=protective_block,
                external_action_allowed=external_action_allowed,
                message=message,
                rationale=candidate.rationale,
                confidence=confidence,
                source_trace=trace,
                evidence=list(candidate.evidence),
                status=ImpulseStatus.PUBLISHED,
                dedupe_fingerprint=fingerprint,
                created_at=observation.observed_at,
            )
            impulses.append(impulse)

            log_cognitive_step(
                self._logger,
                stage="governor.govern",
                action="impulse_governed",
                tenant_id=candidate.tenant_id,
                event_id=candidate.event_id,
                candidate_id=candidate.candidate_id,
                impulse_id=impulse.impulse_id,
                details={
                    "autonomy_level": resolved_level.value,
                    "intervention_requested": intervention_requested,
                    "protective_block": protective_block,
                    "downgraded_from_a4": downgraded_from_a4,
                    "confidence": confidence,
                },
            )

        return impulses

    @staticmethod
    def _compute_confidence(candidate: Candidate) -> float:
        base_conf = (
            float(candidate.choice_confidence)
            if candidate.choice_confidence is not None
            else float(candidate.score)
        )
        noul = (
            float(candidate.noul_probability)
            if candidate.noul_probability is not None
            else float(candidate.score)
        )
        combined = (0.50 * float(candidate.score)) + (0.30 * base_conf) + (0.20 * noul)
        return min(1.0, max(0.0, round(combined, 4)))

    def _resolve_autonomy_level(
        self,
        candidate: Candidate,
    ) -> tuple[AutonomyLevel, bool]:
        proposed = candidate.proposed_autonomy
        downgraded_from_a4 = False

        if proposed == AutonomyLevel.A4:
            if self.config.enable_a4 and candidate.kind == CognitiveKind.RISK:
                return AutonomyLevel.A4, False
            proposed = AutonomyLevel.A3
            downgraded_from_a4 = True

        # Teto de autoridade por tipo de impulso
        if candidate.kind == CognitiveKind.RECALL:
            return (
                AutonomyLevel.A1
                if candidate.score >= self.config.min_candidate_score
                else AutonomyLevel.A0,
                downgraded_from_a4,
            )
        if candidate.kind == CognitiveKind.OPPORTUNITY:
            if candidate.score >= 0.50:
                return AutonomyLevel.A2, downgraded_from_a4
            return AutonomyLevel.A1, downgraded_from_a4

        # CognitiveKind.RISK
        if proposed == AutonomyLevel.A3 or candidate.score >= 0.55:
            return AutonomyLevel.A3, downgraded_from_a4
        return AutonomyLevel.A2, downgraded_from_a4

    def _is_in_cooldown(
        self,
        *,
        tenant_id: str,
        fingerprint: str,
        observed_at: object,
        batch_fingerprints: set[str],
        auth_context: AuthContext | None,
    ) -> bool:
        if fingerprint in batch_fingerprints:
            return True
        if self.storage is None or self.cooldown_seconds <= 0:
            return False
        from datetime import datetime

        assert isinstance(observed_at, datetime)
        since_utc = observed_at - timedelta(seconds=self.cooldown_seconds)
        recent = self.storage.find_recent_impulse_by_fingerprint(
            tenant_id,
            fingerprint,
            since_utc=since_utc,
            auth_context=auth_context,
        )
        return recent is not None

    def _build_suppressed_impulse(
        self,
        *,
        observation: ObservationDecision,
        candidate: Candidate,
        confidence: float,
        reason: str,
        trace: dict[str, object],
        autonomy_level: AutonomyLevel = AutonomyLevel.A0,
        fingerprint: str | None = None,
    ) -> Impulse:
        trace_copy = dict(trace)
        trace_copy["suppressed_reason"] = reason
        return Impulse(
            impulse_id=_build_impulse_id(
                candidate.tenant_id,
                candidate.event_id,
                candidate.candidate_id,
            ),
            tenant_id=candidate.tenant_id,
            event_id=candidate.event_id,
            candidate_id=candidate.candidate_id,
            memory_ids=[candidate.memory_id, *candidate.related_memory_ids],
            kind=candidate.kind,
            autonomy_level=autonomy_level,
            intervention_requested=False,
            protective_block=False,
            external_action_allowed=False,
            message=f"[SUPRIMIDO:{reason}] {candidate.rationale}",
            rationale=candidate.rationale,
            confidence=confidence,
            source_trace=trace_copy,
            evidence=list(candidate.evidence),
            status=ImpulseStatus.SUPPRESSED,
            suppressed_reason=reason,
            dedupe_fingerprint=fingerprint,
            created_at=observation.observed_at,
        )

    @staticmethod
    def _compose_impulse_message(
        *,
        candidate: Candidate,
        autonomy_level: AutonomyLevel,
        downgraded_from_a4: bool,
    ) -> str:
        memory_excerpt = candidate.evidence[-1].excerpt if candidate.evidence else ""
        detail = memory_excerpt or candidate.rationale

        if autonomy_level == AutonomyLevel.A1:
            return f"[RECORDAÇÃO A1] Contexto associado ativado: {detail}"

        if autonomy_level == AutonomyLevel.A2:
            return f"[OPORTUNIDADE A2] Antecipação proativa detectada: {detail}"

        if autonomy_level == AutonomyLevel.A3:
            downgrade_tag = " (A4 desativado -> rebaixado para A3)" if downgraded_from_a4 else ""
            return (
                f"[SOLICITAÇÃO DE PAUSA A3{downgrade_tag}] Risco identificado: {detail} — "
                "Solicita-se pausa e confirmação do agente principal (execução NÃO foi interrompida autonomamente)."
            )

        if autonomy_level == AutonomyLevel.A4:
            return f"[BLOQUEIO PROTETIVO AUTORIZADO A4] Intervenção protetiva acionada: {detail}"

        return f"[OBSERVAÇÃO A0] {detail}"
