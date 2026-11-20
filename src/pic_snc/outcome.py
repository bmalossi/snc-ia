"""Rastreador de Desfechos (OutcomeTracker) do PIC-SNC v0.1.

Responsabilidades:
- Registrar o desfecho de um impulso vinculado ao `impulse_id` e `tenant_id`.
- Capturar a decisão do agente principal (`result`), o resultado observado no mundo real
  (`observed_outcome`), as evidências disponíveis (`evidence`), data UTC e contexto.
- Garantir a invariante epistêmica: a mera aceitação (`accepted`) de um impulso pelo
  agente principal NÃO constitui prova de sua correção (`verified_correct`).
  A confirmação de correção exige resultado observado e/ou evidência verificável independente.
"""

from __future__ import annotations

from pic_snc.logging_utils import get_structured_logger, log_cognitive_step
from pic_snc.schemas import AuthContext, Outcome
from pic_snc.storage import SQLiteStorage, StorageError


class OutcomeVerificationError(ValueError):
    """Violação da regra epistêmica de verificação de desfechos."""


class SQLiteOutcomeTracker:
    """Rastreador de desfechos com separação entre aceitação do agente e correção verificada."""

    def __init__(self, storage: SQLiteStorage) -> None:
        self.storage = storage
        self._logger = get_structured_logger("pic_snc.outcome")

    def record(
        self,
        outcome: Outcome,
        *,
        auth_context: AuthContext | None = None,
    ) -> Outcome:
        """Registra um desfecho validando vínculo com o impulso original e regra epistêmica."""
        impulse = self.storage.get_impulse(
            outcome.tenant_id,
            outcome.impulse_id,
            auth_context=auth_context,
        )
        if impulse is None:
            raise StorageError(
                f"Impulso {outcome.impulse_id!r} não encontrado no tenant {outcome.tenant_id!r}."
            )

        # Invariante epistêmica: verified_correct não pode ser True sem evidência ou resultado observado
        has_verifiable_backing = bool(
            outcome.evidence or (outcome.observed_outcome and outcome.observed_outcome.strip())
        )
        if outcome.verified_correct is not None and not has_verifiable_backing:
            raise OutcomeVerificationError(
                "Não é permitido declarar verified_correct sem observed_outcome ou evidence "
                "verificável. A decisão do agente (ex.: 'accepted') não é prova de correção."
            )

        enriched_context = {
            "original_event_id": impulse.event_id,
            "original_candidate_id": impulse.candidate_id,
            "original_autonomy_level": impulse.autonomy_level.value,
            "original_kind": impulse.kind.value,
            "epistemic_status": (
                "verified"
                if outcome.verified_correct is not None
                else "unverified_agent_feedback_only"
            ),
            **outcome.context,
        }
        normalized_outcome = outcome.model_copy(update={"context": enriched_context})
        saved = self.storage.save_outcome(normalized_outcome, auth_context=auth_context)

        log_cognitive_step(
            self._logger,
            stage="outcome.record",
            action="outcome_recorded",
            tenant_id=saved.tenant_id,
            event_id=impulse.event_id,
            impulse_id=saved.impulse_id,
            details={
                "outcome_id": saved.outcome_id,
                "agent_decision": saved.result.value,
                "verified_correct": saved.verified_correct,
                "epistemic_status": enriched_context["epistemic_status"],
                "evidence_count": len(saved.evidence),
            },
        )
        return saved
