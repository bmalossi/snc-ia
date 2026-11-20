"""Avaliadores Cognitivos do PIC-SNC v0.1 (`MockEvaluator` e `JevEvaluatorAdapter`).

Diretrizes arquiteturais:
- `MockEvaluator`: determinístico, sem dependências externas, resultados reproduzíveis.
- `JevEvaluatorAdapter`: preparado para o SDK oficial da TypeSafe (`typesafe_sdk`),
  utilizando `ts.systemone(model="jev", state=..., questions={...})` com as primitivas
  oficiais `Noul(statement=...)`, `Score(rubric=...)` e `Choice(options=[...])`.
- Diferenciação explícita conforme documentação oficial da TypeSafe (`docs.typesafe.ai`):
  * `Noul.noul`: probabilidade [0, 1] de uma proposição binária ser verdadeira (sem confidence separado).
  * `Score.score`: valor numérico esperado na escala da rubrica (ex.: 0..2), acompanhado de `probabilities` e `confidence` [0, 1].
  * `Choice.choice`: opção escolhida dentre alternativas mutuamente exclusivas, acompanhada de `probabilities` e `confidence` [0, 1].
- Proibido fallback silencioso: falhas de configuração ou execução no `JevEvaluatorAdapter`
  levantam `ConfigurationError` ou `EvaluatorExecutionError`, nunca recorrendo ao `MockEvaluator`.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Protocol

from pic_snc.config import ConfigurationError, EvaluatorMode, SNCConfig
from pic_snc.logging_utils import get_structured_logger, log_cognitive_step
from pic_snc.schemas import (
    ActivatedMemory,
    AutonomyLevel,
    Candidate,
    CognitiveKind,
    EvidenceRef,
    ObservationDecision,
)


class EvaluatorExecutionError(RuntimeError):
    """Falha na execução do avaliador cognitivo (sem fallback silencioso)."""


class TypeSafeSystemOneClient(Protocol):
    """Contrato do cliente oficial `typesafe_sdk.client()`."""

    def systemone(
        self,
        *,
        model: str,
        state: str | dict[str, Any],
        questions: dict[str, Any],
    ) -> Any:
        ...


def _deterministic_candidate_id(tenant_id: str, event_id: str, memory_id: str) -> str:
    digest = hashlib.sha256(
        f"{tenant_id}:{event_id}:{memory_id}".encode("utf-8")
    ).hexdigest()[:12]
    return f"cnd_{digest}"


def _merge_evidence(
    observation: ObservationDecision,
    activated: ActivatedMemory,
) -> list[EvidenceRef]:
    seen: set[str] = set()
    merged: list[EvidenceRef] = []

    for item in [*observation.evidence, *activated.memory.evidence]:
        if item.evidence_id not in seen:
            seen.add(item.evidence_id)
            merged.append(item)

    mem_ref_id = f"evd_mem_{activated.memory.memory_id}"
    if mem_ref_id not in seen:
        merged.append(
            EvidenceRef(
                evidence_id=mem_ref_id,
                source_type="activated_memory",
                reference=f"memory://{activated.memory.tenant_id}/{activated.memory.memory_id}",
                excerpt=activated.memory.content[:140],
                weight=float(activated.activation_score),
                observed_at=observation.observed_at,
            )
        )
    return merged


def _propose_autonomy_level(
    kind: CognitiveKind,
    score: float,
    observation: ObservationDecision,
    activated: ActivatedMemory,
) -> AutonomyLevel:
    if score < 0.45:
        return AutonomyLevel.A0

    if kind == CognitiveKind.RECALL:
        return AutonomyLevel.A1

    if kind == CognitiveKind.OPPORTUNITY:
        return AutonomyLevel.A2 if score >= 0.55 else AutonomyLevel.A1

    # CognitiveKind.RISK
    req_protective_block = bool(
        observation.aggregated_payload.get("request_protective_block")
        or activated.memory.metadata.get("requires_protective_block")
    )
    if req_protective_block and score >= 0.85:
        return AutonomyLevel.A4
    return AutonomyLevel.A3 if score >= 0.60 else AutonomyLevel.A2


class MockEvaluator:
    """Avaliador cognitivo determinístico para execução local sem chave externa."""

    def __init__(
        self,
        *,
        min_candidate_score: float = 0.35,
        min_noul_probability: float = 0.45,
    ) -> None:
        self.min_candidate_score = min_candidate_score
        self.min_noul_probability = min_noul_probability
        self._logger = get_structured_logger("pic_snc.evaluator.mock")

    def evaluate(
        self,
        observation: ObservationDecision,
        activated_memories: list[ActivatedMemory],
    ) -> list[Candidate]:
        """Avalia deterministicamente as memórias ativadas contra o contexto observado."""
        if not observation.should_activate or not activated_memories:
            return []

        candidates: list[Candidate] = []

        for activated in activated_memories:
            mem = activated.memory
            cues = activated.matched_cues
            cue_categories_count = sum(
                1
                for key in (
                    "matched_tags",
                    "matched_semantic_keys",
                    "matched_predictive_cues",
                    "matched_temporal_hint",
                    "relational_links",
                    "contextual_tokens",
                )
                if cues.get(key)
            )

            # Sem pistas concretas suficientes -> evidência insuficiente
            if cue_categories_count == 0:
                continue

            # 1. Noul probability [0, 1]: probabilidade de que a memória é acionável para o estado
            signal_boost = min(0.25, 0.08 * cue_categories_count)
            density_weight = 0.25 * float(observation.accumulated_density)
            noul_prob = min(
                1.0,
                round(
                    (0.55 * float(activated.activation_score)) + signal_boost + density_weight,
                    4,
                ),
            )
            if noul_prob < self.min_noul_probability:
                continue

            # 2. Rubric score [0.0, 2.0] (escala System One 0..2) e score normalizado [0.0, 1.0]
            kind_weight = {
                CognitiveKind.RECALL: 0.92,
                CognitiveKind.OPPORTUNITY: 0.96,
                CognitiveKind.RISK: 1.00,
            }[mem.kind]
            normalized_score = min(
                1.0,
                round(
                    (
                        0.50 * float(activated.activation_score)
                        + 0.30 * float(mem.salience)
                        + 0.20 * noul_prob
                    )
                    * kind_weight,
                    4,
                ),
            )
            if normalized_score < self.min_candidate_score:
                continue

            rubric_score = round(normalized_score * 2.0, 4)

            # 3. Choice confidence [0.0, 1.0]: certeza derivada da concentração de evidências
            evidence_list = _merge_evidence(observation, activated)
            avg_evidence_weight = sum(float(e.weight) for e in evidence_list) / max(
                1, len(evidence_list)
            )
            choice_confidence = min(
                1.0,
                round(
                    0.50 * normalized_score
                    + 0.30 * avg_evidence_weight
                    + 0.20 * min(1.0, cue_categories_count / 2.0),
                    4,
                ),
            )

            proposed_autonomy = _propose_autonomy_level(
                mem.kind, normalized_score, observation, activated
            )

            rationale = (
                f"[MockEvaluator] Memória {mem.memory_id} ({mem.kind.value}) ativada via "
                f"{activated.activation_mode.value} (noul={noul_prob:.2f}, "
                f"rubric={rubric_score:.2f}/2.0, confidence={choice_confidence:.2f})."
            )

            candidate = Candidate(
                candidate_id=_deterministic_candidate_id(
                    observation.tenant_id,
                    observation.trigger_event_id,
                    mem.memory_id,
                ),
                tenant_id=observation.tenant_id,
                event_id=observation.trigger_event_id,
                memory_id=mem.memory_id,
                related_memory_ids=list(activated.related_memory_ids),
                kind=mem.kind,
                activation_mode=activated.activation_mode,
                activation_score=activated.activation_score,
                score=normalized_score,
                noul_probability=noul_prob,
                rubric_score=rubric_score,
                choice_confidence=choice_confidence,
                proposed_autonomy=proposed_autonomy,
                rationale=rationale,
                source_trace={
                    "evaluator": "mock",
                    "event_id": observation.trigger_event_id,
                    "event_source": observation.trigger_source,
                    "memory_id": mem.memory_id,
                    "memory_source": mem.source,
                    "matched_cues": cues,
                    "accumulated_event_ids": observation.accumulated_event_ids,
                },
                evidence=evidence_list,
                created_at=observation.observed_at,
            )
            candidates.append(candidate)

        candidates.sort(key=lambda c: (c.score, c.choice_confidence or 0.0), reverse=True)

        log_cognitive_step(
            self._logger,
            stage="evaluator.mock",
            action="candidates_evaluated",
            tenant_id=observation.tenant_id,
            event_id=observation.trigger_event_id,
            details={
                "activated_input_count": len(activated_memories),
                "candidates_produced": len(candidates),
            },
        )
        return candidates


class JevEvaluatorAdapter:
    """Adaptador oficial para o modelo System One (`jev`) via `typesafe_sdk`.

    Utiliza estritamente os contratos documentados em `https://docs.typesafe.ai/introduction/quickstart`:
    - `from typesafe_sdk import Choice, Noul, Score, client`
    - `ts = client()`
    - `response = ts.systemone(model="jev", state=..., questions={...})`
    - `response.answers.<q>.noul` (probabilidade [0, 1] da proposição Noul)
    - `response.answers.<q>.score` e `.confidence` (Score em rubrica 0..2)
    - `response.answers.<q>.choice`, `.probabilities` e `.confidence` (Choice entre opções)
    """

    def __init__(
        self,
        config: SNCConfig,
        *,
        sdk_client: TypeSafeSystemOneClient | None = None,
    ) -> None:
        if config.evaluator_mode != EvaluatorMode.JEV:
            raise ConfigurationError(
                f"JevEvaluatorAdapter requer evaluator_mode='jev' (atual: {config.evaluator_mode!r})."
            )
        if not config.has_typesafe_key:
            raise ConfigurationError(
                "JevEvaluatorAdapter requer TYPESAFE_API_KEY configurada. "
                "Fallback silencioso para MockEvaluator é proibido."
            )
        self.config = config
        self._sdk_client = sdk_client
        self._logger = get_structured_logger("pic_snc.evaluator.jev")

    def _resolve_sdk_primitives_and_client(
        self,
    ) -> tuple[TypeSafeSystemOneClient, Any, Any, Any]:
        """Carrega `client`, `Choice`, `Noul`, `Score` do pacote oficial `typesafe_sdk`."""
        try:
            from typesafe_sdk import Choice, Noul, Score, client  # type: ignore[import-not-found]
        except ImportError as exc:
            if self._sdk_client is not None:
                # Permite injeção de cliente compatível em testes unitários sem fallback para MockEvaluator
                return self._sdk_client, _ChoiceSpec, _NoulSpec, _ScoreSpec
            raise EvaluatorExecutionError(
                "Pacote oficial 'typesafe-sdk' não está instalado no ambiente e nenhum "
                "sdk_client foi injetado. Fallback silencioso para MockEvaluator é proibido."
            ) from exc

        if self._sdk_client is not None:
            return self._sdk_client, Choice, Noul, Score

        # Garante que TYPESAFE_API_KEY esteja visível para `typesafe_sdk.client()`
        if self.config.typesafe_api_key and not os.getenv("TYPESAFE_API_KEY"):
            os.environ["TYPESAFE_API_KEY"] = self.config.typesafe_api_key
        return client(), Choice, Noul, Score

    def evaluate(
        self,
        observation: ObservationDecision,
        activated_memories: list[ActivatedMemory],
    ) -> list[Candidate]:
        """Avalia cada memória ativada consultando o modelo System One `jev`."""
        if not observation.should_activate or not activated_memories:
            return []

        ts_client, choice_cls, noul_cls, score_cls = (
            self._resolve_sdk_primitives_and_client()
        )
        candidates: list[Candidate] = []

        for activated in activated_memories:
            mem = activated.memory
            state_payload = {
                "observed_event": {
                    "event_id": observation.trigger_event_id,
                    "event_type": observation.trigger_event_type,
                    "source": observation.trigger_source,
                    "payload": observation.aggregated_payload,
                    "detected_signals": observation.detected_signals,
                },
                "activated_memory": {
                    "memory_id": mem.memory_id,
                    "kind": mem.kind.value,
                    "content": mem.content,
                    "source": mem.source,
                    "salience": float(mem.salience),
                    "matched_cues": activated.matched_cues,
                },
            }

            questions = {
                "is_actionable": noul_cls(
                    statement=(
                        "The activated memory provides relevant, evidence-backed context "
                        "that warrants emitting an unsolicited cognitive impulse for the observed event."
                    )
                ),
                "cognitive_kind": choice_cls(
                    options=["recall", "opportunity", "risk", "none"]
                ),
                "relevance_rubric": score_cls(
                    rubric=(
                        "How relevant and impactful is this memory for the observed state? "
                        "0 = irrelevant or insufficient evidence, "
                        "1 = relevant contextual recall or moderate opportunity, "
                        "2 = critical risk or high-value opportunity"
                    )
                ),
            }

            try:
                response = ts_client.systemone(
                    model="jev",
                    state=json.dumps(state_payload, ensure_ascii=False),
                    questions=questions,
                )
            except Exception as exc:
                log_cognitive_step(
                    self._logger,
                    stage="evaluator.jev",
                    action="jev_evaluation_failed",
                    tenant_id=observation.tenant_id,
                    event_id=observation.trigger_event_id,
                    memory_id=mem.memory_id,
                    details={"error_type": type(exc).__name__},
                )
                raise EvaluatorExecutionError(
                    f"Falha na chamada ts.systemone(model='jev'): {exc}"
                ) from exc

            answers = getattr(response, "answers", None)
            if answers is None:
                raise EvaluatorExecutionError(
                    "Resposta inválida do Jev System One: atributo 'answers' ausente."
                )

            noul_ans = getattr(answers, "is_actionable", None)
            choice_ans = getattr(answers, "cognitive_kind", None)
            score_ans = getattr(answers, "relevance_rubric", None)
            if noul_ans is None or choice_ans is None or score_ans is None:
                raise EvaluatorExecutionError(
                    "Resposta incompleta do Jev System One: primitivas esperadas ausentes em 'answers'."
                )

            noul_prob = float(getattr(noul_ans, "noul"))
            selected_choice = str(getattr(choice_ans, "choice")).strip().lower()
            choice_conf = float(getattr(choice_ans, "confidence"))
            choice_probs = getattr(choice_ans, "probabilities", {}) or {}

            raw_rubric_score = float(getattr(score_ans, "score"))
            score_conf = float(getattr(score_ans, "confidence", choice_conf))
            normalized_score = min(1.0, max(0.0, round(raw_rubric_score / 2.0, 4)))
            combined_confidence = min(
                1.0, max(0.0, round((0.60 * choice_conf) + (0.40 * score_conf), 4))
            )

            if (
                selected_choice == "none"
                or noul_prob < 0.45
                or normalized_score < self.config.min_candidate_score
            ):
                continue

            resolved_kind = CognitiveKind(selected_choice)
            proposed_autonomy = _propose_autonomy_level(
                resolved_kind, normalized_score, observation, activated
            )
            evidence_list = _merge_evidence(observation, activated)

            candidates.append(
                Candidate(
                    candidate_id=_deterministic_candidate_id(
                        observation.tenant_id,
                        observation.trigger_event_id,
                        mem.memory_id,
                    ),
                    tenant_id=observation.tenant_id,
                    event_id=observation.trigger_event_id,
                    memory_id=mem.memory_id,
                    related_memory_ids=list(activated.related_memory_ids),
                    kind=resolved_kind,
                    activation_mode=activated.activation_mode,
                    activation_score=activated.activation_score,
                    score=normalized_score,
                    noul_probability=round(noul_prob, 4),
                    rubric_score=round(raw_rubric_score, 4),
                    choice_confidence=combined_confidence,
                    proposed_autonomy=proposed_autonomy,
                    rationale=(
                        f"[JevSystemOne] choice={selected_choice} "
                        f"(confidence={choice_conf:.2f}), noul={noul_prob:.2f}, "
                        f"rubric_score={raw_rubric_score:.2f}/2.0."
                    ),
                    source_trace={
                        "evaluator": "jev",
                        "model": "jev",
                        "event_id": observation.trigger_event_id,
                        "event_source": observation.trigger_source,
                        "memory_id": mem.memory_id,
                        "memory_source": mem.source,
                        "choice_probabilities": choice_probs,
                        "matched_cues": activated.matched_cues,
                    },
                    evidence=evidence_list,
                    created_at=observation.observed_at,
                )
            )

        candidates.sort(key=lambda c: (c.score, c.choice_confidence or 0.0), reverse=True)
        return candidates


class _ChoiceSpec:
    """Representação leve da primitiva `Choice(options=...)` quando `sdk_client` é injetado em testes."""

    def __init__(self, *, options: list[str]) -> None:
        self.options = options


class _NoulSpec:
    """Representação leve da primitiva `Noul(statement=...)` quando `sdk_client` é injetado em testes."""

    def __init__(self, *, statement: str) -> None:
        self.statement = statement


class _ScoreSpec:
    """Representação leve da primitiva `Score(rubric=...)` quando `sdk_client` é injetado em testes."""

    def __init__(self, *, rubric: str) -> None:
        self.rubric = rubric


def create_evaluator(
    config: SNCConfig,
    *,
    sdk_client: TypeSafeSystemOneClient | None = None,
) -> MockEvaluator | JevEvaluatorAdapter:
    """Fábrica que instancia o avaliador explicitamente configurado sem fallback silencioso."""
    if config.evaluator_mode == EvaluatorMode.MOCK:
        return MockEvaluator(min_candidate_score=config.min_candidate_score)
    if config.evaluator_mode == EvaluatorMode.JEV:
        return JevEvaluatorAdapter(config, sdk_client=sdk_client)
    raise ConfigurationError(f"Modo de avaliação não suportado: {config.evaluator_mode!r}")
