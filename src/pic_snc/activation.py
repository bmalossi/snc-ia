"""Motor de Ativação Associativa do PIC-SNC v0.1.

Combina múltiplos sinais determinísticos para ativar memórias sem consulta do agente:
1. Pistas lexicais (`tags`)
2. Chaves semânticas (`semantic_keys` + correspondência contextual)
3. Gatilhos preditivos (`predictive.trigger_cues` + `prior_probability`)
4. Perfil temporal (`temporal`: validade `valid_from`/`valid_until`, recorrência e decaimento)
5. Relações explícitas entre memórias (`relations`: propagação relacional no mesmo tenant)
6. Saliência intrínseca da memória (`salience`)

Preserva ponto de extensão (`SemanticScorer`) para futura integração de embeddings/vetores.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from datetime import datetime
from typing import Any

from pic_snc.logging_utils import get_structured_logger, log_cognitive_step
from pic_snc.schemas import (
    ActivatedMemory,
    ActivationMode,
    AuthContext,
    Memory,
    ObservationDecision,
)
from pic_snc.storage import SQLiteStorage

_WORD_RE = re.compile(r"[a-zA-Z0-9_.\-À-ÿ]{3,}")

_STOP_WORDS: frozenset[str] = frozenset(
    {
        "com",
        "para",
        "por",
        "que",
        "uma",
        "uns",
        "umas",
        "dos",
        "das",
        "nos",
        "nas",
        "aos",
        "pelo",
        "pela",
        "sem",
        "sob",
        "sobre",
        "entre",
        "como",
        "mais",
        "não",
        "nao",
        "sim",
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "are",
        "was",
        "were",
    }
)

SemanticScorer = Callable[[ObservationDecision, Memory], float]


def _tokenize_text(text: str) -> set[str]:
    tokens: set[str] = set()
    for match in _WORD_RE.findall(text.lower()):
        if match not in _STOP_WORDS:
            tokens.add(match)
        if "." in match or "_" in match or "-" in match:
            for part in re.split(r"[._\-]+", match):
                if len(part) >= 3 and part not in _STOP_WORDS:
                    tokens.add(part)
    return tokens


class AssociativeActivationEngine:
    """Motor determinístico de ativação associativa multi-sinal."""

    def __init__(
        self,
        storage: SQLiteStorage,
        *,
        min_salience: float = 0.20,
        min_activation_score: float = 0.30,
        max_activated: int = 10,
        semantic_scorer: SemanticScorer | None = None,
    ) -> None:
        self.storage = storage
        self.min_salience = min_salience
        self.min_activation_score = min_activation_score
        self.max_activated = max_activated
        self.semantic_scorer = semantic_scorer
        self._logger = get_structured_logger("pic_snc.activation")

    def activate(
        self,
        observation: ObservationDecision,
        *,
        auth_context: AuthContext | None = None,
    ) -> list[ActivatedMemory]:
        """Ativa memórias do tenant associadas ao contexto da `ObservationDecision`."""
        if not observation.should_activate:
            return []

        memories = self.storage.list_memories(
            observation.tenant_id,
            min_salience=self.min_salience,
            auth_context=auth_context,
        )
        if not memories:
            return []

        memory_by_id: dict[str, Memory] = {m.memory_id: m for m in memories}
        signal_tokens = self._build_observation_token_set(observation)

        primary_results: dict[str, ActivatedMemory] = {}

        for memory in memories:
            if not self._is_temporally_valid(memory, observation.observed_at):
                continue

            matched_cues, raw_assoc_score, dominant_mode = self._score_memory_associations(
                observation=observation,
                memory=memory,
                signal_tokens=signal_tokens,
            )
            if raw_assoc_score <= 0.0:
                continue

            temporal_factor = self._compute_temporal_factor(memory, observation.observed_at)
            # Combina associação contextual (70%) com saliência intrínseca (30%) modulada pelo perfil temporal
            combined_score = min(
                1.0,
                round(
                    ((0.70 * raw_assoc_score) + (0.30 * float(memory.salience))) * temporal_factor,
                    4,
                ),
            )

            if combined_score >= self.min_activation_score:
                related_ids = [
                    rel.target_id
                    for rel in memory.associative_cues.relations
                    if rel.target_id in memory_by_id
                ]
                primary_results[memory.memory_id] = ActivatedMemory(
                    memory=memory,
                    activation_mode=dominant_mode,
                    activation_score=combined_score,
                    matched_cues=matched_cues,
                    related_memory_ids=related_ids,
                )

        # Propagação relacional (Spreading Activation) entre memórias do mesmo tenant
        relational_additions: dict[str, ActivatedMemory] = {}
        for source_id, activated in list(primary_results.items()):
            for relation in activated.memory.associative_cues.relations:
                target_mem = memory_by_id.get(relation.target_id)
                if target_mem is None:
                    continue
                if not self._is_temporally_valid(target_mem, observation.observed_at):
                    continue

                propagated_score = min(
                    1.0,
                    round(
                        activated.activation_score
                        * float(relation.weight)
                        * (0.65 + 0.35 * float(target_mem.salience)),
                        4,
                    ),
                )
                if propagated_score < self.min_activation_score:
                    continue

                if target_mem.memory_id in primary_results:
                    # Reforça memória já ativada e promove para HYBRID
                    existing = primary_results[target_mem.memory_id]
                    boosted_score = min(
                        1.0,
                        round(existing.activation_score + (0.15 * float(relation.weight)), 4),
                    )
                    updated_cues = dict(existing.matched_cues)
                    updated_cues.setdefault("relational_links", []).append(
                        {
                            "from_memory_id": source_id,
                            "relation_type": relation.relation_type,
                            "weight": float(relation.weight),
                        }
                    )
                    updated_related = list(
                        dict.fromkeys([*existing.related_memory_ids, source_id])
                    )
                    primary_results[target_mem.memory_id] = existing.model_copy(
                        update={
                            "activation_mode": ActivationMode.HYBRID,
                            "activation_score": boosted_score,
                            "matched_cues": updated_cues,
                            "related_memory_ids": updated_related,
                        }
                    )
                else:
                    prev = relational_additions.get(target_mem.memory_id)
                    if prev is None or propagated_score > prev.activation_score:
                        relational_additions[target_mem.memory_id] = ActivatedMemory(
                            memory=target_mem,
                            activation_mode=ActivationMode.RELATIONAL,
                            activation_score=propagated_score,
                            matched_cues={
                                "relational_links": [
                                    {
                                        "from_memory_id": source_id,
                                        "relation_type": relation.relation_type,
                                        "weight": float(relation.weight),
                                    }
                                ]
                            },
                            related_memory_ids=[source_id],
                        )

        all_activated = [
            *primary_results.values(),
            *relational_additions.values(),
        ]
        all_activated.sort(
            key=lambda item: (item.activation_score, float(item.memory.salience)),
            reverse=True,
        )
        selected = all_activated[: self.max_activated]

        log_cognitive_step(
            self._logger,
            stage="activation.activate",
            action="memories_activated",
            tenant_id=observation.tenant_id,
            event_id=observation.trigger_event_id,
            details={
                "total_memories_scanned": len(memories),
                "activated_count": len(selected),
                "activated_ids": [item.memory.memory_id for item in selected],
            },
        )
        return selected

    @staticmethod
    def _build_observation_token_set(observation: ObservationDecision) -> set[str]:
        tokens: set[str] = set()
        for sig in observation.detected_signals:
            tokens.add(sig.lower())
            tokens.update(_tokenize_text(sig))
        tokens.update(_tokenize_text(observation.trigger_event_type))
        return tokens

    @staticmethod
    def _is_temporally_valid(memory: Memory, observed_at: datetime) -> bool:
        temporal = memory.associative_cues.temporal
        if temporal.valid_from is not None and observed_at < temporal.valid_from:
            return False
        if temporal.valid_until is not None and observed_at > temporal.valid_until:
            return False
        return True

    @staticmethod
    def _compute_temporal_factor(memory: Memory, observed_at: datetime) -> float:
        temporal = memory.associative_cues.temporal
        age_seconds = max(0.0, (observed_at - memory.created_at).total_seconds())

        if (
            temporal.time_window_seconds is not None
            and age_seconds > temporal.time_window_seconds
        ):
            # Decaimento suave fora da janela nominal
            periods = age_seconds / float(temporal.time_window_seconds)
            decay = math.exp(-float(temporal.decay_rate) * periods)
            return max(0.50, min(1.0, decay))

        return 1.0

    def _score_memory_associations(
        self,
        *,
        observation: ObservationDecision,
        memory: Memory,
        signal_tokens: set[str],
    ) -> tuple[dict[str, Any], float, ActivationMode]:
        matched_cues: dict[str, Any] = {}
        active_modes: list[tuple[ActivationMode, float]] = []

        # 1. Tags lexicais
        mem_tags = {t.lower() for t in memory.tags}
        matched_tags = sorted(
            t for t in mem_tags if t in signal_tokens or any(t in s for s in signal_tokens)
        )
        if matched_tags:
            matched_cues["matched_tags"] = matched_tags
            tag_ratio = len(matched_tags) / max(1, min(len(mem_tags), 3))
            tag_score = min(1.0, 0.55 + 0.45 * min(1.0, tag_ratio))
            active_modes.append((ActivationMode.LEXICAL, tag_score))

        # 2. Chaves semânticas + correspondência contextual
        sem_keys = [k.lower().strip() for k in memory.associative_cues.semantic_keys if k.strip()]
        matched_sem_keys: list[str] = []
        for sk in sem_keys:
            sk_parts = _tokenize_text(sk)
            if sk in signal_tokens or (sk_parts and sk_parts.issubset(signal_tokens)):
                matched_sem_keys.append(sk)
            elif any(part in signal_tokens for part in sk_parts):
                matched_sem_keys.append(sk)

        content_tokens = _tokenize_text(memory.content)
        content_overlap = sorted(content_tokens & signal_tokens)

        custom_sem_score = (
            float(self.semantic_scorer(observation, memory))
            if self.semantic_scorer is not None
            else 0.0
        )

        if matched_sem_keys or len(content_overlap) >= 2 or custom_sem_score > 0.0:
            if matched_sem_keys:
                matched_cues["matched_semantic_keys"] = matched_sem_keys
            if content_overlap:
                matched_cues["contextual_tokens"] = content_overlap[:8]
            sem_base = 0.0
            if matched_sem_keys:
                sem_base = max(
                    sem_base,
                    min(1.0, 0.60 + 0.20 * len(matched_sem_keys)),
                )
            if len(content_overlap) >= 2:
                sem_base = max(
                    sem_base,
                    min(0.85, 0.35 + 0.12 * len(content_overlap)),
                )
            sem_score = max(sem_base, min(1.0, custom_sem_score))
            active_modes.append((ActivationMode.SEMANTIC, sem_score))

        # 3. Gatilhos preditivos
        pred = memory.associative_cues.predictive
        trigger_cues = [c.lower().strip() for c in pred.trigger_cues if c.strip()]
        matched_triggers: list[str] = []
        for cue in trigger_cues:
            cue_parts = _tokenize_text(cue)
            if cue in signal_tokens or (cue_parts and cue_parts.issubset(signal_tokens)):
                matched_triggers.append(cue)
            elif any(part in signal_tokens for part in cue_parts):
                matched_triggers.append(cue)

        if matched_triggers:
            matched_cues["matched_predictive_cues"] = matched_triggers
            if pred.anticipated_outcome:
                matched_cues["anticipated_outcome"] = pred.anticipated_outcome
            pred_score = min(
                1.0,
                0.55 + (0.30 * float(pred.prior_probability)) + (0.15 * min(2, len(matched_triggers))),
            )
            active_modes.append((ActivationMode.PREDICTIVE, pred_score))

        # 4. Pista temporal explícita (recurrence_hint presente no contexto)
        recurrence = memory.associative_cues.temporal.recurrence_hint
        if recurrence and recurrence.lower() in signal_tokens:
            matched_cues["matched_temporal_hint"] = recurrence.lower()
            active_modes.append((ActivationMode.TEMPORAL, 0.70))

        if not active_modes:
            return {}, 0.0, ActivationMode.LEXICAL

        # Combina múltiplos sinais com bônus sinérgico quando >1 modalidade casa
        active_modes.sort(key=lambda pair: pair[1], reverse=True)
        top_mode, top_score = active_modes[0]
        if len(active_modes) == 1:
            return matched_cues, round(top_score, 4), top_mode

        synergy_bonus = min(0.20, 0.08 * (len(active_modes) - 1))
        hybrid_score = min(1.0, round(top_score + synergy_bonus, 4))
        return matched_cues, hybrid_score, ActivationMode.HYBRID
