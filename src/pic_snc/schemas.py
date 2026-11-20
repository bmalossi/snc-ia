"""Contratos Pydantic do PIC-SNC v0.1.

Mantém distinção explícita entre:
1. Evento Observado (`Event` — `artifact_stage="observed_event"`)
2. Decisão de Observação (`ObservationDecision` — `artifact_stage="observation_decision"`)
3. Memória Latente / Associativa (`Memory` — `artifact_stage="latent_memory"`)
4. Hipótese / Memória Ativada (`Candidate` — `artifact_stage="activated_candidate"`)
5. Decisão Cognitiva Emitida (`Impulse` — `artifact_stage="emitted_decision"`)
6. Desfecho / Retroalimentação (`Outcome` — `artifact_stage="outcome_feedback"`)
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


def _normalize_utc(value: datetime | str) -> datetime:
    """Garante que todo timestamp esteja normalizado em UTC."""
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise TypeError(f"Esperado datetime ou string ISO-8601, recebido: {type(value)!r}")

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _validate_non_empty_trimmed(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("O campo não pode ser vazio ou conter apenas espaços.")
    return cleaned


UtcDatetime = Annotated[datetime, AfterValidator(_normalize_utc)]
NonEmptyStr = Annotated[str, AfterValidator(_validate_non_empty_trimmed)]
UnitIntervalFloat = Annotated[
    float,
    Field(ge=0.0, le=1.0, allow_inf_nan=False),
]


class CognitiveKind(StrEnum):
    """Categorias semânticas de memória, candidato e impulso."""

    RECALL = "recall"
    OPPORTUNITY = "opportunity"
    RISK = "risk"


class EventCompleteness(StrEnum):
    """Estágio de completude para observação progressiva de eventos."""

    PARTIAL = "partial"
    COMPLETE = "complete"
    MILESTONE = "milestone"


class EventProcessingStatus(StrEnum):
    """Estado de processamento independente do evento observado."""

    PENDING = "pending"
    BUFFERED = "buffered"
    PROCESSING = "processing"
    PROCESSED = "processed"
    SKIPPED = "skipped"
    FAILED = "failed"


class ActivationMode(StrEnum):
    """Modalidades de ativação associativa suportadas e extensíveis."""

    LEXICAL = "lexical"
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    PREDICTIVE = "predictive"
    RELATIONAL = "relational"
    HYBRID = "hybrid"


class AutonomyLevel(StrEnum):
    """Níveis de autoridade cognitiva do Governor.

    - A0: Observação silenciosa / registro interno
    - A1: Recordação contextual passiva
    - A2: Antecipação / oportunidade proativa
    - A3: Alerta de risco / solicitação de pausa e revisão (não bloqueia autonomamente)
    - A4: Bloqueio protetivo autorizado (desativado por padrão; separado de ações externas)
    """

    A0 = "A0"
    A1 = "A1"
    A2 = "A2"
    A3 = "A3"
    A4 = "A4"


class ImpulseStatus(StrEnum):
    """Ciclo de vida da decisão cognitiva emitida no barramento."""

    CREATED = "created"
    PENDING = "pending"
    PUBLISHED = "published"
    DELIVERED = "delivered"
    ACKNOWLEDGED = "acknowledged"
    APPLIED = "applied"
    SUPPRESSED = "suppressed"
    DOWNGRADED = "downgraded"
    FAILED = "failed"


class OutcomeResult(StrEnum):
    """Decisão/resposta do agente principal frente ao impulso emitido."""

    ACCEPTED = "accepted"
    IGNORED = "ignored"
    REJECTED = "rejected"
    RESOLVED = "resolved"


class OutboxTopic(StrEnum):
    """Tópicos transacionais do Outbox para processamento assíncrono."""

    EVENT_INGESTED = "event.ingested"
    IMPULSE_EMITTED = "impulse.emitted"
    PROTECTIVE_BLOCK_TRIGGERED = "impulse.protective_block"


class OutboxStatus(StrEnum):
    """Estado do registro na tabela outbox."""

    PENDING = "pending"
    PROCESSING = "processing"
    PUBLISHED = "published"
    FAILED = "failed"


class AuthContext(BaseModel):
    """Contexto de identidade autenticada para isolamento e autorização multi-tenant."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    principal_id: NonEmptyStr
    tenant_id: NonEmptyStr
    authenticated: bool = True
    auth_scheme: NonEmptyStr = "bearer_token"
    scopes: list[str] = Field(default_factory=list)
    issued_at: UtcDatetime = Field(default_factory=lambda: datetime.now(UTC))

    def assert_tenant_access(self, target_tenant_id: str) -> None:
        """Valida que a identidade está autenticada e autorizada para o tenant alvo."""
        if not self.authenticated:
            raise PermissionError("Contexto de segurança não autenticado.")
        if self.tenant_id != target_tenant_id.strip():
            raise PermissionError(
                f"Violação de isolamento multi-tenant: principal do tenant "
                f"{self.tenant_id!r} tentou acessar tenant {target_tenant_id!r}."
            )


class EvidenceRef(BaseModel):
    """Referência auditável de evidência associada a eventos, memórias ou decisões."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_id: NonEmptyStr = Field(default_factory=lambda: f"evd_{uuid4().hex[:12]}")
    source_type: NonEmptyStr
    reference: NonEmptyStr
    excerpt: str | None = None
    weight: UnitIntervalFloat = 1.0
    observed_at: UtcDatetime = Field(default_factory=lambda: datetime.now(UTC))


class MemoryRelation(BaseModel):
    """Vínculo relacional tipado entre uma memória e outra entidade/memória."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_id: NonEmptyStr
    relation_type: NonEmptyStr
    weight: UnitIntervalFloat = 0.5
    metadata: dict[str, Any] = Field(default_factory=dict)


class TemporalProfile(BaseModel):
    """Perfil de ativação temporal (janelas, recorrência, validade e decaimento)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    recurrence_hint: str | None = None
    valid_from: UtcDatetime | None = None
    valid_until: UtcDatetime | None = None
    decay_rate: UnitIntervalFloat = 0.05
    time_window_seconds: int | None = Field(default=None, ge=1)


class PredictiveProfile(BaseModel):
    """Perfil de ativação preditiva (sinais antecedentes e estado antecipado)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trigger_cues: list[str] = Field(default_factory=list)
    anticipated_outcome: str | None = None
    horizon_seconds: int | None = Field(default=None, ge=1)
    prior_probability: UnitIntervalFloat = 0.5


class AssociativeCues(BaseModel):
    """Estrutura extensível de pistas associativas (lexical, semântica, temporal, preditiva, relacional)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tags: list[str] = Field(default_factory=list)
    semantic_keys: list[str] = Field(default_factory=list)
    embedding_ref: str | None = None
    temporal: TemporalProfile = Field(default_factory=TemporalProfile)
    predictive: PredictiveProfile = Field(default_factory=PredictiveProfile)
    relations: list[MemoryRelation] = Field(default_factory=list)


class Event(BaseModel):
    """1. EVENTO OBSERVADO: sinal semântico parcial ou completo capturado do ambiente."""

    model_config = ConfigDict(extra="forbid")

    artifact_stage: Literal["observed_event"] = "observed_event"
    event_id: NonEmptyStr = Field(default_factory=lambda: f"evt_{uuid4().hex[:12]}")
    tenant_id: NonEmptyStr
    event_type: NonEmptyStr
    source: NonEmptyStr
    payload: dict[str, Any] = Field(default_factory=dict)
    completeness: EventCompleteness = EventCompleteness.COMPLETE
    stream_id: str | None = None
    sequence_no: int | None = Field(default=None, ge=0)
    semantic_density: UnitIntervalFloat = 1.0
    requires_evaluation: bool = True
    dedupe_key: str | None = None
    evidence: list[EvidenceRef] = Field(default_factory=list)
    processing_status: EventProcessingStatus = EventProcessingStatus.PENDING
    timestamp: UtcDatetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("dedupe_key", "stream_id")
    @classmethod
    def _clean_optional_str(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    def should_trigger_evaluation(self, min_density: float = 0.40) -> bool:
        """Determina se o evento isolado possui densidade/completude imediata para avaliação."""
        if not self.requires_evaluation:
            return False
        if (
            self.completeness == EventCompleteness.PARTIAL
            and self.semantic_density < min_density
        ):
            return False
        return True


class ObservationDecision(BaseModel):
    """Decisão produzida pelo Observer após acumular e inspecionar contexto progressivo.

    O Observer nunca emite impulsos diretamente: ele produz um `ObservationDecision`
    indicando se há sinal suficiente (`should_activate`) para acionar a ativação associativa.
    """

    model_config = ConfigDict(extra="forbid")

    artifact_stage: Literal["observation_decision"] = "observation_decision"
    decision_id: NonEmptyStr = Field(default_factory=lambda: f"obs_{uuid4().hex[:12]}")
    tenant_id: NonEmptyStr
    stream_id: NonEmptyStr
    trigger_event_id: NonEmptyStr
    trigger_event_type: NonEmptyStr
    trigger_source: NonEmptyStr
    accumulated_event_ids: list[str] = Field(default_factory=list)
    should_activate: bool
    trigger_reason: NonEmptyStr
    accumulated_density: UnitIntervalFloat
    detected_signals: list[str] = Field(default_factory=list)
    aggregated_payload: dict[str, Any] = Field(default_factory=dict)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    stream_finalized: bool = False
    observed_at: UtcDatetime = Field(default_factory=lambda: datetime.now(UTC))


class Memory(BaseModel):
    """2. MEMÓRIA LATENTE: traço cognitivo armazenado com vetores associativos extensíveis."""

    model_config = ConfigDict(extra="forbid")

    artifact_stage: Literal["latent_memory"] = "latent_memory"
    memory_id: NonEmptyStr = Field(default_factory=lambda: f"mem_{uuid4().hex[:12]}")
    tenant_id: NonEmptyStr
    kind: CognitiveKind
    content: NonEmptyStr
    source: NonEmptyStr
    salience: UnitIntervalFloat = 0.5
    tags: list[str] = Field(default_factory=list)
    associative_cues: AssociativeCues = Field(default_factory=AssociativeCues)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: UtcDatetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _sync_tags_and_cues(self) -> Memory:
        normalized_tags = list(
            dict.fromkeys(
                t.strip().lower()
                for t in [*self.tags, *self.associative_cues.tags]
                if t and t.strip()
            )
        )
        self.tags = normalized_tags
        if self.associative_cues.tags != normalized_tags:
            self.associative_cues = self.associative_cues.model_copy(
                update={"tags": normalized_tags}
            )
        return self


class ActivatedMemory(BaseModel):
    """Resultado intermediário do ActivationEngine associando uma Memory ao contexto observado."""

    model_config = ConfigDict(extra="forbid")

    memory: Memory
    activation_mode: ActivationMode
    activation_score: UnitIntervalFloat
    matched_cues: dict[str, Any] = Field(default_factory=dict)
    related_memory_ids: list[str] = Field(default_factory=list)


class Candidate(BaseModel):
    """3. MEMÓRIA ATIVADA / CANDIDATO: hipótese avaliada que vincula um Event a uma Memory.

    Diferencia explicitamente:
    - `noul_probability`: probabilidade [0, 1] da proposição binária (TypeSafe `Noul.noul`)
    - `rubric_score`: valor esperado na escala de rubrica (TypeSafe `Score.score`)
    - `score`: relevância normalizada [0, 1] para governança
    - `choice_confidence`: certeza [0, 1] derivada da distribuição de probabilidades (TypeSafe `confidence`)
    """

    model_config = ConfigDict(extra="forbid")

    artifact_stage: Literal["activated_candidate"] = "activated_candidate"
    candidate_id: NonEmptyStr = Field(default_factory=lambda: f"cnd_{uuid4().hex[:12]}")
    tenant_id: NonEmptyStr
    event_id: NonEmptyStr
    memory_id: NonEmptyStr
    related_memory_ids: list[str] = Field(default_factory=list)
    kind: CognitiveKind
    activation_mode: ActivationMode = ActivationMode.LEXICAL
    activation_score: UnitIntervalFloat = 0.5
    score: UnitIntervalFloat
    noul_probability: UnitIntervalFloat | None = None
    rubric_score: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    choice_confidence: UnitIntervalFloat | None = None
    proposed_autonomy: AutonomyLevel = AutonomyLevel.A1
    rationale: NonEmptyStr
    source_trace: dict[str, Any] = Field(default_factory=dict)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    created_at: UtcDatetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _ensure_source_trace(self) -> Candidate:
        if "event_id" not in self.source_trace:
            self.source_trace["event_id"] = self.event_id
        if "memory_id" not in self.source_trace:
            self.source_trace["memory_id"] = self.memory_id
        return self


class Impulse(BaseModel):
    """4. DECISÃO EMITIDA: impulso cognitivo espontâneo governado e publicado no barramento.

    Em nível A3 (`AutonomyLevel.A3`), `intervention_requested=True` e `protective_block=False`:
    o impulso representa uma solicitação de pausa/revisão, nunca alegando que a ação principal
    foi interrompida sem confirmação do agente ou de seu adaptador.
    """

    model_config = ConfigDict(extra="forbid")

    artifact_stage: Literal["emitted_decision"] = "emitted_decision"
    impulse_id: NonEmptyStr = Field(default_factory=lambda: f"imp_{uuid4().hex[:12]}")
    tenant_id: NonEmptyStr
    event_id: NonEmptyStr
    candidate_id: NonEmptyStr
    memory_ids: list[str] = Field(default_factory=list)
    kind: CognitiveKind
    autonomy_level: AutonomyLevel
    intervention_requested: bool = False
    intervention_applied: bool = False
    protective_block: bool = False
    external_action_allowed: bool = False
    message: NonEmptyStr
    rationale: str | None = None
    confidence: UnitIntervalFloat
    source_trace: dict[str, Any] = Field(default_factory=dict)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    status: ImpulseStatus = ImpulseStatus.PENDING
    suppressed_reason: str | None = None
    dedupe_fingerprint: str | None = None
    created_at: UtcDatetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _validate_authority_invariants(self) -> Impulse:
        if self.protective_block and self.autonomy_level != AutonomyLevel.A4:
            raise ValueError(
                "protective_block=True só é permitido quando autonomy_level == 'A4'."
            )
        if "event_id" not in self.source_trace:
            self.source_trace["event_id"] = self.event_id
        if "candidate_id" not in self.source_trace:
            self.source_trace["candidate_id"] = self.candidate_id
        return self


class Outcome(BaseModel):
    """5. DESFECHO: registro do efeito observado e decisão do agente principal.

    Diretriz epistêmica: a aceitação (`result == 'accepted'`) de um impulso pelo agente
    principal NUNCA é tratada automaticamente como prova de sua correção (`verified_correct`).
    """

    model_config = ConfigDict(extra="forbid")

    artifact_stage: Literal["outcome_feedback"] = "outcome_feedback"
    outcome_id: NonEmptyStr = Field(default_factory=lambda: f"out_{uuid4().hex[:12]}")
    impulse_id: NonEmptyStr
    tenant_id: NonEmptyStr
    result: OutcomeResult
    observed_outcome: str | None = None
    verified_correct: bool | None = None
    notes: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    recorded_at: UtcDatetime = Field(default_factory=lambda: datetime.now(UTC))


class OutboxEntry(BaseModel):
    """Registro transacional do Outbox para processamento e despacho desacoplados."""

    model_config = ConfigDict(extra="forbid")

    outbox_id: NonEmptyStr = Field(default_factory=lambda: f"obx_{uuid4().hex[:12]}")
    tenant_id: NonEmptyStr
    topic: OutboxTopic
    aggregate_type: NonEmptyStr
    aggregate_id: NonEmptyStr
    payload: dict[str, Any]
    status: OutboxStatus = OutboxStatus.PENDING
    attempts: int = Field(default=0, ge=0)
    available_at: UtcDatetime = Field(default_factory=lambda: datetime.now(UTC))
    created_at: UtcDatetime = Field(default_factory=lambda: datetime.now(UTC))
    processed_at: UtcDatetime | None = None
    last_error: str | None = None


class HealthResponse(BaseModel):
    """Contrato de resposta do endpoint GET /health."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["ok"] = "ok"
    version: str = "0.1.0"
    evaluator_mode: str
    a4_enabled: bool
    external_action_policy: str
    environment: str = "development"
    dev_identity_enabled: bool = True
    database_ready: bool = True


class ReadinessResponse(BaseModel):
    """Contrato de resposta do endpoint GET /health/ready."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["ready", "not_ready"]
    database_ok: bool
    evaluator_mode: str
    a4_enabled: bool
    environment: str
    dev_identity_enabled: bool


class EventIngestResponse(BaseModel):
    """Contrato de resposta do endpoint POST /events."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str
    tenant_id: str
    status: Literal["queued", "duplicate_ignored"]
    deduplicated: bool
    processing_status: EventProcessingStatus
    outbox_id: str | None = None


class ImpulsePageResponse(BaseModel):
    """Contrato de resposta paginada do endpoint GET /impulses."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str
    items: list[Impulse]
    count: int
    next_cursor: str | None = None


class ImpulseAckRequest(BaseModel):
    """Payload opcional para confirmação de recebimento (`POST /impulses/{id}/ack`)."""

    model_config = ConfigDict(extra="forbid")

    notes: str | None = None


class ImpulseApplyRequest(BaseModel):
    """Payload para confirmação de aplicação ou respeito à pausa A3 (`POST /impulses/{id}/apply`)."""

    model_config = ConfigDict(extra="forbid")

    pause_respected: bool = True
    action_taken: str | None = None
    notes: str | None = None
    record_outcome: bool = True
