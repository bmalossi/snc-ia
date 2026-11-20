"""Testes unitários e de integração da Etapa 1: Config, Schemas, Logging e SQLite Storage."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from pic_snc.config import (
    ConfigurationError,
    EvaluatorMode,
    ExternalActionPolicy,
    SNCConfig,
)
from pic_snc.logging_utils import (
    get_structured_logger,
    log_cognitive_step,
    sanitize_log_payload,
)
from pic_snc.schemas import (
    ActivationMode,
    AssociativeCues,
    AuthContext,
    AutonomyLevel,
    Candidate,
    CognitiveKind,
    Event,
    EventCompleteness,
    EventProcessingStatus,
    EvidenceRef,
    Impulse,
    ImpulseStatus,
    Memory,
    MemoryRelation,
    OutboxStatus,
    OutboxTopic,
    Outcome,
    OutcomeResult,
    PredictiveProfile,
    TemporalProfile,
)
from pic_snc.storage import SQLiteStorage, TenantIsolationError


# -----------------------------------------------------------------------------
# 1. CONFIGURAÇÃO E POLÍTICAS DE AVALIAÇÃO / AUTORIDADE
# -----------------------------------------------------------------------------
def test_config_defaults_and_explicit_mock_mode() -> None:
    cfg = SNCConfig(evaluator_mode=EvaluatorMode.MOCK)
    assert cfg.evaluator_mode == EvaluatorMode.MOCK
    assert cfg.enable_a4 is False
    assert cfg.external_action_policy == ExternalActionPolicy.DENY
    assert cfg.has_typesafe_key is False


def test_config_forbids_silent_fallback_when_jev_selected_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="Fallback silencioso para MockEvaluator é proibido"):
        SNCConfig(evaluator_mode=EvaluatorMode.JEV, typesafe_api_key=None)

    with pytest.raises(ConfigurationError, match="Fallback silencioso"):
        SNCConfig.from_env(explicit_mode="jev")


def test_config_jev_mode_succeeds_with_key_and_redacts_repr() -> None:
    cfg = SNCConfig(
        evaluator_mode=EvaluatorMode.JEV,
        typesafe_api_key="ts_live_super_secret_key",
    )
    assert cfg.evaluator_mode == EvaluatorMode.JEV
    assert cfg.has_typesafe_key is True
    assert "ts_live_super_secret_key" not in repr(cfg)


def test_logging_redacts_secrets_and_outputs_utc() -> None:
    logger = get_structured_logger("pic_snc.test")
    emitted = log_cognitive_step(
        logger,
        stage="test.stage",
        action="verify_redaction",
        tenant_id="tenant_alpha",
        details={
            "typesafe_api_key": "secret_123",
            "nested": {"authorization": "Bearer xyz", "safe": "ok"},
        },
    )
    assert emitted["details"]["typesafe_api_key"] == "***REDACTED***"
    assert emitted["details"]["nested"]["authorization"] == "***REDACTED***"
    assert emitted["details"]["nested"]["safe"] == "ok"
    assert sanitize_log_payload({"token": "abc"}) == {"token": "***REDACTED***"}


# -----------------------------------------------------------------------------
# 2. SCHEMAS: DISTINÇÃO DE ESTÁGIOS, UTC, INTERVALO [0,1] E EXTENSIBILIDADE
# -----------------------------------------------------------------------------
def test_explicit_stage_distinction_and_utc_normalization() -> None:
    brt = timezone(timedelta(hours=-3))
    local_dt = datetime(2026, 11, 20, 12, 0, 0, tzinfo=brt)

    ev = Event(
        event_id="evt_01",
        tenant_id="tenant_a",
        event_type="invoice.overdue_detected",
        source="erp.billing_webhook",
        payload={"invoice_id": "inv_99"},
        timestamp=local_dt,
    )
    mem = Memory(
        memory_id="mem_01",
        tenant_id="tenant_a",
        kind=CognitiveKind.RISK,
        content="Cliente possui histórico de bloqueio contratual após 15 dias de atraso.",
        source="crm.contract_history",
        salience=0.85,
        created_at=local_dt,
    )
    cand = Candidate(
        candidate_id="cnd_01",
        tenant_id="tenant_a",
        event_id=ev.event_id,
        memory_id=mem.memory_id,
        kind=CognitiveKind.RISK,
        activation_mode=ActivationMode.PREDICTIVE,
        activation_score=0.88,
        score=0.91,
        proposed_autonomy=AutonomyLevel.A3,
        rationale="Atraso recorrente correlacionado com risco contratual.",
        created_at=local_dt,
    )
    imp = Impulse(
        impulse_id="imp_01",
        tenant_id="tenant_a",
        event_id=ev.event_id,
        candidate_id=cand.candidate_id,
        memory_ids=[mem.memory_id],
        kind=CognitiveKind.RISK,
        autonomy_level=AutonomyLevel.A3,
        message="Risco elevado de suspensão contratual para fatura inv_99.",
        confidence=0.90,
        status=ImpulseStatus.PUBLISHED,
        created_at=local_dt,
    )

    assert ev.artifact_stage == "observed_event"
    assert mem.artifact_stage == "latent_memory"
    assert cand.artifact_stage == "activated_candidate"
    assert imp.artifact_stage == "emitted_decision"

    assert ev.timestamp.tzinfo == UTC
    assert ev.timestamp.hour == 15
    assert mem.created_at.tzinfo == UTC
    assert cand.created_at.tzinfo == UTC
    assert imp.created_at.tzinfo == UTC


def test_extensible_associative_memory_contract() -> None:
    mem = Memory(
        memory_id="mem_assoc_1",
        tenant_id="tenant_a",
        kind=CognitiveKind.OPPORTUNITY,
        content="Padrão de expansão trimestral quando uso excede 80% da franquia.",
        source="analytics.usage_patterns",
        salience=0.78,
        tags=["Quota", "Upsell"],
        associative_cues=AssociativeCues(
            tags=["expansion"],
            semantic_keys=["capacity_threshold", "plan_upgrade"],
            embedding_ref="emb://vec_store/tenant_a/mem_assoc_1",
            temporal=TemporalProfile(
                recurrence_hint="quarterly",
                decay_rate=0.02,
                time_window_seconds=86400 * 30,
            ),
            predictive=PredictiveProfile(
                trigger_cues=["usage.above_80_pct", "seat_limit_reached"],
                anticipated_outcome="expansion_opportunity_within_14d",
                horizon_seconds=86400 * 14,
                prior_probability=0.74,
            ),
            relations=[
                MemoryRelation(
                    target_id="mem_pricing_tier_enterprise",
                    relation_type="precedes",
                    weight=0.85,
                )
            ],
        ),
    )
    assert mem.tags == ["quota", "upsell", "expansion"]
    assert mem.associative_cues.semantic_keys == ["capacity_threshold", "plan_upgrade"]
    assert mem.associative_cues.predictive.prior_probability == 0.74
    assert mem.associative_cues.relations[0].relation_type == "precedes"


def test_progressive_observation_does_not_force_evaluation_on_partial_low_density() -> None:
    partial_token_event = Event(
        tenant_id="tenant_a",
        event_type="stream.chunk_received",
        source="agent.stream_tap",
        completeness=EventCompleteness.PARTIAL,
        stream_id="str_1",
        sequence_no=1,
        semantic_density=0.10,
        payload={"fragment": "Ol"},
    )
    complete_semantic_event = Event(
        tenant_id="tenant_a",
        event_type="deployment.ProdConfigModified",
        source="ci.pipeline_observer",
        completeness=EventCompleteness.COMPLETE,
        semantic_density=0.95,
        payload={"service": "payment-gateway", "change": "timeout_reduced"},
    )

    assert partial_token_event.should_trigger_evaluation(min_density=0.40) is False
    assert complete_semantic_event.should_trigger_evaluation(min_density=0.40) is True


@pytest.mark.parametrize("invalid_val", [-0.01, 1.01, float("nan"), float("inf")])
def test_unit_interval_validation_on_salience_score_and_confidence(
    invalid_val: float,
) -> None:
    with pytest.raises(ValidationError):
        Memory(
            tenant_id="tenant_a",
            kind=CognitiveKind.RECALL,
            content="Teste",
            source="src",
            salience=invalid_val,
        )

    with pytest.raises(ValidationError):
        Candidate(
            tenant_id="tenant_a",
            event_id="evt_1",
            memory_id="mem_1",
            kind=CognitiveKind.RECALL,
            score=invalid_val,
            rationale="Teste",
        )

    with pytest.raises(ValidationError):
        Impulse(
            tenant_id="tenant_a",
            event_id="evt_1",
            candidate_id="cnd_1",
            kind=CognitiveKind.RECALL,
            autonomy_level=AutonomyLevel.A1,
            message="Teste",
            confidence=invalid_val,
        )


def test_protective_block_only_allowed_for_a4() -> None:
    with pytest.raises(ValidationError, match="protective_block=True só é permitido"):
        Impulse(
            tenant_id="tenant_a",
            event_id="evt_1",
            candidate_id="cnd_1",
            kind=CognitiveKind.RISK,
            autonomy_level=AutonomyLevel.A3,
            protective_block=True,
            message="Tentativa inválida de bloqueio em A3",
            confidence=0.9,
        )

    valid_a4 = Impulse(
        tenant_id="tenant_a",
        event_id="evt_1",
        candidate_id="cnd_1",
        kind=CognitiveKind.RISK,
        autonomy_level=AutonomyLevel.A4,
        protective_block=True,
        message="Bloqueio protetivo autorizado em A4",
        confidence=0.98,
    )
    assert valid_a4.protective_block is True
    assert valid_a4.external_action_allowed is False


# -----------------------------------------------------------------------------
# 3. SQLITE STORAGE: TRANSAÇÕES, OUTBOX, DEDUPLICAÇÃO E ISOLAMENTO MULTI-TENANT
# -----------------------------------------------------------------------------
@pytest.fixture()
def storage(tmp_path: Path) -> SQLiteStorage:
    return SQLiteStorage(tmp_path / "test_pic_snc.db")


def test_event_ingestion_and_outbox_are_transactional_and_idempotent(
    storage: SQLiteStorage,
) -> None:
    ev1 = Event(
        event_id="evt_tx_1",
        tenant_id="tenant_1",
        event_type="contract.clause_flagged",
        source="legal.analyzer",
        dedupe_key="contract-2026-clause-7",
        payload={"clause": 7, "risk": "unlimited_liability"},
        evidence=[
            EvidenceRef(
                evidence_id="evd_clause_7",
                source_type="document_span",
                reference="doc://contracts/2026#clause-7",
                excerpt="Responsabilidade ilimitada por danos indiretos.",
                weight=0.95,
            )
        ],
    )

    first = storage.ingest_event(ev1)
    assert first.created is True
    assert first.deduplicated is False
    assert first.outbox_entry is not None
    assert first.outbox_entry.topic == OutboxTopic.EVENT_INGESTED

    # Reenvio com mesmo event_id no mesmo tenant
    dup_by_id = storage.ingest_event(ev1)
    assert dup_by_id.created is False
    assert dup_by_id.deduplicated is True
    assert dup_by_id.outbox_entry is None

    # Reenvio com event_id diferente mas mesma dedupe_key no mesmo tenant
    ev_same_dedupe = ev1.model_copy(update={"event_id": "evt_tx_2"})
    dup_by_key = storage.ingest_event(ev_same_dedupe)
    assert dup_by_key.created is False
    assert dup_by_key.deduplicated is True
    assert dup_by_key.event.event_id == "evt_tx_1"

    # Verifica que apenas 1 evento e 1 registro de outbox existem no banco
    events = storage.list_events("tenant_1")
    outbox = storage.list_outbox_entries("tenant_1")
    assert len(events) == 1
    assert len(outbox) == 1
    assert events[0].evidence[0].reference == "doc://contracts/2026#clause-7"


def test_partial_low_density_event_is_buffered_without_outbox(
    storage: SQLiteStorage,
) -> None:
    partial_ev = Event(
        event_id="evt_partial_1",
        tenant_id="tenant_1",
        event_type="typing.progress",
        source="editor.stream",
        completeness=EventCompleteness.PARTIAL,
        semantic_density=0.15,
        payload={"chars": 4},
    )
    res = storage.ingest_event(partial_ev, min_eval_density=0.40)
    assert res.created is True
    assert res.event.processing_status == EventProcessingStatus.BUFFERED
    assert res.outbox_entry is None
    assert storage.list_outbox_entries("tenant_1") == []


def test_multi_tenant_isolation_and_authenticated_context(
    tmp_path: Path,
) -> None:
    strict_storage = SQLiteStorage(
        tmp_path / "strict_tenant.db",
        require_authenticated_context=True,
    )
    auth_a = AuthContext(principal_id="usr_a", tenant_id="tenant_a")
    auth_b = AuthContext(principal_id="usr_b", tenant_id="tenant_b")

    # Sem AuthContext quando obrigatório -> erro
    ev_a = Event(
        event_id="evt_shared_id",
        tenant_id="tenant_a",
        event_type="signal.detected",
        source="sensor.alpha",
        dedupe_key="shared-dedupe-key",
    )
    with pytest.raises(TenantIsolationError, match="AuthContext autenticado é obrigatório"):
        strict_storage.ingest_event(ev_a)

    # AuthContext do tenant_b tentando gravar no tenant_a -> bloqueado
    with pytest.raises(TenantIsolationError, match="Violação de isolamento multi-tenant"):
        strict_storage.ingest_event(ev_a, auth_context=auth_b)

    # Gravação legítima no tenant_a e no tenant_b com mesmo event_id e dedupe_key
    res_a = strict_storage.ingest_event(ev_a, auth_context=auth_a)
    ev_b = ev_a.model_copy(update={"tenant_id": "tenant_b"})
    res_b = strict_storage.ingest_event(ev_b, auth_context=auth_b)

    assert res_a.created is True
    assert res_b.created is True

    mem_a = Memory(
        memory_id="mem_1",
        tenant_id="tenant_a",
        kind=CognitiveKind.RECALL,
        content="Memória exclusiva do Tenant A",
        source="vault.a",
        salience=0.9,
    )
    strict_storage.save_memory(mem_a, auth_context=auth_a)

    # Tenant B não enxerga memórias, eventos ou outbox do Tenant A
    assert strict_storage.list_memories("tenant_b", auth_context=auth_b) == []
    assert len(strict_storage.list_memories("tenant_a", auth_context=auth_a)) == 1

    with pytest.raises(TenantIsolationError):
        strict_storage.list_memories("tenant_a", auth_context=auth_b)


def test_full_lineage_event_memory_candidate_impulse_outcome_and_db_constraints(
    storage: SQLiteStorage,
) -> None:
    ev = Event(
        event_id="evt_lineage_1",
        tenant_id="tenant_x",
        event_type="deploy.window_opened",
        source="release.manager",
    )
    storage.ingest_event(ev)

    mem = Memory(
        memory_id="mem_lineage_1",
        tenant_id="tenant_x",
        kind=CognitiveKind.RECALL,
        content="Deploys na sexta-feira exigem aprovação dupla de SRE.",
        source="runbook.sre_policy",
        salience=0.92,
        associative_cues=AssociativeCues(
            tags=["deploy", "sre"],
            semantic_keys=["release_governance"],
        ),
    )
    storage.save_memory(mem)

    cand = Candidate(
        candidate_id="cnd_lineage_1",
        tenant_id="tenant_x",
        event_id=ev.event_id,
        memory_id=mem.memory_id,
        kind=CognitiveKind.RECALL,
        activation_mode=ActivationMode.HYBRID,
        activation_score=0.90,
        score=0.88,
        proposed_autonomy=AutonomyLevel.A1,
        rationale="Janela de deploy aberta coincide com política de aprovação dupla.",
        source_trace={"event_source": ev.source, "memory_source": mem.source},
    )
    storage.save_candidate(cand)

    imp = Impulse(
        impulse_id="imp_lineage_1",
        tenant_id="tenant_x",
        event_id=ev.event_id,
        candidate_id=cand.candidate_id,
        memory_ids=[mem.memory_id],
        kind=CognitiveKind.RECALL,
        autonomy_level=AutonomyLevel.A1,
        message="Recordação: este deploy requer aprovação dupla de SRE.",
        confidence=0.89,
        source_trace=cand.source_trace,
        status=ImpulseStatus.PUBLISHED,
    )
    saved_imp, imp_outbox = storage.save_impulse(imp)
    assert saved_imp.impulse_id == "imp_lineage_1"
    assert imp_outbox is not None
    assert imp_outbox.topic == OutboxTopic.IMPULSE_EMITTED

    outcome = Outcome(
        outcome_id="out_lineage_1",
        impulse_id=imp.impulse_id,
        tenant_id="tenant_x",
        result=OutcomeResult.ACCEPTED,
        notes="Engenheiro solicitou aprovação secundária antes de prosseguir.",
    )
    storage.save_outcome(outcome)

    assert len(storage.list_candidates_for_event("tenant_x", ev.event_id)) == 1
    assert len(storage.list_impulses("tenant_x", status=ImpulseStatus.PUBLISHED)) == 1
    assert len(storage.list_outcomes("tenant_x", impulse_id=imp.impulse_id)) == 1

    # Verifica restrições nativas do SQLite (CHECK e FOREIGN KEY composta por tenant_id)
    conn = sqlite3.connect(str(storage.db_path))
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        # 1) Salience fora de [0, 1] rejeitada pelo SQLite CHECK constraint
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO memories (
                    tenant_id, memory_id, kind, content, source, salience,
                    tags_json, associative_cues_json, evidence_json, metadata_json, created_at_utc
                ) VALUES ('tenant_x', 'mem_bad', 'recall', 'c', 's', 1.5, '[]', '{}', '[]', '{}', '2026-11-20T00:00:00+00:00')
                """
            )

        # 2) Candidato de outro tenant tentando referenciar evento/memória do tenant_x -> FK violation
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO candidates (
                    tenant_id, candidate_id, event_id, memory_id, related_memory_ids_json,
                    kind, activation_mode, activation_score, score, proposed_autonomy,
                    rationale, source_trace_json, evidence_json, created_at_utc
                ) VALUES (
                    'tenant_intruder', 'cnd_cross', 'evt_lineage_1', 'mem_lineage_1', '[]',
                    'recall', 'lexical', 0.5, 0.5, 'A1', 'r', '{}', '[]', json('[]')
                )
                """
            )
    finally:
        conn.close()
