"""Testes completos da Etapa 2: Interfaces, Observer, ActivationEngine, Evaluator, Governor, Bus, Outcome e Pipeline."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from pic_snc.activation import AssociativeActivationEngine
from pic_snc.bus import SQLiteImpulseBus
from pic_snc.config import ConfigurationError, EvaluatorMode, SNCConfig
from pic_snc.evaluator import (
    EvaluatorExecutionError,
    JevEvaluatorAdapter,
    MockEvaluator,
    create_evaluator,
)
from pic_snc.governor import CognitiveGovernor
from pic_snc.interfaces import (
    ActivationEngine,
    Evaluator,
    Governor,
    ImpulseBus,
    Observer,
    OutcomeTracker,
)
from pic_snc.observer import ObserverPolicy, ProgressiveCognitiveObserver
from pic_snc.outcome import OutcomeVerificationError, SQLiteOutcomeTracker
from pic_snc.pipeline import CognitivePipeline
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
    ImpulseStatus,
    Memory,
    MemoryRelation,
    ObservationDecision,
    OutboxStatus,
    OutboxTopic,
    Outcome,
    OutcomeResult,
    PredictiveProfile,
    TemporalProfile,
)
from pic_snc.storage import SQLiteStorage, TenantIsolationError


@pytest.fixture()
def config(tmp_path: Path) -> SNCConfig:
    return SNCConfig(
        db_path=tmp_path / "etapa2_pic_snc.db",
        evaluator_mode=EvaluatorMode.MOCK,
        enable_a4=False,
    )


@pytest.fixture()
def pipeline(config: SNCConfig) -> CognitivePipeline:
    return CognitivePipeline(config)


def _seed_three_scenario_memories(storage: SQLiteStorage, tenant_id: str = "tenant_corp") -> None:
    """Popula memórias determinísticas para os cenários de recordação, oportunidade e risco."""
    # 1. Memória de Recordação (A1)
    storage.save_memory(
        Memory(
            memory_id="mem_recall_sre_deploy",
            tenant_id=tenant_id,
            kind=CognitiveKind.RECALL,
            content="Política de release: deploys do serviço payment-gateway exigem checklist de migração idempotente e aprovação de SRE.",
            source="runbook.engineering_releases",
            salience=0.88,
            tags=["deploy", "payment-gateway", "sre"],
            associative_cues=AssociativeCues(
                tags=["deploy", "payment-gateway", "sre"],
                semantic_keys=["release_window", "database_migration"],
                temporal=TemporalProfile(recurrence_hint="weekly_release"),
            ),
            evidence=[
                EvidenceRef(
                    evidence_id="evd_runbook_sre_42",
                    source_type="policy_document",
                    reference="doc://runbooks/sre-release-v3#section-4",
                    excerpt="Deploys em payment-gateway requerem checklist idempotente e dupla aprovação.",
                    weight=0.92,
                )
            ],
        )
    )

    # 2. Memória de Oportunidade (A2) + Memória Relacionada
    storage.save_memory(
        Memory(
            memory_id="mem_opp_tier_catalog",
            tenant_id=tenant_id,
            kind=CognitiveKind.OPPORTUNITY,
            content="Plano Enterprise anual oferece desconto de 20% em volume acima de 80% da franquia de API.",
            source="catalog.pricing_tiers",
            salience=0.75,
            tags=["pricing", "enterprise"],
        )
    )
    storage.save_memory(
        Memory(
            memory_id="mem_opp_expansion_80pct",
            tenant_id=tenant_id,
            kind=CognitiveKind.OPPORTUNITY,
            content="Contas que ultrapassam 85% da cota mensal de chamadas com crescimento sustentado têm alta conversão para upgrade Enterprise.",
            source="analytics.revenue_expansion",
            salience=0.90,
            tags=["quota", "expansion", "upgrade"],
            associative_cues=AssociativeCues(
                tags=["quota", "expansion", "upgrade"],
                semantic_keys=["api_usage_spike", "tier_capacity"],
                predictive=PredictiveProfile(
                    trigger_cues=["quota_utilization_high", "usage_85_percent"],
                    anticipated_outcome="enterprise_plan_expansion",
                    horizon_seconds=86400 * 14,
                    prior_probability=0.82,
                ),
                relations=[
                    MemoryRelation(
                        target_id="mem_opp_tier_catalog",
                        relation_type="precedes",
                        weight=0.85,
                    )
                ],
            ),
            evidence=[
                EvidenceRef(
                    evidence_id="evd_analytics_q3",
                    source_type="cohort_study",
                    reference="analytics://cohorts/2026-q3-expansion",
                    excerpt="78% de conversão quando uso excede 85% da franquia sem erros 429.",
                    weight=0.89,
                )
            ],
        )
    )

    # 3. Memória de Risco / Contradição Histórica (A3)
    storage.save_memory(
        Memory(
            memory_id="mem_risk_liability_cap",
            tenant_id=tenant_id,
            kind=CognitiveKind.RISK,
            content="Restrição jurídica obrigatória: contratos com fornecedores nunca podem conter cláusula de responsabilidade ilimitada (unlimited_liability) ou multa rescisória unilateral.",
            source="legal.compliance_charter",
            salience=0.96,
            tags=["contract", "compliance", "unlimited_liability", "clause"],
            associative_cues=AssociativeCues(
                tags=["contract", "compliance", "unlimited_liability", "clause"],
                semantic_keys=["legal_contradiction", "liability_cap_violation"],
                predictive=PredictiveProfile(
                    trigger_cues=["unlimited_liability", "waive_liability_cap"],
                    anticipated_outcome="regulatory_and_financial_exposure",
                    prior_probability=0.95,
                ),
            ),
            evidence=[
                EvidenceRef(
                    evidence_id="evd_legal_charter_09",
                    source_type="compliance_rule",
                    reference="legal://charter/2026#art-9-liability",
                    excerpt="Proibida assinatura com unlimited_liability; teto máximo limitado a 12 meses de contrato.",
                    weight=0.98,
                )
            ],
        )
    )


# -----------------------------------------------------------------------------
# 0. VERIFICAÇÃO DE CONTRATOS E PROTOCOLOS (INTERFACES)
# -----------------------------------------------------------------------------
def test_0_components_implement_runtime_checkable_protocols(
    pipeline: CognitivePipeline,
) -> None:
    assert isinstance(pipeline.observer, Observer)
    assert isinstance(pipeline.activation_engine, ActivationEngine)
    assert isinstance(pipeline.evaluator, Evaluator)
    assert isinstance(pipeline.governor, Governor)
    assert isinstance(pipeline.impulse_bus, ImpulseBus)
    assert isinstance(pipeline.outcome_tracker, OutcomeTracker)


# -----------------------------------------------------------------------------
# 1. RECORDAÇÃO ESPONTÂNEA (A1)
# -----------------------------------------------------------------------------
def test_1_spontaneous_recall_a1(pipeline: CognitivePipeline) -> None:
    _seed_three_scenario_memories(pipeline.storage)

    event = Event(
        event_id="evt_recall_deploy_01",
        tenant_id="tenant_corp",
        event_type="deployment.pipeline_started",
        source="ci.github_actions",
        payload={
            "service": "payment-gateway",
            "action": "deploy",
            "environment": "production",
            "window": "weekly_release",
        },
        evidence=[
            EvidenceRef(
                evidence_id="evd_ci_run_101",
                source_type="ci_pipeline",
                reference="ci://runs/101",
                excerpt="Deploy iniciado para payment-gateway em produção.",
                weight=0.90,
            )
        ],
    )

    result = pipeline.ingest_and_process(event)

    assert result.deduplicated is False
    assert result.observation is not None
    assert result.observation.artifact_stage == "observation_decision"
    assert result.observation.should_activate is True
    assert len(result.published_impulses) >= 1

    top_impulse = result.published_impulses[0]
    assert top_impulse.artifact_stage == "emitted_decision"
    assert top_impulse.kind == CognitiveKind.RECALL
    assert top_impulse.autonomy_level == AutonomyLevel.A1
    assert top_impulse.intervention_requested is False
    assert top_impulse.protective_block is False
    assert "mem_recall_sre_deploy" in top_impulse.memory_ids
    assert "[RECORDAÇÃO A1]" in top_impulse.message


# -----------------------------------------------------------------------------
# 2. OPORTUNIDADE ESPONTÂNEA (A2) COM PROPAGAÇÃO RELACIONAL
# -----------------------------------------------------------------------------
def test_2_spontaneous_opportunity_a2(pipeline: CognitivePipeline) -> None:
    _seed_three_scenario_memories(pipeline.storage)

    event = Event(
        event_id="evt_opp_quota_01",
        tenant_id="tenant_corp",
        event_type="telemetry.quota_utilization_high",
        source="metering.api_gateway",
        payload={
            "account_id": "acc_778",
            "metric": "usage_85_percent",
            "signal": "api_usage_spike",
            "current_utilization": 0.88,
            "trend": "expansion",
        },
    )

    result = pipeline.ingest_and_process(event)

    assert len(result.published_impulses) >= 1
    opp_impulses = [
        imp for imp in result.published_impulses if imp.kind == CognitiveKind.OPPORTUNITY
    ]
    assert len(opp_impulses) >= 1

    primary_opp = opp_impulses[0]
    assert primary_opp.autonomy_level == AutonomyLevel.A2
    assert primary_opp.intervention_requested is False
    assert primary_opp.protective_block is False
    assert "mem_opp_expansion_80pct" in primary_opp.memory_ids
    # Verifica que a relação explícita (`precedes` -> `mem_opp_tier_catalog`) foi propagada
    assert "mem_opp_tier_catalog" in primary_opp.memory_ids
    assert "[OPORTUNIDADE A2]" in primary_opp.message


# -----------------------------------------------------------------------------
# 3. RISCO ESPONTÂNEO (A3)
# -----------------------------------------------------------------------------
def test_3_spontaneous_risk_a3(pipeline: CognitivePipeline) -> None:
    _seed_three_scenario_memories(pipeline.storage)

    event = Event(
        event_id="evt_risk_contract_01",
        tenant_id="tenant_corp",
        event_type="contract.draft_reviewed",
        source="procurement.document_observer",
        payload={
            "vendor": "Acme Cloud Ltd",
            "clause_detected": "unlimited_liability",
            "action": "waive_liability_cap",
            "status": "pending_signature",
        },
        evidence=[
            EvidenceRef(
                evidence_id="evd_contract_draft_55",
                source_type="contract_clause",
                reference="doc://procurement/draft-55#clause-14",
                excerpt="Fornecedor propõe unlimited_liability e waive_liability_cap.",
                weight=0.97,
            )
        ],
    )

    result = pipeline.ingest_and_process(event)

    risk_impulses = [
        imp for imp in result.published_impulses if imp.kind == CognitiveKind.RISK
    ]
    assert len(risk_impulses) == 1
    risk_imp = risk_impulses[0]

    assert risk_imp.autonomy_level == AutonomyLevel.A3
    assert risk_imp.intervention_requested is True
    assert risk_imp.protective_block is False
    assert "mem_risk_liability_cap" in risk_imp.memory_ids
    assert "[SOLICITAÇÃO DE PAUSA A3]" in risk_imp.message


# -----------------------------------------------------------------------------
# 4. OBSERVAÇÃO INCREMENTAL (PROGRESSIVA)
# -----------------------------------------------------------------------------
def test_4_incremental_progressive_observation(pipeline: CognitivePipeline) -> None:
    _seed_three_scenario_memories(pipeline.storage)

    base_time = datetime(2026, 11, 20, 14, 0, 0, tzinfo=UTC)

    # Fragmento 1: parcial e com baixa densidade (0.20) -> NÃO dispara avaliação
    frag1 = Event(
        event_id="evt_stream_f1",
        tenant_id="tenant_corp",
        event_type="conversation.chunk",
        source="agent.workspace_stream",
        completeness=EventCompleteness.PARTIAL,
        stream_id="conv_stream_42",
        sequence_no=1,
        semantic_density=0.20,
        payload={"fragment": "Iniciando preparação da janela weekly_release"},
        timestamp=base_time,
    )
    res1 = pipeline.ingest_and_process(frag1)
    assert res1.observation is not None
    assert res1.observation.should_activate is False
    assert res1.observation.trigger_reason == "insufficient_signal_buffered"
    assert res1.event.processing_status == EventProcessingStatus.BUFFERED
    assert res1.published_impulses == []

    # Fragmento 2: parcial e com baixa densidade (0.25 -> acumulado 0.45 < 0.65) -> NÃO dispara
    frag2 = Event(
        event_id="evt_stream_f2",
        tenant_id="tenant_corp",
        event_type="conversation.chunk",
        source="agent.workspace_stream",
        completeness=EventCompleteness.PARTIAL,
        stream_id="conv_stream_42",
        sequence_no=2,
        semantic_density=0.25,
        payload={"fragment": "para deploy do serviço payment-gateway"},
        timestamp=base_time + timedelta(seconds=2),
    )
    res2 = pipeline.ingest_and_process(frag2)
    assert res2.observation is not None
    assert res2.observation.should_activate is False
    assert res2.published_impulses == []

    # Fragmento 3: parcial (0.25 -> acumulado 0.70 >= 0.65) -> ATINGE LIMIAR e dispara avaliação!
    frag3 = Event(
        event_id="evt_stream_f3",
        tenant_id="tenant_corp",
        event_type="conversation.chunk",
        source="agent.workspace_stream",
        completeness=EventCompleteness.PARTIAL,
        stream_id="conv_stream_42",
        sequence_no=3,
        semantic_density=0.25,
        payload={"fragment": "com database_migration pendente de revisão sre"},
        timestamp=base_time + timedelta(seconds=4),
    )
    res3 = pipeline.ingest_and_process(frag3)
    assert res3.observation is not None
    assert res3.observation.should_activate is True
    assert res3.observation.trigger_reason == "accumulated_context_threshold_reached"
    assert res3.observation.accumulated_event_ids == [
        "evt_stream_f1",
        "evt_stream_f2",
        "evt_stream_f3",
    ]
    assert "payment-gateway" in res3.observation.aggregated_payload["assembled_text"]
    assert len(res3.published_impulses) >= 1
    assert res3.published_impulses[0].autonomy_level == AutonomyLevel.A1

    # Verifica que todos os 3 fragmentos foram marcados como PROCESSED no banco
    for f_id in ("evt_stream_f1", "evt_stream_f2", "evt_stream_f3"):
        stored_ev = pipeline.storage.get_event("tenant_corp", f_id)
        assert stored_ev is not None
        assert stored_ev.processing_status == EventProcessingStatus.PROCESSED


# -----------------------------------------------------------------------------
# 5. AUSÊNCIA DE RELEVÂNCIA
# -----------------------------------------------------------------------------
def test_5_irrelevant_event_publishes_no_impulse(pipeline: CognitivePipeline) -> None:
    _seed_three_scenario_memories(pipeline.storage)

    noise_event = Event(
        event_id="evt_noise_01",
        tenant_id="tenant_corp",
        event_type="ui.theme_toggled",
        source="client.preferences",
        payload={"color_scheme": "solarized", "font_size": "medium"},
    )

    result = pipeline.ingest_and_process(noise_event)
    assert result.observation is not None
    assert result.observation.should_activate is True
    assert result.activated_memories == []
    assert result.candidates == []
    assert result.published_impulses == []
    assert pipeline.storage.list_impulses("tenant_corp") == []


# -----------------------------------------------------------------------------
# 6. IDEMPOTÊNCIA (EVENTOS E BARRAMENTO)
# -----------------------------------------------------------------------------
def test_6_idempotency_duplicate_events_produce_no_duplicate_impulses(
    pipeline: CognitivePipeline,
) -> None:
    _seed_three_scenario_memories(pipeline.storage)

    ev = Event(
        event_id="evt_idem_01",
        tenant_id="tenant_corp",
        event_type="deployment.pipeline_started",
        source="ci.github_actions",
        dedupe_key="deploy-run-2026-999",
        payload={"service": "payment-gateway", "action": "deploy", "team": "sre"},
    )

    first_run = pipeline.ingest_and_process(ev)
    assert first_run.deduplicated is False
    assert len(first_run.published_impulses) == 1

    # Reenvio com mesmo event_id
    second_run = pipeline.ingest_and_process(ev)
    assert second_run.deduplicated is True
    assert second_run.published_impulses == []

    # Reenvio com novo event_id mas mesma dedupe_key
    ev_same_key = ev.model_copy(update={"event_id": "evt_idem_02"})
    third_run = pipeline.ingest_and_process(ev_same_key)
    assert third_run.deduplicated is True
    assert third_run.published_impulses == []

    # Republicação direta do mesmo impulso no barramento também é idempotente
    republished = pipeline.impulse_bus.publish(first_run.published_impulses)
    assert len(republished) == 1

    stored_impulses = pipeline.storage.list_impulses("tenant_corp")
    impulse_outbox = pipeline.storage.list_outbox_entries(
        "tenant_corp",
        topic=OutboxTopic.IMPULSE_EMITTED,
    )
    assert len(stored_impulses) == 1
    assert len(impulse_outbox) == 1


# -----------------------------------------------------------------------------
# 7. SEGURANÇA E ISOLAMENTO MULTI-TENANT
# -----------------------------------------------------------------------------
def test_7_multi_tenant_isolation_across_full_pipeline(tmp_path: Path) -> None:
    cfg = SNCConfig(
        db_path=tmp_path / "tenant_iso.db",
        evaluator_mode=EvaluatorMode.MOCK,
        require_authenticated_context=True,
    )
    pipe = CognitivePipeline(cfg)

    auth_a = AuthContext(principal_id="agent_a", tenant_id="tenant_alpha")
    auth_b = AuthContext(principal_id="agent_b", tenant_id="tenant_beta")

    # Popula memórias sensíveis apenas no tenant_alpha
    pipe.storage.save_memory(
        Memory(
            memory_id="mem_alpha_secret",
            tenant_id="tenant_alpha",
            kind=CognitiveKind.RISK,
            content="Chave de conformidade exclusiva do Tenant Alpha com cláusula unlimited_liability.",
            source="vault.alpha",
            salience=0.95,
            tags=["unlimited_liability", "contract"],
        ),
        auth_context=auth_a,
    )

    # Tenant Beta envia evento com os mesmos sinais ("unlimited_liability", "contract")
    ev_beta = Event(
        event_id="evt_beta_1",
        tenant_id="tenant_beta",
        event_type="contract.review",
        source="beta.observer",
        payload={"clause": "unlimited_liability", "type": "contract"},
    )
    res_beta = pipe.ingest_and_process(ev_beta, auth_context=auth_b)

    # Nenhuma memória do Tenant Alpha pode ser ativada para o Tenant Beta
    assert res_beta.activated_memories == []
    assert res_beta.published_impulses == []

    # Tentativa de processamento cruzado com AuthContext trocado é bloqueada
    with pytest.raises(TenantIsolationError):
        pipe.ingest_and_process(ev_beta, auth_context=auth_a)


# -----------------------------------------------------------------------------
# 8. GOVERNANÇA: A4 DESATIVADO POR PADRÃO + COOLDOWN DE REPETIÇÃO
# -----------------------------------------------------------------------------
def test_8_governance_a4_remains_disabled_by_default_and_cooldown_prevents_repetition(
    pipeline: CognitivePipeline,
) -> None:
    _seed_three_scenario_memories(pipeline.storage)
    base_time = datetime(2026, 11, 20, 16, 0, 0, tzinfo=UTC)

    # Evento solicitando bloqueio protetivo A4 (com enable_a4=False na config padrão)
    ev_a4_req = Event(
        event_id="evt_gov_a4_1",
        tenant_id="tenant_corp",
        event_type="contract.critical_violation",
        source="legal.guard",
        payload={
            "clause": "unlimited_liability",
            "action": "waive_liability_cap",
            "request_protective_block": True,
        },
        timestamp=base_time,
    )
    res1 = pipeline.ingest_and_process(ev_a4_req)

    assert len(res1.candidates) == 1
    assert res1.candidates[0].proposed_autonomy == AutonomyLevel.A4

    # Governor deve bloquear A4 e rebaixar para solicitação de pausa A3
    assert len(res1.published_impulses) == 1
    imp1 = res1.published_impulses[0]
    assert imp1.autonomy_level == AutonomyLevel.A3
    assert imp1.protective_block is False
    assert imp1.intervention_requested is True
    assert imp1.source_trace.get("downgraded_from") == "A4"
    assert imp1.source_trace.get("a4_blocked_by_policy") is True

    # Segundo evento idêntico 10 segundos depois (dentro do cooldown de 120s) -> suprimido!
    ev_repeat = Event(
        event_id="evt_gov_a4_2",
        tenant_id="tenant_corp",
        event_type="contract.critical_violation",
        source="legal.guard",
        payload={
            "clause": "unlimited_liability",
            "action": "waive_liability_cap",
            "request_protective_block": True,
        },
        timestamp=base_time + timedelta(seconds=10),
    )
    res2 = pipeline.ingest_and_process(ev_repeat)
    assert res2.published_impulses == []
    assert len(res2.governed_impulses) == 1
    assert res2.governed_impulses[0].status == ImpulseStatus.SUPPRESSED
    assert res2.governed_impulses[0].suppressed_reason == "cooldown_active_or_duplicate_impulse"


# -----------------------------------------------------------------------------
# 9. FALHA DO AVALIADOR (SEM FALLBACK SILENCIOSO) E CONTRATO OFICIAL TYPESAFE JEV
# -----------------------------------------------------------------------------
def test_9_evaluator_failure_no_silent_fallback_to_mock_and_jev_contract(
    tmp_path: Path,
) -> None:
    # 9a) Sem chave com modo JEV -> ConfigurationError imediato
    with pytest.raises(ConfigurationError, match="Fallback silencioso para MockEvaluator é proibido"):
        SNCConfig(
            db_path=tmp_path / "jev_err.db",
            evaluator_mode=EvaluatorMode.JEV,
            typesafe_api_key=None,
        )

    # 9b) Com chave mas cliente Jev lançando falha -> EvaluatorExecutionError (NUNCA cai para MockEvaluator)
    jev_cfg = SNCConfig(
        db_path=tmp_path / "jev_live.db",
        evaluator_mode=EvaluatorMode.JEV,
        typesafe_api_key="ts_test_valid_format_key",
    )

    class _FailingTypeSafeClient:
        def systemone(
            self,
            *,
            model: str,
            state: str | dict[str, Any],
            questions: dict[str, Any],
        ) -> Any:
            raise ConnectionError("TypeSafe System One API timeout")

    failing_adapter = JevEvaluatorAdapter(jev_cfg, sdk_client=_FailingTypeSafeClient())
    storage = SQLiteStorage(jev_cfg.db_path)
    _seed_three_scenario_memories(storage)

    pipe_fail = CognitivePipeline(jev_cfg, storage=storage, evaluator=failing_adapter)
    ev = Event(
        event_id="evt_jev_fail_1",
        tenant_id="tenant_corp",
        event_type="contract.draft_reviewed",
        source="procurement.observer",
        payload={"clause": "unlimited_liability"},
    )
    with pytest.raises(EvaluatorExecutionError, match="Falha na chamada ts.systemone"):
        pipe_fail.ingest_and_process(ev)

    # Nenhum impulso simulado foi emitido!
    assert storage.list_impulses("tenant_corp") == []

    # 9c) Verifica mapeamento fiel das primitivas oficiais TypeSafe (Noul.noul, Score.score, Choice.choice/confidence)
    class _StubTypeSafeClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def systemone(
            self,
            *,
            model: str,
            state: str | dict[str, Any],
            questions: dict[str, Any],
        ) -> Any:
            self.calls.append({"model": model, "state": state, "questions": questions})
            return SimpleNamespace(
                answers=SimpleNamespace(
                    is_actionable=SimpleNamespace(noul=0.93),
                    cognitive_kind=SimpleNamespace(
                        choice="risk",
                        confidence=0.91,
                        probabilities={"risk": 0.92, "recall": 0.04, "opportunity": 0.02, "none": 0.02},
                    ),
                    relevance_rubric=SimpleNamespace(
                        score=1.84,
                        confidence=0.89,
                        probabilities={"0": 0.02, "1": 0.12, "2": 0.86},
                    ),
                )
            )

    stub_client = _StubTypeSafeClient()
    working_jev = create_evaluator(jev_cfg, sdk_client=stub_client)
    assert isinstance(working_jev, JevEvaluatorAdapter)

    pipe_jev = CognitivePipeline(jev_cfg, storage=storage, evaluator=working_jev)
    ev_jev_ok = Event(
        event_id="evt_jev_ok_1",
        tenant_id="tenant_corp",
        event_type="contract.draft_reviewed",
        source="procurement.observer",
        payload={"clause": "unlimited_liability"},
    )
    res_jev = pipe_jev.ingest_and_process(ev_jev_ok)

    assert len(stub_client.calls) == 1
    assert stub_client.calls[0]["model"] == "jev"
    assert set(stub_client.calls[0]["questions"].keys()) == {
        "is_actionable",
        "cognitive_kind",
        "relevance_rubric",
    }
    assert len(res_jev.candidates) == 1
    cand = res_jev.candidates[0]
    assert cand.noul_probability == 0.93
    assert cand.rubric_score == 1.84
    assert cand.score == 0.92  # 1.84 / 2.0
    assert cand.choice_confidence == round(0.60 * 0.91 + 0.40 * 0.89, 4)
    assert len(res_jev.published_impulses) == 1
    assert res_jev.published_impulses[0].autonomy_level == AutonomyLevel.A3


# -----------------------------------------------------------------------------
# 10. RASTREABILIDADE (EVIDÊNCIAS E LINHAGEM DE ORIGEM)
# -----------------------------------------------------------------------------
def test_10_traceability_impulses_preserve_evidence_and_source_lineage(
    pipeline: CognitivePipeline,
) -> None:
    _seed_three_scenario_memories(pipeline.storage)

    ev = Event(
        event_id="evt_trace_01",
        tenant_id="tenant_corp",
        event_type="contract.clause_flagged",
        source="audit.contract_scanner",
        payload={"clause": "unlimited_liability", "action": "waive_liability_cap"},
        evidence=[
            EvidenceRef(
                evidence_id="evd_scanner_span_99",
                source_type="document_span",
                reference="doc://contracts/2026-acme#line-210",
                excerpt="Cláusula 14.2 remove teto de indenização (unlimited_liability).",
                weight=0.96,
            )
        ],
    )

    res = pipeline.ingest_and_process(ev)
    assert len(res.published_impulses) == 1
    imp = res.published_impulses[0]

    evidence_ids = {e.evidence_id for e in imp.evidence}
    # Preserva evidência do evento observado E evidência da memória ativada
    assert "evd_scanner_span_99" in evidence_ids
    assert "evd_legal_charter_09" in evidence_ids

    assert imp.source_trace["event_id"] == "evt_trace_01"
    assert imp.source_trace["event_source"] == "audit.contract_scanner"
    assert imp.source_trace["memory_id"] == "mem_risk_liability_cap"
    assert imp.source_trace["memory_source"] == "legal.compliance_charter"


# -----------------------------------------------------------------------------
# 11. RECUPERAÇÃO SEGURA APÓS FALHA (CRASH RECOVERY & WORKER RETRY)
# -----------------------------------------------------------------------------
def test_11_safe_recovery_reprocesses_failed_or_pending_event(
    config: SNCConfig,
) -> None:
    storage = SQLiteStorage(config.db_path)
    _seed_three_scenario_memories(storage)

    class _FlakyEvaluator:
        def __init__(self, delegate: MockEvaluator) -> None:
            self.delegate = delegate
            self.should_fail = True

        def evaluate(
            self,
            observation: ObservationDecision,
            activated_memories: list[ Any],
        ) -> list[Candidate]:
            if self.should_fail:
                raise RuntimeError("Falha transitória simulada durante avaliação")
            return self.delegate.evaluate(observation, activated_memories)

    flaky_eval = _FlakyEvaluator(MockEvaluator())
    pipe = CognitivePipeline(config, storage=storage, evaluator=flaky_eval)

    ev = Event(
        event_id="evt_recover_01",
        tenant_id="tenant_corp",
        event_type="deployment.pipeline_started",
        source="ci.runner",
        payload={"service": "payment-gateway", "action": "deploy", "team": "sre"},
    )

    # 1ª tentativa falha durante a avaliação
    with pytest.raises(RuntimeError, match="Falha transitória simulada"):
        pipe.ingest_and_process(ev)

    failed_ev = storage.get_event("tenant_corp", "evt_recover_01")
    assert failed_ev is not None
    assert failed_ev.processing_status == EventProcessingStatus.FAILED

    failed_outbox = storage.list_outbox_entries(
        "tenant_corp",
        status=OutboxStatus.FAILED,
        topic=OutboxTopic.EVENT_INGESTED,
    )
    assert len(failed_outbox) == 1
    assert "Falha transitória simulada" in (failed_outbox[0].last_error or "")

    # Corrige a falha transitória e reprocessa via worker (`process_pending_events(include_failed=True)`)
    flaky_eval.should_fail = False
    recovered_results = pipe.process_pending_events("tenant_corp", include_failed=True)

    assert len(recovered_results) == 1
    rec_res = recovered_results[0]
    assert rec_res.event.processing_status == EventProcessingStatus.PROCESSED
    assert len(rec_res.published_impulses) == 1
    assert rec_res.published_impulses[0].autonomy_level == AutonomyLevel.A1

    # Nova chamada a retry_event em evento já PROCESSED é idempotente
    retry_again = pipe.retry_event("tenant_corp", "evt_recover_01")
    assert retry_again.deduplicated is True
    assert len(storage.list_impulses("tenant_corp")) == 1


# -----------------------------------------------------------------------------
# 12. INTERVENÇÃO A3 (SOLICITAÇÃO DE PAUSA), CICLO DO BUS E EPISTEMOLOGIA DE OUTCOMES
# -----------------------------------------------------------------------------
def test_12_a3_intervention_requests_pause_without_claiming_block_and_outcome_epistemology(
    pipeline: CognitivePipeline,
) -> None:
    _seed_three_scenario_memories(pipeline.storage)

    ev = Event(
        event_id="evt_a3_pause_01",
        tenant_id="tenant_corp",
        event_type="contract.signing_queued",
        source="procurement.workflow",
        payload={"clause": "unlimited_liability", "action": "waive_liability_cap"},
    )
    res = pipeline.ingest_and_process(ev)
    assert len(res.published_impulses) == 1
    imp = res.published_impulses[0]

    # 12a) A3 registra solicitação de pausa, nunca bloqueio executado
    assert imp.autonomy_level == AutonomyLevel.A3
    assert imp.intervention_requested is True
    assert imp.protective_block is False
    assert imp.external_action_allowed is False
    assert imp.source_trace["intervention_mode"] == "pause_request_pending_agent_confirmation"
    assert imp.source_trace["execution_halted"] is False
    assert "NÃO foi interrompida autonomamente" in imp.message

    # 12b) Ciclo de vida do ImpulseBus: PUBLISHED -> DELIVERED -> ACKNOWLEDGED
    received_by_subscriber: list[str] = []
    assert isinstance(pipeline.impulse_bus, SQLiteImpulseBus)
    pipeline.impulse_bus.subscribe(lambda i: received_by_subscriber.append(i.impulse_id))

    delivered = pipeline.impulse_bus.deliver_pending("tenant_corp")
    assert len(delivered) == 1
    assert delivered[0].status == ImpulseStatus.DELIVERED
    assert received_by_subscriber == [imp.impulse_id]

    acked = pipeline.impulse_bus.acknowledge("tenant_corp", imp.impulse_id)
    assert acked.status == ImpulseStatus.ACKNOWLEDGED

    # 12c) OutcomeTracker: aceitação pelo agente NÃO implica verified_correct=True
    unverified_acceptance = pipeline.record_outcome(
        Outcome(
            outcome_id="out_accept_unverified",
            impulse_id=imp.impulse_id,
            tenant_id="tenant_corp",
            result=OutcomeResult.ACCEPTED,
            notes="Agente principal aceitou o alerta e pausou o fluxo para revisão humana.",
        )
    )
    assert unverified_acceptance.result == OutcomeResult.ACCEPTED
    assert unverified_acceptance.verified_correct is None
    assert (
        unverified_acceptance.context["epistemic_status"]
        == "unverified_agent_feedback_only"
    )

    # Tentativa de afirmar verified_correct=True sem observed_outcome ou evidence é rejeitada
    with pytest.raises(OutcomeVerificationError, match="não é prova de correção"):
        pipeline.record_outcome(
            Outcome(
                outcome_id="out_invalid_claim",
                impulse_id=imp.impulse_id,
                tenant_id="tenant_corp",
                result=OutcomeResult.ACCEPTED,
                verified_correct=True,
            )
        )

    # Com evidência verificável e desfecho observado, verified_correct=True é aceito
    verified_outcome = pipeline.record_outcome(
        Outcome(
            outcome_id="out_verified_true",
            impulse_id=imp.impulse_id,
            tenant_id="tenant_corp",
            result=OutcomeResult.RESOLVED,
            observed_outcome="Jurídico confirmou violação no rascunho e renegociou cláusula com teto de 12 meses.",
            verified_correct=True,
            evidence=[
                EvidenceRef(
                    evidence_id="evd_legal_revision_ok",
                    source_type="legal_audit_log",
                    reference="legal://reviews/2026-draft-55-rev2",
                    excerpt="Cláusula 14 retificada para teto de 12 meses.",
                    weight=1.0,
                )
            ],
        )
    )
    assert verified_outcome.verified_correct is True
    assert verified_outcome.context["epistemic_status"] == "verified"
