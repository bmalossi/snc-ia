"""Testes ponta a ponta dos cenários, Agente Consumidor e recuperação/persistência da Etapa 3 (`tests/test_scenarios.py`)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from pic_snc.app import create_app
from pic_snc.config import EnvironmentMode, EvaluatorMode, SNCConfig
from pic_snc.evaluator import EvaluatorExecutionError, JevEvaluatorAdapter
from pic_snc.observer import ObserverPolicy, ProgressiveCognitiveObserver
from pic_snc.pipeline import CognitivePipeline
from pic_snc.scenarios import (
    DemoConsumerAgent,
    run_all_demo_scenarios,
    run_scenario_a_recall,
    run_scenario_b_opportunity,
    run_scenario_c_preventive_intervention,
    seed_demo_memories,
)
from pic_snc.schemas import (
    AutonomyLevel,
    CognitiveKind,
    Event,
    EventCompleteness,
    EventProcessingStatus,
    ImpulseStatus,
)
from pic_snc.worker import CognitiveWorker


def test_1_three_end_to_end_scenarios_and_consumer_isolation(tmp_path: Path) -> None:
    db_path = tmp_path / "scenarios_e2e.db"
    reports = run_all_demo_scenarios(db_path=db_path)

    # Verifica que DemoConsumerAgent não possui métodos de busca explícita na memória
    assert not hasattr(DemoConsumerAgent, "search_memory")
    assert not hasattr(DemoConsumerAgent, "retrieve")

    # Cenário A — Recordação espontânea (A1)
    rep_a = reports["scenario_a"]
    assert rep_a.ingest_response["status"] == "queued"
    assert rep_a.worker_processed_events == 1
    assert len(rep_a.acknowledged_impulses) == 1
    imp_a = rep_a.acknowledged_impulses[0]
    assert imp_a.kind == CognitiveKind.RECALL
    assert imp_a.autonomy_level == AutonomyLevel.A1
    assert rep_a.action_report.action_executed is True
    assert rep_a.action_report.paused_by_a3 is False
    assert rep_a.action_report.applied_impulses[0].status == ImpulseStatus.APPLIED

    # Cenário B — Oportunidade antecipada (A2) com memória relacionada
    rep_b = reports["scenario_b"]
    assert rep_b.worker_processed_events == 1
    opp_imp = next(
        i for i in rep_b.acknowledged_impulses if i.kind == CognitiveKind.OPPORTUNITY
    )
    assert opp_imp.autonomy_level == AutonomyLevel.A2
    assert "mem_demo_opp_quota_expansion" in opp_imp.memory_ids
    assert "mem_demo_opp_enterprise_catalog" in opp_imp.memory_ids
    assert rep_b.action_report.action_executed is True
    assert rep_b.action_report.paused_by_a3 is False

    # Cenário C — Intervenção preventiva (A3): consumidor recebe, reconhece e respeita pausa antes da ação
    rep_c = reports["scenario_c"]
    assert rep_c.worker_processed_events == 1
    risk_imp = next(i for i in rep_c.acknowledged_impulses if i.kind == CognitiveKind.RISK)
    assert risk_imp.autonomy_level == AutonomyLevel.A3
    assert risk_imp.intervention_requested is True
    assert risk_imp.protective_block is False

    assert rep_c.action_report.action_executed is False
    assert rep_c.action_report.paused_by_a3 is True
    applied_c = rep_c.action_report.applied_impulses[0]
    assert applied_c.status == ImpulseStatus.APPLIED
    assert applied_c.intervention_applied is True


def test_2_worker_restart_between_partial_fragments_reconstructs_observer_buffer(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "observer_restart.db"
    cfg = SNCConfig(
        db_path=db_path,
        evaluator_mode=EvaluatorMode.MOCK,
        environment=EnvironmentMode.TEST,
    )
    app = create_app(cfg)
    worker_1 = CognitiveWorker(cfg)
    seed_demo_memories(worker_1.storage, tenant_id="tenant_demo")

    client = TestClient(app)
    headers = {"X-Dev-Tenant-Id": "tenant_demo", "X-Dev-Principal-Id": "agent_stream"}
    t0 = datetime(2026, 11, 20, 15, 0, 0, tzinfo=UTC)

    # Fragmento 1 parcial (baixa densidade 0.25) -> worker_1 coloca em BUFFERED sem avaliar
    r1 = client.post(
        "/events",
        json={
            "event_id": "evt_restart_f1",
            "tenant_id": "tenant_demo",
            "event_type": "conversation.chunk",
            "source": "agent.stream",
            "completeness": "partial",
            "stream_id": "stream_restart_1",
            "sequence_no": 1,
            "semantic_density": 0.25,
            "payload": {"fragment": "Preparando janela weekly_release"},
            "timestamp": t0.isoformat(),
        },
        headers=headers,
    )
    assert r1.status_code == 202
    res_w1 = worker_1.run_once(tenant_id="tenant_demo")
    assert len(res_w1) == 1
    assert res_w1[0].observation is not None
    assert res_w1[0].observation.should_activate is False
    assert res_w1[0].event.processing_status == EventProcessingStatus.BUFFERED

    # Simula reinicialização completa do processo worker (nova instância com memória limpa)
    worker_2 = CognitiveWorker(cfg)

    # Fragmento 2 (baixa densidade individual 0.45, mas somado ao fragmento 1 dá 0.70 >= 0.65)
    r2 = client.post(
        "/events",
        json={
            "event_id": "evt_restart_f2",
            "tenant_id": "tenant_demo",
            "event_type": "conversation.chunk",
            "source": "agent.stream",
            "completeness": "partial",
            "stream_id": "stream_restart_1",
            "sequence_no": 2,
            "semantic_density": 0.45,
            "payload": {"fragment": "para deploy do serviço payment-gateway com sre"},
            "timestamp": (t0 + timedelta(seconds=5)).isoformat(),
        },
        headers=headers,
    )
    assert r2.status_code == 202

    res_w2 = worker_2.run_once(tenant_id="tenant_demo")
    assert len(res_w2) == 1
    obs2 = res_w2[0].observation
    assert obs2 is not None
    assert obs2.should_activate is True
    assert obs2.accumulated_event_ids == ["evt_restart_f1", "evt_restart_f2"]
    assert len(res_w2[0].published_impulses) == 1
    assert res_w2[0].published_impulses[0].autonomy_level == AutonomyLevel.A1

    # Ambos os fragmentos ficam marcados como PROCESSED no banco
    ev1_after = worker_2.storage.get_event("tenant_demo", "evt_restart_f1")
    ev2_after = worker_2.storage.get_event("tenant_demo", "evt_restart_f2")
    assert ev1_after is not None and ev1_after.processing_status == EventProcessingStatus.PROCESSED
    assert ev2_after is not None and ev2_after.processing_status == EventProcessingStatus.PROCESSED


def test_3_expired_observer_buffer_after_restart_is_skipped_without_false_association(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "observer_expired.db"
    cfg = SNCConfig(
        db_path=db_path,
        evaluator_mode=EvaluatorMode.MOCK,
        environment=EnvironmentMode.TEST,
    )
    short_ttl_policy = ObserverPolicy(buffer_ttl_seconds=30)

    # Worker 1 recebe fragmento parcial mencionando "unlimited_liability" se estivesse ativo,
    # ou melhor: fragmento antigo mencionando "payment-gateway" que NÃO pode vazar 10 minutos depois
    pipe_1 = CognitivePipeline(cfg)
    pipe_1.observer = ProgressiveCognitiveObserver(short_ttl_policy, storage=pipe_1.storage)
    worker_1 = CognitiveWorker(cfg, pipeline=pipe_1)
    seed_demo_memories(worker_1.storage, tenant_id="tenant_demo")

    t0 = datetime(2026, 11, 20, 15, 0, 0, tzinfo=UTC)
    pipe_1.submit_event(
        Event(
            event_id="evt_stale_f1",
            tenant_id="tenant_demo",
            event_type="conversation.chunk",
            source="agent.stream",
            completeness=EventCompleteness.PARTIAL,
            stream_id="stream_ttl_1",
            sequence_no=1,
            semantic_density=0.25,
            payload={"fragment": "deploy do serviço payment-gateway"},
            timestamp=t0,
        )
    )
    worker_1.run_once(tenant_id="tenant_demo")

    # Worker 2 inicia 10 minutos depois (TTL de 30s expirado) com fragmento neutro
    pipe_2 = CognitivePipeline(cfg)
    pipe_2.observer = ProgressiveCognitiveObserver(short_ttl_policy, storage=pipe_2.storage)
    worker_2 = CognitiveWorker(cfg, pipeline=pipe_2)

    pipe_2.submit_event(
        Event(
            event_id="evt_fresh_f2",
            tenant_id="tenant_demo",
            event_type="conversation.chunk",
            source="agent.stream",
            completeness=EventCompleteness.COMPLETE,
            stream_id="stream_ttl_1",
            sequence_no=2,
            semantic_density=0.50,
            payload={"fragment": "ajuste visual de tema claro no editor"},
            timestamp=t0 + timedelta(minutes=10),
        )
    )
    res_w2 = worker_2.run_once(tenant_id="tenant_demo")
    assert len(res_w2) == 1
    assert res_w2[0].observation is not None
    # Não juntou o fragmento antigo expirado
    assert res_w2[0].observation.accumulated_event_ids == ["evt_fresh_f2"]
    assert res_w2[0].published_impulses == []

    stale_ev = worker_2.storage.get_event("tenant_demo", "evt_stale_f1")
    assert stale_ev is not None
    assert stale_ev.processing_status == EventProcessingStatus.SKIPPED


def test_4_worker_recovery_of_interrupted_processing_events_and_jev_no_fallback(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "worker_recover.db"
    cfg = SNCConfig(
        db_path=db_path,
        evaluator_mode=EvaluatorMode.MOCK,
        environment=EnvironmentMode.TEST,
    )
    worker = CognitiveWorker(cfg)
    seed_demo_memories(worker.storage, tenant_id="tenant_demo")

    # Simula evento que ficou preso em 'processing' após queda abrupta do worker anterior
    ev = Event(
        event_id="evt_stuck_proc",
        tenant_id="tenant_demo",
        event_type="deployment.pipeline_started",
        source="ci.runner",
        payload={"service": "payment-gateway", "action": "deploy", "window": "weekly_release"},
    )
    worker.storage.ingest_event(ev, min_eval_density=0.0)
    worker.storage.update_event_status("tenant_demo", "evt_stuck_proc", EventProcessingStatus.PROCESSING)

    recovered = worker.recover_on_startup(tenant_id="tenant_demo")
    assert recovered == 1

    res = worker.run_once(tenant_id="tenant_demo")
    assert len(res) == 1
    assert res[0].event.processing_status == EventProcessingStatus.PROCESSED
    assert len(res[0].published_impulses) == 1

    # Valida que falha no JevEvaluatorAdapter via Worker NÃO cai silenciosamente para MockEvaluator
    jev_cfg = SNCConfig(
        db_path=tmp_path / "worker_jev_fail.db",
        evaluator_mode=EvaluatorMode.JEV,
        typesafe_api_key="ts_test_key_no_fallback",
    )

    class _DownJevClient:
        def systemone(self, *, model: str, state: Any, questions: Any) -> Any:
            raise TimeoutError("TypeSafe endpoint unreachable")

    jev_adapter = JevEvaluatorAdapter(jev_cfg, sdk_client=_DownJevClient())
    jev_pipe = CognitivePipeline(jev_cfg, evaluator=jev_adapter)
    seed_demo_memories(jev_pipe.storage, tenant_id="tenant_demo")
    jev_worker = CognitiveWorker(jev_cfg, pipeline=jev_pipe, raise_on_error=True)

    jev_pipe.submit_event(
        Event(
            event_id="evt_jev_err_1",
            tenant_id="tenant_demo",
            event_type="contract.signing_prepared",
            source="procurement.agent",
            payload={"clause_detected": "unlimited_liability"},
        )
    )

    with pytest.raises(EvaluatorExecutionError, match="Falha na chamada ts.systemone"):
        jev_worker.run_once(tenant_id="tenant_demo")

    failed_ev = jev_worker.storage.get_event("tenant_demo", "evt_jev_err_1")
    assert failed_ev is not None
    assert failed_ev.processing_status == EventProcessingStatus.FAILED
    assert jev_worker.storage.list_impulses("tenant_demo") == []


def test_5_individual_scenario_helpers_and_multi_tenant_separation(tmp_path: Path) -> None:
    cfg = SNCConfig(
        db_path=tmp_path / "scenarios_sep.db",
        evaluator_mode=EvaluatorMode.MOCK,
        environment=EnvironmentMode.TEST,
    )
    app = create_app(cfg)
    worker = CognitiveWorker(cfg)
    seed_demo_memories(worker.storage, tenant_id="tenant_one")
    client = TestClient(app)

    rep_a = run_scenario_a_recall(client, worker, tenant_id="tenant_one")
    rep_b = run_scenario_b_opportunity(client, worker, tenant_id="tenant_one")
    rep_c = run_scenario_c_preventive_intervention(client, worker, tenant_id="tenant_one")

    assert rep_a.acknowledged_impulses[0].autonomy_level == AutonomyLevel.A1
    assert rep_b.acknowledged_impulses[0].autonomy_level == AutonomyLevel.A2
    assert rep_c.acknowledged_impulses[0].autonomy_level == AutonomyLevel.A3
    assert rep_c.action_report.paused_by_a3 is True

    # Outro tenant sem memórias próprias não recebe impulsos de tenant_one
    consumer_two = DemoConsumerAgent(client, tenant_id="tenant_two")
    consumer_two.emit_event(
        {
            "event_id": "evt_t2_risk",
            "event_type": "contract.signing_prepared",
            "source": "procurement.agent",
            "payload": {"clause_detected": "unlimited_liability"},
        }
    )
    worker.run_once(tenant_id="tenant_two")
    assert consumer_two.poll_and_acknowledge_impulses() == []
