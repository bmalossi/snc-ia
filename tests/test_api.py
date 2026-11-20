"""Testes da API FastAPI e Worker independente da Etapa 3 (`tests/test_api.py`)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pic_snc.app import create_app
from pic_snc.config import (
    ConfigurationError,
    EnvironmentMode,
    EvaluatorMode,
    SNCConfig,
)
from pic_snc.scenarios import seed_demo_memories
from pic_snc.schemas import (
    AutonomyLevel,
    CognitiveKind,
    EventProcessingStatus,
    ImpulseStatus,
    Memory,
)
from pic_snc.worker import CognitiveWorker


@pytest.fixture()
def api_env(tmp_path: Path) -> tuple[SNCConfig, TestClient, CognitiveWorker]:
    cfg = SNCConfig(
        db_path=tmp_path / "api_test.db",
        evaluator_mode=EvaluatorMode.MOCK,
        enable_a4=False,
        environment=EnvironmentMode.TEST,
        allow_dev_identity_headers=True,
        static_api_tokens={
            "tok_alpha_secret": "tenant_alpha:agent_alpha",
            "tok_beta_secret": "tenant_beta:agent_beta",
        },
    )
    app = create_app(cfg)
    worker = CognitiveWorker(cfg)
    seed_demo_memories(worker.storage, tenant_id="tenant_alpha")
    client = TestClient(app)
    return cfg, client, worker


def test_1_health_and_readiness(api_env: tuple[SNCConfig, TestClient, CognitiveWorker]) -> None:
    _, client, _ = api_env

    health = client.get("/health")
    assert health.status_code == 200
    h_data = health.json()
    assert h_data["status"] == "ok"
    assert h_data["version"] == "0.1.0"
    assert h_data["evaluator_mode"] == "mock"
    assert h_data["a4_enabled"] is False
    assert h_data["external_action_policy"] == "deny"
    assert h_data["database_ready"] is True

    ready = client.get("/health/ready")
    assert ready.status_code == 200
    r_data = ready.json()
    assert r_data["status"] == "ready"
    assert r_data["database_ok"] is True


def test_2_post_events_queues_asynchronously_without_sync_reasoning_and_deduplicates(
    api_env: tuple[SNCConfig, TestClient, CognitiveWorker],
) -> None:
    _, client, worker = api_env
    headers = {"X-Dev-Tenant-Id": "tenant_alpha", "X-Dev-Principal-Id": "agent_a"}

    payload = {
        "event_id": "evt_api_queue_1",
        "tenant_id": "tenant_alpha",
        "event_type": "deployment.pipeline_started",
        "source": "ci.github_actions",
        "dedupe_key": "deploy-api-1",
        "payload": {"service": "payment-gateway", "action": "deploy", "window": "weekly_release"},
    }

    resp = client.post("/events", json=payload, headers=headers)
    assert resp.status_code == 202
    data = resp.json()
    assert data["status"] == "queued"
    assert data["deduplicated"] is False
    assert data["processing_status"] == EventProcessingStatus.PENDING.value
    assert data["outbox_id"] is not None

    # Antes do worker rodar, nenhum impulso foi gerado sincronamente
    imp_before = client.get("/impulses", headers=headers)
    assert imp_before.status_code == 200
    assert imp_before.json()["count"] == 0

    # Reenvio idempotente (mesmo event_id e depois mesma dedupe_key)
    dup1 = client.post("/events", json=payload, headers=headers)
    assert dup1.status_code == 202
    assert dup1.json()["status"] == "duplicate_ignored"
    assert dup1.json()["deduplicated"] is True

    dup2 = client.post(
        "/events",
        json={**payload, "event_id": "evt_api_queue_2"},
        headers=headers,
    )
    assert dup2.status_code == 202
    assert dup2.json()["status"] == "duplicate_ignored"
    assert dup2.json()["deduplicated"] is True
    assert dup2.json()["event_id"] == "evt_api_queue_1"

    # Worker independente processa apenas 1 evento e gera 1 impulso A1
    results = worker.run_once(tenant_id="tenant_alpha")
    assert len(results) == 1
    assert len(results[0].published_impulses) == 1


def test_3_worker_generates_a1_a2_a3_with_pagination_filter_and_lifecycle_ack_apply(
    api_env: tuple[SNCConfig, TestClient, CognitiveWorker],
) -> None:
    _, client, worker = api_env
    headers = {"Authorization": "Bearer tok_alpha_secret"}

    events = [
        {
            "event_id": "evt_w_a1",
            "tenant_id": "tenant_alpha",
            "event_type": "deployment.pipeline_started",
            "source": "ci.runner",
            "payload": {"service": "payment-gateway", "action": "deploy", "window": "weekly_release"},
        },
        {
            "event_id": "evt_w_a2",
            "tenant_id": "tenant_alpha",
            "event_type": "telemetry.quota_utilization_high",
            "source": "metering.gateway",
            "payload": {"metric": "usage_85_percent", "signal": "api_usage_spike", "trend": "expansion"},
        },
        {
            "event_id": "evt_w_a3",
            "tenant_id": "tenant_alpha",
            "event_type": "contract.signing_prepared",
            "source": "procurement.agent",
            "payload": {"clause_detected": "unlimited_liability", "action": "waive_liability_cap"},
        },
    ]
    for ev in events:
        r = client.post("/events", json=ev, headers=headers)
        assert r.status_code == 202

    # Worker processa todos os eventos pendentes
    worker_results = worker.run_once(tenant_id="tenant_alpha")
    assert len(worker_results) == 3

    # Paginação com limit=2 e cursor
    page1 = client.get("/impulses", params={"limit": 2, "status": "published"}, headers=headers)
    assert page1.status_code == 200
    p1_data = page1.json()
    assert p1_data["count"] == 2
    assert p1_data["next_cursor"] is not None

    page2 = client.get(
        "/impulses",
        params={"limit": 2, "status": "published", "cursor": p1_data["next_cursor"]},
        headers=headers,
    )
    assert page2.status_code == 200
    p2_data = page2.json()
    assert p2_data["count"] >= 1

    all_items = [*p1_data["items"], *p2_data["items"]]
    levels = {item["autonomy_level"] for item in all_items}
    assert {AutonomyLevel.A1.value, AutonomyLevel.A2.value, AutonomyLevel.A3.value}.issubset(levels)
    assert AutonomyLevel.A4.value not in levels

    # Ciclo de estado explícito no impulso A3: published -> delivered -> acknowledged -> applied
    a3_item = next(item for item in all_items if item["autonomy_level"] == AutonomyLevel.A3.value)
    a3_id = a3_item["impulse_id"]
    assert a3_item["intervention_requested"] is True
    assert a3_item["intervention_applied"] is False

    delivered_page = client.get(
        "/impulses",
        params={"limit": 10, "mark_delivered": "true"},
        headers=headers,
    )
    assert delivered_page.status_code == 200
    delivered_a3 = next(i for i in delivered_page.json()["items"] if i["impulse_id"] == a3_id)
    assert delivered_a3["status"] == ImpulseStatus.DELIVERED.value

    ack_resp = client.post(
        f"/impulses/{a3_id}/ack",
        json={"notes": "Recebido antes da assinatura."},
        headers=headers,
    )
    assert ack_resp.status_code == 200
    assert ack_resp.json()["status"] == ImpulseStatus.ACKNOWLEDGED.value
    assert ack_resp.json()["intervention_applied"] is False

    apply_resp = client.post(
        f"/impulses/{a3_id}/apply",
        json={
            "pause_respected": True,
            "action_taken": "paused_contract_signature",
            "notes": "Assinatura suspensa para revisão jurídica.",
            "record_outcome": True,
        },
        headers=headers,
    )
    assert apply_resp.status_code == 200
    applied_data = apply_resp.json()
    assert applied_data["status"] == ImpulseStatus.APPLIED.value
    assert applied_data["intervention_applied"] is True

    # Verifica que o Outcome foi registrado sem presumir verified_correct=True
    outcomes = worker.storage.list_outcomes("tenant_alpha", impulse_id=a3_id)
    assert len(outcomes) == 1
    assert outcomes[0].verified_correct is None


def test_4_security_auth_multi_tenant_isolation_and_production_mode(
    api_env: tuple[SNCConfig, TestClient, CognitiveWorker],
    tmp_path: Path,
) -> None:
    _, client, worker = api_env

    # 1. Sem credenciais -> 401
    assert client.get("/impulses").status_code == 401
    assert client.post("/events", json={"tenant_id": "tenant_alpha", "event_type": "x", "source": "s"}).status_code == 401

    # 2. Token inválido -> 401
    assert (
        client.get("/impulses", headers={"Authorization": "Bearer invalid_tok"}).status_code
        == 401
    )

    # 3. tenant_beta tentando enviar evento como tenant_alpha -> 403 Forbidden
    beta_headers = {"Authorization": "Bearer tok_beta_secret"}
    resp_spoof = client.post(
        "/events",
        json={
            "event_id": "evt_spoof_1",
            "tenant_id": "tenant_alpha",
            "event_type": "deployment.pipeline_started",
            "source": "ci.runner",
        },
        headers=beta_headers,
    )
    assert resp_spoof.status_code == 403

    # 4. tenant_beta tentando consultar impulsos de tenant_alpha via query param -> 403 Forbidden
    resp_query_spoof = client.get(
        "/impulses",
        params={"tenant_id": "tenant_alpha"},
        headers=beta_headers,
    )
    assert resp_query_spoof.status_code == 403

    # 5. Cria impulso real para tenant_alpha
    alpha_headers = {"Authorization": "Bearer tok_alpha_secret"}
    client.post(
        "/events",
        json={
            "event_id": "evt_alpha_only",
            "tenant_id": "tenant_alpha",
            "event_type": "contract.signing_prepared",
            "source": "procurement.agent",
            "payload": {"clause_detected": "unlimited_liability"},
        },
        headers=alpha_headers,
    )
    worker.run_once(tenant_id="tenant_alpha")
    alpha_impulses = client.get("/impulses", headers=alpha_headers).json()["items"]
    assert len(alpha_impulses) == 1
    alpha_imp_id = alpha_impulses[0]["impulse_id"]

    # tenant_beta não vê impulso de tenant_alpha e recebe 404 ao tentar ack/apply
    assert client.get("/impulses", headers=beta_headers).json()["count"] == 0
    assert client.post(f"/impulses/{alpha_imp_id}/ack", headers=beta_headers).status_code == 404
    assert (
        client.post(
            f"/impulses/{alpha_imp_id}/apply",
            json={"pause_respected": True},
            headers=beta_headers,
        ).status_code
        == 404
    )

    # 6. Em produção, allow_dev_identity_headers=True é bloqueado e X-Dev-Tenant-Id é rejeitado
    with pytest.raises(ConfigurationError, match="allow_dev_identity_headers"):
        SNCConfig(
            db_path=tmp_path / "prod_bad.db",
            environment=EnvironmentMode.PRODUCTION,
            allow_dev_identity_headers=True,
        )

    prod_cfg = SNCConfig(
        db_path=tmp_path / "prod_ok.db",
        environment=EnvironmentMode.PRODUCTION,
        allow_dev_identity_headers=False,
        static_api_tokens={"prod_tok_a": "tenant_a:svc_a"},
    )
    prod_client = TestClient(create_app(prod_cfg))
    assert prod_client.get("/health").json()["dev_identity_enabled"] is False
    assert (
        prod_client.get("/impulses", headers={"X-Dev-Tenant-Id": "tenant_a"}).status_code
        == 401
    )
    assert (
        prod_client.get("/impulses", headers={"Authorization": "Bearer prod_tok_a"}).status_code
        == 200
    )


def test_5_works_without_any_external_api_key_in_mock_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    cfg = SNCConfig(
        db_path=tmp_path / "no_key_mock.db",
        evaluator_mode=EvaluatorMode.MOCK,
    )
    app = create_app(cfg)
    worker = CognitiveWorker(cfg)
    worker.storage.save_memory(
        Memory(
            memory_id="mem_no_key",
            tenant_id="tenant_local",
            kind=CognitiveKind.RECALL,
            content="Checklist local sem chave externa.",
            source="local.runbook",
            salience=0.85,
            tags=["deploy", "local"],
        )
    )

    client = TestClient(app)
    headers = {"X-Dev-Tenant-Id": "tenant_local"}
    r = client.post(
        "/events",
        json={
            "event_id": "evt_no_key_1",
            "tenant_id": "tenant_local",
            "event_type": "deploy.local",
            "source": "cli.local",
            "payload": {"tag": "deploy", "env": "local"},
        },
        headers=headers,
    )
    assert r.status_code == 202
    worker.run_once(tenant_id="tenant_local")

    impulses = client.get("/impulses", headers=headers).json()["items"]
    assert len(impulses) == 1
    assert impulses[0]["autonomy_level"] == "A1"
