"""Cenários demonstrativos e Agente Consumidor desacoplado do PIC-SNC v0.1.

Demonstra ponta a ponta os três cenários obrigatórios sem NENHUMA busca de memória
por parte do agente principal:
- Cenário A: Recordação espontânea (A1)
- Cenário B: Oportunidade antecipada com memórias relacionadas (A2)
- Cenário C: Intervenção preventiva com solicitação de pausa (A3) e confirmação pelo consumidor

Separação de processos:
- `DemoConsumerAgent`: conversa exclusivamente com a API HTTP (`POST /events`, `GET /impulses`,
  `POST /impulses/{id}/ack`, `POST /impulses/{id}/apply`). Nunca executa `search_memory` ou `retrieve`.
- `CognitiveWorker`: processa eventos enfileirados em segundo plano de forma independente.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from fastapi.testclient import TestClient

from pic_snc.app import create_app
from pic_snc.config import EvaluatorMode, SNCConfig
from pic_snc.schemas import (
    AssociativeCues,
    AutonomyLevel,
    CognitiveKind,
    EvidenceRef,
    Impulse,
    ImpulseStatus,
    Memory,
    MemoryRelation,
    PredictiveProfile,
    TemporalProfile,
)
from pic_snc.storage import SQLiteStorage
from pic_snc.worker import CognitiveWorker


class HttpClientProtocol(Protocol):
    """Protocolo mínimo compatível com `fastapi.testclient.TestClient` e `httpx.Client`."""

    def post(self, url: str, **kwargs: Any) -> Any: ...
    def get(self, url: str, **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class ConsumerActionReport:
    """Relatório da decisão tomada pelo agente consumidor frente aos impulsos."""

    planned_action: str
    action_executed: bool
    paused_by_a3: bool
    applied_impulses: list[Impulse] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ScenarioReport:
    """Relatório ponta a ponta de execução de um cenário demonstrativo."""

    scenario_name: str
    tenant_id: str
    ingest_response: dict[str, Any]
    worker_processed_events: int
    acknowledged_impulses: list[Impulse]
    action_report: ConsumerActionReport


def seed_demo_memories(
    storage: SQLiteStorage,
    tenant_id: str = "tenant_demo",
) -> list[Memory]:
    """Semeia memórias determinísticas para os Cenários A (A1), B (A2) e C (A3)."""
    memories = [
        # Cenário A — Recordação espontânea (A1)
        Memory(
            memory_id="mem_demo_recall_sre",
            tenant_id=tenant_id,
            kind=CognitiveKind.RECALL,
            content=(
                "Política de release: deploys do serviço payment-gateway exigem "
                "checklist de migração idempotente e aprovação de SRE."
            ),
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
                    evidence_id="evd_demo_runbook_sre",
                    source_type="policy_document",
                    reference="doc://runbooks/sre-release-v3#section-4",
                    excerpt="Deploys em payment-gateway requerem checklist idempotente e aprovação de SRE.",
                    weight=0.92,
                )
            ],
        ),
        # Cenário B — Oportunidade antecipada (A2) + memória relacionada
        Memory(
            memory_id="mem_demo_opp_enterprise_catalog",
            tenant_id=tenant_id,
            kind=CognitiveKind.OPPORTUNITY,
            content=(
                "Plano Enterprise anual oferece desconto de 20% para contas com "
                "uso sustentado acima de 80% da franquia."
            ),
            source="catalog.pricing_tiers",
            salience=0.76,
            tags=["pricing", "enterprise"],
        ),
        Memory(
            memory_id="mem_demo_opp_quota_expansion",
            tenant_id=tenant_id,
            kind=CognitiveKind.OPPORTUNITY,
            content=(
                "Contas acima de 85% da cota mensal de API com tendência de "
                "crescimento apresentam alta conversão para upgrade Enterprise."
            ),
            source="analytics.revenue_expansion",
            salience=0.91,
            tags=["quota", "expansion", "upgrade"],
            associative_cues=AssociativeCues(
                tags=["quota", "expansion", "upgrade"],
                semantic_keys=["api_usage_spike", "tier_capacity"],
                predictive=PredictiveProfile(
                    trigger_cues=["quota_utilization_high", "usage_85_percent"],
                    anticipated_outcome="enterprise_plan_expansion",
                    horizon_seconds=86400 * 14,
                    prior_probability=0.84,
                ),
                relations=[
                    MemoryRelation(
                        target_id="mem_demo_opp_enterprise_catalog",
                        relation_type="precedes",
                        weight=0.86,
                    )
                ],
            ),
            evidence=[
                EvidenceRef(
                    evidence_id="evd_demo_cohort_q3",
                    source_type="cohort_study",
                    reference="analytics://cohorts/2026-q3-expansion",
                    excerpt="Conversão de 78% para Enterprise ao atingir 85% da franquia mensal.",
                    weight=0.90,
                )
            ],
        ),
        # Cenário C — Intervenção preventiva (A3)
        Memory(
            memory_id="mem_demo_risk_liability_cap",
            tenant_id=tenant_id,
            kind=CognitiveKind.RISK,
            content=(
                "Restrição jurídica obrigatória: contratos com fornecedores nunca "
                "podem conter cláusula de responsabilidade ilimitada (unlimited_liability)."
            ),
            source="legal.compliance_charter",
            salience=0.97,
            tags=["contract", "compliance", "unlimited_liability", "clause"],
            associative_cues=AssociativeCues(
                tags=["contract", "compliance", "unlimited_liability", "clause"],
                semantic_keys=["legal_contradiction", "liability_cap_violation"],
                predictive=PredictiveProfile(
                    trigger_cues=["unlimited_liability", "waive_liability_cap"],
                    anticipated_outcome="regulatory_and_financial_exposure",
                    prior_probability=0.96,
                ),
            ),
            evidence=[
                EvidenceRef(
                    evidence_id="evd_demo_legal_art9",
                    source_type="compliance_rule",
                    reference="legal://charter/2026#art-9-liability",
                    excerpt="Proibida assinatura com unlimited_liability; teto limitado a 12 meses.",
                    weight=0.98,
                )
            ],
        ),
    ]
    for mem in memories:
        storage.save_memory(mem)
    return memories


class DemoConsumerAgent:
    """Agente consumidor mínimo completamente desacoplado do motor cognitivo.

    Não possui método de busca na memória; apenas emite eventos observados e reage
    a impulsos publicados espontaneamente pelo SNC-IA.
    """

    def __init__(
        self,
        http_client: HttpClientProtocol,
        *,
        tenant_id: str,
        principal_id: str = "demo_consumer_agent",
        bearer_token: str | None = None,
    ) -> None:
        self.http = http_client
        self.tenant_id = tenant_id
        self.principal_id = principal_id
        self._headers = (
            {"Authorization": f"Bearer {bearer_token}"}
            if bearer_token
            else {
                "X-Dev-Tenant-Id": tenant_id,
                "X-Dev-Principal-Id": principal_id,
            }
        )

    def emit_event(self, event_payload: dict[str, Any]) -> dict[str, Any]:
        """Envia um evento observado ao SNC-IA via POST /events (202 Accepted)."""
        body = {"tenant_id": self.tenant_id, **event_payload}
        resp = self.http.post("/events", json=body, headers=self._headers)
        resp.raise_for_status()
        return dict(resp.json())

    def poll_and_acknowledge_impulses(
        self,
        *,
        status_filter: ImpulseStatus | None = ImpulseStatus.PUBLISHED,
        limit: int = 20,
    ) -> list[Impulse]:
        """Consome impulsos publicados pela memória ativa e confirma recebimento (/ack)."""
        params: dict[str, Any] = {"limit": limit, "mark_delivered": "true"}
        if status_filter is not None:
            params["status"] = status_filter.value

        resp = self.http.get("/impulses", params=params, headers=self._headers)
        resp.raise_for_status()
        page = resp.json()

        acknowledged: list[Impulse] = []
        for raw_item in page["items"]:
            imp = Impulse.model_validate(raw_item)
            ack_resp = self.http.post(
                f"/impulses/{imp.impulse_id}/ack",
                json={"notes": f"Recebido pelo agente {self.principal_id}."},
                headers=self._headers,
            )
            ack_resp.raise_for_status()
            acknowledged.append(Impulse.model_validate(ack_resp.json()))
        return acknowledged

    def attempt_planned_action(
        self,
        planned_action: str,
        impulses: list[Impulse],
    ) -> ConsumerActionReport:
        """Simula a execução de uma ação planejada respeitando eventuais pausas A3."""
        a3_requests = [
            imp
            for imp in impulses
            if imp.autonomy_level == AutonomyLevel.A3 and imp.intervention_requested
        ]

        if a3_requests:
            applied: list[Impulse] = []
            for imp in impulses:
                is_pause_target = imp in a3_requests
                apply_resp = self.http.post(
                    f"/impulses/{imp.impulse_id}/apply",
                    json={
                        "pause_respected": True,
                        "action_taken": "paused_for_human_legal_review"
                        if is_pause_target
                        else "contextual_note_attached",
                        "notes": (
                            f"Agente respeitou solicitação A3 e pausou '{planned_action}'."
                            if is_pause_target
                            else "Contexto registrado antes da pausa."
                        ),
                        "record_outcome": True,
                    },
                    headers=self._headers,
                )
                apply_resp.raise_for_status()
                applied.append(Impulse.model_validate(apply_resp.json()))

            return ConsumerActionReport(
                planned_action=planned_action,
                action_executed=False,
                paused_by_a3=True,
                applied_impulses=applied,
            )

        applied = []
        for imp in impulses:
            apply_resp = self.http.post(
                f"/impulses/{imp.impulse_id}/apply",
                json={
                    "pause_respected": True,
                    "action_taken": f"executed:{planned_action}",
                    "notes": f"Ação '{planned_action}' executada considerando impulso {imp.autonomy_level.value}.",
                    "record_outcome": True,
                },
                headers=self._headers,
            )
            apply_resp.raise_for_status()
            applied.append(Impulse.model_validate(apply_resp.json()))

        return ConsumerActionReport(
            planned_action=planned_action,
            action_executed=True,
            paused_by_a3=False,
            applied_impulses=applied,
        )


def run_scenario_a_recall(
    http_client: HttpClientProtocol,
    worker: CognitiveWorker,
    *,
    tenant_id: str = "tenant_demo",
) -> ScenarioReport:
    """Cenário A — Recordação espontânea (A1)."""
    consumer = DemoConsumerAgent(http_client, tenant_id=tenant_id)
    ingest_res = consumer.emit_event(
        {
            "event_id": "evt_demo_a_recall",
            "event_type": "deployment.pipeline_started",
            "source": "ci.github_actions",
            "payload": {
                "service": "payment-gateway",
                "action": "deploy",
                "window": "weekly_release",
            },
            "evidence": [
                {
                    "evidence_id": "evd_demo_ci_101",
                    "source_type": "ci_pipeline",
                    "reference": "ci://runs/demo-101",
                    "excerpt": "Deploy do payment-gateway iniciado na janela semanal.",
                    "weight": 0.90,
                }
            ],
        }
    )

    processed = worker.run_once(tenant_id=tenant_id)
    acked = consumer.poll_and_acknowledge_impulses()
    action_report = consumer.attempt_planned_action("prepare_deploy_checklist", acked)

    return ScenarioReport(
        scenario_name="Cenário A — Recordação espontânea (A1)",
        tenant_id=tenant_id,
        ingest_response=ingest_res,
        worker_processed_events=len(processed),
        acknowledged_impulses=acked,
        action_report=action_report,
    )


def run_scenario_b_opportunity(
    http_client: HttpClientProtocol,
    worker: CognitiveWorker,
    *,
    tenant_id: str = "tenant_demo",
) -> ScenarioReport:
    """Cenário B — Oportunidade antecipada (A2)."""
    consumer = DemoConsumerAgent(http_client, tenant_id=tenant_id)
    ingest_res = consumer.emit_event(
        {
            "event_id": "evt_demo_b_opportunity",
            "event_type": "telemetry.quota_utilization_high",
            "source": "metering.api_gateway",
            "payload": {
                "account_id": "acc_demo_778",
                "metric": "usage_85_percent",
                "signal": "api_usage_spike",
                "trend": "expansion",
            },
        }
    )

    processed = worker.run_once(tenant_id=tenant_id)
    acked = consumer.poll_and_acknowledge_impulses()
    action_report = consumer.attempt_planned_action("notify_account_executive", acked)

    return ScenarioReport(
        scenario_name="Cenário B — Oportunidade antecipada (A2)",
        tenant_id=tenant_id,
        ingest_response=ingest_res,
        worker_processed_events=len(processed),
        acknowledged_impulses=acked,
        action_report=action_report,
    )


def run_scenario_c_preventive_intervention(
    http_client: HttpClientProtocol,
    worker: CognitiveWorker,
    *,
    tenant_id: str = "tenant_demo",
) -> ScenarioReport:
    """Cenário C — Intervenção preventiva com solicitação de pausa (A3)."""
    consumer = DemoConsumerAgent(http_client, tenant_id=tenant_id)
    ingest_res = consumer.emit_event(
        {
            "event_id": "evt_demo_c_risk",
            "event_type": "contract.signing_prepared",
            "source": "procurement.agent_observer",
            "payload": {
                "vendor": "Acme Cloud Ltd",
                "clause_detected": "unlimited_liability",
                "action": "waive_liability_cap",
            },
            "evidence": [
                {
                    "evidence_id": "evd_demo_draft_55",
                    "source_type": "contract_clause",
                    "reference": "doc://procurement/draft-55#clause-14",
                    "excerpt": "Rascunho contém unlimited_liability e waive_liability_cap.",
                    "weight": 0.97,
                }
            ],
        }
    )

    processed = worker.run_once(tenant_id=tenant_id)
    acked = consumer.poll_and_acknowledge_impulses()
    action_report = consumer.attempt_planned_action("auto_sign_vendor_contract", acked)

    return ScenarioReport(
        scenario_name="Cenário C — Intervenção preventiva (A3)",
        tenant_id=tenant_id,
        ingest_response=ingest_res,
        worker_processed_events=len(processed),
        acknowledged_impulses=acked,
        action_report=action_report,
    )


def run_all_demo_scenarios(db_path: Path | None = None) -> dict[str, ScenarioReport]:
    """Executa os três cenários ponta a ponta com API e Worker separados."""
    if db_path is None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            return _execute_scenarios_on_db(Path(tmp_dir) / "pic_snc_demo.db")
    return _execute_scenarios_on_db(db_path)


def _execute_scenarios_on_db(db_path: Path) -> dict[str, ScenarioReport]:
    cfg = SNCConfig(
        db_path=db_path,
        evaluator_mode=EvaluatorMode.MOCK,
        enable_a4=False,
    )
    app = create_app(cfg)
    worker = CognitiveWorker(cfg)
    seed_demo_memories(worker.storage, tenant_id="tenant_demo")

    with TestClient(app) as client:
        return {
            "scenario_a": run_scenario_a_recall(client, worker, tenant_id="tenant_demo"),
            "scenario_b": run_scenario_b_opportunity(client, worker, tenant_id="tenant_demo"),
            "scenario_c": run_scenario_c_preventive_intervention(
                client, worker, tenant_id="tenant_demo"
            ),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PIC-SNC v0.1 — Demonstração dos 3 cenários (A1, A2, A3)"
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Caminho opcional de SQLite para persistir a demonstração.",
    )
    args = parser.parse_args()

    reports = run_all_demo_scenarios(db_path=args.db_path)
    summary = {
        key: {
            "scenario": rep.scenario_name,
            "event_ingest_status": rep.ingest_response["status"],
            "worker_processed_events": rep.worker_processed_events,
            "impulses_acknowledged": [
                {
                    "impulse_id": imp.impulse_id,
                    "kind": imp.kind.value,
                    "autonomy_level": imp.autonomy_level.value,
                    "intervention_requested": imp.intervention_requested,
                    "status_after_ack": imp.status.value,
                }
                for imp in rep.acknowledged_impulses
            ],
            "consumer_action": {
                "planned_action": rep.action_report.planned_action,
                "action_executed": rep.action_report.action_executed,
                "paused_by_a3": rep.action_report.paused_by_a3,
                "applied_states": [
                    {
                        "impulse_id": imp.impulse_id,
                        "status": imp.status.value,
                        "intervention_applied": imp.intervention_applied,
                    }
                    for imp in rep.action_report.applied_impulses
                ],
            },
        }
        for key, rep in reports.items()
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
