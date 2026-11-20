# SNC-IA — PIC-SNC v0.1 (PoC Local)

**Sistema Nervoso Cognitivo para Agentes de IA** — Implementação local do protocolo **PIC-SNC v0.1** em Python 3.12.

---

## 1. Premissa e Arquitetura Resumida

O **SNC-IA** é uma camada cognitiva **ativa e independente** do agente principal:

- O agente principal **nunca** executa chamadas de busca (`search_memory()` ou `retrieve()`).
- O motor **OBSERVA** eventos semânticos (parciais ou completos), **ATIVA** memórias associadas, **AVALIA** candidatos, **GOVERNA** níveis de autoridade (`A0–A3`; `A4` desativado por padrão) e **PUBLICA** impulsos cognitivos espontaneamente via **SQLite + Outbox**.
- A ingestão HTTP (`POST /events`) é desacoplada do processamento cognitivo, que roda em um **worker independente** (`pic_snc.worker`).

```text
Agente / Ambiente ──(POST /events 202)──▶ API FastAPI (app.py) ──▶ SQLite + Outbox
                                                                          │
                                                Worker Independente (worker.py)
                                           Observe ─▶ Activate ─▶ Evaluate ─▶ Govern ─▶ Publish
                                                                          │
Agente Consumidor ◀──(GET /impulses, /ack, /apply)─── API FastAPI ◀───────┘
```

### Componentes Principais (`src/pic_snc/`)

- `schemas.py` — Contratos Pydantic v2 com distinção explícita de estágios (`Event`, `ObservationDecision`, `Memory`, `Candidate`, `Impulse`, `Outcome`, `OutboxEntry`, `AuthContext`, `EvidenceRef`).
- `storage.py` — SQLite + Outbox transacional (`WAL` mode), isolamento por `tenant_id` (chaves primárias e estrangeiras compostas), idempotência por `(tenant_id, event_id)` e `(tenant_id, dedupe_key)`, restrições `CHECK` no banco e paginação por cursor.
- `observer.py` — `ProgressiveCognitiveObserver` com buffer incremental por `(tenant_id, stream_id)`, reconstrução segura pós-reinicialização a partir de eventos `buffered` no SQLite e descarte de fragmentos expirados por TTL (`skipped`).
- `activation.py` — `AssociativeActivationEngine` multi-sinal (tags, `semantic_keys`, contexto, gatilhos preditivos, perfis temporais e propagação relacional).
- `evaluator.py` — `MockEvaluator` determinístico (sem chave externa) e `JevEvaluatorAdapter` preparado para `typesafe_sdk` (`ts.systemone(model="jev", ...)`), sem fallback silencioso.
- `governor.py` — `CognitiveGovernor` com níveis `A0–A3`, `A4` bloqueado por padrão (`enable_a4=False`), solicitação de pausa em `A3` (`intervention_requested=True`, `protective_block=False`) e cooldown contra repetição.
- `bus.py` — `SQLiteImpulseBus` com ciclo de vida explícito (`published` → `delivered` → `acknowledged` → `applied`).
- `outcome.py` — `SQLiteOutcomeTracker` com separação epistêmica entre aceitação do agente e correção verificada (`verified_correct`).
- `pipeline.py` — Orquestrador do fluxo completo com recuperação pós-falha.
- `app.py` — API REST FastAPI.
- `worker.py` — Trabalhador independente com polling, reivindicação atômica (`BEGIN IMMEDIATE`), recuperação na inicialização e parada graciosa.
- `scenarios.py` — `DemoConsumerAgent` e os três cenários determinísticos (Recordação A1, Oportunidade A2 e Intervenção Preventiva A3).

---

## 2. Requisitos e Instalação (Windows + PowerShell + Python 3.12)

Pré-requisito: **Python 3.12** instalado no Windows.

```powershell
# 1. Criar ambiente virtual com Python 3.12
py -3.12 -m venv .venv

# 2. Instalar o projeto e dependências de desenvolvimento
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

---

## 3. Configuração de Ambiente

Variáveis de ambiente suportadas (nenhuma chave é necessária no modo padrão `mock`):

| Variável | Padrão | Descrição |
| :--- | :--- | :--- |
| `PIC_SNC_DB_PATH` | `pic_snc.db` | Caminho do arquivo SQLite compartilhado entre API e worker. |
| `PIC_SNC_EVALUATOR_MODE` | `mock` | Modo do avaliador: `mock` (determinístico local) ou `jev`. |
| `TYPESAFE_API_KEY` | *(vazio)* | Obrigatória **apenas** quando `PIC_SNC_EVALUATOR_MODE=jev` (sem fallback silencioso). |
| `PIC_SNC_ENV` | `development` | Ambiente: `development`, `test` ou `production`. |
| `PIC_SNC_ALLOW_DEV_HEADERS` | `true` (dev/test) | Permite cabeçalho `X-Dev-Tenant-Id` localmente. **Proibido** em `production`. |
| `PIC_SNC_ENABLE_A4` | `false` | Mantém bloqueio protetivo `A4` desativado por padrão. |
| `PIC_SNC_EXTERNAL_ACTION_POLICY` | `deny` | Política separada para ações externas autônomas. |

Exemplo de configuração para execução local no PowerShell:

```powershell
$env:PIC_SNC_DB_PATH = "pic_snc.db"
$env:PIC_SNC_EVALUATOR_MODE = "mock"
$env:PIC_SNC_ENV = "development"
$env:PIC_SNC_ALLOW_DEV_HEADERS = "true"
```

---

## 4. Comandos Exatos de Execução Local

### Executar a API FastAPI (Terminal 1)

```powershell
.\.venv\Scripts\uvicorn.exe pic_snc.app:app --host 127.0.0.1 --port 8000
```

### Executar o Worker Independente (Terminal 2)

```powershell
.\.venv\Scripts\python.exe -m pic_snc.worker --db-path pic_snc.db --poll-interval 0.5
```

Para executar apenas um ciclo único de processamento e encerrar:

```powershell
.\.venv\Scripts\python.exe -m pic_snc.worker --db-path pic_snc.db --once
```

### Executar Todos os Testes (`pytest`)

```powershell
.\.venv\Scripts\pytest.exe -v
```

### Demonstrar os Três Cenários (Recordação A1, Oportunidade A2 e Risco A3)

```powershell
.\.venv\Scripts\python.exe -m pic_snc.scenarios
```

---

## 5. Exemplos de Requisições HTTP (PowerShell)

### 1. Verificar saúde e prontidão

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/health"
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/health/ready"
```

### 2. Enviar um evento observado (`POST /events` → `202 Accepted`)

```powershell
$headers = @{
    "X-Dev-Tenant-Id" = "tenant_demo"
    "X-Dev-Principal-Id" = "agent_local"
}
$body = @{
    event_id = "evt_ps_001"
    tenant_id = "tenant_demo"
    event_type = "contract.signing_prepared"
    source = "procurement.agent"
    dedupe_key = "contract-draft-55"
    payload = @{
        vendor = "Acme Cloud"
        clause_detected = "unlimited_liability"
        action = "waive_liability_cap"
    }
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/events" -Headers $headers -ContentType "application/json" -Body $body
```

### 3. Consultar impulsos produzidos pelo worker (`GET /impulses`)

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/impulses?status=published&mark_delivered=true&limit=20" -Headers $headers
```

### 4. Confirmar recebimento (`/ack`) e respeito à pausa A3 (`/apply`)

```powershell
$impulseId = "<IMPULSE_ID>"

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/impulses/$impulseId/ack" -Headers $headers -ContentType "application/json" -Body '{"notes":"Recebido pelo agente"}'

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/impulses/$impulseId/apply" -Headers $headers -ContentType "application/json" -Body '{"pause_respected":true,"action_taken":"paused_for_legal_review","record_outcome":true}'
```

---

## 6. Política de Acesso (Modo Local vs. Produção)

- **Modo Local (`development` / `test`):** aceita tokens Bearer configurados em `static_api_tokens` ou cabeçalhos explícitos de desenvolvimento (`X-Dev-Tenant-Id` + `X-Dev-Principal-Id`) quando `allow_dev_identity_headers=True`. O `tenant_id` nunca é aceito apenas como parâmetro livre da URL ou do payload se divergir da identidade autenticada (`403 Forbidden`).
- **Futuro Ambiente de Produção (`production`):** `SNCConfig` impede a inicialização caso `allow_dev_identity_headers=True` (`ConfigurationError`). Toda requisição exige `Authorization: Bearer <token>` validado criptograficamente (JWT/OIDC no futuro gateway) para derivar o `AuthContext`.

---

## 7. Limitações Conhecidas da PoC v0.1

1. **Persistência local em SQLite:** adequada para PoC mono-nó local com WAL e `BEGIN IMMEDIATE`; ambientes distribuídos futuros deverão migrar para PostgreSQL + fila gerenciada.
2. **Ativação associativa determinística:** combina sinais lexicais, semânticos, preditivos, temporais e relacionais sem embeddings vetoriais reais nesta versão (o ponto de extensão `SemanticScorer` já está preparado).
3. **Semântica at-least-once com idempotência:** o worker garante processamento idempotente de eventos/impulsos no SQLite, mas entregas HTTP/assinantes externos operam sob semântica *at-least-once*.
4. **Avaliador Jev real desativado por padrão na PoC:** testes e cenários executam no modo `mock` determinístico sem chaves externas.
