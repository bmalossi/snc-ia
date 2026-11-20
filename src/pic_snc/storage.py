"""Camada de persistência SQLite + Outbox transacional do PIC-SNC v0.1.

Garantias arquiteturais:
- Isolamento estrito por `tenant_id` (chaves primárias e estrangeiras compostas por `tenant_id`).
- Idempotência no nível do banco por `(tenant_id, event_id)` e `(tenant_id, dedupe_key)`.
- Restrições `CHECK` no banco para `salience`, `score`, `activation_score` e `confidence` em [0.0, 1.0].
- Gravação transacional atômica entre entidades (`events`, `impulses`) e `outbox`.
- Suporte a `AuthContext` para autorização baseada em identidade autenticada.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pic_snc.logging_utils import get_structured_logger, log_cognitive_step
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
    OutboxEntry,
    OutboxStatus,
    OutboxTopic,
    Outcome,
    OutcomeResult,
)


class StorageError(RuntimeError):
    """Erro base da camada de armazenamento."""


class TenantIsolationError(PermissionError):
    """Violação de isolamento ou autorização multi-tenant."""


@dataclass(frozen=True, slots=True)
class EventIngestResult:
    """Resultado transacional da ingestão de um evento observado."""

    event: Event
    created: bool
    deduplicated: bool
    outbox_entry: OutboxEntry | None = None


SCHEMA_DDL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS events (
    tenant_id TEXT NOT NULL CHECK (length(trim(tenant_id)) > 0),
    event_id TEXT NOT NULL CHECK (length(trim(event_id)) > 0),
    event_type TEXT NOT NULL CHECK (length(trim(event_type)) > 0),
    source TEXT NOT NULL CHECK (length(trim(source)) > 0),
    payload_json TEXT NOT NULL,
    completeness TEXT NOT NULL CHECK (completeness IN ('partial', 'complete', 'milestone')),
    stream_id TEXT,
    sequence_no INTEGER CHECK (sequence_no IS NULL OR sequence_no >= 0),
    semantic_density REAL NOT NULL CHECK (semantic_density >= 0.0 AND semantic_density <= 1.0),
    requires_evaluation INTEGER NOT NULL CHECK (requires_evaluation IN (0, 1)),
    dedupe_key TEXT CHECK (dedupe_key IS NULL OR length(trim(dedupe_key)) > 0),
    evidence_json TEXT NOT NULL,
    processing_status TEXT NOT NULL CHECK (
        processing_status IN ('pending', 'buffered', 'processing', 'processed', 'skipped', 'failed')
    ),
    timestamp_utc TEXT NOT NULL,
    PRIMARY KEY (tenant_id, event_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_events_tenant_dedupe
    ON events (tenant_id, dedupe_key)
    WHERE dedupe_key IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_events_tenant_status
    ON events (tenant_id, processing_status, timestamp_utc);

CREATE INDEX IF NOT EXISTS idx_events_tenant_stream_status
    ON events (tenant_id, stream_id, processing_status, sequence_no, timestamp_utc);

CREATE TABLE IF NOT EXISTS memories (
    tenant_id TEXT NOT NULL CHECK (length(trim(tenant_id)) > 0),
    memory_id TEXT NOT NULL CHECK (length(trim(memory_id)) > 0),
    kind TEXT NOT NULL CHECK (kind IN ('recall', 'opportunity', 'risk')),
    content TEXT NOT NULL CHECK (length(trim(content)) > 0),
    source TEXT NOT NULL CHECK (length(trim(source)) > 0),
    salience REAL NOT NULL CHECK (salience >= 0.0 AND salience <= 1.0),
    tags_json TEXT NOT NULL,
    associative_cues_json TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    PRIMARY KEY (tenant_id, memory_id)
);

CREATE INDEX IF NOT EXISTS idx_memories_tenant_kind_salience
    ON memories (tenant_id, kind, salience DESC);

CREATE TABLE IF NOT EXISTS candidates (
    tenant_id TEXT NOT NULL CHECK (length(trim(tenant_id)) > 0),
    candidate_id TEXT NOT NULL CHECK (length(trim(candidate_id)) > 0),
    event_id TEXT NOT NULL CHECK (length(trim(event_id)) > 0),
    memory_id TEXT NOT NULL CHECK (length(trim(memory_id)) > 0),
    related_memory_ids_json TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('recall', 'opportunity', 'risk')),
    activation_mode TEXT NOT NULL CHECK (
        activation_mode IN ('lexical', 'semantic', 'temporal', 'predictive', 'relational', 'hybrid')
    ),
    activation_score REAL NOT NULL CHECK (activation_score >= 0.0 AND activation_score <= 1.0),
    score REAL NOT NULL CHECK (score >= 0.0 AND score <= 1.0),
    noul_probability REAL CHECK (
        noul_probability IS NULL OR (noul_probability >= 0.0 AND noul_probability <= 1.0)
    ),
    rubric_score REAL CHECK (rubric_score IS NULL OR rubric_score >= 0.0),
    choice_confidence REAL CHECK (
        choice_confidence IS NULL OR (choice_confidence >= 0.0 AND choice_confidence <= 1.0)
    ),
    proposed_autonomy TEXT NOT NULL CHECK (proposed_autonomy IN ('A0', 'A1', 'A2', 'A3', 'A4')),
    rationale TEXT NOT NULL CHECK (length(trim(rationale)) > 0),
    source_trace_json TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    PRIMARY KEY (tenant_id, candidate_id),
    FOREIGN KEY (tenant_id, event_id) REFERENCES events (tenant_id, event_id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, memory_id) REFERENCES memories (tenant_id, memory_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_candidates_tenant_event
    ON candidates (tenant_id, event_id, score DESC);

CREATE TABLE IF NOT EXISTS impulses (
    tenant_id TEXT NOT NULL CHECK (length(trim(tenant_id)) > 0),
    impulse_id TEXT NOT NULL CHECK (length(trim(impulse_id)) > 0),
    event_id TEXT NOT NULL CHECK (length(trim(event_id)) > 0),
    candidate_id TEXT NOT NULL CHECK (length(trim(candidate_id)) > 0),
    memory_ids_json TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('recall', 'opportunity', 'risk')),
    autonomy_level TEXT NOT NULL CHECK (autonomy_level IN ('A0', 'A1', 'A2', 'A3', 'A4')),
    intervention_requested INTEGER NOT NULL DEFAULT 0 CHECK (intervention_requested IN (0, 1)),
    intervention_applied INTEGER NOT NULL DEFAULT 0 CHECK (intervention_applied IN (0, 1)),
    protective_block INTEGER NOT NULL CHECK (protective_block IN (0, 1)),
    external_action_allowed INTEGER NOT NULL CHECK (external_action_allowed IN (0, 1)),
    message TEXT NOT NULL CHECK (length(trim(message)) > 0),
    rationale TEXT,
    confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    source_trace_json TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN (
            'created', 'pending', 'published', 'delivered',
            'acknowledged', 'applied', 'suppressed', 'downgraded', 'failed'
        )
    ),
    suppressed_reason TEXT,
    dedupe_fingerprint TEXT,
    created_at_utc TEXT NOT NULL,
    PRIMARY KEY (tenant_id, impulse_id),
    CHECK (protective_block = 0 OR autonomy_level = 'A4'),
    FOREIGN KEY (tenant_id, event_id) REFERENCES events (tenant_id, event_id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, candidate_id) REFERENCES candidates (tenant_id, candidate_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_impulses_tenant_status_created
    ON impulses (tenant_id, status, created_at_utc DESC);

CREATE INDEX IF NOT EXISTS idx_impulses_tenant_fingerprint
    ON impulses (tenant_id, dedupe_fingerprint, created_at_utc DESC);

CREATE TABLE IF NOT EXISTS outcomes (
    tenant_id TEXT NOT NULL CHECK (length(trim(tenant_id)) > 0),
    outcome_id TEXT NOT NULL CHECK (length(trim(outcome_id)) > 0),
    impulse_id TEXT NOT NULL CHECK (length(trim(impulse_id)) > 0),
    result TEXT NOT NULL CHECK (result IN ('accepted', 'ignored', 'rejected', 'resolved')),
    observed_outcome TEXT,
    verified_correct INTEGER CHECK (verified_correct IS NULL OR verified_correct IN (0, 1)),
    notes TEXT,
    context_json TEXT NOT NULL DEFAULT '{}',
    evidence_json TEXT NOT NULL,
    recorded_at_utc TEXT NOT NULL,
    PRIMARY KEY (tenant_id, outcome_id),
    FOREIGN KEY (tenant_id, impulse_id) REFERENCES impulses (tenant_id, impulse_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_outcomes_tenant_impulse
    ON outcomes (tenant_id, impulse_id, recorded_at_utc DESC);

CREATE TABLE IF NOT EXISTS outbox (
    tenant_id TEXT NOT NULL CHECK (length(trim(tenant_id)) > 0),
    outbox_id TEXT NOT NULL CHECK (length(trim(outbox_id)) > 0),
    topic TEXT NOT NULL CHECK (
        topic IN ('event.ingested', 'impulse.emitted', 'impulse.protective_block')
    ),
    aggregate_type TEXT NOT NULL CHECK (length(trim(aggregate_type)) > 0),
    aggregate_id TEXT NOT NULL CHECK (length(trim(aggregate_id)) > 0),
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'processing', 'published', 'failed')),
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    available_at_utc TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    processed_at_utc TEXT,
    last_error TEXT,
    PRIMARY KEY (tenant_id, outbox_id),
    UNIQUE (tenant_id, topic, aggregate_type, aggregate_id)
);

CREATE INDEX IF NOT EXISTS idx_outbox_tenant_status_available
    ON outbox (tenant_id, status, available_at_utc ASC);
"""


def _serialize_evidence(evidence: list[EvidenceRef]) -> str:
    return json.dumps(
        [item.model_dump(mode="json") for item in evidence],
        ensure_ascii=False,
    )


def _deserialize_evidence(raw_json: str) -> list[EvidenceRef]:
    items = json.loads(raw_json)
    return [EvidenceRef.model_validate(item) for item in items]


class SQLiteStorage:
    """Repositório transacional SQLite + Outbox com isolamento multi-tenant."""

    def __init__(
        self,
        db_path: Path | str,
        *,
        require_authenticated_context: bool = False,
    ) -> None:
        self.db_path = Path(db_path)
        self.require_authenticated_context = require_authenticated_context
        self._logger = get_structured_logger("pic_snc.storage")
        self._initialize_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize_schema(self) -> None:
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            if str(self.db_path) != ":memory:":
                conn.execute("PRAGMA journal_mode = WAL;")
            conn.executescript(SCHEMA_DDL)

    def check_health(self) -> bool:
        """Verifica se o banco SQLite está acessível e operacional."""
        try:
            with self._connect() as conn:
                row = conn.execute("SELECT 1").fetchone()
            return row is not None and row[0] == 1
        except sqlite3.Error:
            return False

    def _verify_access(
        self,
        tenant_id: str,
        auth_context: AuthContext | None = None,
    ) -> str:
        cleaned_tenant = tenant_id.strip()
        if not cleaned_tenant:
            raise TenantIsolationError("tenant_id não pode ser vazio.")
        if self.require_authenticated_context and auth_context is None:
            raise TenantIsolationError(
                "Operação bloqueada: AuthContext autenticado é obrigatório."
            )
        if auth_context is not None:
            try:
                auth_context.assert_tenant_access(cleaned_tenant)
            except PermissionError as exc:
                raise TenantIsolationError(str(exc)) from exc
        return cleaned_tenant

    # -------------------------------------------------------------------------
    # 1. EVENTOS OBSERVADOS + OUTBOX TRANSACIONAL + DEDUPLICAÇÃO
    # -------------------------------------------------------------------------
    def ingest_event(
        self,
        event: Event,
        *,
        auth_context: AuthContext | None = None,
        enqueue_outbox: bool = True,
        min_eval_density: float = 0.40,
    ) -> EventIngestResult:
        """Persiste um evento observado e seu registro de outbox na mesma transação."""
        tenant_id = self._verify_access(event.tenant_id, auth_context)
        eligible_for_eval = event.should_trigger_evaluation(min_density=min_eval_density)
        effective_status = (
            event.processing_status
            if event.processing_status != EventProcessingStatus.PENDING
            else (
                EventProcessingStatus.PENDING
                if eligible_for_eval
                else EventProcessingStatus.BUFFERED
            )
        )
        effective_event = event.model_copy(
            update={"processing_status": effective_status}
        )

        with self._connect() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO events (
                        tenant_id, event_id, event_type, source, payload_json,
                        completeness, stream_id, sequence_no, semantic_density,
                        requires_evaluation, dedupe_key, evidence_json,
                        processing_status, timestamp_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tenant_id,
                        effective_event.event_id,
                        effective_event.event_type,
                        effective_event.source,
                        json.dumps(effective_event.payload, ensure_ascii=False),
                        effective_event.completeness.value,
                        effective_event.stream_id,
                        effective_event.sequence_no,
                        float(effective_event.semantic_density),
                        1 if effective_event.requires_evaluation else 0,
                        effective_event.dedupe_key,
                        _serialize_evidence(effective_event.evidence),
                        effective_event.processing_status.value,
                        effective_event.timestamp.isoformat(),
                    ),
                )
            except sqlite3.IntegrityError:
                existing = self._find_existing_event(
                    conn,
                    tenant_id=tenant_id,
                    event_id=effective_event.event_id,
                    dedupe_key=effective_event.dedupe_key,
                )
                if existing is None:
                    raise
                log_cognitive_step(
                    self._logger,
                    stage="storage.ingest_event",
                    action="event_deduplicated",
                    tenant_id=tenant_id,
                    event_id=existing.event_id,
                    details={"dedupe_key": effective_event.dedupe_key},
                )
                return EventIngestResult(
                    event=existing,
                    created=False,
                    deduplicated=True,
                    outbox_entry=None,
                )

            outbox_entry: OutboxEntry | None = None
            if enqueue_outbox and eligible_for_eval:
                outbox_entry = OutboxEntry(
                    tenant_id=tenant_id,
                    topic=OutboxTopic.EVENT_INGESTED,
                    aggregate_type="event",
                    aggregate_id=effective_event.event_id,
                    payload=effective_event.model_dump(mode="json"),
                )
                self._insert_outbox_row(conn, outbox_entry)

            log_cognitive_step(
                self._logger,
                stage="storage.ingest_event",
                action="event_persisted",
                tenant_id=tenant_id,
                event_id=effective_event.event_id,
                details={
                    "completeness": effective_event.completeness.value,
                    "processing_status": effective_event.processing_status.value,
                    "outbox_enqueued": outbox_entry is not None,
                },
            )
            return EventIngestResult(
                event=effective_event,
                created=True,
                deduplicated=False,
                outbox_entry=outbox_entry,
            )

    def _find_existing_event(
        self,
        conn: sqlite3.Connection,
        *,
        tenant_id: str,
        event_id: str,
        dedupe_key: str | None,
    ) -> Event | None:
        row = conn.execute(
            "SELECT * FROM events WHERE tenant_id = ? AND event_id = ?",
            (tenant_id, event_id),
        ).fetchone()
        if row is None and dedupe_key is not None:
            row = conn.execute(
                "SELECT * FROM events WHERE tenant_id = ? AND dedupe_key = ?",
                (tenant_id, dedupe_key),
            ).fetchone()
        return self._row_to_event(row) if row is not None else None

    def get_event(
        self,
        tenant_id: str,
        event_id: str,
        *,
        auth_context: AuthContext | None = None,
    ) -> Event | None:
        verified_tenant = self._verify_access(tenant_id, auth_context)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM events WHERE tenant_id = ? AND event_id = ?",
                (verified_tenant, event_id),
            ).fetchone()
        return self._row_to_event(row) if row is not None else None

    def update_event_status(
        self,
        tenant_id: str,
        event_id: str,
        status: EventProcessingStatus,
        *,
        auth_context: AuthContext | None = None,
    ) -> None:
        verified_tenant = self._verify_access(tenant_id, auth_context)
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE events SET processing_status = ? WHERE tenant_id = ? AND event_id = ?",
                (status.value, verified_tenant, event_id),
            )
            if cursor.rowcount == 0:
                raise StorageError(
                    f"Evento {event_id!r} não encontrado para o tenant {verified_tenant!r}."
                )

    def list_events(
        self,
        tenant_id: str,
        *,
        status: EventProcessingStatus | None = None,
        limit: int = 50,
        auth_context: AuthContext | None = None,
    ) -> list[Event]:
        verified_tenant = self._verify_access(tenant_id, auth_context)
        with self._connect() as conn:
            if status is not None:
                rows = conn.execute(
                    """
                    SELECT * FROM events
                    WHERE tenant_id = ? AND processing_status = ?
                    ORDER BY timestamp_utc ASC
                    LIMIT ?
                    """,
                    (verified_tenant, status.value, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM events
                    WHERE tenant_id = ?
                    ORDER BY timestamp_utc ASC
                    LIMIT ?
                    """,
                    (verified_tenant, limit),
                ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def list_buffered_stream_events(
        self,
        tenant_id: str,
        stream_id: str,
        *,
        auth_context: AuthContext | None = None,
    ) -> list[Event]:
        """Lista eventos em estado `buffered` de uma stream específica ordenados por sequência/tempo."""
        verified_tenant = self._verify_access(tenant_id, auth_context)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM events
                WHERE tenant_id = ?
                  AND stream_id = ?
                  AND processing_status = 'buffered'
                ORDER BY
                    CASE WHEN sequence_no IS NULL THEN 1 ELSE 0 END ASC,
                    sequence_no ASC,
                    timestamp_utc ASC
                """,
                (verified_tenant, stream_id),
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def claim_pending_events(
        self,
        *,
        tenant_id: str | None = None,
        include_failed: bool = False,
        limit: int = 20,
    ) -> list[Event]:
        """Reivindica eventos pendentes atomicamente (`pending` -> `processing`) via BEGIN IMMEDIATE.

        Usado pelo worker independente para evitar processamento concorrente duplicado.
        """
        statuses = ("pending", "failed") if include_failed else ("pending",)
        placeholders = ", ".join("?" for _ in statuses)

        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if tenant_id is not None:
                cleaned_tenant = self._verify_access(tenant_id, None)
                rows = conn.execute(
                    f"""
                    SELECT * FROM events
                    WHERE tenant_id = ? AND processing_status IN ({placeholders})
                    ORDER BY timestamp_utc ASC
                    LIMIT ?
                    """,
                    (cleaned_tenant, *statuses, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"""
                    SELECT * FROM events
                    WHERE processing_status IN ({placeholders})
                    ORDER BY timestamp_utc ASC
                    LIMIT ?
                    """,
                    (*statuses, limit),
                ).fetchall()

            claimed: list[Event] = []
            for row in rows:
                cursor = conn.execute(
                    """
                    UPDATE events
                    SET processing_status = 'processing'
                    WHERE tenant_id = ? AND event_id = ? AND processing_status IN ({placeholders})
                    """.format(placeholders=placeholders),
                    (row["tenant_id"], row["event_id"], *statuses),
                )
                if cursor.rowcount > 0:
                    ev = self._row_to_event(row).model_copy(
                        update={"processing_status": EventProcessingStatus.PROCESSING}
                    )
                    claimed.append(ev)
        return claimed

    def recover_interrupted_events(
        self,
        *,
        tenant_id: str | None = None,
    ) -> int:
        """Retorna eventos presos em `processing` para `pending` na inicialização do worker."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if tenant_id is not None:
                cleaned_tenant = self._verify_access(tenant_id, None)
                cursor = conn.execute(
                    """
                    UPDATE events
                    SET processing_status = 'pending'
                    WHERE tenant_id = ? AND processing_status = 'processing'
                    """,
                    (cleaned_tenant,),
                )
            else:
                cursor = conn.execute(
                    """
                    UPDATE events
                    SET processing_status = 'pending'
                    WHERE processing_status = 'processing'
                    """
                )
            return int(cursor.rowcount)

    # -------------------------------------------------------------------------
    # 2. MEMÓRIAS LATENTES (COM PISTAS ASSOCIATIVAS EXTENSÍVEIS)
    # -------------------------------------------------------------------------
    def save_memory(
        self,
        memory: Memory,
        *,
        auth_context: AuthContext | None = None,
    ) -> Memory:
        tenant_id = self._verify_access(memory.tenant_id, auth_context)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memories (
                    tenant_id, memory_id, kind, content, source, salience,
                    tags_json, associative_cues_json, evidence_json,
                    metadata_json, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tenant_id,
                    memory.memory_id,
                    memory.kind.value,
                    memory.content,
                    memory.source,
                    float(memory.salience),
                    json.dumps(memory.tags, ensure_ascii=False),
                    json.dumps(
                        memory.associative_cues.model_dump(mode="json"),
                        ensure_ascii=False,
                    ),
                    _serialize_evidence(memory.evidence),
                    json.dumps(memory.metadata, ensure_ascii=False),
                    memory.created_at.isoformat(),
                ),
            )
        return memory

    def get_memory(
        self,
        tenant_id: str,
        memory_id: str,
        *,
        auth_context: AuthContext | None = None,
    ) -> Memory | None:
        verified_tenant = self._verify_access(tenant_id, auth_context)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE tenant_id = ? AND memory_id = ?",
                (verified_tenant, memory_id),
            ).fetchone()
        return self._row_to_memory(row) if row is not None else None

    def list_memories(
        self,
        tenant_id: str,
        *,
        kind: CognitiveKind | None = None,
        min_salience: float = 0.0,
        auth_context: AuthContext | None = None,
    ) -> list[Memory]:
        verified_tenant = self._verify_access(tenant_id, auth_context)
        with self._connect() as conn:
            if kind is not None:
                rows = conn.execute(
                    """
                    SELECT * FROM memories
                    WHERE tenant_id = ? AND kind = ? AND salience >= ?
                    ORDER BY salience DESC, created_at_utc DESC
                    """,
                    (verified_tenant, kind.value, float(min_salience)),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM memories
                    WHERE tenant_id = ? AND salience >= ?
                    ORDER BY salience DESC, created_at_utc DESC
                    """,
                    (verified_tenant, float(min_salience)),
                ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    # -------------------------------------------------------------------------
    # 3. CANDIDATOS ATIVADOS (MEMÓRIA ATIVADA)
    # -------------------------------------------------------------------------
    def save_candidate(
        self,
        candidate: Candidate,
        *,
        auth_context: AuthContext | None = None,
    ) -> Candidate:
        tenant_id = self._verify_access(candidate.tenant_id, auth_context)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO candidates (
                    tenant_id, candidate_id, event_id, memory_id,
                    related_memory_ids_json, kind, activation_mode,
                    activation_score, score, noul_probability, rubric_score,
                    choice_confidence, proposed_autonomy, rationale,
                    source_trace_json, evidence_json, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tenant_id,
                    candidate.candidate_id,
                    candidate.event_id,
                    candidate.memory_id,
                    json.dumps(candidate.related_memory_ids, ensure_ascii=False),
                    candidate.kind.value,
                    candidate.activation_mode.value,
                    float(candidate.activation_score),
                    float(candidate.score),
                    float(candidate.noul_probability)
                    if candidate.noul_probability is not None
                    else None,
                    float(candidate.rubric_score)
                    if candidate.rubric_score is not None
                    else None,
                    float(candidate.choice_confidence)
                    if candidate.choice_confidence is not None
                    else None,
                    candidate.proposed_autonomy.value,
                    candidate.rationale,
                    json.dumps(candidate.source_trace, ensure_ascii=False),
                    _serialize_evidence(candidate.evidence),
                    candidate.created_at.isoformat(),
                ),
            )
        return candidate

    def list_candidates_for_event(
        self,
        tenant_id: str,
        event_id: str,
        *,
        auth_context: AuthContext | None = None,
    ) -> list[Candidate]:
        verified_tenant = self._verify_access(tenant_id, auth_context)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM candidates
                WHERE tenant_id = ? AND event_id = ?
                ORDER BY score DESC
                """,
                (verified_tenant, event_id),
            ).fetchall()
        return [self._row_to_candidate(row) for row in rows]

    # -------------------------------------------------------------------------
    # 4. DECISÕES EMITIDAS (IMPULSES) + OUTBOX TRANSACIONAL
    # -------------------------------------------------------------------------
    def save_impulse(
        self,
        impulse: Impulse,
        *,
        auth_context: AuthContext | None = None,
        enqueue_outbox: bool = True,
    ) -> tuple[Impulse, OutboxEntry | None]:
        """Persiste um impulso governado e, se publicado, seu registro de outbox na mesma transação.

        Idempotente: se o `impulse_id` já existir no tenant, retorna o existente sem duplicar outbox.
        """
        tenant_id = self._verify_access(impulse.tenant_id, auth_context)
        with self._connect() as conn:
            existing_row = conn.execute(
                "SELECT * FROM impulses WHERE tenant_id = ? AND impulse_id = ?",
                (tenant_id, impulse.impulse_id),
            ).fetchone()
            if existing_row is not None:
                return self._row_to_impulse(existing_row), None

            conn.execute(
                """
                INSERT INTO impulses (
                    tenant_id, impulse_id, event_id, candidate_id, memory_ids_json,
                    kind, autonomy_level, intervention_requested, intervention_applied,
                    protective_block, external_action_allowed, message, rationale, confidence,
                    source_trace_json, evidence_json, status, suppressed_reason,
                    dedupe_fingerprint, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tenant_id,
                    impulse.impulse_id,
                    impulse.event_id,
                    impulse.candidate_id,
                    json.dumps(impulse.memory_ids, ensure_ascii=False),
                    impulse.kind.value,
                    impulse.autonomy_level.value,
                    1 if impulse.intervention_requested else 0,
                    1 if impulse.intervention_applied else 0,
                    1 if impulse.protective_block else 0,
                    1 if impulse.external_action_allowed else 0,
                    impulse.message,
                    impulse.rationale,
                    float(impulse.confidence),
                    json.dumps(impulse.source_trace, ensure_ascii=False),
                    _serialize_evidence(impulse.evidence),
                    impulse.status.value,
                    impulse.suppressed_reason,
                    impulse.dedupe_fingerprint,
                    impulse.created_at.isoformat(),
                ),
            )

            outbox_entry: OutboxEntry | None = None
            if enqueue_outbox and impulse.status in {
                ImpulseStatus.PUBLISHED,
                ImpulseStatus.DOWNGRADED,
            }:
                topic = (
                    OutboxTopic.PROTECTIVE_BLOCK_TRIGGERED
                    if impulse.protective_block
                    else OutboxTopic.IMPULSE_EMITTED
                )
                outbox_entry = OutboxEntry(
                    tenant_id=tenant_id,
                    topic=topic,
                    aggregate_type="impulse",
                    aggregate_id=impulse.impulse_id,
                    payload=impulse.model_dump(mode="json"),
                )
                self._insert_outbox_row(conn, outbox_entry)

            log_cognitive_step(
                self._logger,
                stage="storage.save_impulse",
                action="impulse_persisted",
                tenant_id=tenant_id,
                event_id=impulse.event_id,
                candidate_id=impulse.candidate_id,
                impulse_id=impulse.impulse_id,
                details={
                    "autonomy_level": impulse.autonomy_level.value,
                    "status": impulse.status.value,
                    "outbox_enqueued": outbox_entry is not None,
                },
            )
            return impulse, outbox_entry

    def get_impulse(
        self,
        tenant_id: str,
        impulse_id: str,
        *,
        auth_context: AuthContext | None = None,
    ) -> Impulse | None:
        verified_tenant = self._verify_access(tenant_id, auth_context)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM impulses WHERE tenant_id = ? AND impulse_id = ?",
                (verified_tenant, impulse_id),
            ).fetchone()
        return self._row_to_impulse(row) if row is not None else None

    def update_impulse_status(
        self,
        tenant_id: str,
        impulse_id: str,
        status: ImpulseStatus,
        *,
        suppressed_reason: str | None = None,
        auth_context: AuthContext | None = None,
    ) -> Impulse:
        verified_tenant = self._verify_access(tenant_id, auth_context)
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE impulses
                SET status = ?, suppressed_reason = COALESCE(?, suppressed_reason)
                WHERE tenant_id = ? AND impulse_id = ?
                """,
                (status.value, suppressed_reason, verified_tenant, impulse_id),
            )
            if cursor.rowcount == 0:
                raise StorageError(
                    f"Impulso {impulse_id!r} não encontrado para o tenant {verified_tenant!r}."
                )
            row = conn.execute(
                "SELECT * FROM impulses WHERE tenant_id = ? AND impulse_id = ?",
                (verified_tenant, impulse_id),
            ).fetchone()
        assert row is not None
        return self._row_to_impulse(row)

    def find_recent_impulse_by_fingerprint(
        self,
        tenant_id: str,
        dedupe_fingerprint: str,
        *,
        since_utc: datetime,
        auth_context: AuthContext | None = None,
    ) -> Impulse | None:
        verified_tenant = self._verify_access(tenant_id, auth_context)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM impulses
                WHERE tenant_id = ?
                  AND dedupe_fingerprint = ?
                  AND created_at_utc >= ?
                  AND status IN ('published', 'delivered', 'acknowledged', 'applied', 'downgraded')
                ORDER BY created_at_utc DESC
                LIMIT 1
                """,
                (verified_tenant, dedupe_fingerprint, since_utc.isoformat()),
            ).fetchone()
        return self._row_to_impulse(row) if row is not None else None

    def list_impulses(
        self,
        tenant_id: str,
        *,
        status: ImpulseStatus | None = None,
        limit: int = 50,
        auth_context: AuthContext | None = None,
    ) -> list[Impulse]:
        verified_tenant = self._verify_access(tenant_id, auth_context)
        with self._connect() as conn:
            if status is not None:
                rows = conn.execute(
                    """
                    SELECT * FROM impulses
                    WHERE tenant_id = ? AND status = ?
                    ORDER BY created_at_utc DESC
                    LIMIT ?
                    """,
                    (verified_tenant, status.value, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM impulses
                    WHERE tenant_id = ?
                    ORDER BY created_at_utc DESC
                    LIMIT ?
                    """,
                    (verified_tenant, limit),
                ).fetchall()
        return [self._row_to_impulse(row) for row in rows]

    def mark_impulse_applied(
        self,
        tenant_id: str,
        impulse_id: str,
        *,
        intervention_applied: bool = True,
        auth_context: AuthContext | None = None,
    ) -> Impulse:
        """Atualiza o impulso para `applied` e registra se a intervenção/pausa foi respeitada."""
        verified_tenant = self._verify_access(tenant_id, auth_context)
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE impulses
                SET status = ?, intervention_applied = ?
                WHERE tenant_id = ? AND impulse_id = ?
                """,
                (
                    ImpulseStatus.APPLIED.value,
                    1 if intervention_applied else 0,
                    verified_tenant,
                    impulse_id,
                ),
            )
            if cursor.rowcount == 0:
                raise StorageError(
                    f"Impulso {impulse_id!r} não encontrado para o tenant {verified_tenant!r}."
                )
            row = conn.execute(
                "SELECT * FROM impulses WHERE tenant_id = ? AND impulse_id = ?",
                (verified_tenant, impulse_id),
            ).fetchone()
        assert row is not None
        return self._row_to_impulse(row)

    def list_impulses_paginated(
        self,
        tenant_id: str,
        *,
        status: ImpulseStatus | None = None,
        limit: int = 20,
        cursor: str | None = None,
        auth_context: AuthContext | None = None,
    ) -> tuple[list[Impulse], str | None]:
        """Lista impulsos de um tenant com filtro opcional de estado e paginação por cursor."""
        verified_tenant = self._verify_access(tenant_id, auth_context)
        offset = 0
        if cursor:
            if not cursor.startswith("offset:"):
                raise StorageError(f"Cursor inválido: {cursor!r}.")
            try:
                offset = max(0, int(cursor.split(":", 1)[1]))
            except ValueError as exc:
                raise StorageError(f"Cursor inválido: {cursor!r}.") from exc

        fetch_limit = max(1, limit) + 1
        with self._connect() as conn:
            if status is not None:
                rows = conn.execute(
                    """
                    SELECT * FROM impulses
                    WHERE tenant_id = ? AND status = ?
                    ORDER BY created_at_utc DESC, impulse_id ASC
                    LIMIT ? OFFSET ?
                    """,
                    (verified_tenant, status.value, fetch_limit, offset),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM impulses
                    WHERE tenant_id = ?
                    ORDER BY created_at_utc DESC, impulse_id ASC
                    LIMIT ? OFFSET ?
                    """,
                    (verified_tenant, fetch_limit, offset),
                ).fetchall()

        has_more = len(rows) > limit
        page_rows = rows[:limit]
        next_cursor = f"offset:{offset + limit}" if has_more else None
        return [self._row_to_impulse(row) for row in page_rows], next_cursor

    # -------------------------------------------------------------------------
    # 5. OUTCOMES (DESFECHOS)
    # -------------------------------------------------------------------------
    def save_outcome(
        self,
        outcome: Outcome,
        *,
        auth_context: AuthContext | None = None,
    ) -> Outcome:
        tenant_id = self._verify_access(outcome.tenant_id, auth_context)
        verified_int = (
            None
            if outcome.verified_correct is None
            else (1 if outcome.verified_correct else 0)
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO outcomes (
                    tenant_id, outcome_id, impulse_id, result,
                    observed_outcome, verified_correct, notes,
                    context_json, evidence_json, recorded_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tenant_id,
                    outcome.outcome_id,
                    outcome.impulse_id,
                    outcome.result.value,
                    outcome.observed_outcome,
                    verified_int,
                    outcome.notes,
                    json.dumps(outcome.context, ensure_ascii=False),
                    _serialize_evidence(outcome.evidence),
                    outcome.recorded_at.isoformat(),
                ),
            )
        return outcome

    def list_outcomes(
        self,
        tenant_id: str,
        *,
        impulse_id: str | None = None,
        auth_context: AuthContext | None = None,
    ) -> list[Outcome]:
        verified_tenant = self._verify_access(tenant_id, auth_context)
        with self._connect() as conn:
            if impulse_id is not None:
                rows = conn.execute(
                    """
                    SELECT * FROM outcomes
                    WHERE tenant_id = ? AND impulse_id = ?
                    ORDER BY recorded_at_utc DESC
                    """,
                    (verified_tenant, impulse_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM outcomes
                    WHERE tenant_id = ?
                    ORDER BY recorded_at_utc DESC
                    """,
                    (verified_tenant,),
                ).fetchall()
        return [self._row_to_outcome(row) for row in rows]

    # -------------------------------------------------------------------------
    # 6. OUTBOX (PROCESSAMENTO ASSÍNCRONO E INDEPENDENTE)
    # -------------------------------------------------------------------------
    def _insert_outbox_row(
        self,
        conn: sqlite3.Connection,
        entry: OutboxEntry,
    ) -> None:
        conn.execute(
            """
            INSERT OR IGNORE INTO outbox (
                tenant_id, outbox_id, topic, aggregate_type, aggregate_id,
                payload_json, status, attempts, available_at_utc,
                created_at_utc, processed_at_utc, last_error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.tenant_id,
                entry.outbox_id,
                entry.topic.value,
                entry.aggregate_type,
                entry.aggregate_id,
                json.dumps(entry.payload, ensure_ascii=False),
                entry.status.value,
                entry.attempts,
                entry.available_at.isoformat(),
                entry.created_at.isoformat(),
                entry.processed_at.isoformat() if entry.processed_at else None,
                entry.last_error,
            ),
        )

    def list_outbox_entries(
        self,
        tenant_id: str,
        *,
        status: OutboxStatus | None = None,
        topic: OutboxTopic | None = None,
        limit: int = 50,
        auth_context: AuthContext | None = None,
    ) -> list[OutboxEntry]:
        verified_tenant = self._verify_access(tenant_id, auth_context)
        query = "SELECT * FROM outbox WHERE tenant_id = ?"
        params: list[Any] = [verified_tenant]
        if status is not None:
            query += " AND status = ?"
            params.append(status.value)
        if topic is not None:
            query += " AND topic = ?"
            params.append(topic.value)
        query += " ORDER BY available_at_utc ASC, created_at_utc ASC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [self._row_to_outbox(row) for row in rows]

    def mark_outbox_processed(
        self,
        tenant_id: str,
        outbox_id: str,
        *,
        auth_context: AuthContext | None = None,
    ) -> None:
        verified_tenant = self._verify_access(tenant_id, auth_context)
        now_iso = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE outbox
                SET status = ?, processed_at_utc = ?, attempts = attempts + 1, last_error = NULL
                WHERE tenant_id = ? AND outbox_id = ?
                """,
                (OutboxStatus.PUBLISHED.value, now_iso, verified_tenant, outbox_id),
            )
            if cursor.rowcount == 0:
                raise StorageError(
                    f"Registro outbox {outbox_id!r} não encontrado para tenant {verified_tenant!r}."
                )

    def mark_outbox_failed(
        self,
        tenant_id: str,
        outbox_id: str,
        error_message: str,
        *,
        auth_context: AuthContext | None = None,
    ) -> None:
        verified_tenant = self._verify_access(tenant_id, auth_context)
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE outbox
                SET status = ?, attempts = attempts + 1, last_error = ?
                WHERE tenant_id = ? AND outbox_id = ?
                """,
                (OutboxStatus.FAILED.value, error_message, verified_tenant, outbox_id),
            )
            if cursor.rowcount == 0:
                raise StorageError(
                    f"Registro outbox {outbox_id!r} não encontrado para tenant {verified_tenant!r}."
                )

    # -------------------------------------------------------------------------
    # MAPPERS INTERNOS (ROW -> PYDANTIC)
    # -------------------------------------------------------------------------
    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> Event:
        return Event(
            event_id=row["event_id"],
            tenant_id=row["tenant_id"],
            event_type=row["event_type"],
            source=row["source"],
            payload=json.loads(row["payload_json"]),
            completeness=EventCompleteness(row["completeness"]),
            stream_id=row["stream_id"],
            sequence_no=row["sequence_no"],
            semantic_density=float(row["semantic_density"]),
            requires_evaluation=bool(row["requires_evaluation"]),
            dedupe_key=row["dedupe_key"],
            evidence=_deserialize_evidence(row["evidence_json"]),
            processing_status=EventProcessingStatus(row["processing_status"]),
            timestamp=row["timestamp_utc"],
        )

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> Memory:
        return Memory(
            memory_id=row["memory_id"],
            tenant_id=row["tenant_id"],
            kind=CognitiveKind(row["kind"]),
            content=row["content"],
            source=row["source"],
            salience=float(row["salience"]),
            tags=json.loads(row["tags_json"]),
            associative_cues=AssociativeCues.model_validate(
                json.loads(row["associative_cues_json"])
            ),
            evidence=_deserialize_evidence(row["evidence_json"]),
            metadata=json.loads(row["metadata_json"]),
            created_at=row["created_at_utc"],
        )

    @staticmethod
    def _row_to_candidate(row: sqlite3.Row) -> Candidate:
        return Candidate(
            candidate_id=row["candidate_id"],
            tenant_id=row["tenant_id"],
            event_id=row["event_id"],
            memory_id=row["memory_id"],
            related_memory_ids=json.loads(row["related_memory_ids_json"]),
            kind=CognitiveKind(row["kind"]),
            activation_mode=ActivationMode(row["activation_mode"]),
            activation_score=float(row["activation_score"]),
            score=float(row["score"]),
            noul_probability=(
                float(row["noul_probability"])
                if row["noul_probability"] is not None
                else None
            ),
            rubric_score=(
                float(row["rubric_score"])
                if row["rubric_score"] is not None
                else None
            ),
            choice_confidence=(
                float(row["choice_confidence"])
                if row["choice_confidence"] is not None
                else None
            ),
            proposed_autonomy=AutonomyLevel(row["proposed_autonomy"]),
            rationale=row["rationale"],
            source_trace=json.loads(row["source_trace_json"]),
            evidence=_deserialize_evidence(row["evidence_json"]),
            created_at=row["created_at_utc"],
        )

    @staticmethod
    def _row_to_impulse(row: sqlite3.Row) -> Impulse:
        return Impulse(
            impulse_id=row["impulse_id"],
            tenant_id=row["tenant_id"],
            event_id=row["event_id"],
            candidate_id=row["candidate_id"],
            memory_ids=json.loads(row["memory_ids_json"]),
            kind=CognitiveKind(row["kind"]),
            autonomy_level=AutonomyLevel(row["autonomy_level"]),
            intervention_requested=bool(row["intervention_requested"]),
            intervention_applied=bool(row["intervention_applied"]),
            protective_block=bool(row["protective_block"]),
            external_action_allowed=bool(row["external_action_allowed"]),
            message=row["message"],
            rationale=row["rationale"],
            confidence=float(row["confidence"]),
            source_trace=json.loads(row["source_trace_json"]),
            evidence=_deserialize_evidence(row["evidence_json"]),
            status=ImpulseStatus(row["status"]),
            suppressed_reason=row["suppressed_reason"],
            dedupe_fingerprint=row["dedupe_fingerprint"],
            created_at=row["created_at_utc"],
        )

    @staticmethod
    def _row_to_outcome(row: sqlite3.Row) -> Outcome:
        verified_raw = row["verified_correct"]
        verified_bool = None if verified_raw is None else bool(verified_raw)
        return Outcome(
            outcome_id=row["outcome_id"],
            impulse_id=row["impulse_id"],
            tenant_id=row["tenant_id"],
            result=OutcomeResult(row["result"]),
            observed_outcome=row["observed_outcome"],
            verified_correct=verified_bool,
            notes=row["notes"],
            context=json.loads(row["context_json"]) if row["context_json"] else {},
            evidence=_deserialize_evidence(row["evidence_json"]),
            recorded_at=row["recorded_at_utc"],
        )

    @staticmethod
    def _row_to_outbox(row: sqlite3.Row) -> OutboxEntry:
        return OutboxEntry(
            outbox_id=row["outbox_id"],
            tenant_id=row["tenant_id"],
            topic=OutboxTopic(row["topic"]),
            aggregate_type=row["aggregate_type"],
            aggregate_id=row["aggregate_id"],
            payload=json.loads(row["payload_json"]),
            status=OutboxStatus(row["status"]),
            attempts=int(row["attempts"]),
            available_at=row["available_at_utc"],
            created_at=row["created_at_utc"],
            processed_at=row["processed_at_utc"],
            last_error=row["last_error"],
        )
