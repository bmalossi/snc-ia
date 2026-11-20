"""Utilitários de log estruturado em JSON com sanitização de segredos e UTC."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

_SENSITIVE_KEYS = frozenset(
    {
        "typesafe_api_key",
        "api_key",
        "apikey",
        "token",
        "access_token",
        "secret",
        "password",
        "authorization",
    }
)


def sanitize_log_payload(value: Any) -> Any:
    """Mascara recursivamente campos sensíveis antes de serializar para log."""
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for k, v in value.items():
            key_str = str(k)
            if key_str.lower() in _SENSITIVE_KEYS:
                sanitized[key_str] = "***REDACTED***"
            else:
                sanitized[key_str] = sanitize_log_payload(v)
        return sanitized
    if isinstance(value, (list, tuple)):
        return [sanitize_log_payload(item) for item in value]
    return value


class StructuredJsonFormatter(logging.Formatter):
    """Formata registros de log como objetos JSON de linha única em UTC."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        structured_data = getattr(record, "structured_data", None)
        if isinstance(structured_data, dict):
            payload.update(sanitize_log_payload(structured_data))

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


def get_structured_logger(name: str = "pic_snc") -> logging.Logger:
    """Retorna logger configurado com StructuredJsonFormatter sem duplicar handlers."""
    logger = logging.getLogger(name)
    if not any(
        isinstance(getattr(h, "formatter", None), StructuredJsonFormatter)
        for h in logger.handlers
    ):
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredJsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log_cognitive_step(
    logger: logging.Logger,
    *,
    stage: str,
    action: str,
    tenant_id: str,
    level: int = logging.INFO,
    event_id: str | None = None,
    memory_id: str | None = None,
    candidate_id: str | None = None,
    impulse_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Emite um log estruturado do pipeline cognitivo e retorna o dicionário emitido."""
    structured: dict[str, Any] = {
        "stage": stage,
        "action": action,
        "tenant_id": tenant_id,
    }
    if event_id is not None:
        structured["event_id"] = event_id
    if memory_id is not None:
        structured["memory_id"] = memory_id
    if candidate_id is not None:
        structured["candidate_id"] = candidate_id
    if impulse_id is not None:
        structured["impulse_id"] = impulse_id
    if details:
        structured["details"] = sanitize_log_payload(details)

    logger.log(
        level,
        f"[{stage}] {action} (tenant={tenant_id})",
        extra={"structured_data": structured},
    )
    return structured
