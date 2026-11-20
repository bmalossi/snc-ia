"""Configuração central e políticas de autoridade/avaliação do PIC-SNC v0.1."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class ConfigurationError(ValueError):
    """Erro de configuração inválida ou insegura do PIC-SNC."""


class EvaluatorMode(StrEnum):
    """Modo de avaliação cognitiva (seleção explícita obrigatória)."""

    MOCK = "mock"
    JEV = "jev"


class EnvironmentMode(StrEnum):
    """Ambiente de execução do serviço."""

    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


class ExternalActionPolicy(StrEnum):
    """Política separada para execução autônoma de ações externas (independente de A4)."""

    DENY = "deny"
    REQUIRE_EXPLICIT_GRANT = "require_explicit_grant"


def _parse_bool(raw: str | None, default: bool = False) -> bool:
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ConfigurationError(f"Valor booleano inválido na configuração: {raw!r}")


@dataclass(frozen=True, slots=True)
class SNCConfig:
    """Configuração imutável do motor cognitivo PIC-SNC v0.1.

    Regras arquiteturais impostas:
    - `evaluator_mode` deve ser explicitamente selecionado (`mock` ou `jev`).
    - Quando `evaluator_mode == 'jev'`, a ausência de `typesafe_api_key` levanta
      `ConfigurationError` imediatamente (proibido fallback silencioso para mock).
    - `enable_a4` (bloqueio protetivo autorizado) é `False` por padrão.
    - Execução autônoma de ações externas possui política separada (`external_action_policy`),
      `DENY` por padrão.
    - `typesafe_api_key` nunca é exposta em logs ou `repr`.
    """

    db_path: Path = field(default_factory=lambda: Path("pic_snc.db"))
    evaluator_mode: EvaluatorMode = EvaluatorMode.MOCK
    typesafe_api_key: str | None = field(default=None, repr=False)
    enable_a4: bool = False
    external_action_policy: ExternalActionPolicy = ExternalActionPolicy.DENY
    require_authenticated_context: bool = False
    environment: EnvironmentMode = EnvironmentMode.DEVELOPMENT
    allow_dev_identity_headers: bool = True
    static_api_tokens: dict[str, str] = field(default_factory=dict, repr=False)
    min_activation_salience: float = 0.20
    min_candidate_score: float = 0.35
    min_impulse_confidence: float = 0.50
    progressive_eval_min_density: float = 0.40

    def __post_init__(self) -> None:
        try:
            mode = EvaluatorMode(self.evaluator_mode)
        except ValueError as exc:
            raise ConfigurationError(
                f"evaluator_mode inválido: {self.evaluator_mode!r}. "
                "Selecione explicitamente 'mock' ou 'jev'."
            ) from exc
        object.__setattr__(self, "evaluator_mode", mode)

        try:
            env = EnvironmentMode(self.environment)
        except ValueError as exc:
            raise ConfigurationError(
                f"environment inválido: {self.environment!r}. "
                "Use 'development', 'test' ou 'production'."
            ) from exc
        object.__setattr__(self, "environment", env)

        if env == EnvironmentMode.PRODUCTION and self.allow_dev_identity_headers:
            raise ConfigurationError(
                "Identidade de desenvolvimento (allow_dev_identity_headers) não pode estar ativa em produção."
            )

        try:
            policy = ExternalActionPolicy(self.external_action_policy)
        except ValueError as exc:
            raise ConfigurationError(
                f"external_action_policy inválida: {self.external_action_policy!r}."
            ) from exc
        object.__setattr__(self, "external_action_policy", policy)

        if isinstance(self.db_path, str):
            object.__setattr__(self, "db_path", Path(self.db_path))

        cleaned_key = self.typesafe_api_key.strip() if self.typesafe_api_key else None
        object.__setattr__(self, "typesafe_api_key", cleaned_key or None)

        if self.evaluator_mode == EvaluatorMode.JEV and not self.typesafe_api_key:
            raise ConfigurationError(
                "TYPESAFE_API_KEY ausente com evaluator_mode='jev'. "
                "Fallback silencioso para MockEvaluator é proibido por diretriz arquitetural."
            )

        for attr_name in (
            "min_activation_salience",
            "min_candidate_score",
            "min_impulse_confidence",
            "progressive_eval_min_density",
        ):
            value = getattr(self, attr_name)
            if not (0.0 <= float(value) <= 1.0):
                raise ConfigurationError(
                    f"{attr_name} deve estar entre 0.0 e 1.0 (recebido: {value})."
                )

    @property
    def has_typesafe_key(self) -> bool:
        """Indica se há chave configurada sem expor seu valor."""
        return bool(self.typesafe_api_key)

    @classmethod
    def from_env(cls, *, explicit_mode: str | EvaluatorMode | None = None) -> SNCConfig:
        """Carrega configuração a partir de variáveis de ambiente.

        Se `PIC_SNC_EVALUATOR_MODE` estiver definida, respeita seu valor estritamente.
        Se `explicit_mode` não for passado e `PIC_SNC_EVALUATOR_MODE` não estiver definida,
        utiliza `EvaluatorMode.MOCK` apenas para desenvolvimento local, nunca fazendo
        fallback quando `jev` for solicitado.
        """
        raw_mode = (
            str(explicit_mode)
            if explicit_mode is not None
            else os.getenv("PIC_SNC_EVALUATOR_MODE", EvaluatorMode.MOCK.value)
        ).strip().lower()

        raw_env = os.getenv("PIC_SNC_ENV", EnvironmentMode.DEVELOPMENT.value).strip().lower()
        env_mode = EnvironmentMode(raw_env)
        default_dev_headers = env_mode != EnvironmentMode.PRODUCTION
        allow_dev_headers = _parse_bool(
            os.getenv("PIC_SNC_ALLOW_DEV_HEADERS"),
            default=default_dev_headers,
        )

        raw_policy = os.getenv(
            "PIC_SNC_EXTERNAL_ACTION_POLICY", ExternalActionPolicy.DENY.value
        ).strip().lower()

        return cls(
            db_path=Path(os.getenv("PIC_SNC_DB_PATH", "pic_snc.db")),
            evaluator_mode=EvaluatorMode(raw_mode),
            typesafe_api_key=os.getenv("TYPESAFE_API_KEY"),
            enable_a4=_parse_bool(os.getenv("PIC_SNC_ENABLE_A4"), default=False),
            external_action_policy=ExternalActionPolicy(raw_policy),
            require_authenticated_context=_parse_bool(
                os.getenv("PIC_SNC_REQUIRE_AUTH_CONTEXT"), default=False
            ),
            environment=env_mode,
            allow_dev_identity_headers=allow_dev_headers,
            min_activation_salience=float(
                os.getenv("PIC_SNC_MIN_ACTIVATION_SALIENCE", "0.20")
            ),
            min_candidate_score=float(
                os.getenv("PIC_SNC_MIN_CANDIDATE_SCORE", "0.35")
            ),
            min_impulse_confidence=float(
                os.getenv("PIC_SNC_MIN_IMPULSE_CONFIDENCE", "0.50")
            ),
            progressive_eval_min_density=float(
                os.getenv("PIC_SNC_PROGRESSIVE_EVAL_MIN_DENSITY", "0.40")
            ),
        )
