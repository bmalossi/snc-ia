"""Worker independente do motor cognitivo PIC-SNC v0.1.

Responsabilidades:
- Executar em processo separado da API FastAPI.
- Recuperar eventos presos em `processing` na inicialização (`recover_on_startup`).
- Reivindicar eventos pendentes atomicamente do SQLite (`claim_pending_events`).
- Executar o pipeline cognitivo completo (`Observe -> Activate -> Evaluate -> Govern -> Publish`)
  com semântica at-least-once e processamento idempotente.
- Registrar falhas com logs estruturados sem interromper o loop por falhas individuais.
- Suportar parada graciosa via `SIGINT`/`SIGTERM` ou `stop()` e polling configurável.
"""

from __future__ import annotations

import argparse
import logging
import signal
import threading
from pathlib import Path

from pic_snc.config import SNCConfig
from pic_snc.logging_utils import get_structured_logger, log_cognitive_step
from pic_snc.pipeline import CognitivePipeline, PipelineExecutionResult


class CognitiveWorker:
    """Trabalhador independente de processamento cognitivo para o PIC-SNC v0.1."""

    def __init__(
        self,
        config: SNCConfig,
        pipeline: CognitivePipeline | None = None,
        *,
        raise_on_error: bool = False,
    ) -> None:
        self.config = config
        self.pipeline = pipeline or CognitivePipeline(config)
        self.storage = self.pipeline.storage
        self.raise_on_error = raise_on_error
        self._stop_event = threading.Event()
        self._logger = get_structured_logger("pic_snc.worker")

    def recover_on_startup(self, *, tenant_id: str | None = None) -> int:
        """Recupera eventos interrompidos (`processing` -> `pending`) após queda ou reinício."""
        recovered_count = self.storage.recover_interrupted_events(tenant_id=tenant_id)
        self.pipeline._cached_decisions.clear()
        log_cognitive_step(
            self._logger,
            stage="worker.startup",
            action="interrupted_events_recovered",
            tenant_id=tenant_id or "*",
            details={"recovered_count": recovered_count},
        )
        return recovered_count

    def stop(self) -> None:
        """Solicita parada graciosa do loop de polling do worker."""
        self._stop_event.set()
        log_cognitive_step(
            self._logger,
            stage="worker.lifecycle",
            action="stop_requested",
            tenant_id="*",
        )

    def run_once(
        self,
        *,
        tenant_id: str | None = None,
        include_failed: bool = False,
        limit: int = 20,
    ) -> list[PipelineExecutionResult]:
        """Reivindica um lote de eventos pendentes e executa o pipeline cognitivo."""
        claimed = self.storage.claim_pending_events(
            tenant_id=tenant_id,
            include_failed=include_failed,
            limit=limit,
        )
        results: list[PipelineExecutionResult] = []

        for event in claimed:
            try:
                res = self.pipeline._execute_for_persisted_event(event)
                results.append(res)
            except Exception as exc:
                log_cognitive_step(
                    self._logger,
                    stage="worker.run_once",
                    action="event_processing_failed",
                    tenant_id=event.tenant_id,
                    event_id=event.event_id,
                    level=logging.ERROR,
                    details={"error_type": type(exc).__name__, "error": str(exc)},
                )
                if self.raise_on_error:
                    raise

        return results

    def run_forever(
        self,
        *,
        poll_interval_seconds: float = 0.5,
        include_failed: bool = False,
        max_iterations: int | None = None,
        install_signal_handlers: bool = True,
    ) -> None:
        """Executa o loop contínuo de polling com parada graciosa."""
        self._stop_event.clear()

        if install_signal_handlers and threading.current_thread() is threading.main_thread():
            def _handle_signal(signum: int, _frame: object) -> None:
                log_cognitive_step(
                    self._logger,
                    stage="worker.lifecycle",
                    action="signal_received",
                    tenant_id="*",
                    details={"signal": signum},
                )
                self.stop()

            signal.signal(signal.SIGINT, _handle_signal)
            if hasattr(signal, "SIGTERM"):
                signal.signal(signal.SIGTERM, _handle_signal)

        self.recover_on_startup()
        log_cognitive_step(
            self._logger,
            stage="worker.lifecycle",
            action="worker_started",
            tenant_id="*",
            details={
                "poll_interval_seconds": poll_interval_seconds,
                "include_failed": include_failed,
                "evaluator_mode": self.config.evaluator_mode.value,
            },
        )

        iterations = 0
        while not self._stop_event.is_set():
            self.run_once(include_failed=include_failed)
            iterations += 1
            if max_iterations is not None and iterations >= max_iterations:
                break
            self._stop_event.wait(timeout=poll_interval_seconds)

        log_cognitive_step(
            self._logger,
            stage="worker.lifecycle",
            action="worker_stopped",
            tenant_id="*",
            details={"iterations": iterations},
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PIC-SNC v0.1 — Worker Cognitivo Independente"
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Caminho do banco SQLite (padrão: PIC_SNC_DB_PATH ou pic_snc.db).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.5,
        help="Intervalo de polling em segundos (padrão: 0.5).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Executa apenas uma varredura de eventos pendentes e encerra.",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Inclui eventos com status 'failed' para reprocessamento.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    base_cfg = SNCConfig.from_env()
    if args.db_path is not None:
        cfg = SNCConfig(
            db_path=args.db_path,
            evaluator_mode=base_cfg.evaluator_mode,
            typesafe_api_key=base_cfg.typesafe_api_key,
            enable_a4=base_cfg.enable_a4,
            external_action_policy=base_cfg.external_action_policy,
            require_authenticated_context=base_cfg.require_authenticated_context,
            environment=base_cfg.environment,
            allow_dev_identity_headers=base_cfg.allow_dev_identity_headers,
            static_api_tokens=base_cfg.static_api_tokens,
        )
    else:
        cfg = base_cfg

    worker = CognitiveWorker(cfg)
    if args.once:
        worker.recover_on_startup()
        worker.run_once(include_failed=args.include_failed)
    else:
        worker.run_forever(
            poll_interval_seconds=args.poll_interval,
            include_failed=args.include_failed,
        )


if __name__ == "__main__":
    main()
