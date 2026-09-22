"""HTTP serving runtime: POST /v1/systemone, GET /v1/models, GET /healthz.

One scorer call at a time. Benchmark runs are serial anyway, and a single lock keeps GPU execution order, and so results, deterministic.

    python -m compass.server --model-name Qwen/Qwen3.5-4B --port 8000
"""

from __future__ import annotations

import argparse
import json
import threading

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from . import __version__, release
from .calibration import Calibrator
from .contract import LATEST_ALIAS, ContractError, ModelOutputError, Scorer, UnknownModel, decide
from .scorer import SCORERS, build


def _error(status: int, kind: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"type": kind, "message": message}}, status_code=status)


def create_app(scorer: Scorer, calibrator: Calibrator | None = None) -> FastAPI:
    calibrator = calibrator or Calibrator()
    lock = threading.Lock()
    app = FastAPI(title="Compass", version=__version__)

    def run(body):
        with lock:
            return decide(body, scorer, calibrator)

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        try:
            body = json.loads(await request.body())
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            return _error(400, "invalid_json", str(e))
        try:
            return JSONResponse(await run_in_threadpool(run, body))
        except UnknownModel as e:
            return _error(404, "unknown_model", str(e))
        except ContractError as e:
            return _error(422, "invalid_request", str(e))
        except ModelOutputError as e:
            return _error(500, "model_error", str(e))

    @app.get("/v1/models")
    def models():
        return {"data": [{"id": scorer.model_id, "aliases": [LATEST_ALIAS], "calibration": calibrator.source,
                          "backbone": getattr(scorer, "backbone", None), "prompt_version": getattr(scorer, "prompt_version", None),
                          "package": __version__}]}

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "model": scorer.model_id}

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Serve Compass on /v1/systemone.")
    parser.add_argument("--scorer", choices=SCORERS, default="backbone")
    parser.add_argument("--model-name", default=release.BACKBONE, help="backbone: Hugging Face model id")
    parser.add_argument("--revision", default=release.BACKBONE_REVISION, help="backbone: pinned Hub revision")
    parser.add_argument("--model-id", default=None, help=f"id reported in responses; defaults to {release.MODEL_ID} for the pinned backbone")
    parser.add_argument("--device", help="backbone: cuda, mps or cpu; auto-detected when omitted")
    parser.add_argument("--head", help="backbone: trained verification head (Stage B); omitted means vocabulary readout")
    parser.add_argument("--calibration", help="fitted calibration JSON; omitted means uncalibrated")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    calibrator = Calibrator.load(args.calibration) if args.calibration else Calibrator()
    model_id = args.model_id or (release.MODEL_ID if (args.model_name, args.revision) == (release.BACKBONE, release.BACKBONE_REVISION) else None)
    scorer = build(args.scorer, model_name=args.model_name, revision=args.revision, device=args.device, head_path=args.head, model_id=model_id)
    uvicorn.run(create_app(scorer, calibrator), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
