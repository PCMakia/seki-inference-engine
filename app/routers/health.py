"""Liveness (`/health`) and readiness (`/ready`) probes. Unauthenticated."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.backends.failover import FailoverRouter
from app.schemas import HealthResponse

router = APIRouter(tags=["probes"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Process liveness. Returns 200 as long as the gateway is running."""
    return HealthResponse(status="ok")


@router.get("/ready")
async def ready(request: Request) -> JSONResponse:
    """Ready when at least one backend (vLLM or Ollama) answers `/models`."""
    engine: FailoverRouter = request.app.state.router
    payload = await engine.readiness()
    status_code = 200 if payload["status"] == "ready" else 503
    return JSONResponse(status_code=status_code, content=payload)
