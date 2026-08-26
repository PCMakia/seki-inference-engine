"""POST /v1/embeddings — OpenAI Embeddings compatible."""

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from app.auth import require_api_key
from app.backends.exceptions import BackendHTTPError, BackendUnavailableError
from app.backends.failover import FailoverRouter
from app.deps import openai_error
from app.schemas import EmbeddingRequest

router = APIRouter(prefix="/v1", tags=["embeddings"], dependencies=[Depends(require_api_key)])


@router.post("/embeddings")
async def embeddings(body: EmbeddingRequest, request: Request) -> JSONResponse:
    engine: FailoverRouter = request.app.state.router
    payload = body.model_dump(exclude_none=True)

    try:
        result, backend = await engine.embeddings(payload)
        return JSONResponse(content=result, headers={"x-seki-backend": backend})
    except BackendHTTPError as exc:
        return JSONResponse(status_code=exc.status_code, content=exc.body)
    except BackendUnavailableError as exc:
        return JSONResponse(
            status_code=502,
            content=openai_error(
                message=f"All embedding backends failed: {exc}",
                err_type="api_connection_error",
                code="backend_unavailable",
            ),
        )
