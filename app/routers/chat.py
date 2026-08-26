"""POST /v1/chat/completions — OpenAI Chat Completions compatible."""

from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.auth import require_api_key
from app.backends.exceptions import BackendHTTPError, BackendUnavailableError
from app.backends.failover import FailoverRouter
from app.deps import openai_error
from app.schemas import ChatCompletionRequest

router = APIRouter(prefix="/v1", tags=["chat"], dependencies=[Depends(require_api_key)])


@router.post("/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request) -> Any:
    engine: FailoverRouter = request.app.state.router
    payload = body.model_dump(exclude_none=True)

    try:
        if body.stream:
            stream, backend = await engine.chat_completions_stream(payload)
            return StreamingResponse(
                stream,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "x-seki-backend": backend,
                },
            )

        result, backend = await engine.chat_completions(payload)
        return JSONResponse(content=result, headers={"x-seki-backend": backend})
    except BackendHTTPError as exc:
        return JSONResponse(status_code=exc.status_code, content=exc.body)
    except BackendUnavailableError as exc:
        return JSONResponse(
            status_code=502,
            content=openai_error(
                message=f"All inference backends failed: {exc}",
                err_type="api_connection_error",
                code="backend_unavailable",
            ),
        )
