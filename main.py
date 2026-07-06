import time
import uuid
from collections import defaultdict, deque
from typing import Deque, Dict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware

EMAIL = "anshu.saurav@gmail.com"

ALLOWED_ORIGINS = {
    "https://app-b3lmdj.example.com",
    "https://exam.sanand.workers.dev",
}

BUCKET_SIZE = 14
WINDOW_SECONDS = 10.0

app = FastAPI()


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.buckets: Dict[str, Deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS":
            return await call_next(request)

        client_id = request.headers.get("X-Client-Id")
        if client_id:
            now = time.time()
            q = self.buckets[client_id]
            while q and now - q[0] > WINDOW_SECONDS:
                q.popleft()
            if len(q) >= BUCKET_SIZE:
                return JSONResponse(
                    {"error": "rate_limit_exceeded",
                     "detail": f"> {BUCKET_SIZE} requests in {int(WINDOW_SECONDS)}s"},
                    status_code=429,
                )
            q.append(now)

        return await call_next(request)


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response


class ScopedCORSMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        origin = request.headers.get("Origin")
        allowed = origin in ALLOWED_ORIGINS

        if request.method == "OPTIONS":
            if allowed:
                return Response(status_code=204, headers={
                    "Access-Control-Allow-Origin": origin,
                    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                    "Access-Control-Allow-Headers":
                        "X-Request-ID, X-Client-Id, Content-Type",
                    "Access-Control-Expose-Headers": "X-Request-ID",
                    "Access-Control-Max-Age": "600",
                    "Vary": "Origin",
                })
            return Response(status_code=204)

        response = await call_next(request)
        if allowed:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Expose-Headers"] = "X-Request-ID"
            response.headers["Vary"] = "Origin"
        return response


app.add_middleware(RateLimitMiddleware)
app.add_middleware(RequestContextMiddleware)
app.add_middleware(ScopedCORSMiddleware)


@app.get("/ping")
async def ping(request: Request):
    return {"email": EMAIL, "request_id": request.state.request_id}
