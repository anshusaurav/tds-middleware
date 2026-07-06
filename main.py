import base64
import time
import uuid
from collections import defaultdict, deque
from typing import Deque, Dict, Optional, Tuple

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware

EMAIL = "25f1002017@ds.study.iitm.ac.in"

ALLOWED_ORIGINS = {
    "https://app-b3lmdj.example.com",
    "https://exam.sanand.workers.dev",
}

# (limit, window_seconds) keyed by path prefix; longest prefix wins.
PATH_LIMITS: Dict[str, Tuple[int, float]] = {
    "/ping": (14, 10.0),
    "/orders": (20, 10.0),
}
DEFAULT_LIMIT: Tuple[int, float] = (20, 10.0)

TOTAL_ORDERS = 46

app = FastAPI()


def _bucket_scope(path: str) -> str:
    best = ""
    for prefix in PATH_LIMITS:
        if (path == prefix or path.startswith(prefix + "/")) and len(prefix) > len(best):
            best = prefix
    return best or path


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.buckets: Dict[Tuple[str, str], Deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS":
            return await call_next(request)

        client_id = request.headers.get("X-Client-Id")
        if client_id:
            scope = _bucket_scope(request.url.path)
            limit, window = PATH_LIMITS.get(scope, DEFAULT_LIMIT)
            now = time.time()
            q = self.buckets[(client_id, scope)]
            while q and now - q[0] > window:
                q.popleft()
            if len(q) >= limit:
                retry_after = max(1, int(window - (now - q[0])) + 1)
                return JSONResponse(
                    {"error": "rate_limit_exceeded",
                     "detail": f"> {limit} requests in {int(window)}s"},
                    status_code=429,
                    headers={"Retry-After": str(retry_after)},
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
                        "X-Request-ID, X-Client-Id, Idempotency-Key, Content-Type",
                    "Access-Control-Expose-Headers": "X-Request-ID, Retry-After",
                    "Access-Control-Max-Age": "600",
                    "Vary": "Origin",
                })
            return Response(status_code=204)

        response = await call_next(request)
        if allowed:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Expose-Headers"] = "X-Request-ID, Retry-After"
            response.headers["Vary"] = "Origin"
        return response


app.add_middleware(RateLimitMiddleware)
app.add_middleware(RequestContextMiddleware)
app.add_middleware(ScopedCORSMiddleware)


@app.get("/ping")
async def ping(request: Request):
    return {"email": EMAIL, "request_id": request.state.request_id}


# ---------- Orders ----------
_idempotent_orders: Dict[str, dict] = {}


@app.post("/orders", status_code=201)
async def create_order(request: Request):
    key = request.headers.get("Idempotency-Key")
    if not key:
        return JSONResponse(
            {"error": "missing Idempotency-Key header"},
            status_code=400,
        )

    if key in _idempotent_orders:
        return _idempotent_orders[key]

    payload = {}
    try:
        payload = await request.json()
    except Exception:
        payload = {}

    order = {
        "id": str(uuid.uuid4()),
        "status": "created",
        "idempotency_key": key,
    }
    if isinstance(payload, dict) and payload:
        order["data"] = payload

    _idempotent_orders[key] = order
    return order


def _encode_cursor(next_id: int) -> str:
    raw = str(next_id).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_cursor(cursor: str) -> int:
    padded = cursor + "=" * (-len(cursor) % 4)
    try:
        return int(base64.urlsafe_b64decode(padded.encode()).decode())
    except Exception:
        try:
            return int(cursor)
        except Exception:
            return 1


@app.get("/orders")
async def list_orders(
    limit: int = Query(10, ge=1, le=100),
    cursor: Optional[str] = None,
):
    start = _decode_cursor(cursor) if cursor else 1
    if start < 1:
        start = 1

    end = min(start + limit, TOTAL_ORDERS + 1)
    items = [{"id": i, "status": "created"} for i in range(start, end)]

    next_cursor: Optional[str] = None
    if end <= TOTAL_ORDERS:
        next_cursor = _encode_cursor(end)

    return {
        "items": items,
        "orders": items,
        "next_cursor": next_cursor,
        "next": next_cursor,
    }
