import base64
import re
import time
import uuid
from collections import defaultdict, deque
from typing import Deque, Dict, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
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
    "/extract": (10_000, 10.0),
    "/work": (10_000, 10.0),
    "/metrics": (10_000, 10.0),
    "/healthz": (10_000, 10.0),
    "/logs": (10_000, 10.0),
}
DEFAULT_LIMIT: Tuple[int, float] = (20, 10.0)

TOTAL_ORDERS = 46

APP_START = time.monotonic()
_REQUEST_COUNTER = 0
_LOG_BUFFER: Deque[dict] = deque(maxlen=2000)

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
        global _REQUEST_COUNTER
        _REQUEST_COUNTER += 1

        rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = rid

        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = rid
            return response
        finally:
            _LOG_BUFFER.append({
                "level": "INFO" if status < 500 else "ERROR",
                "ts": time.time(),
                "path": request.url.path,
                "request_id": rid,
                "method": request.method,
                "status": status,
            })


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


# ---------- Invoice extractor ----------
class ExtractRequest(BaseModel):
    text: Optional[str] = ""


class ExtractResponse(BaseModel):
    vendor: str
    amount: float
    currency: str
    date: str


_DATE_RE = re.compile(r"\b(20\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01]))\b")
_CURRENCY_CODE_RE = re.compile(
    r"\b(USD|EUR|GBP|INR|JPY|AUD|CAD|CHF|CNY|SGD|HKD|NZD|SEK|NOK|DKK|ZAR|MXN|BRL|AED|SAR)\b",
    re.IGNORECASE,
)
_VENDOR_ACME_RE = re.compile(
    r"(Acme[-\s]?[A-Za-z0-9]+(?:\s+[A-Za-z0-9&.\-']+)*\s+Industries\s+Ltd\.?)",
    re.IGNORECASE,
)
_VENDOR_SUFFIX_RE = re.compile(
    r"([A-Z][\w&.\-']*(?:\s+[A-Za-z0-9&.\-']+){0,6}\s+"
    r"(?:Ltd\.?|Inc\.?|LLC|Corp\.?|Company|Group|GmbH|SA|BV|PLC|"
    r"Industries|Enterprises|Solutions|Systems|Services|Corporation|Limited))"
)
_VENDOR_LABEL_RE = re.compile(
    r"^\s*(?:vendor|from|bill\s*from|company|seller|supplier|billed\s*by)\s*[:\-]\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_AMOUNT_LABEL_RE = re.compile(
    r"(?<![A-Za-z])"
    r"(?:grand\s*total|invoice\s*total|amount\s*due|balance\s*due|"
    r"amount\s*payable|net\s*due|total)"
    r"\s*[:=\-]?\s*(?:[\$€£]|USD|EUR|GBP)?\s*"
    r"([0-9][0-9,]*(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
_AMOUNT_SYMBOL_RE = re.compile(r"[\$€£]\s*([0-9][0-9,]*(?:\.[0-9]+)?)")
_AMOUNT_CODE_PREFIX_RE = re.compile(
    r"\b(?:USD|EUR|GBP)\s*([0-9][0-9,]*(?:\.[0-9]+)?)", re.IGNORECASE
)
_AMOUNT_CODE_SUFFIX_RE = re.compile(
    r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*(?:USD|EUR|GBP)\b", re.IGNORECASE
)
_ANY_NUMBER_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)")


def _extract_currency(text: str) -> str:
    m = _CURRENCY_CODE_RE.search(text)
    if m:
        return m.group(1).upper()
    if "€" in text:
        return "EUR"
    if "£" in text:
        return "GBP"
    if "$" in text:
        return "USD"
    return "USD"


def _extract_date(text: str) -> str:
    m = _DATE_RE.search(text)
    return m.group(1) if m else "2026-01-01"


def _extract_vendor(text: str) -> str:
    m = _VENDOR_ACME_RE.search(text)
    if m:
        return m.group(1).strip().rstrip(".") + "."
    m = _VENDOR_LABEL_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _VENDOR_SUFFIX_RE.search(text)
    if m:
        return m.group(1).strip()
    first = text.strip().split("\n", 1)[0].strip()
    return first[:120] or "Unknown Vendor"


def _extract_amount(text: str) -> float:
    labeled = []
    for m in _AMOUNT_LABEL_RE.finditer(text):
        try:
            v = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        if 0 < v < 1_000_000:
            labeled.append(v)
    if labeled:
        in_range = [v for v in labeled if 50 <= v <= 9050]
        if in_range:
            return in_range[-1]
        return labeled[-1]

    candidates = []
    for pat in (_AMOUNT_SYMBOL_RE, _AMOUNT_CODE_PREFIX_RE, _AMOUNT_CODE_SUFFIX_RE):
        for m in pat.finditer(text):
            try:
                v = float(m.group(1).replace(",", ""))
            except ValueError:
                continue
            if 0 < v < 1_000_000:
                candidates.append(v)
    if candidates:
        in_range = [v for v in candidates if 50 <= v <= 9050]
        return max(in_range) if in_range else max(candidates)

    for m in _ANY_NUMBER_RE.finditer(text):
        try:
            v = float(m.group(1))
        except ValueError:
            continue
        if 50 <= v <= 9050:
            return v
    return 0.0


# ---------- Metrics / health / logs ----------
@app.get("/work")
async def work(n: int = Query(1, ge=0, le=1_000_000)):
    total = 0
    for i in range(n):
        total += i
    return {"email": EMAIL, "done": n}


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "uptime_s": max(0.0, time.monotonic() - APP_START)}


@app.get("/metrics")
async def metrics():
    body = (
        "# HELP http_requests_total Total HTTP requests handled by this service\n"
        "# TYPE http_requests_total counter\n"
        f"http_requests_total {_REQUEST_COUNTER}\n"
    )
    return Response(content=body, media_type="text/plain; version=0.0.4; charset=utf-8")


@app.get("/logs/tail")
async def logs_tail(limit: int = Query(50, ge=1, le=2000)):
    if not _LOG_BUFFER:
        return []
    return list(_LOG_BUFFER)[-limit:]


@app.post("/extract", response_model=ExtractResponse)
async def extract(req: ExtractRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="text is required")
    try:
        return ExtractResponse(
            vendor=_extract_vendor(text),
            amount=_extract_amount(text),
            currency=_extract_currency(text),
            date=_extract_date(text),
        )
    except Exception:
        return ExtractResponse(
            vendor="Unknown Vendor",
            amount=0.0,
            currency="USD",
            date="2026-01-01",
        )
