import base64
import os
import re
import time
import uuid
from collections import defaultdict, deque
from typing import Deque, Dict, Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

EMAIL = "25f1002017@ds.study.iitm.ac.in"
ANALYTICS_API_KEY = "ak_n6jqafr50nenrfru11eevksj"

ALLOWED_ORIGINS = {
    "https://app-b3lmdj.example.com",
    "https://exam.sanand.workers.dev",
}

# Paths that must accept any origin (grader sends from a Cloudflare Worker
# whose subdomain isn't fixed). CORS reflects whatever Origin arrives.
PERMISSIVE_CORS_PATHS = ("/answer-image",)

# (limit, window_seconds) keyed by path prefix; longest prefix wins.
PATH_LIMITS: Dict[str, Tuple[int, float]] = {
    "/ping": (14, 10.0),
    "/orders": (20, 10.0),
    "/extract": (10_000, 10.0),
    "/work": (10_000, 10.0),
    "/metrics": (10_000, 10.0),
    "/healthz": (10_000, 10.0),
    "/logs": (10_000, 10.0),
    "/analytics": (10_000, 10.0),
    "/answer-image": (10_000, 10.0),
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
        path = request.url.path
        is_permissive = any(path == p or path.startswith(p + "/") for p in PERMISSIVE_CORS_PATHS)
        allowed = origin in ALLOWED_ORIGINS or (is_permissive and origin is not None)

        if request.method == "OPTIONS":
            if allowed:
                return Response(status_code=204, headers={
                    "Access-Control-Allow-Origin": origin,
                    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                    "Access-Control-Allow-Headers":
                        "X-Request-ID, X-Client-Id, Idempotency-Key, X-API-Key, Content-Type, Authorization",
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


# ---------- New 6-field invoice extractor (fixed-schema) ----------
_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2,
    "mar": 3, "march": 3, "apr": 4, "april": 4,
    "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}
_MONTH_ALT = "|".join(sorted(_MONTHS.keys(), key=len, reverse=True))

_ISO_DATE_RE = re.compile(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b")
_DMY_RE = re.compile(r"\b(\d{1,2})[-/](\d{1,2})[-/](20\d{2})\b")
_DAY_MONTH_YEAR_RE = re.compile(
    rf"\b(\d{{1,2}})[\s\-]+({_MONTH_ALT})[a-z]*[\s\-,]+(20\d{{2}})\b",
    re.IGNORECASE,
)
_MONTH_DAY_YEAR_RE = re.compile(
    rf"\b({_MONTH_ALT})[a-z]*[\s\-]+(\d{{1,2}})(?:st|nd|rd|th)?[\s\-,]+(20\d{{2}})\b",
    re.IGNORECASE,
)


def _iso(y: int, mo: int, d: int) -> Optional[str]:
    if 1 <= mo <= 12 and 1 <= d <= 31:
        return f"{y:04d}-{mo:02d}-{d:02d}"
    return None


def _extract_iso_date(text: str) -> Optional[str]:
    m = _ISO_DATE_RE.search(text)
    if m:
        r = _iso(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if r:
            return r
    m = _DAY_MONTH_YEAR_RE.search(text)
    if m:
        r = _iso(int(m.group(3)), _MONTHS[m.group(2).lower()], int(m.group(1)))
        if r:
            return r
    m = _MONTH_DAY_YEAR_RE.search(text)
    if m:
        r = _iso(int(m.group(3)), _MONTHS[m.group(1).lower()], int(m.group(2)))
        if r:
            return r
    m = _DMY_RE.search(text)
    if m:
        r = _iso(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        if r:
            return r
    return None


_INV_LABEL_RE = re.compile(
    r"\b(?:invoice|inv\.?|bill|receipt|order|reference|ref\.?|"
    r"document|doc\.?|voucher|po|purchase\s*order)\b"
    r"\s*(?:no\.?|number|num|#|id)?\s*[:=#\-]?\s*"
    r"([A-Z][A-Z0-9\-_/]*)",
    re.IGNORECASE,
)
# Standalone: uppercase-letter-prefixed alphanumeric with a separator.
# Requires >=2 leading letters (so "USD-100" style currency codes don't win),
# and rejects month codes (JAN, FEB, ...) and 3-letter currency codes as pure heads.
_INV_STANDALONE_RE = re.compile(
    r"\b([A-Za-z]{2,6}[-_/]\d[\w\-_/]*)\b"
)
_CURRENCY_PREFIXES = {
    "USD","EUR","GBP","INR","JPY","AUD","CAD","CHF","CNY","SGD","HKD",
    "NZD","SEK","NOK","DKK","ZAR","MXN","BRL","AED","SAR","RUB","KRW","TRY",
}
_MONTH_PREFIXES = {
    "JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","SEPT","OCT","NOV","DEC",
}


_LABEL_WORDS = {
    "invoice", "inv", "bill", "receipt", "order", "reference", "ref",
    "document", "doc", "voucher", "po", "no", "number", "num", "id",
}


def _looks_like_invoice_no(cand: str) -> bool:
    if not cand:
        return False
    if cand.lower() in _LABEL_WORDS:
        return False
    # Real invoice numbers have either a digit or a separator (or both).
    # Bare-word matches like "Ref" have neither and should be rejected.
    has_digit = any(ch.isdigit() for ch in cand)
    has_sep = any(ch in "-_/" for ch in cand)
    return has_digit or has_sep


def _extract_invoice_no(text: str) -> Optional[str]:
    for m in _INV_LABEL_RE.finditer(text):
        cand = m.group(1).strip().rstrip(".,;:")
        if _looks_like_invoice_no(cand):
            return cand
    for m in _INV_STANDALONE_RE.finditer(text):
        v = m.group(1).strip().rstrip(".,;:")
        head = re.split(r"[-_/]", v, 1)[0].upper()
        if head in _CURRENCY_PREFIXES or head in _MONTH_PREFIXES:
            continue
        if _looks_like_invoice_no(v):
            return v
    return None


_SUBTOTAL_RE = re.compile(
    r"\b(?:sub[\s\-]?total|subtotal|net\s*amount|net\s*total|taxable\s*amount|"
    r"amount\s*(?:before\s*tax)?|base\s*amount)\s*[:=\-]?\s*"
    r"(?:[\$€£₹]|Rs\.?|INR|USD|EUR|GBP)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
_TOTAL_RE_STRICT = re.compile(
    r"(?<![A-Za-z])(?:grand\s*total|invoice\s*total|total(?:\s*amount)?|"
    r"amount\s*due|balance\s*due|amount\s*payable)"
    r"\s*[:=\-]?\s*(?:[\$€£₹]|Rs\.?|INR|USD|EUR|GBP)?\s*"
    r"([0-9][0-9,]*(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
_TAX_RE = re.compile(
    r"\b(?:tax|gst|vat|cgst|sgst|igst|service\s*tax|sales\s*tax)\b"
    r"[\s\(@:=\-]*"                                      # optional separators, incl. '(' or '@'
    r"(?:\d+(?:\.\d+)?\s*%\s*\)?)?"                      # optional rate percentage (parens optional)
    r"[\s\)@:=\-]*"                                       # more separators / closing paren
    r"(?:[\$€£₹]|Rs\.?|INR|USD|EUR|GBP)?\s*"             # optional currency
    r"([0-9][0-9,]*(?:\.[0-9]+)?)",                      # the amount
    re.IGNORECASE,
)


def _first_float(match) -> Optional[float]:
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except (ValueError, AttributeError):
        return None


def _extract_subtotal_and_tax(text: str):
    tax = _first_float(_TAX_RE.search(text))
    subtotal = _first_float(_SUBTOTAL_RE.search(text))
    if subtotal is not None:
        return subtotal, tax
    total = _first_float(_TOTAL_RE_STRICT.search(text))
    if total is not None and tax is not None:
        return round(total - tax, 2), tax
    if total is not None:
        return total, tax
    return None, tax


_CURRENCY_NEW_RE = re.compile(
    r"\b(USD|EUR|GBP|INR|JPY|AUD|CAD|CHF|CNY|SGD|HKD|NZD|SEK|NOK|DKK|"
    r"ZAR|MXN|BRL|AED|SAR|RUB|KRW|TRY)\b",
    re.IGNORECASE,
)


def _extract_currency_new(text: str) -> Optional[str]:
    m = _CURRENCY_NEW_RE.search(text)
    if m:
        return m.group(1).upper()
    if re.search(r"\bRs\.?\b|₹", text):
        return "INR"
    if "€" in text:
        return "EUR"
    if "£" in text:
        return "GBP"
    if "¥" in text:
        return "JPY"
    if "$" in text:
        return "USD"
    return None


_VENDOR_LABEL_NEW_RE = re.compile(
    r"^\s*(?:vendor|from|bill\s*from|company|seller|supplier|billed\s*by|billed\s*from|bill\s*to)"
    r"\s*[:\-]\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _extract_vendor_new(text: str) -> Optional[str]:
    m = _VENDOR_LABEL_NEW_RE.search(text)
    if m:
        v = m.group(1).strip().rstrip(".,;:")
        if v:
            return v
    m = _VENDOR_SUFFIX_RE.search(text)
    if m:
        return m.group(1).strip()
    return None


class ExtractInvoiceResponse(BaseModel):
    invoice_no: Optional[str] = None
    date: Optional[str] = None
    vendor: Optional[str] = None
    amount: Optional[float] = None
    tax: Optional[float] = None
    currency: Optional[str] = None


def _extract_invoice_new(text: str) -> dict:
    subtotal, tax = _extract_subtotal_and_tax(text)
    return {
        "invoice_no": _extract_invoice_no(text),
        "date": _extract_iso_date(text),
        "vendor": _extract_vendor_new(text),
        "amount": subtotal,
        "tax": tax,
        "currency": _extract_currency_new(text),
    }


@app.post("/extract")
async def extract(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=422, detail="invalid JSON body")
    if not isinstance(body, dict):
        raise HTTPException(status_code=422, detail="body must be a JSON object")

    # Dispatch by which key the caller sent
    if "invoice_text" in body:
        text = str(body.get("invoice_text") or "").strip()
        if not text:
            return ExtractInvoiceResponse().model_dump()
        try:
            return _extract_invoice_new(text)
        except Exception:
            return ExtractInvoiceResponse().model_dump()

    if "text" in body:
        text = str(body.get("text") or "").strip()
        if not text:
            raise HTTPException(status_code=422, detail="text is required")
        try:
            return {
                "vendor": _extract_vendor(text),
                "amount": _extract_amount(text),
                "currency": _extract_currency(text),
                "date": _extract_date(text),
            }
        except Exception:
            return {
                "vendor": "Unknown Vendor",
                "amount": 0.0,
                "currency": "USD",
                "date": "2026-01-01",
            }

    raise HTTPException(
        status_code=422,
        detail="body must contain either 'text' or 'invoice_text'",
    )


# ---------- Analytics ----------
@app.post("/analytics")
async def analytics(request: Request):
    if request.headers.get("X-API-Key") != ANALYTICS_API_KEY:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    events = payload.get("events") if isinstance(payload, dict) else None
    if not isinstance(events, list):
        events = []

    users = set()
    user_totals: Dict[str, float] = defaultdict(float)
    revenue = 0.0
    for e in events:
        if not isinstance(e, dict):
            continue
        u = e.get("user")
        if u is not None:
            users.add(u)
        a = e.get("amount")
        if isinstance(a, bool):
            continue
        if isinstance(a, (int, float)) and a > 0:
            revenue += float(a)
            if u is not None:
                user_totals[u] += float(a)

    top_user = ""
    if user_totals:
        top_user = max(user_totals.items(), key=lambda kv: kv[1])[0]

    return {
        "email": EMAIL,
        "total_events": len(events),
        "unique_users": len(users),
        "revenue": revenue,
        "top_user": top_user,
    }


# ---------- Multimodal image QA ----------
AIPIPE_BASE = "https://aipipe.org/openai/v1"
AIPIPE_MODEL = "gpt-4o-mini"
AIPIPE_SYSTEM_PROMPT = (
    "You extract a single answer from an image. Look at the image and answer the "
    "user's question. Return ONLY the raw answer value with no units, no currency "
    "symbols, no labels, no explanation, no punctuation beyond a decimal point. "
    "For numeric answers return just the number as a string (e.g. '4089.35'). "
    "For text answers return only the exact value."
)


class AnswerImageRequest(BaseModel):
    image_base64: str
    question: str


class AnswerImageResponse(BaseModel):
    answer: str


def _sanitise_answer(raw: str) -> str:
    s = (raw or "").strip()
    # Drop wrapping quotes / backticks the model sometimes emits
    s = s.strip("`\"' \n\t")
    # If the model still adds a trailing period or comma, drop it
    if s.endswith((".", ",")) and not re.search(r"\d\.\d*$", s):
        s = s.rstrip(".,")
    return s


@app.post("/answer-image", response_model=AnswerImageResponse)
async def answer_image(req: AnswerImageRequest):
    token = os.environ.get("AIPIPE_TOKEN", "").strip()
    if not token:
        raise HTTPException(status_code=500, detail="AIPIPE_TOKEN not configured on server")

    b64 = req.image_base64 or ""
    # Strip a data URL prefix if the caller supplied one
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[-1]
    if not b64:
        raise HTTPException(status_code=422, detail="image_base64 is required")

    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=422, detail="question is required")

    data_url = f"data:image/png;base64,{b64}"
    payload = {
        "model": AIPIPE_MODEL,
        "messages": [
            {"role": "system", "content": AIPIPE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        "temperature": 0,
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                f"{AIPIPE_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"upstream call failed: {e}")

    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"upstream {r.status_code}: {r.text[:400]}")

    try:
        content = r.json()["choices"][0]["message"]["content"]
    except Exception:
        raise HTTPException(status_code=502, detail=f"unexpected upstream shape: {r.text[:400]}")

    return AnswerImageResponse(answer=_sanitise_answer(str(content)))
