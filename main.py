import os
import json
import re
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional, List

import httpx
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageOps
from io import BytesIO

# ─── Config ────────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL       = os.getenv("OLLAMA_MODEL",    "qwen2.5:3b")
OLLAMA_TIMEOUT_SEC = float(os.getenv("OLLAMA_TIMEOUT_SEC", "300"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "2500"))
OCR_LANG          = os.getenv("OCR_LANG",        "en")
OCR_CONF_THRESH   = float(os.getenv("OCR_CONF_THRESH", "0.4"))
MAX_IMAGE_PX      = int(os.getenv("MAX_IMAGE_PX", "2048"))

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s  %(levelname)-7s  %(message)s",
)
log = logging.getLogger("receipt-extractor")


# ─── Pydantic models ───────────────────────────────────────────────────────────

class LineItem(BaseModel):
    name:   str
    qty:    Optional[float] = None
    rate:   Optional[float] = None
    amount: Optional[float] = None

class TaxEntry(BaseModel):
    name:   str
    rate:   Optional[str]   = None
    amount: Optional[float] = None

class ReceiptData(BaseModel):
    merchant_name: Optional[str]   = None
    bill_number:   Optional[str]   = None
    date:          Optional[str]   = None
    items:         List[LineItem]  = []
    sub_total:     Optional[float] = None
    discount:      Optional[float] = None
    taxes:         List[TaxEntry]  = []
    total_amount:  Optional[float] = None
    currency:      Optional[str]   = None
    payment_mode:  Optional[str]   = None

class ExtractResponse(BaseModel):
    success:    bool
    data:       Optional[ReceiptData] = None
    ocr_text:   Optional[str]         = None
    latency_ms: Optional[int]         = None
    error:      Optional[str]         = None


# ─── PaddleOCR: version-aware singleton ───────────────────────────────────────

_ocr_engine  = None
_paddle_major = 2          # filled in by get_ocr()


def _detect_paddle_major() -> int:
    try:
        import paddleocr as _pm
        return int(getattr(_pm, "__version__", "2.0.0").split(".")[0])
    except Exception:
        return 2


def get_ocr():
    """
    Lazy singleton.  Handles PaddleOCR v2 and v3 which have different __init__
    signatures — v3 removed show_log / use_gpu / enable_mkldnn.
    """
    global _ocr_engine, _paddle_major
    if _ocr_engine is not None:
        return _ocr_engine

    _paddle_major = _detect_paddle_major()
    log.info("Detected PaddleOCR major version: %d", _paddle_major)

    from paddleocr import PaddleOCR

    if _paddle_major >= 3:
        # v3.x — skip doc-orientation + unwarping (EXIF + flat receipts); keep
        # server det/rec models and textline orientation for accuracy.
        _ocr_engine = PaddleOCR(
            lang=OCR_LANG,
            enable_mkldnn=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
    else:
        # v2.x — richer init options
        _ocr_engine = PaddleOCR(
            use_angle_cls=True,
            lang=OCR_LANG,
            use_gpu=False,
            show_log=False,
            enable_mkldnn=True,
        )

    log.info("PaddleOCR ready.")
    return _ocr_engine


# ─── Image pre-processing ─────────────────────────────────────────────────────

def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)   # fix phone-photo rotation
    img = img.convert("RGB")
    w, h  = img.size
    scale = MAX_IMAGE_PX / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.array(img)


# ─── OCR runner — handles v2 and v3 output formats ────────────────────────────

def run_ocr(img_arr: np.ndarray) -> str:
    """
    Run OCR and return reconstructed plain text.

    PaddleOCR v2  →  ocr(img, cls=True)  →  list[ list[ [bbox,(text,conf)] ] ]
    PaddleOCR v3  →  predict(img)        →  list[ dict{dt_polys, rec_texts, …} ]

    Both paths feed into the same row-clustering logic.
    """
    engine = get_ocr()
    blocks = []

    if _paddle_major >= 3:
        # v3 uses predict()
        results = engine.predict(img_arr)
        if not results:
            return ""
        page = results[0]  # single-image → first (only) page

        rec_texts  = page.get("rec_texts",  [])
        rec_scores = page.get("rec_scores", [1.0] * len(rec_texts))
        dt_polys   = page.get("dt_polys",   [])

        for i, text in enumerate(rec_texts):
            conf = float(rec_scores[i]) if i < len(rec_scores) else 1.0
            if conf < OCR_CONF_THRESH:
                continue
            if i < len(dt_polys):
                box = dt_polys[i]           # shape (4, 2) or list of 4 [x,y]
                xs  = [float(p[0]) for p in box]
                ys  = [float(p[1]) for p in box]
                blocks.append({
                    "text":     text,
                    "x_min":    min(xs),
                    "y_center": (min(ys) + max(ys)) / 2,
                    "height":   max(ys) - min(ys),
                })
            else:
                # no bbox — append sequentially so text isn't lost
                blocks.append({
                    "text":     text,
                    "x_min":    0.0,
                    "y_center": float(len(blocks) * 20),
                    "height":   16.0,
                })

    else:
        # v2 uses ocr()
        result = engine.ocr(img_arr, cls=True)
        if not result or not result[0]:
            return ""
        for item in result[0]:
            box, (text, conf) = item
            if conf < OCR_CONF_THRESH:
                continue
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            blocks.append({
                "text":     text,
                "x_min":    min(xs),
                "y_center": (min(ys) + max(ys)) / 2,
                "height":   max(ys) - min(ys),
            })

    return _cluster_and_join(blocks)


def _cluster_and_join(blocks: list) -> str:
    """
    Sort blocks top→bottom, cluster into rows by Y proximity, then join each
    row left→right with two spaces.  Keeps 'Item  Qty  Rate  Amount' on one line
    so the LLM can parse multi-column receipt tables correctly.
    """
    if not blocks:
        return ""

    blocks.sort(key=lambda b: b["y_center"])

    rows: list[list[dict]] = [[blocks[0]]]
    for blk in blocks[1:]:
        cur_row   = rows[-1]
        avg_y     = sum(b["y_center"] for b in cur_row) / len(cur_row)
        avg_h     = sum(b["height"]   for b in cur_row) / len(cur_row)
        threshold = max(avg_h * 0.6, 6)

        if abs(blk["y_center"] - avg_y) <= threshold:
            cur_row.append(blk)
        else:
            rows.append([blk])

    lines = []
    for row in rows:
        row.sort(key=lambda b: b["x_min"])
        lines.append("  ".join(b["text"] for b in row))

    return "\n".join(lines)


# ─── Ollama call ──────────────────────────────────────────────────────────────

_PROMPT = """\
You are a receipt data extraction assistant.
The text below was extracted by OCR from a receipt image.
Extract the fields and return ONLY a valid JSON object — no markdown, no explanation.

OCR TEXT:
{ocr_text}

Return this exact JSON schema (null for missing fields, plain numbers for amounts):
{{
  "merchant_name": "string or null",
  "bill_number":   "string or null",
  "date":          "string or null",
  "items": [
    {{"name": "string", "qty": number_or_null, "rate": number_or_null, "amount": number_or_null}}
  ],
  "sub_total":    number_or_null,
  "discount":     number_or_null,
  "taxes": [
    {{"name": "CGST/SGST/VAT/Sales Tax/etc", "rate": "percent string or null", "amount": number_or_null}}
  ],
  "total_amount": number_or_null,
  "currency":     "INR or USD or other 3-letter code or null",
  "payment_mode": "Cash or Card or UPI or null"
}}
"""

_REPAIR_PROMPT = """\
The JSON below is malformed or truncated. Fix it and return ONLY a valid JSON object
with the same receipt fields and data. No markdown, no explanation.

BROKEN JSON:
{raw}
"""


def _strip_json_fences(raw: str) -> str:
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    return m.group() if m else cleaned


def safe_parse_json(raw: str) -> dict:
    """Parse LLM JSON with json-repair fallback for truncated/malformed output."""
    cleaned = _strip_json_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        log.warning("JSON parse failed (%s), attempting json-repair", exc)
        try:
            from json_repair import repair_json
            repaired = repair_json(cleaned)
            return json.loads(repaired)
        except Exception as repair_exc:
            log.error(
                "JSON repair failed. Raw LLM output (first 800 chars):\n%s",
                raw[:800],
            )
            raise ValueError(f"Invalid JSON from LLM: {exc}") from repair_exc


async def _ollama_generate(prompt: str, *, temperature: float = 0.05) -> str:
    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT_SEC) as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model":  OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": temperature,
                    "top_p":       0.9,
                    "num_predict": OLLAMA_NUM_PREDICT,
                },
            },
        )
        resp.raise_for_status()
    return resp.json()["response"]


async def call_ollama(ocr_text: str) -> str:
    return await _ollama_generate(_PROMPT.format(ocr_text=ocr_text))


async def call_ollama_repair(raw: str) -> str:
    return await _ollama_generate(
        _REPAIR_PROMPT.format(raw=raw[:4000]),
        temperature=0.0,
    )


async def extract_receipt_json(ocr_text: str) -> dict:
    """Run LLM extraction with json-repair and a second LLM pass if needed."""
    raw = await call_ollama(ocr_text)
    try:
        return safe_parse_json(raw)
    except ValueError:
        log.info("LLM JSON still invalid after repair — retrying with fix pass")
        raw = await call_ollama_repair(raw)
        return safe_parse_json(raw)


def _to_float(v) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    cleaned = re.sub(r"[^\d.]", "", str(v))
    try:
        return float(cleaned)
    except ValueError:
        return None


# ─── App ──────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_ocr()          # warm up on startup
    yield


app = FastAPI(
    title="Receipt Extractor API",
    description="PaddleOCR + Ollama — fully offline receipt data extraction",
    version="1.1.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.post("/extract", response_model=ExtractResponse)
async def extract_receipt(
    file:  UploadFile = File(...),
    debug: bool       = Query(False, description="Include raw OCR text in response"),
):
    allowed = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
    if (file.content_type or "").lower() not in allowed:
        raise HTTPException(415, f"Unsupported type '{file.content_type}'. Use JPEG/PNG/WEBP.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(400, "Empty file.")

    t0 = time.monotonic()

    # Step 1 — OCR
    t_ocr = time.monotonic()
    try:
        img_arr  = preprocess(image_bytes)
        ocr_text = run_ocr(img_arr)
    except Exception as exc:
        log.exception("OCR stage failed")
        raise HTTPException(500, f"OCR error: {exc}")
    ocr_ms = _ms(t_ocr)

    if not ocr_text.strip():
        return ExtractResponse(success=False, error="OCR returned no text.", latency_ms=_ms(t0))

    log.info("OCR: %d chars from '%s' in %d ms", len(ocr_text), file.filename, ocr_ms)
    log.debug("OCR text:\n%s", ocr_text)

    # Step 2 — LLM
    t_llm = time.monotonic()
    try:
        data_dict = await extract_receipt_json(ocr_text)
    except httpx.ConnectError:
        raise HTTPException(503, f"Cannot reach Ollama at {OLLAMA_BASE_URL} — is `ollama serve` running?")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(502, f"Ollama {exc.response.status_code}: {exc.response.text[:200]}")
    except Exception as exc:
        log.exception("LLM stage failed after %d ms", _ms(t_llm))
        return ExtractResponse(
            success=False, ocr_text=ocr_text if debug else None,
            error=f"LLM error: {exc}", latency_ms=_ms(t0),
        )
    llm_ms = _ms(t_llm)
    log.info("LLM: parsed receipt in %d ms (total %d ms)", llm_ms, _ms(t0))

    # Step 3 — Build typed response
    try:
        receipt = ReceiptData(
            merchant_name = data_dict.get("merchant_name"),
            bill_number   = data_dict.get("bill_number"),
            date          = data_dict.get("date"),
            items  = [LineItem(**i) for i in (data_dict.get("items")  or [])],
            sub_total     = _to_float(data_dict.get("sub_total")),
            discount      = _to_float(data_dict.get("discount")),
            taxes  = [TaxEntry(**t) for t in (data_dict.get("taxes")  or [])],
            total_amount  = _to_float(data_dict.get("total_amount")),
            currency      = data_dict.get("currency"),
            payment_mode  = data_dict.get("payment_mode"),
        )
    except Exception as exc:
        return ExtractResponse(
            success=False, ocr_text=ocr_text if debug else None,
            error=f"Mapping error: {exc}", latency_ms=_ms(t0),
        )

    return ExtractResponse(
        success=True, data=receipt,
        ocr_text=ocr_text if debug else None,
        latency_ms=_ms(t0),
    )


@app.get("/health")
async def health():
    ollama_ok, models = False, []
    try:
        async with httpx.AsyncClient(timeout=4.0) as c:
            r = await c.get(f"{OLLAMA_BASE_URL}/api/tags")
            models   = [m["name"] for m in r.json().get("models", [])]
            ollama_ok = True
    except Exception as exc:
        models = [str(exc)]

    return {
        "status":         "ok",
        "paddle_version": _paddle_major,
        "ollama_ok":      ollama_ok,
        "model":          OLLAMA_MODEL,
        "model_pulled":   OLLAMA_MODEL in models,
        "available":      models,
    }


def _ms(t: float) -> int:
    return int((time.monotonic() - t) * 1000)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)