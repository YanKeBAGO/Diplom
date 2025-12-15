#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from dateutil import parser as dateparser
import numpy as np
import pandas as pd
from transformers import pipeline
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm


from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# ---------------------------
# Logging
# ---------------------------

def log(stage: str, msg: str):
    print(f"[{stage}] {msg}", flush=True)

def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


# ---------------------------
# Telegram JSON parsing helpers
# ---------------------------

def parse_dt(s: str):
    try:
        return dateparser.parse(s)
    except Exception:
        return None

def normalize_text(x: str) -> str:
    x = x.replace("\u200b", " ").replace("\u00a0", " ")
    x = re.sub(r"\s+", " ", x).strip()
    return x

PII_PATTERNS = [
    (re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.IGNORECASE), "<EMAIL>"),
    (re.compile(r"\b\+?\d[\d\-\s\(\)]{7,}\d\b"), "<PHONE>"),
    (re.compile(r"@\w+"), "<USERNAME>"),
    (re.compile(r"\bhttps?://\S+\b", re.IGNORECASE), "<URL>"),
]

def anonymize_text(x: str) -> str:
    for pat, repl in PII_PATTERNS:
        x = pat.sub(repl, x)
    return x

def extract_message_text(msg_field):
    """
    Telegram export:
      - text can be string
      - or list of dicts/items (entities)
    """
    if msg_field is None:
        return ""
    if isinstance(msg_field, str):
        return msg_field
    if isinstance(msg_field, list):
        parts = []
        for item in msg_field:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        return "".join(parts)
    return str(msg_field)

def load_telegram_result_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Case A: direct messages
    if isinstance(data, dict) and "messages" in data and isinstance(data["messages"], list):
        return data, data["messages"]

    # Case B: chats->list->messages
    if isinstance(data, dict) and "chats" in data:
        lst = (data.get("chats") or {}).get("list", [])
        for chat in lst:
            if isinstance(chat, dict) and "messages" in chat and isinstance(chat["messages"], list):
                return chat, chat["messages"]

    raise ValueError("Не нашёл список messages в result.json. Проверь формат экспорта Telegram.")


@dataclass
class Msg:
    mid: int
    dt: datetime
    sender: str
    text_raw: str
    text: str


# ---------------------------
# ML setup
# ---------------------------

def build_pipelines(device: str):
    device_idx = -1 if device == "cpu" else 0

    log("ML", "Загружаю модели (первый запуск может скачивать веса)...")

    sentiment = pipeline(
        "text-classification",
        model="cointegrated/rubert-tiny-sentiment-balanced",
        tokenizer="cointegrated/rubert-tiny-sentiment-balanced",
        device=device_idx,
        truncation=True,
    )

    emotions = pipeline(
        "text-classification",
        model="cointegrated/rubert-tiny2-cedr-emotion-detection",
        tokenizer="cointegrated/rubert-tiny2-cedr-emotion-detection",
        device=device_idx,
        truncation=True,
        top_k=None,
    )

    toxicity = pipeline(
        "text-classification",
        model="cointegrated/rubert-tiny-toxicity",
        tokenizer="cointegrated/rubert-tiny-toxicity",
        device=device_idx,
        truncation=True,
        top_k=None,
    )

    log("ML", "Модели загружены.")
    return sentiment, emotions, toxicity

def run_in_batches(pipe, texts, batch_size=32):
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        out.extend(pipe(chunk))
    return out


# ---------------------------
# Metrics / profiling
# ---------------------------

def compute_roles(df: pd.DataFrame):
    roles = {}

    initiator = df.sort_values("dt").iloc[0]["sender"] if len(df) else None
    counts = df["sender"].value_counts().to_dict()
    qrate = df.groupby("sender")["text"].apply(lambda s: float(np.mean([("?" in x) for x in s]))).to_dict()

    polmap = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    df["sent_pol"] = df["sent_label"].map(polmap).fillna(0.0)

    emo_cols = [c for c in df.columns if c.startswith("emo_")]
    tox_cols = [c for c in df.columns if c.startswith("tox_")]

    df["emo_max"] = df[emo_cols].max(axis=1) if emo_cols else 0.0
    df["tox_max"] = df[tox_cols].max(axis=1) if tox_cols else 0.0

    by_sender = df.groupby("sender")
    sent_abs = by_sender["sent_pol"].apply(lambda s: float(np.mean(np.abs(s)))).to_dict()
    emo_driver = by_sender["emo_max"].mean().to_dict()
    tox_source = by_sender["tox_max"].mean().to_dict()

    roles["initiator"] = initiator
    roles["more_active"] = max(counts, key=counts.get) if counts else None
    roles["questioner"] = max(qrate, key=qrate.get) if qrate else None
    roles["emotional_driver"] = max(emo_driver, key=emo_driver.get) if emo_driver else None
    roles["toxic_source"] = max(tox_source, key=tox_source.get) if tox_source else None

    roles["per_sender"] = {}
    for s in counts.keys():
        roles["per_sender"][s] = {
            "messages": int(counts.get(s, 0)),
            "question_rate": float(qrate.get(s, 0.0)),
            "avg_abs_sentiment": float(sent_abs.get(s, 0.0)),
            "avg_emotion_peak": float(emo_driver.get(s, 0.0)),
            "avg_toxicity_peak": float(tox_source.get(s, 0.0)),
        }

    return roles

def response_time_stats(df: pd.DataFrame):
    df2 = df.sort_values("dt").reset_index(drop=True)
    gaps = []
    for i in range(1, len(df2)):
        if df2.loc[i, "sender"] != df2.loc[i - 1, "sender"]:
            dt0 = df2.loc[i - 1, "dt"]
            dt1 = df2.loc[i, "dt"]
            if isinstance(dt0, datetime) and isinstance(dt1, datetime):
                sec = (dt1 - dt0).total_seconds()
                if sec >= 0:
                    gaps.append(sec)

    if not gaps:
        return {"count": 0, "avg_sec": None, "median_sec": None, "p90_sec": None}

    gaps = np.array(gaps, dtype=float)
    return {
        "count": int(len(gaps)),
        "avg_sec": float(np.mean(gaps)),
        "median_sec": float(np.median(gaps)),
        "p90_sec": float(np.percentile(gaps, 90)),
    }


# ---------------------------
# PDF report (with Cyrillic)
# ---------------------------

def register_cyrillic_font(font_path: str):
    # Register once (safe to call multiple times)
    pdfmetrics.registerFont(TTFont("DejaVu", font_path))

def wrap_text(c: canvas.Canvas, text: str, max_width: float, font_name: str, font_size: int):
    """
    Simple word-wrapping for ReportLab.
    """
    words = (text or "").split()
    if not words:
        return [""]

    lines = []
    current = words[0]
    for w in words[1:]:
        trial = current + " " + w
        if c.stringWidth(trial, font_name, font_size) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = w
    lines.append(current)
    return lines

def build_pdf_report(out_pdf: str, summary: dict, font_path: str = "DejaVuSans.ttf"):
    if not os.path.exists(font_path):
        raise FileNotFoundError(
            f"Не найден шрифт {font_path}. Скачай DejaVuSans.ttf и положи рядом с main.py"
        )

    register_cyrillic_font(font_path)

    c = canvas.Canvas(out_pdf, pagesize=A4)
    w, h = A4

    left = 18 * mm
    right = 18 * mm
    top = 18 * mm
    bottom = 18 * mm

    max_width = w - left - right
    y = h - top

    def newline(dy=6 * mm):
        nonlocal y
        y -= dy
        if y < bottom:
            c.showPage()
            # after new page, font still needs set each draw
            y = h - top

    def draw_block(text: str, size=11, dy=6 * mm):
        nonlocal y
        c.setFont("DejaVu", size)
        for line in wrap_text(c, text, max_width, "DejaVu", size):
            c.drawString(left, y, line)
            newline(dy)

    c.setTitle("Психолингвистический отчёт по переписке")

    draw_block("Психолингвистический отчёт по переписке", size=14, dy=7 * mm)
    newline(2 * mm)

    totals = summary.get("totals", {})
    roles = summary.get("roles", {})
    rt = summary.get("response_time", {})

    draw_block(f"Всего сообщений: {totals.get('messages')}")
    draw_block(f"Участников: {totals.get('participants')}")
    draw_block(f"Период: {totals.get('date_min')} — {totals.get('date_max')}")
    newline(2 * mm)

    draw_block("Роли (эвристики):", size=12, dy=7 * mm)
    for k in ["initiator", "more_active", "questioner", "emotional_driver", "toxic_source"]:
        draw_block(f"- {k}: {roles.get(k)}")

    newline(2 * mm)
    draw_block("Статистика по участникам:", size=12, dy=7 * mm)

    per_sender = (roles.get("per_sender") or {})
    for sender, sdata in per_sender.items():
        newline(1 * mm)
        draw_block(f"{sender}:", size=11, dy=6 * mm)
        draw_block(f"  сообщений: {sdata.get('messages')}")
        draw_block(f"  вопросительность: {float(sdata.get('question_rate', 0.0)):.3f}")
        draw_block(f"  средн. |тональность|: {float(sdata.get('avg_abs_sentiment', 0.0)):.3f}")
        draw_block(f"  пик эмоций: {float(sdata.get('avg_emotion_peak', 0.0)):.3f}")
        draw_block(f"  пик токсичности: {float(sdata.get('avg_toxicity_peak', 0.0)):.3f}")

    newline(2 * mm)
    draw_block("Динамика (время ответа):", size=12, dy=7 * mm)
    draw_block(f"  измерений: {rt.get('count')}")
    draw_block(f"  avg (sec): {rt.get('avg_sec')}")
    draw_block(f"  median (sec): {rt.get('median_sec')}")
    draw_block(f"  p90 (sec): {rt.get('p90_sec')}")

    c.save()


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="ML-анализ Telegram result.json: тональность/эмоции/токсичность + профиль + PDF.")
    ap.add_argument("json_path", help="Путь к result.json (экспорт Telegram)")
    ap.add_argument("--outdir", default="out", help="Папка для результатов")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Устройство для инференса (cuda если есть GPU)")
    ap.add_argument("--batch", type=int, default=32, help="Batch size")
    ap.add_argument("--limit", type=int, default=0, help="Ограничить число сообщений (0 = все)")
    ap.add_argument("--font", default="DejaVuSans.ttf", help="TTF-шрифт для PDF (кириллица)")
    args = ap.parse_args()

    safe_mkdir(args.outdir)

    log("LOAD", f"Читаю JSON: {args.json_path}")
    chat_meta, raw_messages = load_telegram_result_json(args.json_path)
    log("LOAD", f"Сообщений в файле: {len(raw_messages)}")

    msgs = []
    skipped = 0

    for m in raw_messages:
        if not isinstance(m, dict):
            skipped += 1
            continue

        if m.get("type") != "message":
            continue

        sender = m.get("from") or m.get("from_id") or "UNKNOWN"
        dt = parse_dt(m.get("date", ""))

        if dt is None:
            skipped += 1
            continue

        text_raw = extract_message_text(m.get("text"))
        text_raw = normalize_text(text_raw)
        if not text_raw:
            continue

        text = anonymize_text(text_raw)
        mid = int(m.get("id", 0)) if str(m.get("id", "0")).isdigit() else 0

        msgs.append(Msg(mid=mid, dt=dt, sender=str(sender), text_raw=text_raw, text=text))

    if args.limit and args.limit > 0:
        msgs = msgs[:args.limit]

    log("LOAD", f"Отобрано текстовых сообщений: {len(msgs)} (пропущено/битых: {skipped})")
    if not msgs:
        raise SystemExit("Нет текстовых сообщений для анализа.")

    df = pd.DataFrame([{
        "mid": x.mid,
        "dt": x.dt,
        "sender": x.sender,
        "text_raw": x.text_raw,
        "text": x.text,
        "len": len(x.text),
        "words": len(x.text.split()),
    } for x in msgs])

    sentiment_pipe, emotions_pipe, toxicity_pipe = build_pipelines(args.device)
    texts = df["text"].tolist()

    log("SENT", f"Считаю тональность батчами по {args.batch}...")
    sent_out = run_in_batches(sentiment_pipe, texts, batch_size=args.batch)
    df["sent_label"] = [o.get("label") for o in sent_out]
    df["sent_score"] = [float(o.get("score", 0.0)) for o in sent_out]
    log("SENT", "Готово.")

    log("EMO", f"Считаю эмоции батчами по {args.batch}...")
    emo_out = run_in_batches(emotions_pipe, texts, batch_size=args.batch)
    all_emo_labels = sorted({p["label"] for item in emo_out for p in item})
    for lab in all_emo_labels:
        df[f"emo_{lab}"] = 0.0
    for i, item in enumerate(emo_out):
        for p in item:
            df.at[i, f"emo_{p['label']}"] = float(p["score"])
    log("EMO", f"Готово. Метки эмоций: {all_emo_labels}")

    log("TOX", f"Считаю токсичность батчами по {args.batch}...")
    tox_out = run_in_batches(toxicity_pipe, texts, batch_size=args.batch)
    all_tox_labels = sorted({p["label"] for item in tox_out for p in item})
    for lab in all_tox_labels:
        df[f"tox_{lab}"] = 0.0
    for i, item in enumerate(tox_out):
        for p in item:
            df.at[i, f"tox_{p['label']}"] = float(p["score"])
    log("TOX", f"Готово. Метки токсичности: {all_tox_labels}")

    log("PROFILE", "Считаю профиль и метрики...")
    date_min = df["dt"].min()
    date_max = df["dt"].max()

    totals = {
        "messages": int(len(df)),
        "participants": int(df["sender"].nunique()),
        "date_min": date_min.isoformat() if isinstance(date_min, datetime) else str(date_min),
        "date_max": date_max.isoformat() if isinstance(date_max, datetime) else str(date_max),
    }

    roles = compute_roles(df.copy())
    rt = response_time_stats(df)

    summary = {
        "totals": totals,
        "roles": roles,
        "response_time": rt,
        "model_ids": {
            "sentiment": "cointegrated/rubert-tiny-sentiment-balanced",
            "emotions": "cointegrated/rubert-tiny2-cedr-emotion-detection",
            "toxicity": "cointegrated/rubert-tiny-toxicity",
        }
    }
    log("PROFILE", "Готово.")

    out_json = os.path.join(args.outdir, "report.json")
    out_csv = os.path.join(args.outdir, "messages_scored.csv")
    out_pdf = os.path.join(args.outdir, "report.pdf")

    log("SAVE", f"Пишу {out_json}")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log("SAVE", f"Пишу {out_csv}")
    df.sort_values("dt").to_csv(out_csv, index=False, encoding="utf-8")

    log("SAVE", f"Генерирую {out_pdf}")
    build_pdf_report(out_pdf, summary, font_path=args.font)

    log("DONE", f"Готово. Результаты в папке: {args.outdir}")
    log("DONE", "Файлы: report.json, messages_scored.csv, report.pdf")


if __name__ == "__main__":
    main()
