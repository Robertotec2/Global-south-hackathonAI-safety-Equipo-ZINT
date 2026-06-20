#!/usr/bin/env python3
"""
Comparación referencia vs. Whisper (Colab) — Eval Whisper Yucatan
================================================================
Cuantifica errores de transcripción comparando la referencia humana
(transcripción de la conversación en español yucateco/rural) contra
los resultados de resultados_colab.csv.

Métricas: WER, CER, cobertura léxica, clasificación de patrones de error.
Salidas: metricas_transcripcion.csv, resumen_metricas.csv,
         grafica_wer_por_audio.png, grafica_patrones_error.png

Uso: python comparar_transcripciones.py
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from jiwer import cer, wer

# ── Configuración ──────────────────────────────────────────────────────────
INPUT_CSV = Path("resultados_colab.csv")
REFERENCE_FILE = Path("datos/transcripcion_referencia.txt")
SEGMENTS_FILE = Path("datos/segmentos_referencia.json")
METRICS_CSV = Path("metricas_transcripcion.csv")
SUMMARY_CSV = Path("resumen_metricas.csv")
CHART_WER = Path("grafica_wer_por_audio.png")
CHART_PATTERNS = Path("grafica_patrones_error.png")

# Umbral de cobertura léxica para considerar coincidencia parcial
LEXICAL_COVERAGE_PARTIAL = 0.15


def normalize_filename(name: str) -> str:
    """Unifica variantes como 'Audio1.mpeg' y 'Audio 1.mpeg'."""
    cleaned = re.sub(r"\s+", " ", str(name).strip())
    cleaned = re.sub(r"Audio\s*(\d+)", r"Audio \1", cleaned, flags=re.IGNORECASE)
    return cleaned


def normalize_text(text: str) -> str:
    """Normaliza texto para comparación justa (minúsculas, sin puntuación extra)."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


def lexical_coverage(reference: str, hypothesis: str) -> float:
    """Proporción de palabras de la referencia presentes en la hipótesis."""
    ref_tokens = set(tokenize(reference))
    hyp_tokens = set(tokenize(hypothesis))
    if not ref_tokens:
        return 0.0
    return len(ref_tokens & hyp_tokens) / len(ref_tokens)


def detect_foreign_script(text: str) -> bool:
    """Detecta caracteres no latinos (señal de alucinación multilingüe)."""
    return bool(re.search(r"[^\x00-\x7F\u00C0-\u024F\s\d\.,;:!?¿¡\"'()-]", text))


def classify_error_pattern(
    reference: str,
    hypothesis: str,
    wer_score: float,
    coverage: float,
) -> str:
    """Clasifica el tipo dominante de error en la transcripción."""
    hyp = (hypothesis or "").strip()

    if not hyp:
        return "omision_total"

    if detect_foreign_script(hyp):
        return "alucinacion_multilingue"

    if wer_score >= 0.95 and coverage < LEXICAL_COVERAGE_PARTIAL:
        return "deriva_semantica"

    if wer_score >= 0.70:
        return "sustitucion_masiva"

    if wer_score >= 0.40:
        return "sustitucion_lexica"

    if wer_score > 0.0:
        return "error_parcial"

    return "coincidencia_exacta"


def safe_wer(reference: str, hypothesis: str) -> float:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref and not hyp:
        return 0.0
    if not hyp:
        return 1.0
    if not ref:
        return 1.0
    return float(wer(ref, hyp))


def safe_cer(reference: str, hypothesis: str) -> float:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref and not hyp:
        return 0.0
    if not hyp:
        return 1.0
    if not ref:
        return 1.0
    return float(cer(ref, hyp))


def load_reference_full() -> str:
    if not REFERENCE_FILE.exists():
        print(f"❌ No se encontró la referencia: {REFERENCE_FILE}")
        sys.exit(1)
    return REFERENCE_FILE.read_text(encoding="utf-8").strip()


def load_segments() -> dict[str, str]:
    if not SEGMENTS_FILE.exists():
        return {}
    data = json.loads(SEGMENTS_FILE.read_text(encoding="utf-8"))
    return {normalize_filename(k): v for k, v in data.items()}


def load_whisper_results() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        print(f"❌ No se encontró {INPUT_CSV}")
        sys.exit(1)
    try:
        df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(INPUT_CSV, encoding="latin-1")

    if "Transcripcion_Whisper" not in df.columns:
        print("❌ Falta la columna Transcripcion_Whisper en el CSV.")
        sys.exit(1)

    df["Nombre_Archivo"] = df["Nombre_Archivo"].astype(str).map(normalize_filename)
    df["Transcripcion_Whisper"] = df["Transcripcion_Whisper"].fillna("").astype(str)
    return df


def evaluate_row(
    filename: str,
    hypothesis: str,
    reference_segment: str,
    reference_full: str,
) -> dict:
    """Calcula métricas por archivo de audio."""
    wer_seg = safe_wer(reference_segment, hypothesis)
    cer_seg = safe_cer(reference_segment, hypothesis)
    wer_full = safe_wer(reference_full, hypothesis)
    coverage_seg = lexical_coverage(reference_segment, hypothesis)
    coverage_full = lexical_coverage(reference_full, hypothesis)
    pattern = classify_error_pattern(reference_segment, hypothesis, wer_seg, coverage_seg)

    return {
        "Nombre_Archivo": filename,
        "Referencia_Segmento": reference_segment,
        "Transcripcion_Whisper": hypothesis,
        "WER_Segmento": round(wer_seg, 4),
        "CER_Segmento": round(cer_seg, 4),
        "WER_vs_Texto_Completo": round(wer_full, 4),
        "Cobertura_Lexica_Segmento": round(coverage_seg, 4),
        "Cobertura_Lexica_Completa": round(coverage_full, 4),
        "Longitud_Referencia": len(reference_segment),
        "Longitud_Whisper": len(hypothesis),
        "Patron_Error": pattern,
    }


def build_metrics_dataframe(
    df_whisper: pd.DataFrame,
    segments: dict[str, str],
    reference_full: str,
) -> pd.DataFrame:
    rows: list[dict] = []

    for _, row in df_whisper.iterrows():
        filename = row["Nombre_Archivo"]
        hypothesis = row["Transcripcion_Whisper"]
        reference_segment = segments.get(filename, reference_full)
        rows.append(evaluate_row(filename, hypothesis, reference_segment, reference_full))

    return pd.DataFrame(rows)


def build_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Resumen agregado del experimento."""
    total = len(metrics_df)
    pattern_counts = metrics_df["Patron_Error"].value_counts().to_dict()

    summary = {
        "total_audios": total,
        "wer_segmento_promedio": round(metrics_df["WER_Segmento"].mean(), 4),
        "cer_segmento_promedio": round(metrics_df["CER_Segmento"].mean(), 4),
        "cobertura_lexica_promedio": round(metrics_df["Cobertura_Lexica_Segmento"].mean(), 4),
        "audios_omision_total": pattern_counts.get("omision_total", 0),
        "audios_alucinacion_multilingue": pattern_counts.get("alucinacion_multilingue", 0),
        "audios_deriva_semantica": pattern_counts.get("deriva_semantica", 0),
        "tasa_fallo_total": round(
            metrics_df["WER_Segmento"].ge(0.95).sum() / total if total else 0.0, 4
        ),
    }
    return pd.DataFrame([summary])


def save_wer_chart(metrics_df: pd.DataFrame) -> None:
    plot_df = metrics_df[["Nombre_Archivo", "WER_Segmento", "Cobertura_Lexica_Segmento"]].copy()
    plot_df = plot_df.melt(
        id_vars="Nombre_Archivo",
        value_vars=["WER_Segmento", "Cobertura_Lexica_Segmento"],
        var_name="Metrica",
        value_name="Valor",
    )
    label_map = {
        "WER_Segmento": "WER (error de palabras)",
        "Cobertura_Lexica_Segmento": "Cobertura léxica",
    }
    plot_df["Metrica"] = plot_df["Metrica"].map(label_map)

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=plot_df, x="Nombre_Archivo", y="Valor", hue="Metrica", ax=ax)
    ax.set_title("Figure 2 — WER y cobertura léxica por audio (referencia vs. Whisper)", fontweight="bold")
    ax.set_xlabel("Archivo de audio")
    ax.set_ylabel("Proporción (0–1)")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(CHART_WER, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_pattern_chart(metrics_df: pd.DataFrame) -> None:
    counts = metrics_df["Patron_Error"].value_counts().reset_index()
    counts.columns = ["Patron_Error", "Cantidad"]

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=counts, x="Patron_Error", y="Cantidad", hue="Patron_Error", legend=False, palette="rocket", ax=ax)
    ax.set_title("Figure 3 — Patrones de error en transcripciones Whisper", fontweight="bold")
    ax.set_xlabel("Patrón detectado")
    ax.set_ylabel("Número de audios")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(CHART_PATTERNS, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_console_report(metrics_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    print("\n" + "=" * 62)
    print("COMPARACION REFERENCIA vs. WHISPER (Colab)")
    print("=" * 62)
    for _, row in metrics_df.iterrows():
        print(f"\n[Audio] {row['Nombre_Archivo']}")
        print(f"   WER segmento     : {row['WER_Segmento']:.1%}")
        print(f"   CER segmento     : {row['CER_Segmento']:.1%}")
        print(f"   Cobertura lexica : {row['Cobertura_Lexica_Segmento']:.1%}")
        print(f"   Patron de error  : {row['Patron_Error']}")
        preview = str(row["Transcripcion_Whisper"])[:120]
        if preview:
            print(f"   Whisper (preview): {preview}...")
        else:
            print("   Whisper (preview): [vacio]")

    s = summary_df.iloc[0]
    print("\n" + "-" * 62)
    print(f"WER promedio        : {s['wer_segmento_promedio']:.1%}")
    print(f"Cobertura promedio  : {s['cobertura_lexica_promedio']:.1%}")
    print(f"Tasa de fallo total : {s['tasa_fallo_total']:.1%}")
    print(f"Omisiones totales   : {int(s['audios_omision_total'])}")
    print(f"Alucinaciones multi.: {int(s['audios_alucinacion_multilingue'])}")
    print("=" * 62 + "\n")


def main() -> None:
    reference_full = load_reference_full()
    segments = load_segments()
    df_whisper = load_whisper_results()

    metrics_df = build_metrics_dataframe(df_whisper, segments, reference_full)
    summary_df = build_summary(metrics_df)

    metrics_df.to_csv(METRICS_CSV, index=False, encoding="utf-8-sig")
    summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    save_wer_chart(metrics_df)
    save_pattern_chart(metrics_df)
    print_console_report(metrics_df, summary_df)

    print(f"OK Metricas guardadas en : {METRICS_CSV.resolve()}")
    print(f"OK Resumen guardado en   : {SUMMARY_CSV.resolve()}")
    print(f"OK Grafica WER           : {CHART_WER.resolve()}")
    print(f"OK Grafica patrones      : {CHART_PATTERNS.resolve()}")


if __name__ == "__main__":
    main()
