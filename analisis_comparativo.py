#!/usr/bin/env python3
"""
Eval Whisper Yucatan — Benchmark determinista Whisper vs. Gemini 1.5
====================================================================
Compara transcripciones automaticas contra ground truth humano y cuantifica:
  - WER y CER (jiwer)
  - Reduccion de error (Gemini vs. Whisper)
  - Tipologia de errores: sustituciones, omisiones, inserciones (process_words)

Entrada : benchmark_whisper_vs_gemini.csv
          Columnas: Archivo, Transcripcion_Whisper, Transcripcion_Gemini
Ground  : transcripciones_oficiales/TranscripcionOficialAudio{N}.txt
Salidas : resultados_finales_benchmark.csv
          comparativa_wer.png, tipologia_errores.png

Uso: python analisis_comparativo.py
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from jiwer import cer, process_words, wer

# ── Configuracion ────────────────────────────────────────────────────────────
INPUT_CSV = Path("benchmark_whisper_vs_gemini.csv")
OFFICIAL_DIR = Path("transcripciones_oficiales")
OUTPUT_CSV = Path("resultados_finales_benchmark.csv")
CHART_WER = Path("comparativa_wer.png")
CHART_TYPOLOGY = Path("tipologia_errores.png")

OFFICIAL_PREFIX = "TranscripcionOficial"
OFFICIAL_PREFIX_ALT = "TranscripcionOfical"

OUTPUT_COLUMNS = [
    "Archivo",
    "Texto_Oficial",
    "Transcripcion_Whisper",
    "Transcripcion_Gemini",
    "WER_Whisper",
    "WER_Gemini",
    "Reduccion_Error",
    "Sustituciones_Whisper",
    "Omisiones_Whisper",
    "Inserciones_Whisper",
    "Sustituciones_Gemini",
    "Omisiones_Gemini",
    "Inserciones_Gemini",
]


def extract_audio_base(archivo: str) -> str:
    """
    Limpia extension y normaliza el nombre base del audio.
    'Audio1.mpeg' -> 'Audio1' | 'Audio 7.mp3' -> 'Audio7'
    """
    stem = Path(str(archivo).strip()).stem
    match = re.search(r"(\d+)", stem, flags=re.IGNORECASE)
    if match:
        return f"Audio{match.group(1)}"
    return re.sub(r"\s+", "", stem)


def official_transcript_path(audio_base: str) -> Path | None:
    """Busca el txt oficial con convencion estricta (+ alias historico Ofical)."""
    candidates = [
        OFFICIAL_DIR / f"{OFFICIAL_PREFIX}{audio_base}.txt",
        OFFICIAL_DIR / f"{OFFICIAL_PREFIX_ALT}{audio_base}.txt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def normalize_text(text: str) -> str:
    """Normaliza texto para metricas ASR reproducibles."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def safe_wer(reference: str, hypothesis: str) -> float:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref and not hyp:
        return 0.0
    if not hyp or not ref:
        return 1.0
    return float(wer(ref, hyp))


def safe_cer(reference: str, hypothesis: str) -> float:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref and not hyp:
        return 0.0
    if not hyp or not ref:
        return 1.0
    return float(cer(ref, hyp))


def error_reduction_pct(wer_whisper: float, wer_gemini: float) -> float:
    """((WER_Whisper - WER_Gemini) / WER_Whisper) * 100"""
    if wer_whisper == 0.0:
        return 0.0 if wer_gemini == 0.0 else -100.0
    return ((wer_whisper - wer_gemini) / wer_whisper) * 100.0


def word_error_typology(reference: str, hypothesis: str) -> dict[str, int]:
    """
    Extrae conteos de sustituciones, omisiones e inserciones con jiwer.process_words.
    """
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)

    if not ref and not hyp:
        return {"substitutions": 0, "deletions": 0, "insertions": 0, "hits": 0}

    if not hyp:
        ref_words = ref.split()
        return {
            "substitutions": 0,
            "deletions": len(ref_words),
            "insertions": 0,
            "hits": 0,
        }

    if not ref:
        hyp_words = hyp.split()
        return {
            "substitutions": 0,
            "deletions": 0,
            "insertions": len(hyp_words),
            "hits": 0,
        }

    alignment = process_words(ref, hyp)
    return {
        "substitutions": alignment.substitutions,
        "deletions": alignment.deletions,
        "insertions": alignment.insertions,
        "hits": alignment.hits,
    }


def load_benchmark_csv() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        print(f"ERROR: No se encontro '{INPUT_CSV.resolve()}'.")
        print("   Coloca el export de Colab: benchmark_whisper_vs_gemini.csv")
        sys.exit(1)

    try:
        df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(INPUT_CSV, encoding="latin-1")

    required = {"Archivo", "Transcripcion_Whisper", "Transcripcion_Gemini"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: Faltan columnas: {sorted(missing)}")
        print(f"   Encontradas: {list(df.columns)}")
        sys.exit(1)

    if df.empty:
        print("ERROR: benchmark_whisper_vs_gemini.csv esta vacio.")
        sys.exit(1)

    df["Archivo"] = df["Archivo"].astype(str).str.strip()
    df["Transcripcion_Whisper"] = df["Transcripcion_Whisper"].fillna("").astype(str)
    df["Transcripcion_Gemini"] = df["Transcripcion_Gemini"].fillna("").astype(str)
    return df


def load_official_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1").strip()


def analyze_row(archivo: str, whisper: str, gemini: str) -> dict | None:
    audio_base = extract_audio_base(archivo)
    official_path = official_transcript_path(audio_base)

    if official_path is None:
        expected = f"{OFFICIAL_PREFIX}{audio_base}.txt"
        print(
            f"WARNING: Sin ground truth para '{archivo}' "
            f"(esperado: {expected}). Fila omitida."
        )
        return None

    texto_oficial = load_official_text(official_path)

    wer_w = safe_wer(texto_oficial, whisper)
    wer_g = safe_wer(texto_oficial, gemini)
    cer_w = safe_cer(texto_oficial, whisper)
    cer_g = safe_cer(texto_oficial, gemini)

    typo_w = word_error_typology(texto_oficial, whisper)
    typo_g = word_error_typology(texto_oficial, gemini)

    return {
        "Archivo": archivo,
        "Texto_Oficial": texto_oficial,
        "Transcripcion_Whisper": whisper,
        "Transcripcion_Gemini": gemini,
        "WER_Whisper": round(wer_w, 4),
        "WER_Gemini": round(wer_g, 4),
        "CER_Whisper": round(cer_w, 4),
        "CER_Gemini": round(cer_g, 4),
        "Reduccion_Error": round(error_reduction_pct(wer_w, wer_g), 2),
        "Sustituciones_Whisper": typo_w["substitutions"],
        "Omisiones_Whisper": typo_w["deletions"],
        "Inserciones_Whisper": typo_w["insertions"],
        "Sustituciones_Gemini": typo_g["substitutions"],
        "Omisiones_Gemini": typo_g["deletions"],
        "Inserciones_Gemini": typo_g["insertions"],
    }


def build_results_dataframe(df_benchmark: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    for _, row in df_benchmark.iterrows():
        result = analyze_row(
            archivo=row["Archivo"],
            whisper=row["Transcripcion_Whisper"],
            gemini=row["Transcripcion_Gemini"],
        )
        if result is not None:
            rows.append(result)

    if not rows:
        print("ERROR: Ninguna fila evaluada. Revisa transcripciones_oficiales/.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    return df


def export_results_csv(results_df: pd.DataFrame) -> None:
    """Exporta solo las columnas requeridas por el benchmark."""
    results_df[OUTPUT_COLUMNS].to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")


def save_wer_chart(results_df: pd.DataFrame) -> None:
    """Grafica 1: WER promedio Whisper vs Gemini."""
    summary = pd.DataFrame(
        {
            "Modelo": ["Whisper-large-v3", "Gemini 1.5"],
            "WER_Promedio": [
                results_df["WER_Whisper"].mean(),
                results_df["WER_Gemini"].mean(),
            ],
        }
    )

    sns.set_theme(style="whitegrid", context="talk", font_scale=1.05)
    fig, ax = plt.subplots(figsize=(8, 6))

    palette = {"Whisper-large-v3": "#E45756", "Gemini 1.5": "#4C78A8"}
    bars = sns.barplot(
        data=summary,
        x="Modelo",
        y="WER_Promedio",
        hue="Modelo",
        palette=palette,
        legend=False,
        ax=ax,
        width=0.55,
    )

    ax.set_title(
        "Eval Whisper Yucatan — WER promedio\n"
        "Espanol rural + maya yucateco (ground truth humano)",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Modelo ASR")
    ax.set_ylabel("Word Error Rate (WER)")
    ax.set_ylim(0, min(1.05, summary["WER_Promedio"].max() * 1.2 + 0.05))

    for container in bars.containers:
        ax.bar_label(container, fmt=lambda v: f"{v:.1%}", padding=4, fontweight="bold")

    plt.tight_layout()
    fig.savefig(CHART_WER, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_typology_chart(results_df: pd.DataFrame) -> None:
    """
    Grafica 2: barras apiladas con proporcion de S / D / I por modelo.
    Evidencia si Whisper sufre mas sordera digital (omisiones).
    """
    whisper_totals = {
        "Sustituciones": results_df["Sustituciones_Whisper"].sum(),
        "Omisiones": results_df["Omisiones_Whisper"].sum(),
        "Inserciones": results_df["Inserciones_Whisper"].sum(),
    }
    gemini_totals = {
        "Sustituciones": results_df["Sustituciones_Gemini"].sum(),
        "Omisiones": results_df["Omisiones_Gemini"].sum(),
        "Inserciones": results_df["Inserciones_Gemini"].sum(),
    }

    records: list[dict] = []
    for modelo, totals in [("Whisper-large-v3", whisper_totals), ("Gemini 1.5", gemini_totals)]:
        grand = sum(totals.values()) or 1
        for tipo, count in totals.items():
            records.append(
                {
                    "Modelo": modelo,
                    "Tipo_Error": tipo,
                    "Conteo": int(count),
                    "Proporcion": count / grand,
                }
            )

    plot_df = pd.DataFrame(records)
    tipo_order = ["Sustituciones", "Omisiones", "Inserciones"]
    palette = {
        "Sustituciones": "#F58518",
        "Omisiones": "#E45756",
        "Inserciones": "#72B7B2",
    }

    sns.set_theme(style="whitegrid", context="talk", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(9, 6))

    bottom = {model: 0.0 for model in plot_df["Modelo"].unique()}
    x_positions = range(len(bottom))
    model_labels = list(bottom.keys())

    for tipo in tipo_order:
        values = []
        for model in model_labels:
            row = plot_df[(plot_df["Modelo"] == model) & (plot_df["Tipo_Error"] == tipo)]
            values.append(float(row["Proporcion"].iloc[0]) if not row.empty else 0.0)

        ax.bar(
            x_positions,
            values,
            bottom=[bottom[m] for m in model_labels],
            label=tipo,
            color=palette[tipo],
            width=0.55,
            edgecolor="white",
            linewidth=0.8,
        )
        for i, model in enumerate(model_labels):
            bottom[model] += values[i]

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(model_labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Proporcion de errores")
    ax.set_xlabel("Modelo ASR")
    ax.set_title(
        "Tipologia de errores — Sustituciones / Omisiones / Inserciones\n"
        "Omisiones altas en Whisper = mayor sordera digital",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.legend(title="Tipo de error", loc="upper right")
    plt.tight_layout()
    fig.savefig(CHART_TYPOLOGY, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_summary(results_df: pd.DataFrame) -> None:
    n = len(results_df)
    print("\n" + "=" * 68)
    print("EVAL WHISPER YUCATAN — BENCHMARK DETERMINISTA")
    print("=" * 68)
    print(f"Audios evaluados       : {n}")
    print(f"WER promedio Whisper   : {results_df['WER_Whisper'].mean():.1%}")
    print(f"WER promedio Gemini    : {results_df['WER_Gemini'].mean():.1%}")
    print(f"CER promedio Whisper   : {results_df['CER_Whisper'].mean():.1%}")
    print(f"CER promedio Gemini    : {results_df['CER_Gemini'].mean():.1%}")
    print(f"Reduccion error prom.  : {results_df['Reduccion_Error'].mean():+.1f}%")
    print(f"Omisiones totales W    : {results_df['Omisiones_Whisper'].sum()}")
    print(f"Omisiones totales G    : {results_df['Omisiones_Gemini'].sum()}")
    print("-" * 68)

    for _, row in results_df.iterrows():
        print(
            f"{row['Archivo']:<14} | WER W:{row['WER_Whisper']:.1%} G:{row['WER_Gemini']:.1%} "
            f"| Red:{row['Reduccion_Error']:+.1f}% "
            f"| Omis W:{row['Omisiones_Whisper']} G:{row['Omisiones_Gemini']}"
        )

    print("=" * 68)
    print(f"CSV    : {OUTPUT_CSV.resolve()}")
    print(f"WER    : {CHART_WER.resolve()}")
    print(f"Tipol. : {CHART_TYPOLOGY.resolve()}")
    print("=" * 68 + "\n")


def main() -> None:
    if not OFFICIAL_DIR.exists():
        print(f"ERROR: Falta la carpeta '{OFFICIAL_DIR}/'.")
        sys.exit(1)

    df_benchmark = load_benchmark_csv()
    print(f"Benchmark : {INPUT_CSV} ({len(df_benchmark)} filas)")
    print(f"Ground truth: {OFFICIAL_DIR.resolve()}\n")

    results_df = build_results_dataframe(df_benchmark)

    export_results_csv(results_df)
    save_wer_chart(results_df)
    save_typology_chart(results_df)
    print_summary(results_df)


if __name__ == "__main__":
    main()
