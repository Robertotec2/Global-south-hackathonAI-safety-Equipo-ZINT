#!/usr/bin/env python3
"""
Benchmark comparativo Whisper vs. Gemini — Eval Whisper Yucatan
==============================================================
Compara transcripciones automáticas contra ground truth humano
(transcripciones_oficiales/) y cuantifica WER, reducción de error
y genera visualización para el reporte del hackathon.

Entrada : benchmark_whisper_vs_gemini.csv
          Columnas: Archivo, Transcripcion_Whisper, Transcripcion_Gemini
Ground  : transcripciones_oficiales/TranscripcionOficialAudio{N}.txt
Salida  : resultados_finales_benchmark.csv, comparativa_modelos.png

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
from jiwer import wer

# ── Configuración ──────────────────────────────────────────────────────────
INPUT_CSV = Path("benchmark_whisper_vs_gemini.csv")
OFFICIAL_DIR = Path("transcripciones_oficiales")
OUTPUT_CSV = Path("resultados_finales_benchmark.csv")
CHART_PATH = Path("comparativa_modelos.png")

# Nombres alternativos por si existen transcripciones con typo histórico
OFFICIAL_PREFIX = "TranscripcionOficial"
OFFICIAL_PREFIX_ALT = "TranscripcionOfical"


def extract_audio_base(archivo: str) -> str:
    """
    Extrae la base del nombre de audio para mapear al ground truth.
    Ejemplos:
        'Audio1.mpeg'   -> 'Audio1'
        'Audio 1.mp3'   -> 'Audio1'
        'Audio7.mpeg'   -> 'Audio7'
    """
    stem = Path(str(archivo).strip()).stem
    match = re.search(r"(\d+)", stem, flags=re.IGNORECASE)
    if match:
        return f"Audio{match.group(1)}"
    return re.sub(r"\s+", "", stem)


def official_transcript_path(audio_base: str) -> Path | None:
    """Resuelve la ruta del archivo de transcripción oficial."""
    candidates = [
        OFFICIAL_DIR / f"{OFFICIAL_PREFIX}{audio_base}.txt",
        OFFICIAL_DIR / f"{OFFICIAL_PREFIX_ALT}{audio_base}.txt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def normalize_text(text: str) -> str:
    """Normaliza texto para comparación WER justa."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def safe_wer(reference: str, hypothesis: str) -> float:
    """Calcula WER con manejo de entradas vacías."""
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)

    if not ref and not hyp:
        return 0.0
    if not hyp:
        return 1.0
    if not ref:
        return 1.0

    return float(wer(ref, hyp))


def error_reduction_pct(wer_whisper: float, wer_gemini: float) -> float:
    """
    Porcentaje de reducción de error de Gemini respecto a Whisper.

    Fórmula: ((WER_Whisper - WER_Gemini) / WER_Whisper) * 100

    Valores positivos  -> Gemini cometió menos errores que Whisper.
    Valores negativos  -> Gemini fue peor que Whisper.
    """
    if wer_whisper == 0.0:
        if wer_gemini == 0.0:
            return 0.0
        return -100.0
    return ((wer_whisper - wer_gemini) / wer_whisper) * 100.0


def load_benchmark_csv() -> pd.DataFrame:
    """Lee el CSV de benchmark exportado desde Colab."""
    if not INPUT_CSV.exists():
        print(f"ERROR: No se encontro '{INPUT_CSV.resolve()}'.")
        print("   Exporta desde Colab el archivo benchmark_whisper_vs_gemini.csv")
        sys.exit(1)

    try:
        df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(INPUT_CSV, encoding="latin-1")

    required = {"Archivo", "Transcripcion_Whisper", "Transcripcion_Gemini"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: Faltan columnas en el CSV: {sorted(missing)}")
        print(f"   Columnas encontradas: {list(df.columns)}")
        sys.exit(1)

    if df.empty:
        print("ERROR: El archivo benchmark esta vacio.")
        sys.exit(1)

    df["Archivo"] = df["Archivo"].astype(str).str.strip()
    df["Transcripcion_Whisper"] = df["Transcripcion_Whisper"].fillna("").astype(str)
    df["Transcripcion_Gemini"] = df["Transcripcion_Gemini"].fillna("").astype(str)
    return df


def load_official_text(path: Path) -> str:
    """Carga el texto oficial desde archivo .txt."""
    try:
        return path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1").strip()


def analyze_row(archivo: str, whisper: str, gemini: str) -> dict | None:
    """Procesa una fila del benchmark; retorna None si falta ground truth."""
    audio_base = extract_audio_base(archivo)
    official_path = official_transcript_path(audio_base)

    if official_path is None:
        expected = OFFICIAL_DIR / f"{OFFICIAL_PREFIX}{audio_base}.txt"
        print(
            f"ADVERTENCIA: Sin ground truth para '{archivo}' "
            f"(esperado: {expected.name}). Fila omitida."
        )
        return None

    texto_oficial = load_official_text(official_path)
    wer_whisper = safe_wer(texto_oficial, whisper)
    wer_gemini = safe_wer(texto_oficial, gemini)
    reduccion = error_reduction_pct(wer_whisper, wer_gemini)

    return {
        "Archivo": archivo,
        "Texto_Oficial": texto_oficial,
        "Transcripcion_Whisper": whisper,
        "Transcripcion_Gemini": gemini,
        "WER_Whisper": round(wer_whisper, 4),
        "WER_Gemini": round(wer_gemini, 4),
        "Reduccion_Error": round(reduccion, 2),
        "Archivo_Oficial": official_path.name,
    }


def build_results_dataframe(df_benchmark: pd.DataFrame) -> pd.DataFrame:
    """Itera el benchmark y construye el DataFrame de resultados."""
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
        print("ERROR: Ninguna fila pudo evaluarse. Revisa transcripciones_oficiales/.")
        sys.exit(1)

    return pd.DataFrame(rows)


def save_comparison_chart(results_df: pd.DataFrame) -> None:
    """
    Figure 1 — Barras con WER promedio Whisper vs. Gemini.
    Diseño limpio para reporte PDF del hackathon.
    """
    avg_whisper = results_df["WER_Whisper"].mean()
    avg_gemini = results_df["WER_Gemini"].mean()

    summary = pd.DataFrame(
        {
            "Modelo": ["Whisper-large-v3", "Gemini"],
            "WER_Promedio": [avg_whisper, avg_gemini],
        }
    )

    sns.set_theme(style="whitegrid", context="talk", font_scale=1.05)
    fig, ax = plt.subplots(figsize=(8, 6))

    palette = ["#E45756", "#4C78A8"]
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
        "Benchmark Eval Whisper Yucatan\nWER promedio vs. transcripciones oficiales",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Modelo de transcripcion", fontsize=12)
    ax.set_ylabel("Word Error Rate (WER)", fontsize=12)
    ax.set_ylim(0, min(1.05, max(avg_whisper, avg_gemini) * 1.25 + 0.05))

    for container in bars.containers:
        ax.bar_label(
            container,
            fmt=lambda v: f"{v:.1%}",
            padding=4,
            fontsize=11,
            fontweight="bold",
        )

    ax.axhline(y=0, color="black", linewidth=0.8)
    plt.tight_layout()
    fig.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_summary(results_df: pd.DataFrame) -> None:
    """Resumen en consola."""
    n = len(results_df)
    avg_w = results_df["WER_Whisper"].mean()
    avg_g = results_df["WER_Gemini"].mean()
    avg_red = results_df["Reduccion_Error"].mean()
    gemini_wins = (results_df["WER_Gemini"] < results_df["WER_Whisper"]).sum()

    print("\n" + "=" * 62)
    print("BENCHMARK COMPARATIVO — Eval Whisper Yucatan")
    print("=" * 62)
    print(f"Audios evaluados     : {n}")
    print(f"WER promedio Whisper : {avg_w:.1%}")
    print(f"WER promedio Gemini  : {avg_g:.1%}")
    print(f"Reduccion error prom.: {avg_red:+.1f}%")
    print(f"Gemini gano en       : {gemini_wins}/{n} audios")
    print("-" * 62)

    for _, row in results_df.iterrows():
        print(
            f"{row['Archivo']:<16} | WER W:{row['WER_Whisper']:.1%} "
            f"G:{row['WER_Gemini']:.1%} | Reduccion:{row['Reduccion_Error']:+.1f}%"
        )

    print("=" * 62)
    print(f"CSV  : {OUTPUT_CSV.resolve()}")
    print(f"Chart: {CHART_PATH.resolve()}")
    print("=" * 62 + "\n")


def main() -> None:
    if not OFFICIAL_DIR.exists():
        print(f"ERROR: Crea la carpeta '{OFFICIAL_DIR}/' con los archivos oficiales.")
        print("   Ejemplo: transcripciones_oficiales/TranscripcionOficialAudio1.txt")
        sys.exit(1)

    df_benchmark = load_benchmark_csv()
    print(f"Benchmark cargado : {INPUT_CSV} ({len(df_benchmark)} filas)")
    print(f"Ground truth dir  : {OFFICIAL_DIR.resolve()}\n")

    results_df = build_results_dataframe(df_benchmark)

    try:
        results_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    except OSError as exc:
        print(f"ERROR al guardar CSV: {exc}")
        sys.exit(1)

    try:
        save_comparison_chart(results_df)
    except Exception as exc:
        print(f"ADVERTENCIA: No se pudo generar la grafica: {exc}")

    print_summary(results_df)


if __name__ == "__main__":
    main()
