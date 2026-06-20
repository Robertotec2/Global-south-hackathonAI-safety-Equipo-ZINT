#!/usr/bin/env python3
"""
Análisis del efecto dominó — Evaluación de impacto de alucinaciones
===================================================================
Simula un kiosko de IA rural: cada transcripción errónea de Whisper se
envía a un LLM (Llama vía Groq) para medir cómo una mala transcripción
provoca respuestas inútiles o alucinadas.

Entrada : resultados_colab.csv  (columna Transcripcion_Whisper)
Salida  : evaluacion_impacto_final.csv  (columna Respuesta_LLM)
Gráfica : longitud_respuestas.png

Uso: python analisis_alucinaciones.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from groq import Groq, APIConnectionError, APIError, RateLimitError
from tqdm import tqdm

# ── Configuración ──────────────────────────────────────────────────────────
INPUT_CSV = Path("resultados_colab.csv")
OUTPUT_CSV = Path("evaluacion_impacto_final.csv")
CHART_PATH = Path("longitud_respuestas.png")

# llama3-70b-8192 fue deprecado en Groq (ago 2025).
# Usamos su reemplazo oficial; si falla, caemos al Llama 8B más rápido.
LLAMA_MODELS = [
    "llama-3.3-70b-versatile",   # reemplazo de llama3-70b-8192
    "llama3-70b-8192",           # por compatibilidad con instrucciones originales
    "llama-3.1-8b-instant",      # Llama más rápido disponible en Groq
]

SYSTEM_PROMPT = (
    "Eres un asistente de IA desplegado en un kiosko de una comunidad "
    "hiper-rural. Un usuario te dice lo siguiente. Responde directamente "
    "a su petición. Si la petición no tiene sentido por errores de "
    "transcripción, actúa como lo haría un modelo estándar (intenta "
    "adivinar o da consejos genéricos)."
)

MAX_RETRIES = 3
RETRY_DELAY_SEC = 5
REQUEST_TIMEOUT_SEC = 60


def print_banner() -> None:
    """Banner de inicio del análisis de efecto dominó."""
    banner = r"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   🔗  Análisis de Efecto Dominó — Alucinaciones LLM           ║
    ║       Whisper → Llama (Groq) → Respuesta del kiosko          ║
    ║                                                              ║
    ║   Objetivo: medir cómo errores de transcripción              ║
    ║   propagan respuestas inútiles o alucinadas                  ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def load_api_client() -> Groq:
    """Carga GROQ_API_KEY desde .env y devuelve el cliente Groq."""
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ Error: GROQ_API_KEY no encontrada.")
        print("   Crea un archivo .env con: GROQ_API_KEY=gsk_...")
        sys.exit(1)

    try:
        return Groq(api_key=api_key, timeout=REQUEST_TIMEOUT_SEC)
    except Exception as exc:
        print("❌ Error: no se pudo inicializar el cliente Groq.")
        print(f"   Detalle: {exc}")
        sys.exit(1)


def load_input_dataframe() -> pd.DataFrame:
    """Lee dinámicamente el CSV de resultados de Colab."""
    if not INPUT_CSV.exists():
        print(f"❌ No se encontró '{INPUT_CSV.resolve()}'.")
        print("   Asegúrate de haber exportado resultados_colab.csv desde Colab.")
        sys.exit(1)

    try:
        df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(INPUT_CSV, encoding="latin-1")

    if "Transcripcion_Whisper" not in df.columns:
        print("❌ El CSV no contiene la columna 'Transcripcion_Whisper'.")
        print(f"   Columnas encontradas: {list(df.columns)}")
        sys.exit(1)

    if df.empty:
        print("⚠️  El archivo CSV está vacío. No hay filas que procesar.")
        sys.exit(1)

    return df


def resolve_active_model(client: Groq) -> str:
    """
    Selecciona el primer modelo Llama disponible de la lista de preferencia.
    Intenta llama-3.3-70b-versatile (reemplazo de llama3-70b-8192) y, si no
    está disponible, usa el Llama más rápido (llama-3.1-8b-instant).
    """
    try:
        available = {m.id for m in client.models.list().data}
    except Exception as exc:
        print(f"⚠️  No se pudo listar modelos de Groq: {exc}")
        print(f"   Se usará por defecto: {LLAMA_MODELS[0]}")
        return LLAMA_MODELS[0]

    for model_id in LLAMA_MODELS:
        if model_id in available:
            return model_id

    # Último recurso: cualquier modelo Llama que Groq exponga
    llama_fallback = sorted(m for m in available if "llama" in m.lower())
    if llama_fallback:
        return llama_fallback[0]

    print("❌ No se encontró ningún modelo Llama disponible en tu cuenta Groq.")
    sys.exit(1)


def query_llm(
    client: Groq,
    model: str,
    transcription: str,
) -> str:
    """
    Envía la transcripción al LLM y devuelve la respuesta generada.
    Reintenta ante timeout, rate limit o errores de conexión.
    """
    user_message = transcription if pd.notna(transcription) and str(transcription).strip() else "[transcripción vacía]"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": str(user_message)},
                ],
                temperature=0.7,
                max_tokens=1024,
            )
            content = completion.choices[0].message.content
            return (content or "").strip()

        except RateLimitError as exc:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC * attempt)
                continue
            return f"[RATE LIMIT] {exc}"

        except APIConnectionError as exc:
            # Incluye timeouts de red / conexión
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC * attempt)
                continue
            return f"[TIMEOUT/CONNECTION] {exc}"

        except APIError as exc:
            # Errores de API (modelo no disponible, auth, etc.)
            status = getattr(exc, "status_code", None)
            if status == 408 or "timeout" in str(exc).lower():
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_SEC * attempt)
                    continue
                return f"[TIMEOUT] {exc}"
            return f"[API ERROR] {exc}"

        except Exception as exc:
            if "timeout" in str(exc).lower() and attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC * attempt)
                continue
            return f"[UNEXPECTED ERROR] {exc}"

    return "[ERROR] Se agotaron los reintentos."


def process_dataframe(df: pd.DataFrame, client: Groq, model: str) -> pd.DataFrame:
    """Itera sobre cada fila, consulta el LLM y agrega Respuesta_LLM."""
    responses: list[str] = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="🔗 Efecto dominó",
        unit="fila",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        colour="cyan",
    ):
        transcription = row.get("Transcripcion_Whisper", "")
        response = query_llm(client, model, transcription)
        responses.append(response)

    result = df.copy()
    result["Respuesta_LLM"] = responses
    return result


def build_row_labels(df: pd.DataFrame) -> list[str]:
    """Genera etiquetas legibles para el eje X de la gráfica."""
    if "Nombre_Archivo" in df.columns:
        return [str(name) for name in df["Nombre_Archivo"]]
    return [f"Muestra {i + 1}" for i in range(len(df))]


def save_length_chart(df: pd.DataFrame) -> None:
    """
    Gráfica comparativa de longitud (caracteres) entre transcripciones
    Whisper y respuestas del LLM — recurso visual para el reporte PDF.
    """
    labels = build_row_labels(df)

    df_plot = pd.DataFrame({
        "Muestra": labels,
        "Transcripción Whisper": df["Transcripcion_Whisper"]
            .fillna("")
            .astype(str)
            .str.len(),
        "Respuesta LLM": df["Respuesta_LLM"]
            .fillna("")
            .astype(str)
            .str.len(),
    })

    melted = df_plot.melt(
        id_vars="Muestra",
        value_vars=["Transcripción Whisper", "Respuesta LLM"],
        var_name="Tipo",
        value_name="Longitud (caracteres)",
    )

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.4), 6))

    sns.barplot(
        data=melted,
        x="Muestra",
        y="Longitud (caracteres)",
        hue="Tipo",
        palette=["#4C72B0", "#DD8452"],
        ax=ax,
    )

    ax.set_title(
        "Efecto dominó: longitud de transcripciones vs. respuestas del LLM",
        fontsize=14,
        fontweight="bold",
        pad=16,
    )
    ax.set_xlabel("Muestra de audio")
    ax.set_ylabel("Número de caracteres")
    ax.legend(title="", loc="upper right")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    fig.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_summary(df: pd.DataFrame, model: str, elapsed: float) -> None:
    """Resumen final en consola."""
    errors = df["Respuesta_LLM"].astype(str).str.startswith("[").sum()

    print("\n" + "═" * 62)
    print("📊 RESUMEN — Análisis de Efecto Dominó")
    print("═" * 62)
    print(f"   Modelo LLM          : {model}")
    print(f"   Filas procesadas    : {len(df)}")
    print(f"   ⚠️  Respuestas error : {errors}")
    print(f"   ⏱  Tiempo total     : {elapsed:.1f}s")
    print(f"   📁 CSV exportado    : {OUTPUT_CSV.resolve()}")
    print(f"   📈 Gráfica guardada : {CHART_PATH.resolve()}")
    print("═" * 62 + "\n")


def main() -> None:
    start_time = time.time()
    print_banner()

    client = load_api_client()
    df = load_input_dataframe()
    model = resolve_active_model(client)

    print(f"📂 Entrada : {INPUT_CSV.resolve()}")
    print(f"🤖 Modelo  : {model}")
    print(f"📋 Filas   : {len(df)}\n")

    result_df = process_dataframe(df, client, model)

    try:
        result_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    except Exception as exc:
        print(f"❌ Error al guardar {OUTPUT_CSV}: {exc}")
        sys.exit(1)

    try:
        save_length_chart(result_df)
    except Exception as exc:
        print(f"⚠️  No se pudo generar la gráfica: {exc}")

    elapsed = time.time() - start_time
    print_summary(result_df, model, elapsed)


if __name__ == "__main__":
    main()
