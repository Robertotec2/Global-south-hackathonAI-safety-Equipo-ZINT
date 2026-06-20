#!/usr/bin/env python3
"""
Mimikyu Protocol v1.0 — Evaluación de Sordera Digital
Evalúa sesgo acústico y exclusión algorítmica en transcripción Whisper
frente a dialectos rurales (ej. Yucatán).

Backend: Groq API (gratuita) — modelo whisper-large-v3

Uso: python main.py
Requisitos: audios_yucatan/ con archivos .mp3, .wav o .m4a
Salida: resultados_evaluacion.csv
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq, APIError, APIConnectionError, RateLimitError

# ── Configuración ──────────────────────────────────────────────────────────
AUDIO_DIR = Path("audios_yucatan")
OUTPUT_CSV = Path("resultados_evaluacion.csv")
WHISPER_MODEL = "whisper-large-v3"  # Modelo Whisper en Groq
SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a"}


def print_banner() -> None:
    """Muestra banner de inicio visualmente llamativo."""
    banner = r"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   👾  Iniciando Evaluación de Sordera Digital                ║
    ║       (Mimikyu Protocol) v1.0...                             ║
    ║                                                              ║
    ║   Backend: Groq · whisper-large-v3                           ║
    ║   Objetivo: detectar sesgo acústico frente a                 ║
    ║   dialectos rurales yucatecos                                ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def load_api_client() -> Groq:
    """Carga GROQ_API_KEY desde .env y devuelve cliente Groq."""
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ Error: GROQ_API_KEY no encontrada.")
        print("   Crea un archivo .env con: GROQ_API_KEY=gsk_...")
        sys.exit(1)

    try:
        client = Groq(api_key=api_key)
    except Exception as exc:
        print("❌ Error: no se pudo inicializar el cliente Groq.")
        print(f"   Detalle: {exc}")
        sys.exit(1)

    return client


def discover_audio_files(directory: Path) -> list[Path]:
    """Encuentra dinámicamente todos los audios soportados en la carpeta."""
    if not directory.exists():
        print(f"❌ La carpeta '{directory}' no existe.")
        print(f"   Créala y coloca archivos {', '.join(sorted(SUPPORTED_EXTENSIONS))}.")
        sys.exit(1)

    files = sorted(
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not files:
        print(f"⚠️  No se encontraron audios en '{directory}'.")
        print(f"   Extensiones válidas: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        sys.exit(1)

    return files


def transcribe_file(client: Groq, audio_path: Path) -> tuple[str, str]:
    """
    Envía un archivo a Groq Whisper (whisper-large-v3).
    Retorna (transcripcion, estado) donde estado es 'Éxito' o 'Error'.
    """
    try:
        with audio_path.open("rb") as audio_file:
            response = client.audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=audio_file,
                # language="es",           # Opcional: forzar español
                # response_format="json",  # Por defecto devuelve JSON con .text
            )

        transcription = (response.text or "").strip()
        return transcription, "Éxito"

    except RateLimitError as exc:
        return f"[RATE LIMIT] {exc}", "Error"

    except APIConnectionError as exc:
        return f"[CONNECTION ERROR] {exc}", "Error"

    except APIError as exc:
        # Errores de la API Groq (auth, formato, tamaño, etc.)
        return f"[API ERROR] {exc}", "Error"

    except OSError as exc:
        # Errores de lectura del archivo local
        return f"[FILE ERROR] {exc}", "Error"

    except Exception as exc:
        # Cualquier otro fallo inesperado
        return f"[UNEXPECTED ERROR] {exc}", "Error"


def format_progress(current: int, total: int, filename: str) -> str:
    """Barra de progreso simple en consola."""
    pct = int((current / total) * 100) if total else 0
    bar_len = 30
    filled = int(bar_len * current / total) if total else 0
    bar = "█" * filled + "░" * (bar_len - filled)
    return f"[{bar}] {pct:3d}% ({current}/{total}) → {filename}"


def export_results(rows: list[dict]) -> None:
    """Exporta resultados a CSV con pandas."""
    df = pd.DataFrame(rows, columns=["Nombre_Archivo", "Transcripcion_Whisper", "Estado"])
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")


def print_summary(rows: list[dict], elapsed: float) -> None:
    """Resumen final en consola."""
    total = len(rows)
    ok = sum(1 for r in rows if r["Estado"] == "Éxito")
    err = total - ok

    print("\n" + "═" * 62)
    print("📊 RESUMEN DE EVALUACIÓN — Mimikyu Protocol v1.0 (Groq)")
    print("═" * 62)
    print(f"   Modelo              : {WHISPER_MODEL}")
    print(f"   Archivos procesados : {total}")
    print(f"   ✅ Éxito            : {ok}")
    print(f"   ❌ Error            : {err}")
    print(f"   ⏱  Tiempo total     : {elapsed:.1f}s")
    print(f"   📁 CSV exportado    : {OUTPUT_CSV.resolve()}")
    print("═" * 62 + "\n")


def main() -> None:
    start_time = time.time()
    print_banner()

    print(f"🕐 Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 Carpeta de audios: {AUDIO_DIR.resolve()}\n")

    client = load_api_client()
    audio_files = discover_audio_files(AUDIO_DIR)
    total = len(audio_files)

    print(f"🔍 Se encontraron {total} archivo(s) de audio.\n")

    results: list[dict] = []

    for index, audio_path in enumerate(audio_files, start=1):
        filename = audio_path.name

        print(format_progress(index - 1 if index > 1 else 0, total, filename))
        print(f"   ⏳ Procesando: {filename} ...")

        transcription, status = transcribe_file(client, audio_path)

        results.append({
            "Nombre_Archivo": filename,
            "Transcripcion_Whisper": transcription,
            "Estado": status,
        })

        icon = "✅" if status == "Éxito" else "❌"
        print(f"   {icon} Finalizado: {filename} [{status}]")
        print(format_progress(index, total, filename))
        print()

    export_results(results)
    elapsed = time.time() - start_time
    print_summary(results, elapsed)


if __name__ == "__main__":
    main()
