# Eval_Whisper_Yucatan

**Global South Hackathon AI Safety — Equipo ZINT**

## ¿Qué estamos haciendo?

Este proyecto evalúa el sesgo acústico y la exclusión algorítmica del modelo de transcripción **Whisper** (via Groq API) frente a dialectos rurales del sureste mexicano, específicamente el español yucateco.

La hipótesis central es que los sistemas de reconocimiento de voz entrenados mayoritariamente con datos del norte global presentan una "sordera digital" hacia variantes lingüísticas periféricas: acentos rurales, préstamos mayas, entonaciones regionales y patrones de habla propios de comunidades históricamente subrepresentadas en los datos de entrenamiento.

### Objetivo

Medir cuantitativamente el desempeño de `whisper-large-v3` sobre muestras de audio reales de hablantes yucatecos rurales y documentar patrones de error sistemático que evidencien sesgo acústico.

### Flujo del experimento

```
audios_yucatan/          ← Muestras de audio (MP3, WAV, M4A)
       │
       ▼
 Groq API (whisper-large-v3)
       │
       ▼
resultados_evaluacion.csv  ← Transcripciones + estado por archivo
```

### Salida

Un CSV con columnas:
- `Nombre_Archivo` — identificador de la muestra de audio
- `Transcripcion_Whisper` — texto generado por el modelo
- `Estado` — `Éxito` / `Error`

Los resultados permiten analizar errores de transcripción, omisiones y distorsiones frente a dialectos rurales.

---

## Stack técnico

| Componente | Detalle |
|---|---|
| Modelo de transcripción | `whisper-large-v3` (OpenAI, vía Groq) |
| Backend API | [Groq](https://console.groq.com/) — tier gratuito |
| Lenguaje | Python 3.11+ |
| Librerías | `groq`, `pandas`, `python-dotenv` |

---

## Cómo usamos Cursor

**[Cursor](https://www.cursor.com/)** es el editor de código con IA que usamos como entorno principal de desarrollo en este proyecto. Nos ayuda a:

- **Iterar rápido sobre el código de evaluación:** el autocompletado contextual y el chat en línea permiten modificar la lógica de transcripción, añadir métricas o ajustar el manejo de errores sin salir del editor.
- **Debugging asistido:** cuando la API de Groq devuelve errores inesperados o los archivos de audio no se procesan correctamente, Cursor nos ayuda a rastrear el problema directamente en el código.
- **Explorar la documentación de Groq y Whisper en contexto:** en lugar de alternar entre el editor y el navegador, consultamos la API desde el chat del editor mientras escribimos el código.
- **Refactorizar y mantener consistencia:** al trabajar en equipo, Cursor facilita mantener un estilo uniforme y detectar duplicaciones o mejoras en el pipeline de evaluación.

En resumen, Cursor actúa como copiloto de desarrollo: acelera la escritura, reduce el tiempo de depuración y permite que el equipo se enfoque en el problema de investigación (sesgo acústico) en lugar de en la infraestructura del código.

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/robertobalmessol1s/Global-south-hackathonAI-safety-Equipo-ZINT.git
cd Global-south-hackathonAI-safety-Equipo-ZINT

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar la API key de Groq
echo "GROQ_API_KEY=gsk_..." > .env

# 4. Colocar audios en la carpeta
# Copia tus archivos .mp3, .wav o .m4a en audios_yucatan/

# 5. Ejecutar la evaluación
python main.py
```

---

## Contexto: Global South AI Safety Hackathon

Este proyecto fue desarrollado por el **Equipo ZINT** en el marco del [Global South AI Safety Hackathon](https://globalsouthhackathon.com/), una iniciativa que promueve la participación del sur global en la investigación de seguridad en IA.

Nuestra propuesta parte de una pregunta concreta: **¿son seguros los sistemas de IA para todos por igual?** Los modelos de lenguaje y transcripción se evalúan casi siempre sobre datos del norte global. Las comunidades rurales latinoamericanas, con sus dialectos, lenguas originarias y contextos acústicos distintos, quedan fuera de esas métricas. Esa invisibilidad no es neutral: es un vector de exclusión.

---

## Equipo ZINT

Proyecto desarrollado en Yucatán, México.
