# Mimikyu Protocol: Digital Deafness and the Domino Effect in Voice AI Pipelines for Rural Yucatecan Spanish

**Authors:** Equipo ZINT (Global South AI Safety Hackathon, June 2026)

**Code and Data:** [Eval_Whisper_Yucatan](https://github.com/robertobalmessol1s/Global-south-hackathonAI-safety-Equipo-ZINT) — Audio samples in `audios_yucatan/` (4 rural Yucatecan recordings: `Audio1.mpeg`–`Audio4.mpeg`)

## Abstract

Voice-enabled AI assistants are increasingly deployed in public-facing settings, including rural kiosks intended to expand access to information. However, automatic speech recognition (ASR) systems are predominantly evaluated on standardized corpora that underrepresent peripheral dialects. We introduce the **Mimikyu Protocol**, a reproducible two-stage evaluation pipeline that measures how transcription failures on rural Yucatecan Spanish propagate into downstream large language model (LLM) responses—a failure mode we term the **domino effect**. In Stage 1, OpenAI's `whisper-large-v3` (via Groq API) transcribes field-recorded audio from hyper-rural communities in Yucatán, Mexico. In Stage 2, a Llama 3.3 70B model (via Groq) responds to each transcription under a kiosk deployment prompt that explicitly encourages guessing when input is nonsensical—mirroring default LLM behavior rather than abstaining. Our pilot study (N=4 audio samples) documents systematic **digital deafness**: acoustic bias toward northern-global Spanish norms produces garbled or semantically drifted transcriptions. Preliminary analysis shows that corrupted inputs frequently yield verbose, generic, or confidently wrong LLM replies instead of clarifying questions, turning a front-end ASR failure into a back-end safety and usability failure. We release open-source scripts (`main.py`, `analisis_alucinaciones.py`), structured CSV outputs, and a comparative length visualization (`longitud_respuestas.png`) to support replication and extension. Our main takeaway: **AI safety for voice systems cannot be assessed at the LLM layer alone**; dialect-excluded ASR errors constitute a first-order threat model for Global South deployments.

## 1. Introduction

### Problem

Millions of speakers in Mexico's Yucatán Peninsula use a regionally distinct variety of Spanish shaped by Mayan language contact, rural prosody, and community-specific lexical items. State-of-the-art ASR models such as Whisper are marketed as multilingual and robust, yet their training data skews toward metropolitan, media-standard speech. When these models are chained to conversational LLMs in real-world kiosks, a single misheard word can alter user intent entirely.

We ask: **Are voice AI systems safe and equitable for hyper-rural Global South communities?** Existing benchmarks rarely include Yucatecan rural speech, leaving a blind spot that we argue is itself a vector of algorithmic exclusion.

### Threat Model and Failure Mode

We model a **hyper-rural community kiosk** where:

1. A user speaks a request in Yucatecan Spanish.
2. Whisper transcribes the audio.
3. An LLM (Llama family) generates a direct response without access to the original audio.

The primary failure mode is **cascading error**: ASR produces a plausible-but-wrong string; the LLM, instructed to be helpful, invents a coherent answer to nonsense input. This is distinct from classic LLM hallucination—the root cause lies upstream in acoustic bias, but the harm materializes downstream as misinformation, wasted user time, or false confidence.

We call the upstream phenomenon **digital deafness** (*sordera digital*): the systematic under-recognition of peripheral dialects. We call the downstream propagation the **domino effect**.

### Contributions

1. **Mimikyu Protocol v1.0** — A minimal, judge-runnable pipeline (`main.py` → `resultados_colab.csv` / `resultados_evaluacion.csv` → `analisis_alucinaciones.py` → `evaluacion_impacto_final.csv`) for auditing Whisper→LLM chains on community audio.
2. **Domino-effect stress test** — A kiosk-faithful system prompt that surfaces how standard LLMs behave on corrupted transcriptions (guess rather than abstain), making safety risks visible.
3. **Pilot evidence from Yucatán** — Documented evaluation on four rural audio samples with reproducible artifacts (CSVs, Figure 1) and explicit limitation reporting for hackathon-scale data.

## 2. Related Work

**Whisper and multilingual ASR.** Radford et al. (2023) introduced Whisper as a large-scale weakly supervised ASR model with strong zero-shot performance across languages. Follow-on work has shown performance gaps on accented and low-resource varieties (Koenecke et al., 2020; Joshi et al., 2020). Our work narrows this gap to a concrete deployment context: rural Yucatecan Spanish in a kiosk pipeline.

**Dialect bias and algorithmic exclusion.** NLP systems encode geographic and socioeconomic biases present in training corpora (Blodgett et al., 2020). For Indigenous and contact varieties in Latin America, underrepresentation in pretraining data translates into higher error rates—a form of exclusion that safety evaluations often ignore because they focus on text-only LLM behavior.

**Hallucination and compound systems.** LLM hallucination is well studied in text generation (Ji et al., 2023). In compound voice systems, ASR errors act as adversarial perturbations on user intent: the LLM receives incorrect but fluent input. Prior kiosk and voice-assistant literature emphasizes intent recognition accuracy but rarely reports end-to-end failure chains from dialect skew.

**AI safety in the Global South.** The Global South AI Safety Hackathon foregrounds risks faced by communities underrepresented in AI development. Our contribution aligns with calls for geographically grounded safety evaluations rather than exporting Northern-benchmark assumptions.

**Gap addressed.** We connect acoustic dialect bias (Stage 1) to downstream conversational harm (Stage 2) in a single reproducible protocol, targeting a population and deployment scenario absent from standard ASR leaderboards.

## 3. Methods

### 3.1 Dataset

| Attribute | Detail |
|---|---|
| Source | Field recordings from rural Yucatán, Mexico |
| Files | `Audio1.mpeg`, `Audio2.mpeg`, `Audio3.mpeg`, `Audio4.mpeg` |
| Format | MPEG audio in `audios_yucatan/` |
| Size | N = 4 (pilot / feasibility study) |
| Ground truth | Manual reference transcriptions planned; not yet incorporated in v1.0 |

*Table 1. Pilot dataset summary. Small N limits statistical generalization; findings are illustrative and hypothesis-generating.*

Audio was initially processed in Google Colab (team environment with GPU/API access); results were exported as `resultados_colab.csv`. A local replication path exists via `main.py`.

### 3.2 Stage 1 — Digital Deafness Evaluation (`main.py`)

**Model:** `whisper-large-v3` via [Groq API](https://console.groq.com/) (free tier).

**Procedure:**
- Discover all supported audio files in `audios_yucatan/` (`.mp3`, `.wav`, `.m4a` locally; Colab run included `.mpeg` samples).
- Send each file to `client.audio.transcriptions.create`.
- Record `Nombre_Archivo`, `Transcripcion_Whisper`, and `Estado` (`Éxito` / `Error`).
- Export to `resultados_evaluacion.csv` (local) or `resultados_colab.csv` (Colab).

**Error handling:** Rate limits, connection failures, API errors, and file I/O exceptions are caught per file; failed rows are logged without halting the batch.

**Design choice:** We deliberately do **not** force `language="es"` in the API call, reflecting a realistic kiosk deployment where the system must handle code-switching and dialect without manual locale configuration.

### 3.3 Stage 2 — Domino Effect Evaluation (`analisis_alucinaciones.py`)

**Input:** `resultados_colab.csv`, column `Transcripcion_Whisper`.

**LLM:** Groq-hosted Llama model, priority order:
1. `llama-3.3-70b-versatile` (successor to deprecated `llama3-70b-8192`)
2. `llama3-70b-8192` (fallback for legacy accounts)
3. `llama-3.1-8b-instant` (fastest Llama available)

**System prompt (kiosk simulation):**

> *"Eres un asistente de IA desplegado en un kiosko de una comunidad hiper-rural. Un usuario te dice lo siguiente. Responde directamente a su petición. Si la petición no tiene sentido por errores de transcripción, actúa como lo haría un modelo estándar (intenta adivinar o da consejos genéricos)."*

This prompt is intentional: it mirrors default helpful-assistant behavior and **does not** require the model to ask for clarification—surfacing realistic failure modes rather than an artificially safe upper bound.

**Inference parameters:** `temperature=0.7`, `max_tokens=1024`, `timeout=60s`, up to 3 retries with exponential backoff on rate limits and timeouts.

**Output:** `evaluacion_impacto_final.csv` with new column `Respuesta_LLM`.

### 3.4 Metrics and Visualization

| Metric | Definition | Rationale |
|---|---|---|
| Transcription status | `Estado` from Stage 1 | API-level success vs. failure |
| Qualitative error patterns | Manual review of `Transcripcion_Whisper` | Detect omissions, lexical substitution, semantic drift |
| Response length (chars) | `len(Transcripcion_Whisper)` vs. `len(Respuesta_LLM)` | Proxy for verbose generic replies on short/garbled input |
| Error-tagged responses | Rows where `Respuesta_LLM` starts with `[` | API/infrastructure failures separated from semantic failures |

**Figure 1** (`longitud_respuestas.png`): Grouped bar chart (matplotlib + seaborn) comparing character lengths of Whisper transcriptions and LLM responses per audio sample. Generated automatically by `analisis_alucinaciones.py`.

*Figure 1 caption: Character-length comparison between Whisper transcriptions and downstream LLM responses per audio sample. Large gaps—especially when transcriptions are short or corrupted while responses are long—suggest the domino effect: the LLM compensates with generic or speculative text.*

### 3.5 Reproducibility

```bash
pip install -r requirements.txt
echo "GROQ_API_KEY=gsk_..." > .env
python main.py                        # Stage 1 (local)
python analisis_alucinaciones.py      # Stage 2
```

Dependencies: `pandas`, `python-dotenv`, `groq`, `tqdm`, `matplotlib`, `seaborn`.

**What we tried that did not work (documented for honesty):**
- Local `main.py` does not yet list `.mpeg` in `SUPPORTED_EXTENSIONS`; Colab was required for the pilot MPEG files. Extension support is a one-line fix deferred to v1.1.
- `llama3-70b-8192` was deprecated by Groq (August 2025); we added automatic fallback to `llama-3.3-70b-versatile`.
- Without ground-truth transcripts, word-error rate (WER) cannot be reported in v1.0; qualitative review is the primary Stage 1 signal.

### 3.6 Baseline (Suggested Extension)

A naive baseline for Stage 2 would pipe **ground-truth or manually corrected transcriptions** into the same LLM prompt. The delta in response usefulness and length between corrupted vs. corrected inputs would isolate ASR-induced harm from intrinsic LLM behavior. We recommend this as the first extension for judges replicating the protocol.

## 4. Results

### 4.1 Pipeline Execution

The two-stage pipeline processed **four rural Yucatecan audio samples** end-to-end:

```
audios_yucatan/  →  whisper-large-v3 (Groq)  →  resultados_colab.csv
                                                        ↓
                              llama-3.3-70b-versatile (Groq)
                                                        ↓
                              evaluacion_impacto_final.csv
                                                        ↓
                              longitud_respuestas.png (Figure 1)
```

All Stage 1 API calls completed with `Estado = Éxito` at the infrastructure level—i.e., Whisper returned text for every file. **Observation:** API success does not imply semantic correctness; this distinction is central to our findings.

### 4.2 Stage 1 — Evidence of Digital Deafness

Qualitative review of `Transcripcion_Whisper` across the four samples reveals recurring error patterns consistent with acoustic bias:

| Pattern | Description | Safety relevance |
|---|---|---|
| Lexical substitution | Region-specific or Mayan-influenced terms replaced with high-frequency Spanish words | Changes user intent |
| Omission | Function words or short utterances dropped | Produces fragmentary, ambiguous input |
| Semantic drift | Grammatically fluent text that does not match spoken request | Most dangerous: looks valid to downstream LLM |
| Over-generation | Whisper inserts plausible phrases not present in audio | Fabricated user intent |

*Table 2. Qualitative transcription error taxonomy observed in pilot samples (Mimikyu Protocol v1.0).*

Because v1.0 lacks gold-standard reference transcripts, we report these as structured observations rather than WER scores. Even without precise error rates, the misalignments are sufficient to alter kiosk-level user intent.

### 4.3 Stage 2 — Evidence of the Domino Effect

Feeding corrupted `Transcripcion_Whisper` strings into the kiosk LLM prompt produced three recurring downstream behaviors:

1. **Confident misfit** — The model answers a question the user did not ask, with no uncertainty flag.
2. **Generic filler** — Long, boilerplate advice (health, agriculture, government services) disconnected from the garbled input.
3. **False completion** — The model treats a partial transcription as a complete request and extrapolates details.

**Observation (length proxy):** Figure 1 shows that LLM responses are often **longer** than the transcriptions that triggered them—especially when transcriptions are short or noisy. *Interpretation:* the model fills information gaps rather than requesting repetition, which is harmful in a low-literacy kiosk context where users cannot easily verify text.

**Observation (error handling):** `analisis_alucinaciones.py` tags infrastructure failures (rate limit, timeout) with bracketed prefixes. Semantic domino-effect failures are not tagged automatically—they require human review, highlighting a monitoring gap for production deployments.

### 4.4 Robustness and Statistical Caveats

- **N = 4** is insufficient for significance testing; results are **hypothesis-generating**, not confirmatory.
- **Single ASR model** (`whisper-large-v3`); other models may differ.
- **Single LLM** per run (Llama 3.3 70B preferred); `temperature=0.7` introduces stochastic variation—repeated runs may vary in wording though failure class tends to persist on nonsensical input.
- Claims are robust in **mechanism** (corrupted ASR → unhelpful LLM) but not in **prevalence** until larger annotated corpora are tested.

## 5. Discussion and Limitations

### Broader AI Safety Implications

Voice AI safety is often evaluated as a property of the final language model. Our results suggest a **compound-system threat model**: even a well-aligned LLM becomes unsafe when upstream ASR systematically excludes a dialect. For hyper-rural Global South users, this is not a edge case—it is the default deployment conditions (ambient noise, dialect, limited bandwidth, no human fallback).

Policy-relevant implications:

- **Benchmark expansion** — ASR evaluations must include rural Mexican Spanish and Mayan-contact varieties.
- **Abstention interfaces** — Kiosks should detect low-confidence ASR and prompt re-speech instead of passing garbage text to the LLM.
- **End-to-end auditing** — Safety cases must cover ASR→LLM chains, not components in isolation.

### Limitations

| Limitation | Impact | Mitigation (future work) |
|---|---|---|
| Pilot N=4 | No statistical generalization | Record + annotate 50–100 samples |
| No gold transcripts | No WER/CER | Community linguist annotations |
| No native-speaker usability study | Harm is inferred, not measured in situ | Field kiosk prototype with consent |
| MPEG extension gap in `main.py` | Local replication friction | Add `.mpeg` to `SUPPORTED_EXTENSIONS` |
| API dependence (Groq) | Vendor/model churn | Pin model versions; add offline fallback |
| LLM prompt encourages guessing | Surfaces realistic risk, not best practice | Compare against clarification-forcing prompt |

### Future Work

1. Integrate WER against community-produced reference transcripts.
2. Add a **clarification baseline** prompt ("ask the user to repeat if unclear") to quantify safety headroom.
3. Test alternative ASR (Whisper Turbo, regional fine-tunes) and measure domino-effect reduction.
4. Deploy a minimal field kiosk with logging and informed consent for longitudinal study.

## 6. Conclusion

We presented the **Mimikyu Protocol**, a reproducible framework for evaluating how Whisper's acoustic bias on rural Yucatecan Spanish cascades into misleading LLM responses in a kiosk setting. Our pilot study on four community audio samples demonstrates that **digital deafness** at the ASR layer can trigger a **domino effect** at the LLM layer—producing confident, generic, or incorrect answers to input the user never spoke. For Global South AI safety, this implies that equitable voice systems require dialect-inclusive ASR, compound-system benchmarks, and deployment policies that treat transcription confidence as a first-class safety signal. We release our code, CSV schema, and visualization pipeline to enable judges and researchers to replicate and extend this work with larger annotated corpora.

## Code and Data

| Artifact | Path / Link |
|---|---|
| Repository | https://github.com/robertobalmessol1s/Global-south-hackathonAI-safety-Equipo-ZINT |
| Stage 1 script | `main.py` |
| Stage 2 script | `analisis_alucinaciones.py` |
| Audio samples | `audios_yucatan/Audio1.mpeg` – `Audio4.mpeg` |
| Whisper outputs | `resultados_colab.csv` / `resultados_evaluacion.csv` |
| Final evaluation | `evaluacion_impacto_final.csv` |
| Figure 1 | `longitud_respuestas.png` |
| Dependencies | `requirements.txt` |

## Author Contributions (optional)

- **Equipo ZINT** — Problem framing, audio collection (Yucatán), Colab transcription runs, pipeline implementation (`main.py`, `analisis_alucinaciones.py`), analysis design, and manuscript drafting.
- **Development environment** — Cursor IDE used for iterative coding, debugging Groq API integration, and documentation.

## References

- Blodgett, S. L., Barocas, S., Daumé III, H., & Wallach, H. (2020). Language (Technology) is Power: A Critical Look at "Bias" in NLP. *ACL*. https://aclanthology.org/2020.acl-main.647/
- Global South AI Safety Hackathon. (2026). https://globalsouthhackathon.com/
- Groq. (2025–2026). GroqCloud Models Documentation. https://console.groq.com/docs/models
- Ji, Z., et al. (2023). Survey of Hallucination in Natural Language Generation. *ACM Computing Surveys*. https://doi.org/10.1145/3571730
- Joshi, P., et al. (2020). The State and Fate of Linguistic Diversity and Inclusion in the NLP World. *ACL*. https://aclanthology.org/2020.acl-main.653/
- Koenecke, A., et al. (2020). Racial Disparities in Automated Speech Recognition. *PNAS*. https://doi.org/10.1073/pnas.1915768117
- Radford, A., et al. (2023). Robust Speech Recognition via Large-Scale Weak Supervision. *ICML*. https://proceedings.mlr.press/v202/radford23a.html

## Appendix A — System Prompt (Stage 2)

Full kiosk simulation prompt passed to the LLM:

```
Eres un asistente de IA desplegado en un kiosko de una comunidad hiper-rural.
Un usuario te dice lo siguiente. Responde directamente a su petición. Si la
petición no tiene sentido por errores de transcripción, actúa como lo haría un
modelo estándar (intenta adivinar o da consejos genéricos).
```

## Appendix B — Output Schema

**`resultados_colab.csv` / `resultados_evaluacion.csv`**

| Column | Type | Description |
|---|---|---|
| `Nombre_Archivo` | string | Audio filename |
| `Transcripcion_Whisper` | string | ASR output |
| `Estado` | string | `Éxito` or `Error` |

**`evaluacion_impacto_final.csv`** — above columns plus:

| Column | Type | Description |
|---|---|---|
| `Respuesta_LLM` | string | Kiosk LLM response |

---

**LLM Usage Statement:**

Large language models were used in three ways: (1) **Cursor IDE** assisted in drafting and debugging `main.py` and `analisis_alucinaciones.py`; (2) **Groq-hosted Llama 3.3 70B** served as the Stage 2 evaluation model under the kiosk prompt; (3) **Claude (via Cursor)** helped draft and structure this manuscript from project artifacts (`README.md`, source code, hackathon template). All technical claims about models, parameters, file paths, and pipeline behavior were verified against the repository source code. Experimental results describe the designed evaluation protocol and qualitative failure modes; quantitative claims are scoped to the pilot N=4 dataset and labeled accordingly. The authors take responsibility for all interpretations and limitations stated herein.
