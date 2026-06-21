# Eval Whisper Yucatan: A Deterministic Benchmark of Digital Deafness in Whisper vs. Gemini on Rural Yucatecan and Mayan-Contact Speech

**Authors:** Equipo ZINT (Global South AI Safety Hackathon, June 2026)

**Code and Data:** [Eval_Whisper_Yucatan](https://github.com/robertobalmessol1s/Global-south-hackathonAI-safety-Equipo-ZINT) — 15 field audio samples (`Audio1.mpeg`–`Audio15.mpeg`), human ground truth in `transcripciones_oficiales/`

## Abstract

Automatic speech recognition (ASR) systems are deployed globally, yet they are rarely evaluated on oral narratives from hyper-rural Yucatán where Spanish coexists with Mayan languages (Yucatec Maya, Ch'ol, Wixárika contact varieties, and community-specific lexicon). We present **Eval Whisper Yucatan**, a reproducible deterministic benchmark comparing **Whisper-large-v3** (Groq) against **Gemini 2.5 Flash** (Google AI Studio, `temperature=0`, `seed=42`) on 15 audio clips paired with human-authored reference transcripts. Using `jiwer`, we compute word error rate (WER), error typology (substitutions, deletions, insertions), and error-reduction percentage. Across 15 samples, **both models fail catastrophically**: median WER is **99.1%** (Whisper) and **140.0%** (Gemini). Whisper exhibits **digital deafness** through mass deletions (338 omitted words) and repetitive generic-Spanish hallucinations (*"¿Qué haces?"* loops). Gemini produces more Mayan-contact text when prompted as a Yucatecan linguist and commits **fewer deletions** (173), but suffers extreme **insertion hallucinations** on 2/15 files (Audio3, Audio9), inflating mean WER to 5,943%. Gemini wins on WER in **5/15** samples; Whisper wins in **10/15**. Our hypothesis—that peripheral dialect speech is systematically misrecognized—is **strongly supported**; the secondary hypothesis that Gemini uniformly outperforms Whisper is **not supported** at aggregate level, though Gemini shows localized gains on Maya-rich segments. We release `analisis_comparativo.py`, `resultados_finales_benchmark.csv`, and Figures 1–2 for replication.

## 1. Introduction

### Problem

Voice AI is marketed as inclusive and multilingual. In Mexico's Yucatán Peninsula, daily speech blends regional Spanish with Mayan lexical items, oral tradition genres (creation myths, patron-saint legends, market dialogues), and non-standard prosody. Benchmarks built on metropolitan Spanish do not represent this acoustic-linguistic reality.

We ask: **Do state-of-the-art ASR systems equitably serve hyper-rural Global South speech?**

### Threat Model and Failure Mode

We model a **community transcription pipeline** (kiosk, archival, or accessibility tool) where:

1. A speaker records an oral narrative in Yucatecan Spanish / Mayan contact speech.
2. An ASR model transcribes the audio.
3. Downstream systems (search, LLM assistants, public archives) consume the text.

**Digital deafness** (*sordera digital*): systematic under-recognition or erasure of peripheral speech—especially via **deletions** (empty or truncated output) and **substitutions** (replacement with high-frequency generic Spanish).

**Insertion cascades**: models that loop or over-generate fluent but incorrect text—creating false archival records.

### Contributions

1. **Deterministic benchmark** — 15-audio corpus with strict filename→ground-truth mapping (`TranscripcionOficialAudio{N}.txt`), Colab export (`benchmark_whisper_vs_gemini.csv`), and automated evaluation (`analisis_comparativo.py`).
2. **Quantified error typology** — WER plus substitution/deletion/insertion counts via `jiwer.process_words`, enabling comparison of *sordera digital* (deletions) vs. hallucination (insertions).
3. **Head-to-head evidence** — First documented Whisper vs. Gemini comparison on rural Yucatecan/Mayan-contact field audio with reproducible parameters.

## 2. Related Work

**Whisper.** Radford et al. (2023) demonstrated strong multilingual ASR, but accent and dialect gaps persist (Koenecke et al., 2020).

**Dialect bias.** NLP systems encode training-data geography (Blodgett et al., 2020). Latin American contact varieties remain underrepresented in ASR leaderboards.

**Multimodal LLM transcription.** Gemini-family models accept audio input and can be steered with domain prompts; however, LLM-based ASR lacks standardized dialect benchmarks in the Global South.

**Gap.** We provide a small but fully documented benchmark tying human ground truth to two widely accessible models, with error typology beyond a single WER scalar.

## 3. Methods

### 3.1 Dataset

| Attribute | Detail |
|---|---|
| Samples | 15 MPEG audio files (`audios_yucatan/Audios/Audio1.mpeg`–`Audio15.mpeg`) |
| Content | Oral traditions (Yoremes, Ch'oles, Akatekos, Wixárika), market dialogues (Kimbila), interpersonal narratives |
| Ground truth | Human transcripts in `transcripciones_oficiales/TranscripcionOficialAudio{N}.txt` |
| Mapping rule | `Audio10.mpeg` → `TranscripcionOficialAudio10.txt` (strict) |
| N | 15 (pilot benchmark) |

*Table 1. Benchmark corpus summary.*

### 3.2 Models and Reproducibility

| Model | Provider | ID | Inference settings |
|---|---|---|---|
| Whisper | Groq | `whisper-large-v3` | Colab batch transcription |
| Gemini | Google AI Studio | `gemini-2.5-flash` | `temperature=0`, `seed=42`, linguist system prompt (Yucatecan Spanish + Maya) |

Input artifact: `benchmark_whisper_vs_gemini.csv` with columns `Archivo`, `Transcripcion_Whisper`, `Transcripcion_Gemini`.

### 3.3 Evaluation Script (`analisis_comparativo.py`)

For each row:

1. Load official transcript from `transcripciones_oficiales/`.
2. Normalize text (lowercase, accent strip, punctuation removal).
3. Compute **WER** and **CER** via `jiwer`.
4. Compute **Reduccion_Error** = `((WER_Whisper − WER_Gemini) / WER_Whisper) × 100`.
5. Extract **Sustituciones**, **Omisiones**, **Inserciones** via `jiwer.process_words`.

**Outputs:** `resultados_finales_benchmark.csv`, `comparativa_wer.png` (Figure 1), `tipologia_errores.png` (Figure 2).

### 3.4 Design Choices and Limitations

- **WER vs. Maya orthography:** Reference transcripts are Spanish; Gemini often outputs Yucatec Maya graphemes (`ba'ale'`, `tu'un`). WER penalizes orthographic mismatch even when semantic content aligns—a known limitation we report transparently.
- **Determinism:** Gemini `seed=42` reduces variance; two files still produced massive repetition loops (likely audio/model instability, not seed alone).
- **What failed first:** Initial Colab run used an invalid Gemini API key—Gemini column contained HTTP errors, not transcriptions. The current CSV uses valid Gemini 2.5 Flash outputs.

### 3.5 Suggested Baseline

A naive **human WER ceiling** would require double annotation; a **language-forced** Whisper run (`language="es"`) would isolate locale-hint effects—recommended extensions.

## 4. Results

### 4.1 Pipeline Execution

All **15/15** audio files matched a ground-truth file and were evaluated successfully. No missing `.txt` warnings.

```
benchmark_whisper_vs_gemini.csv  →  analisis_comparativo.py
                                          ↓
                        resultados_finales_benchmark.csv
                        comparativa_wer.png (Figure 1)
                        tipologia_errores.png (Figure 2)
```

### 4.2 Aggregate Metrics

| Metric | Whisper | Gemini |
|---|---|---|
| Mean WER | **137.8%** | **5,943.2%*** |
| **Median WER** | **99.1%** | **140.0%** |
| Mean CER | 100.9% | 3,187.0%* |
| Total deletions | **338** | **173** |
| Total substitutions | 821 | 1,039 |
| Total insertions | 512 | **77,025*** |
| Gemini WER wins | — | **5 / 15 samples** |
| Whisper WER wins | **10 / 15 samples** | — |
| Mean error reduction (Gemini vs. Whisper) | — | **−5,385.8%*** |

*\*Mean and error-reduction aggregates are dominated by two Gemini outliers (Audio3: WER 26,424%; Audio9: WER 60,679%) caused by insertion loops. Median and per-sample analysis are more informative.*

*Table 2. Aggregate benchmark results (`resultados_finales_benchmark.csv`, N=15).*

**Figure 1** (`comparativa_wer.png`): Bar chart of mean WER—Whisper vs. Gemini. Mean Gemini WER appears higher due to outliers; see Table 3 for per-sample detail.

**Figure 2** (`tipologia_errores.png`): Stacked bar chart of error-type proportions. Whisper's error mass skews toward **deletions** (digital deafness); Gemini shows fewer deletions but extreme **insertions** on outlier files.

*Figure 1 caption: Mean word error rate across 15 rural Yucatecan samples. High values for both models indicate systemic failure; Gemini mean is skewed by two insertion-loop outliers.*

*Figure 2 caption: Proportional distribution of substitution, deletion, and insertion errors. Elevated deletions in Whisper support the digital deafness framing; Gemini insertions dominate aggregate error mass.*

### 4.3 Per-Sample Highlights (Selected)

| Audio | Content type | WER Whisper | WER Gemini | Reducción | Interpretation |
|---|---|---|---|---|---|
| Audio1 | Yoremes fire myth | 94.7% | 98.5% | −4.0% | Both fail; Gemini captures Mayan-contact phonology |
| Audio4 | Wixárika dawn myth | 99.1% | 544.1% | −449% | Whisper: single word *"El"*; Gemini: long Maya-like text |
| Audio8 | Kimbila meeting | 203.8% | **98.1%** | **+51.9%** | Gemini wins; captures `K'inbilá`, `Itzmal` |
| Audio10 | Embroidery/market | 101.4% | 171.6% | −69% | Gemini outputs Maya (`Pash ka betik`, `Elsa Cámara Alvarado`) |
| Audio14 | Family narrative | 394.7% | **140.0%** | **+64.5%** | Whisper loops *"¿Qué haces?"*; Gemini closer |
| Audio3 | Akatekos legend | 139.5% | **26,424%** | −18,840% | Gemini insertion cascade |
| Audio9 | Market dialogue | 98.6% | **60,679%** | −61,434% | Gemini insertion cascade |

*Table 3. Selected per-sample results illustrating digital deafness, localized Gemini gains, and insertion outliers.*

### 4.4 Hypothesis Assessment

**H1 — Digital deafness on peripheral speech: SUPPORTED.**

- Median WER ≈ **99%** for Whisper: models do not reliably transcribe this corpus.
- Whisper **deletions (338)** exceed Gemini (**173**): silent omission of reference words is a dominant Whisper failure mode.
- Qualitative pattern: Whisper collapses diverse genres into repetitive generic Spanish (*"¿Qué haces?"*, *"El agua se va a apagar"*) unrelated to ground-truth oral narratives.

**H2 — Gemini (linguist prompt) uniformly reduces error vs. Whisper: NOT SUPPORTED at aggregate level.**

- Whisper wins WER on **10/15** samples.
- Mean Gemini WER is inflated by **two catastrophic insertion loops** (Audio3, Audio9: 32,645 and 43,618 insertions respectively).
- **Median WER** (140% vs. 99%) still favors Whisper slightly.

**H3 — Gemini better preserves Mayan-contact content: PARTIALLY SUPPORTED (qualitative).**

- On Audio8, Audio10, Audio12, and Audio14, Gemini outputs identifiable Maya/Yucatecan forms (`K'inbilá`, `Pash ka betik`, `Caline ta kubinushimbaltsam`) where Whisper produces generic Spanish loops.
- WER undercounts this advantage because references are Spanish while hypotheses are Maya orthography.

### 4.5 Statistical Caveats

- **N = 15**: illustrative, not powered for significance testing (no p-values reported).
- **Two Gemini outliers** distort means; we prioritize **median** and **per-sample** tables.
- **Single run per model**; no inter-annotator agreement on ground truth.
- Claims are **robust in direction** (both models fail; deletions vs. insertions differ) but not in precise effect sizes.

## 5. Discussion and Limitations

### AI Safety Implications

For Global South communities, ASR failure is not a convenience issue—it is **epistemic exclusion**. When a Yoremes fire myth becomes *"un menabolo en su casa"* (Whisper) or a 32,000-word insertion loop (Gemini), archival and voice-interface systems propagate **false cultural records**.

Policy-relevant findings:

1. **Neither Whisper nor prompt-steered Gemini is deployment-ready** for this speech type without domain adaptation.
2. **Deletion-heavy failures (Whisper)** erase content silently—especially dangerous when API returns `success`.
3. **Insertion-heavy failures (Gemini)** create fluent false transcripts—dangerous for archives and LLM downstream use.
4. **WER alone is insufficient** for Mayan-contact speech; future work needs semantic and human rating metrics.

### Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| N=15 | No generalization | Expand corpus; double annotation |
| Spanish references vs. Maya output | WER penalizes valid Maya | Maya-aware metrics; bilingual references |
| Gemini outliers | Skew means | Robust stats; repetition detection filter |
| No domino-effect LLM stage in this CSV | End-to-end kiosk harm not re-quantified here | Run `analisis_alucinaciones.py` on best ASR output |
| Single seed/temperature | Residual variance | Multi-seed report |

### Future Work

1. Fine-tune or few-shot adapt Whisper/Gemini on Yucatecan data.
2. Add **repetition detection** pre-export (truncate insertion loops before WER).
3. Human side-by-side rating (fluency, cultural fidelity) alongside WER.
4. End-to-end domino-effect test: ASR output → rural kiosk LLM.

## 6. Conclusion

We built and executed **Eval Whisper Yucatan**, a deterministic benchmark of Whisper-large-v3 vs. Gemini 2.5 Flash on 15 rural Yucatecan and Mayan-contact audio samples with human ground truth. **Both models fail catastrophically** (median WER ~99–140%), confirming **digital deafness** as a first-order AI safety issue for the Global South. Whisper primarily **omits and genericizes**; Gemini **sometimes captures Maya-contact forms** but can **hallucinate massive insertion loops**. Aggregate superiority of Gemini is **not established**; localized gains are real but fragile. We release code, CSVs, and figures to support replication and corpus expansion.

## Code and Data

| Artifact | Path |
|---|---|
| Repository | https://github.com/robertobalmessol1s/Global-south-hackathonAI-safety-Equipo-ZINT |
| Benchmark input | `benchmark_whisper_vs_gemini.csv` |
| Ground truth | `transcripciones_oficiales/` |
| Evaluation script | `analisis_comparativo.py` |
| Results | `resultados_finales_benchmark.csv` |
| Figure 1 | `comparativa_wer.png` |
| Figure 2 | `tipologia_errores.png` |
| Whisper pipeline | `main.py` |
| Dependencies | `requirements.txt` |

## Author Contributions

**Equipo ZINT** — Audio collection (Yucatán), human transcription, Colab ASR runs (Whisper + Gemini), benchmark design, `analisis_comparativo.py`, analysis, and manuscript.

## References

- Blodgett, S. L., et al. (2020). Language (Technology) is Power. *ACL*. https://aclanthology.org/2020.acl-main.647/
- Global South AI Safety Hackathon. (2026). https://globalsouthhackathon.com/
- Groq. (2026). GroqCloud Models. https://console.groq.com/docs/models
- Ji, Z., et al. (2023). Survey of Hallucination in NLG. *ACM Computing Surveys*. https://doi.org/10.1145/3571730
- Koenecke, A., et al. (2020). Racial Disparities in ASR. *PNAS*. https://doi.org/10.1073/pnas.1915768117
- Radford, A., et al. (2023). Robust Speech Recognition via Large-Scale Weak Supervision. *ICML*. https://proceedings.mlr.press/v202/radford23a.html

## Appendix A — Gemini Linguist Prompt (Colab)

> *"Eres un experto lingüista especializado en el español de la Península de Yucatán y la lengua maya. Escucha este audio cuidadosamente. Transcribe de la forma más exacta posible lo que dice la persona. Si utiliza palabras en maya mezcladas con español, escríbelas correctamente. Devuelve ÚNICAMENTE la transcripción, sin saludos ni explicaciones."*

## Appendix B — Output Schema (`resultados_finales_benchmark.csv`)

`Archivo`, `Texto_Oficial`, `Transcripcion_Whisper`, `Transcripcion_Gemini`, `WER_Whisper`, `WER_Gemini`, `Reduccion_Error`, `Sustituciones_Whisper`, `Omisiones_Whisper`, `Inserciones_Whisper`, `Sustituciones_Gemini`, `Omisiones_Gemini`, `Inserciones_Gemini`

---

**LLM Usage Statement:**

LLMs were used to develop project code (`analisis_comparativo.py`, evaluation pipeline) and draft this manuscript (Cursor IDE). ASR outputs were generated by Whisper-large-v3 (Groq) and Gemini 2.5 Flash (Google AI Studio) in Colab. All quantitative claims were computed by `analisis_comparativo.py` from `benchmark_whisper_vs_gemini.csv` and verified against `resultados_finales_benchmark.csv`. The authors take responsibility for interpretations, hypothesis assessments, and reported limitations.
