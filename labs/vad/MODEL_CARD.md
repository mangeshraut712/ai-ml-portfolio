# Model card — VAD pipeline (`labs/vad`)

Last updated: 2026-09-17. Numbers below are the ones reproduced by
`make verify-vad-pass` (the CI gate), not hand-entered.

## What it does

Frame-level voice activity detection on 16 kHz mono PCM. Input: a WAV file.
Output: a boolean speech flag per 30 ms frame plus merged speech segments.
No external inference API is called at any point.

## Architecture

```text
raw audio → spectral-gate denoiser → 16-bit PCM 30 ms frames → WebRTC GMM VAD (aggressiveness 2) → hangover / gap fill → flags
```

| Stage | File | Notes |
| --- | --- | --- |
| Denoiser | `denoiser.py` | STFT magnitude gating; noise floor estimated from the quietest frames |
| Classifier | `vad_engine.py` | `webrtcvad` Gaussian-mixture VAD, aggressiveness **2** (the FULL_PASS gate; 0–3 selectable) |
| Post-processing | `postprocess.py` | fill non-speech gaps shorter than 2 frames, pad speech regions (hangover) |
| Orchestration | `pipeline.py` | `VADPipeline(aggressiveness=2)` |

An experimental neural / energy A/B backend exists (`--backend neural|compare`)
for local comparison only; it is **not** part of the gate.

## Data

- **Real audio:** ~50 WAVs, 16 kHz mono, from Hugging Face dataset
  `danielrosehill/Small-STT-Eval-Audio-Dataset`, pinned to revision
  `d395fcce66e8843d8ec6ec1036f009ade9329b23` (`fetch_samples.py`).
- **Exact labels (synthetic split):** `build_labeled_benchmark.py` concatenates
  silence with real speech excerpts so every frame has an objective label.
  20 clips, seed **42**; the second half has additive white Gaussian noise at
  SNR drawn from {5, 10, 15} dB. "Clean" = `snr_db is None`; "noisy" =
  `snr_db` set.
- **Soft labels (real samples):** energy-reference flags (`metrics.energy_reference_flags`)
  over the real WAVs — a secondary, weaker reference.
- Labels live in `sample_data/vad_labeled/ground_truth.json`.

## Evaluation

Gate thresholds are fixed in `challenge_pass.py` (`GATES`). All rows are frame-level F1.

| Split | Measured | Gate | Status |
| --- | ---: | ---: | --- |
| Exact, clean — mean F1 | **0.9569** | ≥ 0.92 | pass |
| Exact, clean — min per-file F1 | see gate report | ≥ 0.90 | pass |
| Exact, noisy (5–15 dB SNR) — mean F1 | **0.7768** | ≥ 0.75 | pass, narrow margin |
| Soft (energy reference) — mean F1 | **0.8135** | ≥ 0.78 | pass |
| Steady-state p95 latency per 30 ms frame | **~19 ms** | < 30 ms (real time) | pass |
| `sample_001.wav` frames processed | **258/258** | complete | pass |

Baseline: the gate thresholds themselves are the acceptance bar the pipeline
was built against (the interview task's "FULL_PASS"). A naive always-speech
predictor scores F1 = 2p/(1+p) for speech proportion p; on these clips it is
well below every gate. The soft split is reported but is a weaker reference
than the exact split by construction.

## Known limitations and failure modes

1. **Noise is the failure mode.** Mean F1 drops from 0.9569 (clean) to
   0.7768 on the 5–15 dB white-noise clips, with only 0.027 headroom over
   the gate. The spectral gate cannot separate speech from noise bins near
   the estimated floor, and the GMM misses low-SNR speech onsets. Expect
   worse than 0.7768 on real non-stationary noise (traffic, babble), which
   the synthetic split does not contain.
2. **Synthetic exact labels.** Exact frame labels come from silence + speech
   concatenation, not from human annotation of natural recordings.
3. **Single language / domain.** The real audio is a small STT evaluation
   set; no claim is made for other languages, far-field microphones, or
   music-heavy content.
4. **Fixed aggressiveness.** Level 2 is tuned to pass the gate; level 3
   trades recall for precision and is not evaluated here.
5. **Not a learned model.** WebRTC VAD is a pre-trained GMM shipped with
   `webrtcvad`; there is no training loop in this repository.

## Reproduce

```bash
make install
make verify-vad-pass      # downloads pinned samples if missing, runs the gate
make run-vad-ui           # Streamlit timeline for a single file
```

## Intended use

Interview / portfolio demonstration of a low-latency, dependency-light VAD
with an explicit acceptance gate. Not intended for production deployment
without re-evaluation on the target acoustic conditions.
