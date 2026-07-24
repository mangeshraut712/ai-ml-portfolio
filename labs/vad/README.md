# High-Performance Voice Activity Detector (VAD) from Scratch

A clean, modular VAD implementation optimizing low-latency speech detection over
raw noisy channels. Designed for **zero external API usage** during intense
runtime constraints — the same constraint set used in Sarvam AI's ML Engineer
on-site / proctored coding challenge.

> Interview context: candidates were given ~50 audio files and ~2.5 hours to
> build a VAD from scratch. Architecture choice was open; this example follows
> the Denoiser + WebRTC (GMM) approach that maximized detection accuracy.

## Architectural Blueprint

This pipeline breaks speech segregation into a deterministic two-stage engine:

1. **Spectral Gating Denoiser** — estimates a per-frequency noise floor from
   the quietest frames and softly attenuates bins near that floor using STFT
   magnitude gating (keeps speech energy intact for the downstream classifier).
2. **GMM Classifier (WebRTC-VAD)** — takes processed 16-bit PCM chunks across
   fixed 30 ms windows, calculating sub-band energy profiles to evaluate speech
   likelihood via Gaussian Mixture Models.

```text
raw audio → STFT spectral gate → 16-bit PCM frames → WebRTC GMM → speech flags
```

## Project Layout

```text
labs/vad/
├── denoiser.py                 # Spectral gating
├── vad_engine.py               # WebRTC GMM VAD
├── postprocess.py              # Hangover / gap-fill
├── metrics.py                  # Precision / recall / F1
├── pipeline.py                 # End-to-end orchestration
├── main.py                     # CLI
├── evaluate_dataset.py         # Unlabeled batch metrics
├── build_labeled_benchmark.py  # Exact + soft ground truth
├── evaluate_against_labels.py  # Labeled F1 scoring
├── challenge_pass.py           # FULL_PASS acceptance gate
├── fetch_samples.py
├── streamlit_app.py
├── INTERVIEW_DEFENSE.md
├── CHALLENGE_CHECK.md
└── README.md

sample_data/sample_audio/           # 50× 16 kHz WAVs
sample_data/vad_labeled/            # ground_truth.json + synthetic/
notebooks/vad_evaluation.ipynb
```

## Quick Start

```bash
# From repo root
make install

# Append speech deps if you have not re-run install after pulling this example
.venv/bin/pip install webrtcvad librosa soundfile huggingface_hub

# Download ~50 evaluation WAVs (16 kHz mono PCM)
.venv/bin/python labs/vad/fetch_samples.py

# Run on one file
make verify-vad-challenge

# Batch metrics across all 50 WAVs
make verify-vad-batch

# Interactive Streamlit UI
make run-vad-ui
# or:
.venv/bin/python labs/vad/main.py \
  sample_data/sample_audio/sample_001.wav
```

### CLI options

```bash
.venv/bin/python labs/vad/main.py path/to/audio.wav \
  --aggressiveness 3 \
  --frame-ms 30 \
  --json
```

| Flag | Meaning |
|---|---|
| `--aggressiveness 0..3` | WebRTC strictness (default `2`, 3 = most restrictive) |
| `--frame-ms 10\|20\|30` | Frame window size required by WebRTC |
| `--no-denoise` | Skip spectral gating (A/B accuracy tests) |
| `--json` | Machine-readable segment dump |

## Evaluation Criteria (mirrors the interview rubric)

| Metric | What to defend in review |
|---|---|
| **Accuracy** | Speech vs silence frame quality on the provided set |
| **Code quality** | Modular stages, clear contracts, no API calls |
| **Future improvements** | Roadmap below — what you would ship next under less time pressure |

Open `notebooks/vad_evaluation.ipynb` for the full evaluation walkthrough
(batch metrics, timelines, denoise A/B, aggressiveness sweep).

Interview talking points (latency budget, architecture trade-offs, Round-2
topics): [`INTERVIEW_DEFENSE.md`](./INTERVIEW_DEFENSE.md).

## Challenge Check (FULL PASS)

```bash
make build-vad-labels   # exact synthetic GT + soft labels for 50 WAVs
make verify-vad-pass    # single acceptance gate → FULL_PASS / FAIL
```

| Dimension | Evidence |
|---|---|
| Accuracy | Exact clean F1 ≈ **0.96**; noisy F1 ≥ 0.75; soft F1 ≥ 0.78 |
| Completeness | Every expected frame classified (e.g. **258/258** on sample_001) |
| Latency | Steady p95 ≈ **19 ms**, RTF ≈ **0.002** |
| Quality | Modular stages, hangover post-process, 9 pytest cases |
| Defendability | [`INTERVIEW_DEFENSE.md`](./INTERVIEW_DEFENSE.md) + notebook + Streamlit |

Details: [`CHALLENGE_CHECK.md`](./CHALLENGE_CHECK.md).

The tuned baseline uses `aggressiveness=2` with hangover smoothing.



## Neural VAD A/B (experimental)

An energy / optional-ONNX stub (`neural_vad.py`) implements the same interface as
WebRTC for local A/B comparison. **It is not part of the FULL_PASS gate.**

| Backend | Status |
|---|---|
| `webrtc` (default) | **FULL_PASS** — `challenge_pass.py` / `make verify-vad-pass` |
| `neural` | Experimental energy gate (+ ONNX if model + `onnxruntime` present) |
| `compare` | Side-by-side speech % + frame agreement |

```bash
# WebRTC (acceptance path)
.venv/bin/python labs/vad/main.py sample_data/sample_audio/sample_001.wav

# Neural stub
.venv/bin/python labs/vad/main.py sample_data/sample_audio/sample_001.wav --backend neural

# A/B
.venv/bin/python labs/vad/main.py sample_data/sample_audio/sample_001.wav --backend compare
.venv/bin/python labs/vad/ab_compare.py sample_data/sample_audio/sample_001.wav
```

Do not change `challenge_pass.py` aggressiveness away from WebRTC **2** for gated runs.

## Scalability and Future Roadmap

To solve production constraints that do not fit a 2.5-hour hackathon timeline:

* **Deep Neural Networks** — transition the GMM model into a lightweight
  Silero-VAD or an ONNX-runtime optimized CRNN to detect non-stationary noise
  (babble, background music).
* **Dynamic Noise Profiling** — swap fixed spectral thresholds for adaptive
  noise estimation (e.g. Minimum Statistics) to track drifting noise floors.
* **Hardware Invariance** — move frame splitting into C++ / pybind bindings for
  thread-safe bare-metal execution with zero Python overhead.
* **Latency budgets** — pair this VAD with streaming STT and measure p95
  end-to-end latency (interview discussion target: &lt; 800 ms).

## Interview Defense Notes

See **[`INTERVIEW_DEFENSE.md`](./INTERVIEW_DEFENSE.md)** for the full script.
Short checklist:

* Why denoise *before* WebRTC (GMM is sensitive to stationary noise floors).
* Why aggressiveness **2** beat **3** on this set (81% → ~95% on sample_001).
* Steady-state p95 ~20 ms / RTF ~0.002 → leaves budget for STT p95 &lt; 800 ms.
* How you would stress-test STT behind this VAD (concurrency, WER under noise).
* Latent-space decomposition (content vs speaker/style) for Bulbul-style TTS.

## License

Same as the parent cookbook repository.
