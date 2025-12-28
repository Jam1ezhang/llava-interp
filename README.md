# LLaVA Interpretability

This repository contains utilities and experiment scripts for interpreting video-capable vision-language models (e.g. Qwen2-VL). The focus is on counterfactual causal tracing, token-level patching, and representation probes over video QA datasets.

## Table of Contents
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Repository Structure](#repository-structure)
- [Usage](#usage)
  - [Run Video QA Inference](#run-video-qa-inference)
  - [Counterfactual Causal Tracing](#counterfactual-causal-tracing)
  - [Have-1/Have-2 Probes](#have-1have-2-probes)
  - [Mean Vector Utilities](#mean-vector-utilities)
  - [Token Indexing Smoke Test](#token-indexing-smoke-test)
- [Citation](#citation)
- [Contact](#contact)

## Installation

### Prerequisites
- Python 3.8+
- `pip`

### Steps
1. **Clone the repository:**
   ```bash
   git clone https://github.com/clemneo/llava-interp
   cd llava-interp
   ```

2. **Install required Python packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

The scripts in `scripts/` expect a video QA annotations JSON file and a directory of videos. A single annotation entry should look like:

```json
{
  "video_name": "video_0001.mp4",
  "question": "What is the person doing?",
  "candidates": ["Walking", "Running", "Cooking"],
  "answer": "Running",
  "answer_number": 1,
  "question_id": "0001",
  "task_name": "action",
  "start": 0.0,
  "end": 2.5,
  "frames_with_answer_object": [3, 4, 5]
}
```

Only `video_name` and `question` are strictly required, but including `candidates`, `answer`, and timing metadata enables the causal tracing and probing utilities.

## Repository Structure

- `src/`
  - `HookedLVLM.py`: Wrapper around the base VLM for hooking activations and patching.
  - `activation_patching.py`: Utilities for caching and patching hidden states.
  - `counterfactuals.py`: Counterfactual video transformations (swap, reverse, motion-only, etc.).
  - `token_indexing.py`: Helpers to locate visual token spans and frame spans.
  - `VideoDatasets.py`: `VideoQADataset` loader for the annotation JSON files.
  - `video_utils.py`: Frame loading and prompt formatting.
- `scripts/`
  - `video_inference.py`: Run baseline video QA inference.
  - `counterfactual_causal_tracing.py`: Perform counterfactual causal tracing experiments.
  - `train_have_probe.py`: Train Have-1 / Have-2 probes from tracing outputs.
  - `calculate_mean_vector.py`: Compute mean vectors from cached activation tensors.
  - `test_token_indexing_smoke.py`: Small sanity check for token indexing.

## Usage

### Run Video QA Inference

```bash
python scripts/video_inference.py \
  --annotations data/clean_questions.json \
  --video_root /path/to/videos \
  --output outputs/inference.json \
  --model_id Qwen/Qwen2-VL-7B-Instruct \
  --device cuda:0 \
  --num_frames 8
```

### Counterfactual Causal Tracing

```bash
python scripts/counterfactual_causal_tracing.py \
  --annotations data/clean_questions.json \
  --video_root /path/to/videos \
  --output outputs/tracing.json \
  --representations_output outputs/representations.json \
  --model_id Qwen/Qwen2-VL-7B-Instruct \
  --token_slice visual_frames:0:2 \
  --swap_spans 0:2,2:4 \
  --dump_representations
```

Notes:
- Use `--token_slice visual` to patch all visual tokens, `text` for text tokens, or `visual_frames:start:end` to target specific frame spans.
- Add `--include_motion_only` to include motion-only counterfactuals.

### Have-1/Have-2 Probes

```bash
python scripts/train_have_probe.py \
  --tracing_results outputs/tracing.json \
  --representations outputs/representations.json \
  --output outputs/have1_metrics.json \
  --probe_type have1 \
  --representation layers_mean \
  --layer_idx 12
```

For Have-2 probes, supply annotations and a label key:

```bash
python scripts/train_have_probe.py \
  --tracing_results outputs/tracing.json \
  --representations outputs/representations.json \
  --annotations data/clean_questions.json \
  --label_key question_type \
  --output outputs/have2_metrics.json \
  --probe_type have2
```

### Mean Vector Utilities

If you cache per-sample activation tensors (e.g. `*.pt` files), you can compute a mean vector:

```bash
python scripts/calculate_mean_vector.py /path/to/activation_cache --device cuda
```

This writes `mean_vector.pt` into the provided directory.

### Token Indexing Smoke Test

```bash
python scripts/test_token_indexing_smoke.py
```

## Citation
To cite our work, please use the following BibTeX entry:
```
@misc{neo2024interpretingvisualinformationprocessing,
      title={Towards Interpreting Visual Information Processing in Vision-Language Models}, 
      author={Clement Neo and Luke Ong and Philip Torr and Mor Geva and David Krueger and Fazl Barez},
      year={2024},
      eprint={2410.07149},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.07149}, 
}
```

## Contact
For questions or issues, please open a GitHub issue on this repository.
