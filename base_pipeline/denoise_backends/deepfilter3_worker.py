"""Method-local DeepFilterNet3 worker executed inside the pinned method environment.

The shared evaluation environment cannot host the pinned stack because
``deepfilternet==0.5.6`` requires ``numpy<2`` and ``deepfilterlib==0.5.6`` ships no
CPython 3.12 wheel.  This worker therefore runs under the method environment built
by ``build_deepfilternet3.sh`` and is invoked deterministically by
``base_pipeline/denoise_backends/deepfilternet3.py``.

It performs exactly one pinned ``init_df()``/``enhance()`` call for one sample on
CPU and reports the observed runtime, geometry, and device.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path

_REQUIRED_PACKAGES = (
    "deepfilternet",
    "deepfilterlib",
    "numpy",
    "torch",
    "torchaudio",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deepfilter3_worker")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input", required=True, help="float32 mono .npy at the native rate")
    parser.add_argument("--output", required=True, help="float32 mono .npy to write")
    parser.add_argument("--report", required=True, help="JSON runtime report to write")
    parser.add_argument("--threads", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    # CPU is pinned before torch initialises so no accelerator can be selected.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", str(arguments.threads))

    import numpy as np
    import torch

    from df.enhance import enhance, init_df

    torch.set_num_threads(arguments.threads)
    torch.set_grad_enabled(False)
    torch.manual_seed(0)

    model_dir = Path(arguments.model_dir)
    if not model_dir.is_dir():
        raise SystemExit(f"model directory is missing: {model_dir}")

    audio = np.load(arguments.input, allow_pickle=False)
    if audio.dtype != np.float32 or audio.ndim != 1 or audio.size == 0:
        raise SystemExit("worker input must be non-empty mono float32")
    if not np.isfinite(audio).all():
        raise SystemExit("worker input must be finite")

    # The shipped DeepFilterNet3 config.ini predates some configuration keys in
    # the pinned 0.5.6 code, so upstream's documented default is used and
    # reported: missing values take the code's own defaults, nothing else.
    load_started = time.perf_counter()
    model, df_state, suffix = init_df(
        model_base_dir=str(model_dir),
        post_filter=False,
        log_level="ERROR",
        log_file=None,
        config_allow_defaults=True,
        epoch="best",
    )
    model_load_s = time.perf_counter() - load_started

    fft_size = int(df_state.fft_size())
    hop_size = int(df_state.hop_size())
    samples = torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0)

    enhance_started = time.perf_counter()
    # pad=True applies the method-documented compensation: n_fft trailing zeros
    # are added and the resulting n_fft - hop algorithmic delay is removed.
    enhanced = enhance(model, df_state, samples, pad=True)
    enhance_s = time.perf_counter() - enhance_started

    enhanced_audio = enhanced.detach().squeeze(0).cpu().numpy().astype(np.float32, copy=False)
    if enhanced_audio.size != audio.size:
        raise SystemExit(
            f"upstream enhance changed the sample count: {enhanced_audio.size} != {audio.size}"
        )
    if not np.isfinite(enhanced_audio).all():
        raise SystemExit("worker output must be finite")

    np.save(arguments.output, enhanced_audio, allow_pickle=False)
    report = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "package_versions": {
            name: importlib.metadata.version(name) for name in sorted(_REQUIRED_PACKAGES)
        },
        "device": "cpu",
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_threads": int(torch.get_num_threads()),
        "model_suffix": suffix,
        "config_allow_defaults": True,
        "post_filter": False,
        "nb_df": int(getattr(model, "nb_df", 0)),
        "native_sample_rate": 48_000,
        "fft_size": fft_size,
        "hop_size": hop_size,
        "algorithm_delay_samples": fft_size - hop_size,
        "padding_samples": fft_size,
        "input_samples": int(audio.size),
        "output_samples": int(enhanced_audio.size),
        "model_load_s": model_load_s,
        "enhance_s": enhance_s,
        "worker_wall_s": model_load_s + enhance_s,
    }
    Path(arguments.report).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
