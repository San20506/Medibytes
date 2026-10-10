"""Stage a fixed sample of the Hugging Face dataset `united-we-care/United-Syn-Med`.

    python -m clinical_eval.syn_med fetch --n 40          # development sample: audio + sample.json
    python -m clinical_eval.syn_med fetch --n 40 --heldout   # a disjoint sample, annotated before any run

The dataset is synthetic speech: one short carrier sentence per clip with a drug
(brand or generic) in it, and the sentence as text. The audio ships only inside
multi-gigabyte tarballs, so this streams the first `--pool` members of the test
tarball and stops, instead of downloading 4 GB for 40 clips.

The sample is a seeded draw from that pool, so it is the same on every machine.
Only `.mp3` members are written, under a name that contains no path, because the
archive is downloaded data and its member names are not trusted.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import tarfile
from pathlib import Path
from typing import Sequence
from base_pipeline.paths import data_root

REPO = "united-we-care/United-Syn-Med"
TARBALL = "data/audio/test.tar.gz"
DATA_ROOT_DEFAULT = data_root()
CORPUS_ID = "united-syn-med-40"
HELDOUT_CORPUS_ID = "united-syn-med-40b"   # disjoint pool: tar members after the first 600
SEED = 1729
_SAFE = re.compile(r"^[A-Za-z0-9._-]+\.mp3$")


def corpus_root(data_root: Path = DATA_ROOT_DEFAULT, corpus_id: str = CORPUS_ID) -> Path:
    return data_root / "corpus" / corpus_id


def _transcriptions() -> dict[str, str]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO, "data/test.csv", repo_type="dataset")
    csv.field_size_limit(10**8)
    with open(path, encoding="utf-8", newline="") as f:
        return {row["file_name"]: row["transcription"] for row in csv.DictReader(f)}


def fetch(n: int, pool: int, data_root: Path, skip: int = 0, corpus_id: str = CORPUS_ID) -> Path:
    import requests
    from huggingface_hub import get_token, hf_hub_url

    texts = _transcriptions()
    url = hf_hub_url(REPO, TARBALL, repo_type="dataset")
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    out = corpus_root(data_root, corpus_id)
    audio = out / "audio"
    audio.mkdir(parents=True, exist_ok=True)

    # Read the pool in memory (a few hundred 40-70 KB clips), draw, then write only the draw.
    members: dict[str, bytes] = {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with tarfile.open(fileobj=resp.raw, mode="r|gz") as tar:
            seen = 0
            for member in tar:
                name = Path(member.name).name
                if not (member.isfile() and _SAFE.match(name) and name in texts):
                    continue
                seen += 1
                if seen <= skip:
                    continue
                members[name] = tar.extractfile(member).read()
                if len(members) >= pool:
                    break
    rng = random.Random(SEED)
    chosen = sorted(rng.sample(sorted(members), n))
    sample = []
    for i, name in enumerate(chosen, 1):
        clip_id = f"SYN_{i:03d}"
        (audio / f"{clip_id}.mp3").write_bytes(members[name])
        sample.append({"clip_id": clip_id, "source_file": name, "transcription": texts[name]})
    (out / "sample.json").write_text(json.dumps(
        {"repo": REPO, "tarball": TARBALL, "seed": SEED, "pool": len(members), "clips": sample},
        indent=1, ensure_ascii=False), encoding="utf-8")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="clinical_eval.syn_med", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--n", type=int, default=40)
    f.add_argument("--pool", type=int, default=600)
    f.add_argument("--skip", type=int, default=0, help="tar members to pass over first")
    f.add_argument("--corpus-id", help="stage under this id (e.g. a larger precision-check draw)")
    f.add_argument("--heldout", action="store_true",
                   help=f"skip the first 600 members and stage {HELDOUT_CORPUS_ID}")
    f.add_argument("--data-root", default=str(DATA_ROOT_DEFAULT))
    a = ap.parse_args(argv)
    skip, cid = (600, HELDOUT_CORPUS_ID) if a.heldout else (a.skip, a.corpus_id or CORPUS_ID)
    print(fetch(a.n, a.pool, Path(a.data_root), skip, cid))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
