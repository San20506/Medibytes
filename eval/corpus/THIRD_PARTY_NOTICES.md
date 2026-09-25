# Third-Party Notices

## Google FLEURS

The `pilot-15-v1` corpus includes 12 English (`en-US-proxy`), Hindi, and Tamil
recordings from The FLEURS dataset by Google Research and contributors. FLEURS
is licensed under Creative Commons Attribution 4.0 International (`CC-BY-4.0`).
The immutable dataset revision is `168de341b3db6859a9bac1c50a2ef5e3b47647e0`. See the FLEURS dataset
card and license at <https://huggingface.co/datasets/google/fleurs>.

## MUCS 2021 Hindi-English Test

The `pilot-15-v1` corpus includes three code-mixed segments from the official
SLR104 Hindi-English test archive. MUCS is licensed under Creative Commons
Attribution-ShareAlike 4.0 International (`CC-BY-SA-4.0`). Source:
<https://openslr.elda.org/resources/104/Hindi-English_test.tar.gz>; dataset description: <https://www.openslr.org/104/>.

## Build Tools

- NumPy 2.2.6, BSD-3-Clause.
- PyArrow 21.0.0, Apache-2.0.
- huggingface-hub 0.36.0, Apache-2.0.
- imageio-ffmpeg 0.6.0, BSD-2-Clause, and its bundled FFmpeg executable.
- Bundled FFmpeg SHA-256: `e7e7fb30477f717e6f55f9180a70386c62677ef8a4d4d1a5d948f4098aa3eb99`. FFmpeg licensing and source
  details are available from <https://ffmpeg.org/legal.html>.

## Pinned Artifact Digests

- `sources/fleurs/en_us/validation/0000.parquet`: `7c3eeb11a9597bd52cdc1b0d637e85389fe094cfd8763913e7bf4fdf7a853959`
- `sources/fleurs/hi_in/validation/0000.parquet`: `2c6286b884069103be9a8051b12dd9dfbfe1c0c3d4834ad7ecaab1a7689518c5`
- `sources/fleurs/ta_in/validation/0000.parquet`: `9dffc417039ecdef0b080fd3a1386a71a233532f9438fe8901e382ce6a656908`
- `sources/mucs/Hindi-English_test.tar.gz`: `93e358b3bf8233a897fcd353c1f4f98fdda6b8c01b7eed17a70c7dd26e984b37`

The generated corpus retains the source attribution and share-alike terms.
Source transcripts are marked `reference_status=source_reference`; this
non-clinical smoke corpus is not clinical gold and does not establish
production medical performance.
