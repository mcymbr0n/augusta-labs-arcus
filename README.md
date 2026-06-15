# Arcus Prize — "Ode Triunfal"

My investigation of Augusta Labs' Arcus recruitment challenge, *Ode Triunfal*.

The challenge distributes a ~191 MB PyTorch artifact (`ode.pt`) and asks for a
flag over `ssh augustalabs.ai`. This repo is the full record of how I took the
model apart: the reasoning, the experiments, and the dead ends.

## What I found

`ode.pt` is a 50M-parameter byte-level nanoGPT trained on digitized
public-domain Luso-Brazilian literature. It has memorized one flag — a **decoy**,
triggered by the literal string `<|alvaro_de_campos|>`: the one Pessoa heteronym
deliberately missing from the model's special tokens, and the author of the very
poem on screen. The decoy is a corrupted version of the Ode's closing
onomatopoeia that degenerates into a snore — a thematic "you've been fooled". It
fails submission.

After an extensive, varied search I could not extract a genuine flag from the
public artifact. I document why, honestly — including where my own tools proved
inconclusive.

**Full write-up: [`writeup/writeup.md`](writeup/writeup.md)**

## How this repo is organized

- `writeup/` — the write-up (the main deliverable)
- `notes/` — my working notes: a chronological log, server/artifact recon, and
  the running list of hypotheses and why each was discarded
- `scripts/` — the probes (`probe_*.py`); every claim in the write-up is backed
  by one of these

The model file `ode.pt` is not included (it's distributed via the challenge's
own GitHub release).

## Reproducing

Load the model via `scripts/model_lib.py` (safe `weights_only=True` loading),
then run any probe:
```
PYTHONIOENCODING=utf-8 python -X utf8 scripts/<probe>.py
```
