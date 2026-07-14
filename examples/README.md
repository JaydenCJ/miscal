# miscal examples

Two committed, fully deterministic sample logs from a simulated
support-ticket intent classifier:

- **`sample_run.jsonl`** — 200 decisions from an *overconfident* model
  (stated confidence far above observed accuracy), with confidences
  verbalized the way real LLMs log them: floats, `"92%"`, `"9/10"`, and
  anchor words like `"very likely"`.
- **`sample_run_v2.jsonl`** — 200 decisions after a hypothetical prompt fix;
  much better calibrated. Useful for `miscal compare` and for seeing the
  `--max-ece` gate flip from exit 1 to exit 0.

Try them from the repository root:

```bash
miscal report examples/sample_run.jsonl
miscal diagram examples/sample_run.jsonl -o reliability.svg
miscal compare examples/sample_run.jsonl examples/sample_run_v2.jsonl
miscal fit examples/sample_run.jsonl
miscal report examples/sample_run.jsonl --max-ece 0.10   # exits 1
```

## Regenerating

`make_sample.py` rebuilds both files byte-for-byte (seeded RNG, no wall
clock):

```bash
python examples/make_sample.py
```

The `honesty` parameter controls how truthful the stated confidences are:
the true probability of being correct is `0.5 + honesty * (conf - 0.5)`, so
`honesty=0.35` produces the overconfident run and `honesty=0.85` the fixed
one. If you change the generator, re-capture the outputs quoted in the
READMEs and `scripts/smoke.sh` — `tests/test_readme_example.py` pins them.
