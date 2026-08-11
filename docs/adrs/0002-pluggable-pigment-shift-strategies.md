# ADR 0002: Three Interchangeable Pigment-Shift Strategies, Not One "Correct" Model

**Status:** Implemented
**Scope:** `src/absorpig/data.py` (`pigment_shift`, `pigment_shift_linear_uni`,
`pigment_shift_linear_individual`, `routine(shift_method=...)`)

---

## 1. Context

In-vivo pigment absorption spectra are red- or blue-shifted relative to the reference
(in-vitro/solvent) spectra shipped in `data/pigment_spectra_*.csv`, because the protein
environment inside the cell shifts absorption peaks. `routine()` exposes three ways to
correct for this before fitting: `"total"` (one constant shift per pigment, from
`pigment_shifts.csv`), `"linear uni"` (a single linearly-varying shift applied uniformly
across all pigments, the default), and `"linear individual"` (a per-pigment linear shift,
from separately fitted `shift_offset`/`shift_end` values).

## 2. Decision

Keep all three strategies as a `shift_method` string switch on `routine()`, rather than
picking one as "the" model and deleting the others.

## 3. Rationale

There is no universally-correct shift model across organisms/pigment sets — how much and
how a spectrum shifts in vivo is an empirical property of the specific measurement setup,
and the "right" amount of flexibility (one global shift vs. per-pigment shifts) trades off
against overfitting when few reference spectra are available. Exposing the choice lets a
caller pick the strategy that matches how much shift data they actually trust for their
organism, rather than the package silently assuming one model fits all cases. `"linear
uni"` is the default because it is the best generally-applicable compromise observed so
far — not because the other two are deprecated.

## 4. Consequences

- Adding a fourth shift strategy is expected to happen again if a new use case doesn't
  fit these three — extend the `shift_method` switch in `routine()`, don't replace it.
- Callers must still supply the right shift data (`shift_values` /
  `shift_offset`+`shift_end`) for whichever method they pick; the function does not infer
  which method is appropriate for a given dataset.
