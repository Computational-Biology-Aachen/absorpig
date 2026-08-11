# absorpig: Architecture Context

This is the entry point for understanding *why* `absorpig` is shaped the way it is —
written down ahead of a maintainer handoff, alongside the equivalent `docs/adrs/`
directories in the sibling `mxlpy`, `mxlbricks`, `mxlmodels`, `parameteriser`, and
`schemegen` repos.

## What This Tool Does

`absorpig` extracts pigment composition from a measured whole-cell absorption spectrum:
it fits reference pigment spectra to the measurement after correcting for two physical
distortions — the optical package effect and in-vivo spectral shift.

→ [ADR 0001 — Symbolic derivation of the package-effect correction, cached to disk](0001-symbolic-package-effect-with-disk-cache.md)
→ [ADR 0002 — Three interchangeable pigment-shift strategies, not one "correct" model](0002-pluggable-pigment-shift-strategies.md)

## Where It Sits in the Tool Family

→ [ADR 0003 — A standalone data-analysis tool, not a member of the Mxl tool family](0003-standalone-analysis-tool-not-tool-family.md)

`absorpig` has no code dependency on `mxlpy` or any other package in this family, and is
held to a deliberately lighter tooling standard than `mxlbricks`/`mxlmodels`. Its outputs
(pigment concentrations, photosystem composition) are consumed as *data* by reference
models elsewhere — e.g. `mxlmodels`' `pfennig2024_synechocystis` model ships the
`data/pfennig2024/` inputs this kind of analysis produces — but that is a one-time,
build-time data hand-off, not a runtime or code dependency in either direction.

## Threads That Cross Multiple ADRs

- **Expose scientific judgment calls as options, don't hide them behind a single
  default.** ADR 0002's three shift strategies exist because no single model of in-vivo
  spectral shift is correct across organisms — the package makes the choice explicit
  rather than silently picking one.
- **Cache expensive derivations, not just expensive numeric results.** ADR 0001 memoizes
  a symbolic integration result, not just a final number, because the same closed form is
  reused across many different numeric inputs.

## See Also

- [`mxlmodels`' `docs/adrs/CONTEXT.md`](https://github.com/Computational-Biology-Aachen/mxl-models/blob/main/docs/adrs/CONTEXT.md)
  for the reference-model package that consumes `absorpig`-derived pigment/photosystem
  data as input.
- [`mxlpy`'s `docs/adrs/0011-strict-tooling-for-downstream-scientists.md`](https://github.com/Computational-Biology-Aachen/MxlPy/blob/main/docs/adrs/0011-strict-tooling-for-downstream-scientists.md)
  for the tooling posture `absorpig` deliberately does *not* adopt, and why that's the
  right call here.
