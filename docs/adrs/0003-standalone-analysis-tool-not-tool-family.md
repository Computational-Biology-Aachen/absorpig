# ADR 0003: A Standalone Data-Analysis Tool, Not a Member of the Mxl Tool Family

**Status:** Implemented
**Scope:** whole package — most visibly `pyproject.toml`'s lax Ruff config and the
absence of `docs/adrs`-style infra (no `mkdocs.yml`, `.pre-commit-config.yaml`,
`CONTRIBUTING.md`, or changelog tooling) prior to this ADR set

---

## 1. Context

`absorpig` extracts pigment composition from a measured whole-cell absorption spectrum
(deconvolution against reference pigment spectra, corrected for package effect and
in-vivo shift — see [ADR 0001](0001-symbolic-package-effect-with-disk-cache.md),
[ADR 0002](0002-pluggable-pigment-shift-strategies.md)). It has no dependency on `mxlpy`
and is not an ODE/kinetic-modelling tool at all — it is signal processing over measured
spectra, one step upstream of modelling. Its output (pigment concentrations, photosystem
composition) feeds parameterization of reference models elsewhere (e.g. the
`pfennig2024_synechocystis` model's `data/pfennig2024/` inputs in `mxlmodels`), but that
link is a data hand-off, not a code dependency in either direction.

## 2. Decision

Do not fold `absorpig` into the `mxlpy`/`mxlbricks`/`mxlmodels` dependency graph, and do
not hold it to the same tooling posture (`Ruff ALL` + Pyright strict + Bandit +
`mkdocs`-published docs) as the packages other scientists build directly on top of.
`pyproject.toml`'s Ruff config ignores docstring rules (`"D"`) wholesale and there is no
`pyright`/`bandit` configuration at all.

## 3. Rationale

`absorpig` is a self-contained analysis script package with a CLI entry point, authored
and used by a small group (Tobias Pfennig, Marvin van Aalst, David Fuente Herraiz) for a
specific measurement pipeline — not shared infrastructure that other packages import and
build correctness guarantees on top of. The rigor trade-off in
[mxlpy ADR 0011](https://github.com/Computational-Biology-Aachen/MxlPy/blob/main/docs/adrs/0011-strict-tooling-for-downstream-scientists.md)
(pay strict-tooling cost centrally because many downstream users would otherwise each
pay it repeatedly) does not apply here: there are no downstream packages importing
`absorpig`'s internals, so the calculus favors lighter tooling and faster iteration over
library-grade rigor.

## 4. Consequences

- Don't propose adopting `mxlbricks`/`mxlmodels`-style `Ruff ALL` + Pyright strict +
  Bandit wholesale here without a concrete reason (e.g. absorpig gaining downstream
  consumers) — it would slow down a workflow that doesn't currently need the guarantee.
- If `absorpig` ever becomes a real dependency of another package in this family (rather
  than a data-hand-off predecessor), this decision should be revisited alongside adding
  the missing `CONTRIBUTING.md`/CI-test/changelog infra the other packages have.
- Despite the standalone status, keep result *correctness* (the fitting math itself) held
  to a high bar regardless of tooling posture — the informal tooling is about process
  overhead, not about tolerating wrong numbers.
