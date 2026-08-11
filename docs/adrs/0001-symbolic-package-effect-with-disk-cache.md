# ADR 0001: Symbolic Derivation of the Package-Effect Correction, Cached to Disk

**Status:** Implemented
**Scope:** `src/absorpig/data.py` (`get_package_effect`, `_get_qexp_integralfraction_symb`,
`_calc_qth_symbn`)

---

## 1. Context

Measured whole-cell absorption spectra are distorted by the "package effect": pigments
packed inside a cell/organelle absorb less efficiently per molecule than the same
pigments free in solution, because self-shading flattens the absorption curve. Correcting
for it requires integrating a detector-response-like function (`Qth`, `Qexp`) over a
log-normal cell-size distribution — an integral that has no simple closed form and is
expensive to evaluate numerically at the precision needed.

## 2. Decision

The integral is derived once, symbolically, with `sympy` (`_get_qexp_integralfraction_symb`,
`_calc_qth_symbn`), then evaluated numerically per call by substituting the fitted
log-normal parameters. The symbolic result of the (slow) indefinite integration is
memoized to disk as a pickle (`files.qexp_integralfraction`,
`data/Qexp_itegralfraction_subs.pickle`) on first use and loaded directly on every
subsequent call.

## 3. Rationale

Deriving the expression symbolically once and reusing it numerically many times (once per
fitted spectrum) is both more accurate and dramatically faster than re-deriving or
re-integrating numerically on every call — `sympy.integrate` on this expression is slow
enough that doing it inline per spectrum would make the CLI/notebook workflow
impractical. Caching the *symbolic result* (not just numeric outputs) to a pickle keeps
the win even across process restarts, at the cost of the pickle being effectively an
opaque, sympy-version-coupled build artifact.

## 4. Consequences

- The `.pickle` in `data/` is a derived artifact, not a hand-authored one — if the
  symbolic derivation logic in `_get_qexp_integralfraction_symb` changes, the stale
  pickle must be deleted (there is no cache-invalidation check on the function body).
- Upgrading `sympy` across a major version is the main risk to this cache; if unpickling
  ever fails or silently returns something wrong, deleting the file and letting it
  regenerate is the fix, not debugging the pickle itself.
- This same "derive symbolically once, cache the closed form" idea is *not* applied to
  `pigment_shift_linear_*` — those are already simple enough that eager numeric
  computation is fine. Don't reach for this pattern unless a derivation is genuinely
  both expensive and reused.
