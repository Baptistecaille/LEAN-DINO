# Lean extractor

Not a finished program. It is the shortest path to the four numbers that
`scripts/measure_corpus.py` needs, and those numbers decide the rest of the project.

## Do this first

Run on **one module** (`Mathlib/Algebra/Group/Basic.lean`), emit JSONL, then run
`measure_corpus.py`. Do not extract all of Mathlib until the proof-length distribution
is known.

## The part that needs real work

`sourceOf` is now implemented (`moduleFilePath` + `findDeclarationRanges?` +
`FileMap.ofPosition` + `String.extract`), but it has never been run against a real
Mathlib checkout in this environment (no `lake`/`lean` toolchain here). Treat it as
unverified until step 1 of docs/DESIGN.md's order of work has actually been run:

1. `findDeclarationRanges? declName` -> a `DeclarationRanges` with start/end positions
2. resolve the module name to a filesystem path via `Lean.SearchPath.findModuleWithExt`
   (uses the active Lake search path)
3. read the file and slice it between the two positions, after converting each
   `Position` to a byte offset with `FileMap.ofPosition` (the ranges are line/column,
   not offsets)

This is the single most important function in the extractor. If it returns the
elaborated proof term instead of source text, every downstream length assumption
breaks (see docs/DESIGN.md invariant 1). **Before trusting it at corpus scale**: run it on
`Mathlib/Algebra/Group/Basic.lean`, and eyeball a handful of `proof_source` values
against the actual file to confirm the slice boundaries land where expected (off-by-
one line/column bugs are the likely failure mode, and they produce output that still
looks like valid Lean, not an obvious crash).

## Verify before trusting

- Every metaprogramming name here should be checked against your Mathlib pin.
  `approxDepth`, `isTypeCorrect`, `findDeclarationRanges?`, `getModuleFor?`,
  `Lean.SearchPath.findModuleWithExt` have all moved or changed signature across
  versions.
- `permuteHypotheses` must keep its `isTypeCorrect` check. That check *is* the
  certificate; without it the word "certified" is not earned.
