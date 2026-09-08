/-
  MathEBT extractor -- STARTING POINT, expect to iterate.

  This compiles against the Lean 4 / Mathlib API as of writing but the metaprogramming
  API moves; treat every `import` and every name below as a hypothesis to verify
  against your Mathlib pin. Run it on ONE module first (see docs/DESIGN.md, step 1).

  INVARIANT (docs/DESIGN.md A1): `proof_source` is SOURCE TEXT obtained from
  `findDeclarationRanges?` plus the file contents. It is NOT `ppExpr` of the proof
  term. Pretty-printing an elaborated Mathlib proof yields thousands of tokens of
  kernel noise and would make max_seq_len meaningless.
-/
import Lean

-- Deliberately NOT `import Mathlib`: that forces building the entire library (~17k
-- files) at compile time of this executable. `main` below dynamically imports just
-- the one module it's asked for via `Lean.importModules`, which only needs that
-- module's .olean (and its transitive closure) to already be built -- a much smaller
-- target, and the right one per docs/DESIGN.md ("run on ONE module first").

open Lean Elab Meta

/-- `Environment.getModuleFor?` was removed in favor of the two-step
    `getModuleIdxFor?` + `header.moduleNames` lookup (checked against the actual
    Lean v4.33.0-rc2 source, since the API moved -- see the extractor README).
    Re-added here, in the `Lean.Environment` namespace (NOT nested under `MathEBT`,
    or dot notation `env.getModuleFor?` would not find it), so call sites can keep
    using the familiar `env.getModuleFor? n` syntax. -/
def Lean.Environment.getModuleFor? (env : Environment) (declName : Name) : Option Name :=
  env.getModuleIdxFor? declName |>.bind (env.header.moduleNames[·]?)

namespace MathEBT

structure DeclRecord where
  decl_name      : String
  lean_namespace : String
  module         : String
  file_path      : String
  kind           : String
  type_implicit  : String
  type_explicit  : String
  proof_source   : String
  ast_hash       : String
  type_ast_hash  : String
  premises_raw   : Array String
  used_constants : Array String
  type_alpha     : Option String
  type_perm_hyp  : Option String
  type_subterms  : Array String
  hypotheses     : Array String
  proof_steps    : Array String
  deriving ToJson

/-- Print an expression with a given explicitness. Universes are always erased. -/
def ppWith (e : Expr) (explicit : Bool) : MetaM String := do
  withOptions (fun o =>
      (o.setBool `pp.explicit explicit).setBool `pp.universes false) do
    return toString (← Meta.ppExpr e)

/-- Expr already uses De Bruijn indices, so alpha-normalisation only means erasing
    binder NAMES (which are pretty-printing hints). Hashing is therefore cheap. -/
partial def eraseBinderNames : Expr → Expr
  | .forallE _ d b bi => .forallE `x (eraseBinderNames d) (eraseBinderNames b) bi
  | .lam _ d b bi     => .lam `x (eraseBinderNames d) (eraseBinderNames b) bi
  | .letE _ t v b nd  => .letE `x (eraseBinderNames t) (eraseBinderNames v)
                                  (eraseBinderNames b) nd
  | .app f a          => .app (eraseBinderNames f) (eraseBinderNames a)
  | .mdata _ e        => eraseBinderNames e
  | .proj s i e       => .proj s i (eraseBinderNames e)
  | e                 => e

def astHash (e : Expr) : String := toString (eraseBinderNames e).hash

/-- Alpha-renaming view: re-print with different binder names. Certified by
    construction (binder names carry no semantics). NOTE: correspondingly weak as a
    training signal -- it teaches only that variable names do not matter. -/
def alphaView (e : Expr) : MetaM String := do
  let names := #[`u, `v, `w, `a', `b', `c', `h₁, `h₂, `h₃]
  let rec go (e : Expr) (i : Nat) : Expr :=
    match e with
    | .forallE _ d b bi => .forallE (names[i % names.size]!) d (go b (i+1)) bi
    | .lam _ d b bi     => .lam (names[i % names.size]!) d (go b (i+1)) bi
    | e                 => e
  ppWith (go e 0) false

/-- Hypothesis permutation. ONLY adjacent, mutually independent binders, and the
    result is type-checked. Without the dependency test this produces ill-typed
    terms: in `∀ (a : α) (h : P a), _` the two binders cannot be swapped. -/
def permuteHypotheses (e : Expr) : MetaM (Option String) := do
  forallTelescope e fun xs body => do
    if xs.size < 2 then return none
    for i in [0 : xs.size - 1] do
      let ty ← inferType xs[i+1]!
      -- xs[i+1] independent of xs[i]?
      if !(ty.containsFVar xs[i]!.fvarId!) then
        -- `Array.swap!` was removed in favor of a proof-obligated `swap`; `set!`
        -- (a total, silently-clamping setter) does the same job without a proof.
        let perm := (xs.set! i xs[i+1]!).set! (i+1) xs[i]!
        let candidate ← mkForallFVars perm body
        -- MANDATORY: the certificate is this check, not our reasoning about it.
        if (← isTypeCorrect candidate) && (← isDefEq (← inferType candidate) (← inferType e)) then
          return some (← ppWith candidate false)
    return none

/-- Subterms of the type, used to build genuinely local views. -/
def collectSubterms (e : Expr) (maxN : Nat := 8) : MetaM (Array String) := do
  let mut out : Array String := #[]
  let mut queue := #[e]
  while !queue.isEmpty && out.size < maxN do
    let x := queue.back!
    queue := queue.pop
    match x with
    | .app f a =>
        queue := (queue.push f).push a
        if !a.hasLooseBVars && a.approxDepth > 1 then
          out := out.push (← ppWith a false)
    | .forallE _ d b _ => queue := (queue.push d).push b
    | _ => pure ()
  return out

/-- Split source-level tactic proof into steps.

    measure_corpus.py on Mathlib.Algebra.Group.Basic showed only 3.1% of
    declarations had >=2 steps when splitting on `;` alone -- most Mathlib tactic
    blocks are newline-separated, not `;`-separated, so that was under-segmenting
    almost everything. Splitting on both fixes the common case; still crude for
    nested `by ... <;> ...` or multi-line `simp [...]` args, but matches what the
    corpus actually looks like now instead of a guess. -/
def splitProofSteps (src : String) : Array String :=
  (src.replace "\n" ";").splitOn ";" |>.toArray.map String.trim |>.filter (·.length > 0)

/-- Resolve a module name to the `.lean` file it was compiled from.
    `searchPathRef` (populated by `initSearchPath`) only ever reads `LEAN_PATH`, i.e.
    where `.olean`s live -- `.lean` sources live in a separate tree and are found via
    `LEAN_SRC_PATH` (`Lean.getSrcSearchPath`), which `lake env` sets to each
    dependency's package root. Using `searchPathRef` here silently finds nothing,
    since olean build directories don't contain `.lean` files. -/
def moduleFilePath (modName : Name) : IO (Option System.FilePath) := do
  let sp ← Lean.getSrcSearchPath
  Lean.SearchPath.findModuleWithExt sp "lean" modName

/-- Source text of a declaration, via its declaration range.

    `findDeclarationRanges?` gives a `Position` (line/column), not a byte offset, so
    the position has to be resolved against a `FileMap` of the *same* file contents
    before it can be used with `String.extract`. Getting this step wrong (e.g. reading
    a different file, or an oleandiff'd source) silently produces garbage slices that
    still look plausible -- verify a handful by eye against the module before trusting
    it at corpus scale.

    `FileMap.ofPosition` returns a `String.Pos.Raw` (untyped byte offset), but
    `String.extract` in this Lean version takes `contents.Pos` (a position typed to
    the specific string) -- `String.pos!` does that conversion, panicking only if the
    offset is not a valid codepoint boundary in `contents`, which it always is here
    since it came from `FileMap.ofPosition` on the same `contents`. -/
def sourceOf (declName : Name) : MetaM (Option (String × System.FilePath)) := do
  let some ranges ← findDeclarationRanges? declName | return none
  let some modName := (← getEnv).getModuleFor? declName | return none
  let some path ← moduleFilePath modName | return none
  let contents ← IO.FS.readFile path
  let fileMap := contents.toFileMap
  let startPos := contents.pos! (fileMap.ofPosition ranges.range.pos)
  let endPos := contents.pos! (fileMap.ofPosition ranges.range.endPos)
  return some (contents.extract startPos endPos, path)

def extractDecl (declName : Name) : MetaM (Option DeclRecord) := do
  let env ← getEnv
  let some info := env.find? declName | return none
  unless info.isTheorem || info.isDefinition do return none
  let ty := info.type
  let some (src, path) ← sourceOf declName | return none
  let hyps ← forallTelescope ty fun xs _ =>
    xs.mapM fun x => do ppWith (← inferType x) false
  return some {
    decl_name      := declName.getString!
    lean_namespace := declName.getPrefix.toString
    module         := (env.getModuleFor? declName).map toString |>.getD ""
    file_path      := path.toString
    kind           := if info.isTheorem then "theorem" else "def"
    type_implicit  := ← ppWith ty false
    type_explicit  := ← ppWith ty true
    proof_source   := src
    ast_hash       := astHash ty
    type_ast_hash  := astHash ty
    -- `ConstantInfo.value?` defaults to `allowOpaque := false`, which returns `none`
    -- for EVERY theorem (only `def`s get a value back) -- confirmed by
    -- measure_corpus.py showing 0.9% premise availability on a corpus that is
    -- almost entirely theorems. `allowOpaque := true` returns the theorem's proof
    -- term too, which is what premise selection actually needs.
    premises_raw   := (info.value? (allowOpaque := true)).map (·.getUsedConstants.map toString)
                        |>.getD #[]
    used_constants := ty.getUsedConstants.map toString
    type_alpha     := some (← alphaView ty)
    type_perm_hyp  := ← permuteHypotheses ty
    type_subterms  := ← collectSubterms ty
    hypotheses     := hyps
    proof_steps    := splitProofSteps src
  }

/-- All declarations whose module is exactly `modName`, in the environment's
    declaration order. Filters out auto-generated names (`.match_1`, `.eq_def`,
    `_proof_1`, `.injEq`, ...) and anything not from `modName` itself (imports
    pull in a huge transitive environment; we only want this file's own decls). -/
def declsInModule (modName : Name) : CoreM (Array Name) := do
  let env ← getEnv
  let mut out := #[]
  for (n, _) in env.constants.toList do
    if env.getModuleFor? n == some modName && !n.isInternal then
      out := out.push n
  return out

/-- Extract the given declarations (already known to belong to one module) as
    compressed JSON lines. Pure computation over the already-imported environment;
    the caller decides where the lines go (stdout for the single-module CLI, one
    file per module for `--all`). -/
def extractDeclsLines (names : Array Name) : CoreM (Array String × Nat × Nat) := do
  let mut lines : Array String := #[]
  let mut skipped := 0
  for declName in names do
    -- One bad declaration (a `ppWith`/`isDefEq` internal exception on some
    -- obscure term shape) must not abort a multi-hour, thousands-of-modules run --
    -- catch and skip it instead of propagating.
    match ← (try (extractDecl declName).run' {} {} catch _ => pure none) with
    | some rec => lines := lines.push (Lean.Json.compress (Lean.toJson rec))
    | none => skipped := skipped + 1
  return (lines, lines.size, skipped)

/-- Entry point: extract every declaration of `modName`, print one JSON object per
    line to stdout. Run via `lake env lean --run` or a small `#eval` driver (see
    lean_extractor/README.md for the exact invocation against your Mathlib pin). -/
def extractModule (modName : Name) : CoreM Unit := do
  let names ← declsInModule modName
  let (lines, n, skipped) ← extractDeclsLines names
  for line in lines do
    IO.println line
  IO.eprintln s!"{modName}: extracted {n}, skipped {skipped} (no source range / not thm-or-def)"

/-- Module -> declaration names, built in ONE pass over the environment's constants.
    `declsInModule` rescans the whole `env.constants` map per call, which is fine
    called once (the single-module CLI path) but is O(modules * total_constants) if
    called once per module -- with the full Mathlib environment loaded (~8300
    modules, well over a million constants counting instances/auxiliary defs) that
    measured at minutes per (mostly-empty) module, i.e. an estimated 1-2 WEEKS for
    the full corpus. Building the index once is the same total work as a single
    `declsInModule` call. -/
def buildModuleIndex (env : Environment) : Std.HashMap Name (Array Name) := Id.run do
  let mut idx : Std.HashMap Name (Array Name) := {}
  for (n, _) in env.constants.toList do
    if n.isInternal then continue
    match env.getModuleFor? n with
    | some m => idx := idx.insert m ((idx.getD m #[]).push n)
    | none => pure ()
  return idx

/-- Full-corpus extraction (docs/DESIGN.md step 4). Imports `Mathlib` ONCE -- re-importing
    per module would re-elaborate shared transitive dependencies thousands of times
    over -- builds the module index ONCE (see `buildModuleIndex`), then iterates
    every discovered `Mathlib.*` module, writing one JSONL file per module. One file
    per module makes this resumable: a killed/crashed run can just be re-launched and
    it skips modules whose output already exists, rather than starting the
    multi-hour extraction over from nothing. -/
def extractAll (outDir : System.FilePath) : IO Unit := do
  let env ← Lean.importModules #[{ module := `Mathlib }] {} (trustLevel := 1024)
  -- `Context.maxHeartbeats` is in RAW units, displayed-as-thousands (the "200000"
  -- in the timeout message is `raw/1000`) -- first attempt at raising this set the
  -- raw field to 4000000 meaning to get 20x the default, but that's 4000 displayed,
  -- ~50x TIGHTER than default, not looser; it failed 2715/8308 modules outright.
  -- 0 means unlimited. We already catch per-declaration and per-module exceptions
  -- and kill the whole run if disk gets critical, so an unbounded budget here is
  -- bounded in practice by those, not by an arbitrary guess at the right number.
  let ctx : Lean.Core.Context := { fileName := "Mathlib", fileMap := default, maxHeartbeats := 0 }
  let mathlibModules := env.header.moduleNames.filter (·.toString.startsWith "Mathlib.")
  IO.eprintln s!"{mathlibModules.size} Mathlib modules found; indexing declarations..."
  let idx := buildModuleIndex env
  IO.eprintln s!"index built, {idx.size} modules have at least one declaration"
  IO.FS.createDirAll outDir
  let mut totalDecls := 0
  let mut totalSkipped := 0
  let mut doneModules := 0
  let mut failedModules : Array Name := #[]
  for modName in mathlibModules do
    doneModules := doneModules + 1
    let outPath := outDir / s!"{modName}.jsonl"
    if ← outPath.pathExists then
      continue
    let names := idx.getD modName #[]
    -- Belt-and-suspenders: extractDeclsLines already catches per-declaration
    -- exceptions, but a module-level catch here means even something that slips
    -- past that (or a genuinely stuck computation that only fails at a checkpoint
    -- outside the per-decl try) costs us one module's data, not the whole run.
    -- `.toIO'` converts `Lean.Exception` into `IO.Error` (see `CoreM.toIO`), so the
    -- exception caught HERE has type `IO.Error`, not `Lean.Exception` -- it has no
    -- `.toMessageData`, just `ToString`.
    match ← (try some <$> (extractDeclsLines names).toIO' ctx { env := env }
              catch e => do
                IO.eprintln s!"module {modName} FAILED, skipping: {e}"
                pure none) with
    | none => failedModules := failedModules.push modName
    | some (lines, n, skipped) => do
      totalDecls := totalDecls + n
      totalSkipped := totalSkipped + skipped
      IO.FS.writeFile outPath (if lines.isEmpty then "" else String.intercalate "\n" lines.toList ++ "\n")
    if doneModules % 100 == 0 then
      IO.eprintln s!"[{doneModules}/{mathlibModules.size}] {totalDecls} declarations so far \
        ({totalSkipped} skipped, {failedModules.size} modules failed outright)"
  IO.eprintln s!"DONE: {totalDecls} declarations across {mathlibModules.size} modules \
    ({totalSkipped} skipped, {failedModules.size} modules failed outright)"
  unless failedModules.isEmpty do
    IO.eprintln s!"failed modules: {failedModules.toList.map toString}"

end MathEBT

/-- CLI entry point.
    `extract <ModuleName>`               -- one module, JSONL to stdout (step 1)
    `extract --all <outDir>`             -- full Mathlib, one JSONL file per module (step 4) -/
def main (args : List String) : IO Unit := do
  Lean.initSearchPath (← Lean.findSysroot)
  match args with
  | ["--all", outDir] => MathEBT.extractAll outDir
  | [modStr] => do
    let modName := modStr.toName
    let some path ← MathEBT.moduleFilePath modName | do
      IO.eprintln s!"could not resolve module {modName} to a file via the search path"
      IO.Process.exit 1
    let env ← Lean.importModules #[{ module := modName }] {} (trustLevel := 1024)
    let ctx : Lean.Core.Context := { fileName := path.toString, fileMap := default }
    (MathEBT.extractModule modName).toIO' ctx { env := env }
  | _ => do
    IO.eprintln "usage: extract <ModuleName>  |  extract --all <outDir>"
    IO.Process.exit 1
