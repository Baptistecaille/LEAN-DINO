"""The contract between the Lean extractor and everything downstream.

One JSONL line per declaration. If a field is missing here, the Lean side must not
emit it and the Python side must not read it -- schema drift between the two
languages is the single most expensive bug class in this project.

NAMING (see docs/DESIGN.md): `decl_name` is the Lean declaration name (`add_comm`),
`lean_namespace` is the Lean namespace (`Nat`), `module` is the source module
(`Mathlib.Algebra.Group.Basic`). The module is NOT a prefix of the declaration name.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[PROOF]", "[SEP]", "[HYP]", "[MASK]"]


@dataclass
class Certificate:
    """Provenance of one generated view. `kind` is the certification level."""

    view_name: str
    transformation: str
    kind: str  # same_expr_different_pp | defeq_by_construction | proved_iff | unchecked
    lean_lemma: str | None = None
    source_hash: str = ""
    target_hash: str = ""
    mathlib_version: str = ""


@dataclass
class Declaration:
    decl_name: str
    lean_namespace: str
    module: str
    file_path: str
    kind: str  # theorem | lemma | def | instance

    # --- surface forms -------------------------------------------------------
    type_implicit: str
    type_explicit: str
    # Lean SOURCE TEXT via declRange -- never the pretty-printed elaborated term.
    proof_source: str

    # --- identity ------------------------------------------------------------
    ast_hash: str
    type_ast_hash: str

    # --- premise selection ---------------------------------------------------
    premises_raw: list[str] = field(default_factory=list)
    premises_filtered: list[str] = field(default_factory=list)
    used_constants: list[str] = field(default_factory=list)

    # --- certified view material (empty => that view is unavailable) ---------
    type_alpha: str | None = None
    type_perm_hyp: str | None = None
    # Pretty-printed subterms of the type, used to build genuinely local views.
    type_subterms: list[str] = field(default_factory=list)
    # Hypotheses of the goal telescope, printed one by one.
    hypotheses: list[str] = field(default_factory=list)
    # Proof source split into tactic steps; local views take contiguous slices.
    proof_steps: list[str] = field(default_factory=list)

    certificates: list[Certificate] = field(default_factory=list)

    # --- derived -------------------------------------------------------------
    @property
    def domain_label(self) -> str:
        """Linear-probe label: second component of the module path.

        `Mathlib.Analysis.Calculus.Deriv` -> `Analysis`. Comes from the MODULE,
        never from `lean_namespace` (see docs/DESIGN.md).
        """
        parts = self.module.split(".")
        return parts[1] if len(parts) > 1 else parts[0]

    def full_text(self) -> str:
        return f"[PROOF] {self.type_implicit} [SEP] {self.proof_source} [SEP]"

    def type_only_text(self, explicit: bool = False) -> str:
        t = self.type_explicit if explicit else self.type_implicit
        return f"[PROOF] {t} [SEP]"


def _to_decl(d: dict) -> Declaration:
    certs = [Certificate(**c) for c in d.pop("certificates", [])]
    known = Declaration.__dataclass_fields__.keys()
    decl = Declaration(**{k: v for k, v in d.items() if k in known}, certificates=certs)
    if not decl.premises_filtered and decl.premises_raw:
        # The Lean extractor only emits premises_raw (every constant used in the
        # elaborated proof term -- includes typeclass instances, structure
        # projections, etc). premises_filtered is the dedup'd, self-reference-free
        # version that premise selection actually uses; trivial-lemma removal
        # (rfl, Eq.symm, ...) happens later in eval.retrieval.filter_premises,
        # since that list is a property of the evaluation, not of the corpus.
        seen: list[str] = []
        for p in decl.premises_raw:
            if p != decl.decl_name and p not in seen:
                seen.append(p)
        decl.premises_filtered = seen
    return decl


def read_jsonl(path: str | Path) -> Iterator[Declaration]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield _to_decl(json.loads(line))


def read_dir(raw_dir: str | Path) -> list[Declaration]:
    """Read every corpus shard in `raw_dir`.

    Dot-prefixed files are skipped. A corpus tarball packed on macOS carries an
    AppleDouble sidecar (`._Mathlib.Foo.jsonl`) for every file with an extended
    attribute; unpacked on Linux those become real, binary files that pathlib's
    glob happily matches, and reading one dies with
    `UnicodeDecodeError: ... byte 0xa3 in position 45`.
    Rejected alternative: opening with errors="replace", which would also
    swallow genuine encoding bugs coming out of the Lean extractor.
    """
    out: list[Declaration] = []
    for p in sorted(Path(raw_dir).glob("*.jsonl")):
        if p.name.startswith("."):
            continue
        out.extend(read_jsonl(p))
    return out


def write_jsonl(path: str | Path, decls: list[Declaration]) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        for d in decls:
            fh.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")
