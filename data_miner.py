"""
╔══════════════════════════════════════════════════════════════════════╗
║              Google Style Docstring — Data Miner v1.0               ║
║  Extracts high-quality function/docstring pairs → JSONL dataset      ║
╚══════════════════════════════════════════════════════════════════════╝

USAGE:
  1. Clone target repos into ./target_repos/
  2. Run:  python data_miner.py
  3. Output: dataset_<timestamp>.jsonl  +  mining_report.txt

TARGET REPOS (clone these manually first):
  git clone https://github.com/tiangolo/fastapi           target_repos/fastapi
  git clone https://github.com/pydantic/pydantic          target_repos/pydantic
  git clone https://github.com/googleapis/python-bigquery target_repos/python-bigquery
  git clone https://github.com/googleapis/google-cloud-python target_repos/google-cloud-python
"""

import ast
import json
import os
import re
import textwrap
import hashlib
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional


# ─────────────────────────── CONFIGURATION ───────────────────────────

# Directories to scan (relative to this script's location)
REPO_DIRS: list[str] = [
    "target_repos/fastapi",
    "target_repos/pydantic",
    "target_repos/python-bigquery",
    "target_repos/google-cloud-python",
]

OUTPUT_JSONL   = f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
REPORT_FILE    = "mining_report.txt"

# Quality-control thresholds
MIN_FUNCTION_LINES   = 3       # body must have at least N lines
MAX_FUNCTION_LINES   = 150     # skip gigantic functions
MIN_DOCSTRING_WORDS  = 10      # docstring must have at least N words
MAX_DOCSTRING_CHARS  = 4000    # skip pathologically long docstrings
REQUIRED_SECTIONS    = {"Args", "Returns"}   # both must be present
OPTIONAL_BONUS       = {"Raises", "Example", "Examples", "Note", "Notes"}

# Instruction template written into every JSONL record
INSTRUCTION_TEMPLATE = (
    "Write a Python function with a complete Google-style docstring "
    "that includes Args and Returns sections."
)


# ─────────────────────────── DATA MODEL ──────────────────────────────

@dataclass
class FunctionRecord:
    instruction: str
    input: str           # the function signature + body (code)
    output: str          # the docstring (documentation)
    # ── metadata (not sent to the LLM, kept for provenance) ──
    source_file: str     = ""
    function_name: str   = ""
    quality_score: float = 0.0
    has_raises: bool     = False
    has_examples: bool   = False
    line_count: int      = 0
    sha256: str          = ""

    def to_jsonl_dict(self) -> dict:
        """Return a clean dict suitable for JSONL export."""
        return {
            "instruction": self.instruction,
            "input":       self.input,
            "output":      self.output,
            # lightweight metadata kept alongside training fields
            "_meta": {
                "source":        self.source_file,
                "function":      self.function_name,
                "quality_score": round(self.quality_score, 3),
                "line_count":    self.line_count,
                "has_raises":    self.has_raises,
                "has_examples":  self.has_examples,
            }
        }


# ─────────────────────────── STATISTICS ──────────────────────────────

@dataclass
class MiningStats:
    files_scanned:       int = 0
    files_failed:        int = 0
    functions_found:     int = 0
    rejected_no_doc:     int = 0
    rejected_incomplete: int = 0
    rejected_too_short:  int = 0
    rejected_too_long:   int = 0
    rejected_duplicate:  int = 0
    accepted:            int = 0
    repos_processed:     list = field(default_factory=list)


# ──────────────────────── CORE EXTRACTOR ─────────────────────────────

class GoogleStyleMiner:
    """
    Walks Python source files, parses ASTs, and extracts
    (code, docstring) pairs that satisfy Google Style standards.

    Args:
        repo_dirs: List of directory paths to search recursively.
        output_path: Destination .jsonl file path.
        verbose: If True, prints per-file progress.
    """

    def __init__(
        self,
        repo_dirs: list[str],
        output_path: str,
        verbose: bool = True,
    ) -> None:
        self.repo_dirs   = [Path(d) for d in repo_dirs]
        self.output_path = Path(output_path)
        self.verbose     = verbose
        self.stats       = MiningStats()
        self._seen_hashes: set[str] = set()   # deduplication

    # ── public entry point ────────────────────────────────────────────

    def run(self) -> MiningStats:
        """
        Execute the full mining pipeline.

        Returns:
            MiningStats dataclass with counters for every stage.
        """
        records: list[FunctionRecord] = []

        for repo_dir in self.repo_dirs:
            if not repo_dir.exists():
                print(f"  [SKIP] {repo_dir} — directory not found.")
                continue

            print(f"\n📂  Scanning: {repo_dir}")
            self.stats.repos_processed.append(str(repo_dir))
            repo_records = self._process_repo(repo_dir)
            records.extend(repo_records)
            print(f"  ✔  {len(repo_records):,} records collected from {repo_dir.name}")

        # sort by quality score descending before writing
        records.sort(key=lambda r: r.quality_score, reverse=True)

        self._write_jsonl(records)
        self.stats.accepted = len(records)
        return self.stats

    # ── repo / file traversal ─────────────────────────────────────────

    def _process_repo(self, repo_dir: Path) -> list[FunctionRecord]:
        records: list[FunctionRecord] = []
        py_files = sorted(repo_dir.rglob("*.py"))

        for py_file in py_files:
            # skip test files — they rarely have Google-style docs
            if any(part.startswith("test") for part in py_file.parts):
                continue

            self.stats.files_scanned += 1
            try:
                file_records = self._process_file(py_file)
                records.extend(file_records)
            except Exception as exc:
                self.stats.files_failed += 1
                if self.verbose:
                    print(f"  [ERR] {py_file}: {exc}")

        return records

    def _process_file(self, py_file: Path) -> list[FunctionRecord]:
        source = py_file.read_text(encoding="utf-8", errors="ignore")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            self.stats.files_failed += 1
            return []

        source_lines = source.splitlines()
        records: list[FunctionRecord] = []

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            self.stats.functions_found += 1
            record = self._extract_record(node, source_lines, str(py_file))
            if record is not None:
                records.append(record)

        return records

    # ── extraction & quality control ──────────────────────────────────

    def _extract_record(
        self,
        node: ast.FunctionDef,
        source_lines: list[str],
        filepath: str,
    ) -> Optional[FunctionRecord]:
        """
        Extract and validate a single function node.

        Args:
            node: The AST FunctionDef node.
            source_lines: All lines of the source file.
            filepath: Absolute path of the source file (for metadata).

        Returns:
            A FunctionRecord if the function passes all quality gates,
            or None if it is rejected.
        """
        # ── 1. pull raw source of the function ───────────────────────
        func_lines = source_lines[node.lineno - 1 : node.end_lineno]
        func_source = textwrap.dedent("\n".join(func_lines)).strip()
        line_count  = len(func_lines)

        if line_count < MIN_FUNCTION_LINES:
            self.stats.rejected_too_short += 1
            return None
        if line_count > MAX_FUNCTION_LINES:
            self.stats.rejected_too_long += 1
            return None

        # ── 2. extract docstring ──────────────────────────────────────
        raw_docstring = ast.get_docstring(node)
        if not raw_docstring:
            self.stats.rejected_no_doc += 1
            return None

        docstring = raw_docstring.strip()

        if len(docstring.split()) < MIN_DOCSTRING_WORDS:
            self.stats.rejected_incomplete += 1
            return None
        if len(docstring) > MAX_DOCSTRING_CHARS:
            self.stats.rejected_too_long += 1
            return None

        # ── 3. check required Google-style sections ───────────────────
        found_sections = self._parse_sections(docstring)
        if not REQUIRED_SECTIONS.issubset(found_sections):
            self.stats.rejected_incomplete += 1
            return None

        # ── 4. deduplication via SHA-256 of normalised source ─────────
        digest = hashlib.sha256(func_source.encode()).hexdigest()
        if digest in self._seen_hashes:
            self.stats.rejected_duplicate += 1
            return None
        self._seen_hashes.add(digest)

        # ── 5. compute quality score ──────────────────────────────────
        bonus_sections = OPTIONAL_BONUS & found_sections
        quality_score  = self._score(
            docstring, line_count, found_sections, bonus_sections
        )

        # ── 6. build record ───────────────────────────────────────────
        return FunctionRecord(
            instruction   = INSTRUCTION_TEMPLATE,
            input         = func_source,
            output        = f'"""\n{docstring}\n"""',
            source_file   = filepath,
            function_name = node.name,
            quality_score = quality_score,
            has_raises    = "Raises" in bonus_sections,
            has_examples  = bool({"Example", "Examples"} & bonus_sections),
            line_count    = line_count,
            sha256        = digest,
        )

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _parse_sections(docstring: str) -> set[str]:
        """
        Detect Google-style section headers inside a docstring.

        Args:
            docstring: Raw docstring text to inspect.

        Returns:
            A set of section names found (e.g. {'Args', 'Returns'}).
        """
        # Google style: "SectionName:" at the start of an indented line
        pattern = re.compile(r"^\s{0,4}([A-Z][a-zA-Z]+):\s*$", re.MULTILINE)
        return {m.group(1) for m in pattern.finditer(docstring)}

    @staticmethod
    def _score(
        docstring: str,
        line_count: int,
        sections: set[str],
        bonus: set[str],
    ) -> float:
        """
        Compute a 0-1 quality score for a function record.

        Args:
            docstring: The full docstring text.
            line_count: Number of lines in the function body.
            sections: All section names detected in the docstring.
            bonus: Optional bonus sections present.

        Returns:
            A float quality score between 0.0 and 1.0.
        """
        score = 0.5  # baseline: passed required filters

        # bonus for optional sections (each +0.1, max 0.3)
        score += min(len(bonus) * 0.1, 0.3)

        # reward meaningful docstring length (sweet spot: 50-300 words)
        words = len(docstring.split())
        if 50 <= words <= 300:
            score += 0.1
        elif words > 300:
            score += 0.05   # still decent, just long

        # reward medium-sized functions (10-60 lines)
        if 10 <= line_count <= 60:
            score += 0.1

        return min(score, 1.0)

    # ── output ───────────────────────────────────────────────────────

    def _write_jsonl(self, records: list[FunctionRecord]) -> None:
        """
        Serialise all accepted records to a JSONL file.

        Args:
            records: List of FunctionRecord instances to write.

        Returns:
            None
        """
        with self.output_path.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec.to_jsonl_dict(), ensure_ascii=False) + "\n")
        print(f"\n✅  Dataset written → {self.output_path}  ({len(records):,} records)")


# ──────────────────────── REPORT GENERATOR ───────────────────────────

def write_report(stats: MiningStats, output_jsonl: str) -> None:
    """
    Write a human-readable mining report to disk.

    Args:
        stats: MiningStats collected during the run.
        output_jsonl: Path of the generated dataset file.

    Returns:
        None
    """
    total_rejected = (
        stats.rejected_no_doc
        + stats.rejected_incomplete
        + stats.rejected_too_short
        + stats.rejected_too_long
        + stats.rejected_duplicate
    )
    acceptance_rate = (
        stats.accepted / max(stats.functions_found, 1) * 100
    )

    lines = [
        "=" * 60,
        "  Google Style Docstring — Mining Report",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        "REPOSITORIES PROCESSED:",
        *[f"  • {r}" for r in stats.repos_processed],
        "",
        "FILE STATISTICS:",
        f"  Files scanned  : {stats.files_scanned:>8,}",
        f"  Files failed   : {stats.files_failed:>8,}",
        "",
        "FUNCTION STATISTICS:",
        f"  Found          : {stats.functions_found:>8,}",
        f"  Accepted       : {stats.accepted:>8,}",
        f"  Acceptance rate: {acceptance_rate:>7.1f}%",
        "",
        "REJECTION BREAKDOWN:",
        f"  No docstring   : {stats.rejected_no_doc:>8,}",
        f"  Incomplete doc : {stats.rejected_incomplete:>8,}",
        f"  Too short      : {stats.rejected_too_short:>8,}",
        f"  Too long       : {stats.rejected_too_long:>8,}",
        f"  Duplicate      : {stats.rejected_duplicate:>8,}",
        f"  Total rejected : {total_rejected:>8,}",
        "",
        "OUTPUT:",
        f"  Dataset file   : {output_jsonl}",
        "=" * 60,
    ]

    report_text = "\n".join(lines)
    print("\n" + report_text)
    Path(REPORT_FILE).write_text(report_text, encoding="utf-8")
    print(f"\n📄  Report saved → {REPORT_FILE}")


# ─────────────────────────────── MAIN ────────────────────────────────

def main() -> None:
    print(__doc__)
    print("─" * 60)

    # validate that at least one repo dir exists
    existing = [d for d in REPO_DIRS if Path(d).exists()]
    if not existing:
        print(
            "⚠️  No repository directories found.\n"
            "    Clone at least one repo into ./target_repos/ and re-run.\n\n"
            "    Example:\n"
            "      mkdir target_repos\n"
            "      git clone https://github.com/tiangolo/fastapi target_repos/fastapi\n"
        )
        return

    miner = GoogleStyleMiner(
        repo_dirs   = REPO_DIRS,
        output_path = OUTPUT_JSONL,
        verbose     = True,
    )

    stats = miner.run()
    write_report(stats, OUTPUT_JSONL)


if __name__ == "__main__":
    main()