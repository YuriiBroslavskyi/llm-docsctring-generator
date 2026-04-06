"""
╔══════════════════════════════════════════════════════════════════════╗
║                    JSONL Dataset Merger v1.0                        ║
║     Compares two .jsonl files, deduplicates, merges into one        ║
╚══════════════════════════════════════════════════════════════════════╝

USAGE:
  python jsonl_merger.py --a dataset_claude.jsonl --b dataset_gemini.jsonl
  python jsonl_merger.py --a dataset_claude.jsonl --b dataset_gemini.jsonl --out merged.jsonl

OUTPUT:
  merged_<timestamp>.jsonl   — final deduplicated dataset
  merge_report.txt           — full comparison report
"""

import json
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field


# ─────────────────────────── CONFIG ──────────────────────────────────

DEFAULT_OUTPUT = f"merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
REPORT_FILE    = "merge_report.txt"

# Which field(s) to use for deduplication fingerprint
# "input" = function source code (recommended — catches identical code
#            even if docstrings differ slightly between the two miners)
DEDUP_FIELD = "input"


# ─────────────────────────── STATS ───────────────────────────────────

@dataclass
class MergeStats:
    file_a:              str = ""
    file_b:              str = ""
    total_a:             int = 0
    total_b:             int = 0
    unique_to_a:         int = 0
    unique_to_b:         int = 0
    duplicates_removed:  int = 0
    conflicts_resolved:  int = 0   # same code, different docstring → best wins
    final_count:         int = 0


# ──────────────────────────── LOADER ─────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    """
    Load every line of a JSONL file into a list of dicts.

    Args:
        path: Path to the .jsonl file.

    Returns:
        List of parsed JSON objects; malformed lines are skipped with a warning.
    """
    records = []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  [WARN] {path.name} line {i} skipped — {exc}")
    return records


# ──────────────────────── FINGERPRINT ────────────────────────────────

def fingerprint(record: dict) -> str:
    """
    Compute a SHA-256 fingerprint of the dedup field value.

    Args:
        record: A single JSONL record dict.

    Returns:
        Hex SHA-256 string of the normalised dedup field.
    """
    raw = record.get(DEDUP_FIELD, "")
    normalised = " ".join(raw.split())   # collapse whitespace
    return hashlib.sha256(normalised.encode()).hexdigest()


# ──────────────────────── QUALITY SCORE ──────────────────────────────

def quality_score(record: dict) -> float:
    """
    Extract or estimate a quality score from a record.

    Prefers an explicit '_meta.quality_score' field (written by
    data_miner.py); falls back to a heuristic based on output length.

    Args:
        record: A single JSONL record dict.

    Returns:
        Float quality score (higher is better).
    """
    meta = record.get("_meta", {})
    if isinstance(meta, dict) and "quality_score" in meta:
        return float(meta["quality_score"])

    # heuristic fallback: longer docstring ≈ more complete
    output = record.get("output", "")
    words  = len(output.split())
    score  = min(words / 200, 0.9)   # cap at 0.9 — no provenance

    # bonus if common Google-style sections are present
    for section in ("Args:", "Returns:", "Raises:", "Example"):
        if section in output:
            score += 0.05

    return min(score, 1.0)


# ──────────────────────────── MERGER ─────────────────────────────────

class JsonlMerger:
    """
    Loads two JSONL datasets, deduplicates them, resolves docstring
    conflicts by keeping the higher-quality record, and writes the
    result to a new JSONL file.

    Args:
        path_a: Path to the first dataset (e.g. Claude's output).
        path_b: Path to the second dataset (e.g. Gemini's output).
        output_path: Destination path for the merged dataset.
    """

    def __init__(self, path_a: Path, path_b: Path, output_path: Path) -> None:
        self.path_a      = path_a
        self.path_b      = path_b
        self.output_path = output_path
        self.stats       = MergeStats(
            file_a = str(path_a),
            file_b = str(path_b),
        )

    def run(self) -> MergeStats:
        """
        Execute the full merge pipeline.

        Returns:
            MergeStats with counts for every merge decision.
        """
        print(f"\n📂  Loading  A → {self.path_a.name}")
        records_a = load_jsonl(self.path_a)
        print(f"📂  Loading  B → {self.path_b.name}")
        records_b = load_jsonl(self.path_b)

        self.stats.total_a = len(records_a)
        self.stats.total_b = len(records_b)
        print(f"\n  A: {self.stats.total_a:,} records")
        print(f"  B: {self.stats.total_b:,} records")

        merged = self._merge(records_a, records_b)

        self.stats.final_count = len(merged)
        self._write(merged)
        return self.stats

    # ── merge logic ───────────────────────────────────────────────────

    def _merge(
        self,
        records_a: list[dict],
        records_b: list[dict],
    ) -> list[dict]:
        """
        Merge two record lists using fingerprint-based deduplication.

        When two records share the same fingerprint (identical code),
        the one with the higher quality score wins.

        Args:
            records_a: Records from dataset A.
            records_b: Records from dataset B.

        Returns:
            Deduplicated and sorted list of merged records.
        """
        # index: fingerprint → (record, source_label, score)
        index: dict[str, tuple[dict, str, float]] = {}

        def _insert(records: list[dict], label: str) -> None:
            for rec in records:
                fp    = fingerprint(rec)
                score = quality_score(rec)

                if fp not in index:
                    index[fp] = (rec, label, score)
                else:
                    # conflict: same code, pick the better docstring
                    _, existing_label, existing_score = index[fp]
                    if score > existing_score:
                        index[fp] = (rec, label, score)
                        self.stats.conflicts_resolved += 1
                    else:
                        self.stats.duplicates_removed += 1

        _insert(records_a, "A")
        _insert(records_b, "B")

        # count provenance
        from_a = sum(1 for _, (_, lbl, _) in index.items() if lbl == "A")
        from_b = sum(1 for _, (_, lbl, _) in index.items() if lbl == "B")
        self.stats.unique_to_a = from_a
        self.stats.unique_to_b = from_b

        # sort by quality score descending
        sorted_records = sorted(
            (rec for rec, _, _ in index.values()),
            key=quality_score,
            reverse=True,
        )
        return sorted_records

    # ── writer ────────────────────────────────────────────────────────

    def _write(self, records: list[dict]) -> None:
        """
        Write the merged records to a JSONL file.

        Args:
            records: Sorted, deduplicated list of record dicts.

        Returns:
            None
        """
        with self.output_path.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"\n✅  Merged dataset → {self.output_path}  ({len(records):,} records)")


# ───────────────────────── REPORT ────────────────────────────────────

def write_report(stats: MergeStats, output_path: str) -> None:
    """
    Print and save a human-readable merge report.

    Args:
        stats: MergeStats collected during the merge run.
        output_path: Path of the generated merged dataset file.

    Returns:
        None
    """
    overlap = stats.total_a + stats.total_b - stats.final_count
    overlap_pct = overlap / max(stats.total_a + stats.total_b, 1) * 100

    lines = [
        "=" * 60,
        "  JSONL Dataset Merge Report",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        "INPUT FILES:",
        f"  A  {stats.file_a}",
        f"  B  {stats.file_b}",
        "",
        "RECORD COUNTS:",
        f"  Dataset A          : {stats.total_a:>8,}",
        f"  Dataset B          : {stats.total_b:>8,}",
        f"  Combined (raw)     : {stats.total_a + stats.total_b:>8,}",
        "",
        "DEDUPLICATION:",
        f"  Exact duplicates   : {stats.duplicates_removed:>8,}",
        f"  Conflicts resolved : {stats.conflicts_resolved:>8,}  (same code, best doc kept)",
        f"  Overlap rate       : {overlap_pct:>7.1f}%",
        "",
        "FINAL DATASET:",
        f"  Records from A     : {stats.unique_to_a:>8,}",
        f"  Records from B     : {stats.unique_to_b:>8,}",
        f"  Total              : {stats.final_count:>8,}",
        f"  Gain over larger   : +{stats.final_count - max(stats.total_a, stats.total_b):,}",
        "",
        "OUTPUT:",
        f"  {output_path}",
        "=" * 60,
    ]

    text = "\n".join(lines)
    print("\n" + text)
    Path(REPORT_FILE).write_text(text, encoding="utf-8")
    print(f"\n📄  Report saved → {REPORT_FILE}")


# ──────────────────────────── CLI ─────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge two JSONL docstring datasets into one deduplicated file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--a",   required=True, help="Path to dataset A (e.g. Claude's output)")
    parser.add_argument("--b",   required=True, help="Path to dataset B (e.g. Gemini's output)")
    parser.add_argument("--out", default=DEFAULT_OUTPUT, help=f"Output path (default: {DEFAULT_OUTPUT})")
    return parser.parse_args()


# ─────────────────────────── MAIN ────────────────────────────────────

def main() -> None:
    args = parse_args()

    path_a = Path(args.a)
    path_b = Path(args.b)

    for p in (path_a, path_b):
        if not p.exists():
            print(f"❌  File not found: {p}")
            raise SystemExit(1)

    merger = JsonlMerger(
        path_a      = path_a,
        path_b      = path_b,
        output_path = Path(args.out),
    )

    stats = merger.run()
    write_report(stats, args.out)


if __name__ == "__main__":
    main()