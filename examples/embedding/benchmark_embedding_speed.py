#!/usr/bin/env python3
"""Benchmark embedding-style title classification across multiple IBL engines.

This script supports two modes:
- compare: build a larger dataset, precompute embedding similarities, and benchmark
  multiple engines by launching isolated subprocesses.
- single: run one engine against a prepared artifact (used internally by compare).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
import json
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

import numpy as np


ENGINE_IBL6 = "ibl6_local"
ENGINE_IBL4 = "ibl4_speedy_repo"
ENGINE_SPEEDY = "speedyibl_repo"
ENGINE_PYIBL5 = "pyibl5_pip"
ALL_ENGINES = (ENGINE_IBL6, ENGINE_IBL4, ENGINE_SPEEDY, ENGINE_PYIBL5)

LABEL_SAFE = 0
LABEL_SCAM = 1


def _label_name(label_id: int) -> str:
    return "scam" if int(label_id) == LABEL_SCAM else "safe"


def _load_base_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as infile:
        rows = list(csv.DictReader(infile))
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    for row in rows:
        if "title" not in row or "label" not in row:
            raise ValueError(f"CSV must contain title and label columns: {csv_path}")
    return rows


def _generate_examples(rows: list[dict[str, str]], target_size: int, scam_ratio: float, seed: int) -> list[dict[str, str]]:
    if target_size < 2:
        raise ValueError("target_size must be at least 2")
    if not (0.05 <= scam_ratio <= 0.95):
        raise ValueError("scam_ratio must be between 0.05 and 0.95")

    rng = random.Random(seed)
    scam_base = [row["title"] for row in rows if row["label"].strip().lower() == "scam"]
    safe_base = [row["title"] for row in rows if row["label"].strip().lower() == "safe"]
    if not scam_base or not safe_base:
        raise ValueError("Base CSV must include both scam and safe labels")

    scam_prefixes = [
        "Urgent alert",
        "Immediate action required",
        "Security warning",
        "Account verification notice",
        "Payment issue",
        "Limited time claim",
        "Mailbox quota exceeded",
        "Fraud prevention request",
    ]
    safe_prefixes = [
        "Team update",
        "Project note",
        "Meeting reminder",
        "Weekly digest",
        "Invoice update",
        "Planning follow-up",
        "Operations notice",
        "Department summary",
    ]
    scam_suffixes = [
        "verify within 24 hours",
        "click to confirm now",
        "reset credentials today",
        "unlock access immediately",
        "prevent suspension now",
    ]
    safe_suffixes = [
        "for next sprint planning",
        "for your review",
        "before friday meeting",
        "for monthly reporting",
        "for project archive",
    ]

    target_scam = int(round(target_size * scam_ratio))
    target_safe = target_size - target_scam

    def make_rows(base_titles: list[str], prefixes: list[str], suffixes: list[str], label: str, count: int) -> list[dict[str, str]]:
        generated: list[dict[str, str]] = []
        for i in range(count):
            base_title = base_titles[i % len(base_titles)]
            prefix = prefixes[i % len(prefixes)]
            suffix = suffixes[i % len(suffixes)]
            nonce = 100000 + i
            day = ["Mon", "Tue", "Wed", "Thu", "Fri"][i % 5]
            if label == "scam":
                title = f"{prefix}: {base_title} ref {nonce} {suffix} {day}"
            else:
                title = f"{prefix}: {base_title} memo {nonce} {suffix} {day}"
            generated.append({"title": title, "label": label})
        return generated

    examples = make_rows(scam_base, scam_prefixes, scam_suffixes, "scam", target_scam)
    examples.extend(make_rows(safe_base, safe_prefixes, safe_suffixes, "safe", target_safe))
    rng.shuffle(examples)
    return examples


def _write_examples_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["title", "label"])
        writer.writeheader()
        writer.writerows(rows)


def _build_similarity_artifact(
    examples: list[dict[str, str]],
    model_name: str,
    artifact_path: Path,
) -> dict[str, int | str | float]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "This benchmark requires sentence-transformers. Install it with: "
            "pip install sentence-transformers"
        ) from exc

    unique_titles = sorted({row["title"] for row in examples})
    title_to_id = {title: i for i, title in enumerate(unique_titles)}
    label_ids = np.array([LABEL_SCAM if row["label"] == "scam" else LABEL_SAFE for row in examples], dtype=np.int8)
    title_ids = np.array([title_to_id[row["title"]] for row in examples], dtype=np.int32)

    encoder = SentenceTransformer(model_name)
    vectors = encoder.encode(unique_titles, convert_to_numpy=True, show_progress_bar=True)
    vectors = np.asarray(vectors, dtype=np.float32)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit_vectors = vectors / norms
    similarity = np.matmul(unit_vectors, unit_vectors.T)
    similarity = np.clip((similarity + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact_path,
        title_ids=title_ids,
        labels=label_ids,
        similarity=similarity,
    )

    return {
        "rows": int(len(examples)),
        "unique_titles": int(len(unique_titles)),
        "embedding_dim": int(vectors.shape[1]),
        "model_name": model_name,
    }


def _load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_pyibl_style_engine(
    make_agent: Callable[[], object],
    configure_similarity: Callable[[object], None],
    choose_fn: Callable[[object, int], int],
    populate_fn: Callable[[object, int, int, float], None],
    respond_fn: Callable[[object, float], None],
    title_ids: np.ndarray,
    labels: np.ndarray,
    runs: int,
    epochs: int,
    seed: int,
) -> dict[str, float]:
    correct = 0
    total = 0
    started = time.perf_counter()

    for run in range(runs):
        run_seed = seed + run
        random.seed(run_seed)
        np.random.seed(run_seed)

        agent = make_agent()
        configure_similarity(agent)

        # Give each label option a starting utility for each title id.
        for title_id, label in zip(title_ids, labels):
            populate_fn(agent, int(title_id), int(label), 1.0)
            populate_fn(agent, int(title_id), int(1 - int(label)), 0.0)

        for _epoch in range(epochs):
            order = np.arange(len(title_ids))
            np.random.shuffle(order)
            for index in order:
                title_id = int(title_ids[index])
                expected_label = int(labels[index])
                predicted_label = choose_fn(agent, title_id)
                reward = 1.0 if predicted_label == expected_label else 0.0
                respond_fn(agent, reward)
                correct += int(predicted_label == expected_label)
                total += 1

    elapsed = time.perf_counter() - started
    return {
        "elapsed_seconds": float(elapsed),
        "accuracy": float(correct / total if total else 0.0),
        "total_decisions": float(total),
        "decisions_per_second": float(total / elapsed if elapsed > 0 else 0.0),
    }


def _run_engine_single(
    engine: str,
    artifact_path: Path,
    repo_root: Path,
    runs: int,
    epochs: int,
    seed: int,
) -> dict[str, float | str]:
    data = np.load(artifact_path, allow_pickle=False)
    title_ids = data["title_ids"].astype(np.int32)
    labels = data["labels"].astype(np.int8)
    similarity = data["similarity"].astype(np.float32)

    def lookup_similarity(left: int, right: int) -> float:
        return float(similarity[int(left), int(right)])

    if engine == ENGINE_IBL6:
        sys.path.insert(0, str(repo_root))
        import pyibl as pyibl_local

        Agent = pyibl_local.Agent

        def make_agent():
            return Agent(attributes=["title_id", "label"], mismatch_penalty=1, default_utility=0.5)

        def configure_similarity(agent):
            def sim_fn(x, y):
                return lookup_similarity(int(x), int(y))

            try:
                agent.similarity("title_id", sim_fn)
            except TypeError:
                agent.similarity(["title_id"], sim_fn)

        def choose_fn(agent, title_id):
            options = [
                {"title_id": title_id, "label": "scam"},
                {"title_id": title_id, "label": "safe"},
            ]
            try:
                choice = agent.choose(options)
            except TypeError:
                choice = agent.choose(*options)
            return LABEL_SCAM if choice["label"] == "scam" else LABEL_SAFE

        def populate_fn(agent, title_id, label, utility):
            option = {"title_id": title_id, "label": _label_name(label)}
            try:
                agent.populate([option], utility)
            except TypeError:
                agent.populate(utility, option)

        def respond_fn(agent, reward):
            agent.respond(reward)

        metrics = _run_pyibl_style_engine(
            make_agent,
            configure_similarity,
            choose_fn,
            populate_fn,
            respond_fn,
            title_ids,
            labels,
            runs,
            epochs,
            seed,
        )

    elif engine == ENGINE_PYIBL5:
        import pyibl as pyibl_pip

        Agent = pyibl_pip.Agent

        def make_agent():
            return Agent(attributes=["title_id", "label"], mismatch_penalty=1, default_utility=0.5)

        def configure_similarity(agent):
            def sim_fn(x, y):
                return lookup_similarity(int(x), int(y))

            try:
                agent.similarity("title_id", sim_fn)
            except TypeError:
                agent.similarity(["title_id"], sim_fn)

        def choose_fn(agent, title_id):
            options = [
                {"title_id": title_id, "label": "scam"},
                {"title_id": title_id, "label": "safe"},
            ]
            try:
                choice = agent.choose(options)
            except TypeError:
                choice = agent.choose(*options)
            return LABEL_SCAM if choice["label"] == "scam" else LABEL_SAFE

        def populate_fn(agent, title_id, label, utility):
            option = {"title_id": title_id, "label": _label_name(label)}
            try:
                agent.populate([option], utility)
            except TypeError:
                agent.populate(utility, option)

        def respond_fn(agent, reward):
            agent.respond(reward)

        metrics = _run_pyibl_style_engine(
            make_agent,
            configure_similarity,
            choose_fn,
            populate_fn,
            respond_fn,
            title_ids,
            labels,
            runs,
            epochs,
            seed,
        )

    elif engine == ENGINE_IBL4:
        pyibl4_path = repo_root / "SpeedyIBL" / "Codes" / "pyibl.py"
        if not pyibl4_path.exists():
            raise FileNotFoundError(f"Missing legacy pyibl implementation at {pyibl4_path}")

        sys.path.insert(0, str(repo_root))
        pyibl4 = _load_module_from_path("speedyibl_pyibl4", pyibl4_path)

        # The bundled IBL4 code in SpeedyIBL targets an older pyactup signature
        # where Memory accepted learning_time_increment.
        memory_init_params = inspect.signature(pyibl4.pyactup.Memory.__init__).parameters
        if "learning_time_increment" not in memory_init_params:
            base_memory = pyibl4.pyactup.Memory
            reset_params = inspect.signature(base_memory.reset).parameters

            class MemoryCompat(base_memory):
                def __init__(self, *args, learning_time_increment=None, **kwargs):
                    super().__init__(*args, **kwargs)

                def reset(self, *args, optimized_learning=None, **kwargs):
                    if optimized_learning is not None:
                        try:
                            self.optimized_learning = optimized_learning
                        except Exception:
                            pass

                    supported = {}
                    if "preserve_prepopulated" in reset_params and "preserve_prepopulated" in kwargs:
                        supported["preserve_prepopulated"] = kwargs["preserve_prepopulated"]
                    if "index" in reset_params and "index" in kwargs:
                        supported["index"] = kwargs["index"]
                    return base_memory.reset(self, *args, **supported)

            pyibl4.pyactup.Memory = MemoryCompat

        Agent = pyibl4.Agent

        def make_agent():
            return Agent(attributes=["title_id", "label"], mismatch_penalty=1, default_utility=0.5)

        def configure_similarity(_agent):
            def sim_fn(x, y):
                return lookup_similarity(int(x), int(y))

            pyibl4.similarity(sim_fn, "title_id")

        def choose_fn(agent, title_id):
            option_scam = {"title_id": title_id, "label": "scam"}
            option_safe = {"title_id": title_id, "label": "safe"}
            choice = agent.choose(option_scam, option_safe)
            return LABEL_SCAM if choice["label"] == "scam" else LABEL_SAFE

        def populate_fn(agent, title_id, label, utility):
            option = {"title_id": title_id, "label": _label_name(label)}
            agent.populate(utility, option)

        def respond_fn(agent, reward):
            agent.respond(reward)

        metrics = _run_pyibl_style_engine(
            make_agent,
            configure_similarity,
            choose_fn,
            populate_fn,
            respond_fn,
            title_ids,
            labels,
            runs,
            epochs,
            seed,
        )

    elif engine == ENGINE_SPEEDY:
        speedy_path = repo_root / "SpeedyIBL" / "Codes" / "speedyibl.py"
        if not speedy_path.exists():
            raise FileNotFoundError(f"Missing SpeedyIBL implementation at {speedy_path}")

        speedy_mod = _load_module_from_path("speedyibl_repo_module", speedy_path)
        Agent = speedy_mod.Agent

        started = time.perf_counter()
        correct = 0
        total = 0

        for run in range(runs):
            run_seed = seed + run
            random.seed(run_seed)
            np.random.seed(run_seed)

            agent = Agent(default_utility=0.5, mismatchPenalty=1)

            def speedy_similarity(left, right):
                left_id = int(np.asarray(left).reshape(-1)[0])
                right_id = int(np.asarray(right).reshape(-1)[0])
                return np.array([lookup_similarity(left_id, right_id)])

            agent.similarity([0], speedy_similarity)

            for title_id, label in zip(title_ids, labels):
                option_true = (_label_name(int(label)), (int(title_id),))
                option_false = (_label_name(int(1 - int(label))), (int(title_id),))
                agent.prepopulate(option_true, 1.0)
                agent.prepopulate(option_false, 0.0)

            for _epoch in range(epochs):
                order = np.arange(len(title_ids))
                np.random.shuffle(order)
                for index in order:
                    title_id = int(title_ids[index])
                    expected_label = int(labels[index])
                    options = [
                        ("scam", (title_id,)),
                        ("safe", (title_id,)),
                    ]
                    choice = agent.choose(options)
                    predicted_label = LABEL_SCAM if choice[0] == "scam" else LABEL_SAFE
                    reward = 1.0 if predicted_label == expected_label else 0.0
                    agent.respond(reward)
                    correct += int(predicted_label == expected_label)
                    total += 1

        elapsed = time.perf_counter() - started
        metrics = {
            "elapsed_seconds": float(elapsed),
            "accuracy": float(correct / total if total else 0.0),
            "total_decisions": float(total),
            "decisions_per_second": float(total / elapsed if elapsed > 0 else 0.0),
        }

    else:
        raise ValueError(f"Unknown engine {engine}")

    metrics["engine"] = engine
    return metrics


def _run_compare(args: argparse.Namespace) -> int:
    script_path = Path(__file__).resolve()
    repo_root = Path(args.repo_root).resolve()
    base_csv = Path(args.base_csv).resolve()

    base_rows = _load_base_rows(base_csv)
    expanded_rows = _generate_examples(base_rows, args.dataset_size, args.scam_ratio, args.seed)

    if args.write_expanded_csv:
        _write_examples_csv(expanded_rows, Path(args.write_expanded_csv).resolve())

    with tempfile.TemporaryDirectory(prefix="pyibl-embedding-bench-") as tmpdir:
        artifact_path = Path(tmpdir) / "embedding_similarity_artifact.npz"
        artifact_info = _build_similarity_artifact(expanded_rows, args.model_name, artifact_path)

        engines = args.engines or list(ALL_ENGINES)
        results = []
        skipped = []

        for engine in engines:
            python_executable = sys.executable
            if engine == ENGINE_IBL6 and args.ibl6_python:
                python_executable = args.ibl6_python
            elif engine == ENGINE_IBL4 and args.ibl4_python:
                python_executable = args.ibl4_python
            elif engine == ENGINE_SPEEDY and args.speedy_python:
                python_executable = args.speedy_python
            elif engine == ENGINE_PYIBL5:
                if not args.pyibl5_python:
                    skipped.append({"engine": engine, "reason": "--pyibl5-python not provided"})
                    continue
                python_executable = args.pyibl5_python

            cmd = [
                python_executable,
                str(script_path),
                "--mode",
                "single",
                "--engine",
                engine,
                "--artifact",
                str(artifact_path),
                "--runs",
                str(args.runs),
                "--epochs",
                str(args.epochs),
                "--seed",
                str(args.seed),
                "--repo-root",
                str(repo_root),
            ]

            completed = subprocess.run(cmd, capture_output=True, text=True)
            if completed.returncode != 0:
                skipped.append(
                    {
                        "engine": engine,
                        "reason": "subprocess failed",
                        "stderr": completed.stderr.strip(),
                    }
                )
                continue

            payload = completed.stdout.strip().splitlines()[-1]
            results.append(json.loads(payload))

    print()
    print("Embedding benchmark configuration")
    print(f"  dataset rows:      {artifact_info['rows']}")
    print(f"  unique titles:     {artifact_info['unique_titles']}")
    print(f"  embedding dim:     {artifact_info['embedding_dim']}")
    print(f"  model:             {artifact_info['model_name']}")
    print(f"  runs x epochs:     {args.runs} x {args.epochs}")
    print()

    if results:
        sorted_results = sorted(results, key=lambda item: item["elapsed_seconds"])
        print("Results (fastest first)")
        print("engine,elapsed_seconds,decisions_per_second,accuracy")
        for result in sorted_results:
            print(
                f"{result['engine']},{result['elapsed_seconds']:.4f},"
                f"{result['decisions_per_second']:.2f},{result['accuracy']:.4f}"
            )
    else:
        print("No engines completed successfully.")

    if skipped:
        print()
        print("Skipped/failed engines")
        for item in skipped:
            print(f"  {item['engine']}: {item['reason']}")
            if "stderr" in item and item["stderr"]:
                print(f"    {item['stderr']}")

    return 0 if results else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Embedding-speed benchmark for IBL engines")

    repo_default = Path(__file__).resolve().parents[2]
    base_csv_default = Path(__file__).resolve().with_name("scam_titles.csv")

    parser.add_argument("--mode", choices=["compare", "single"], default="compare")
    parser.add_argument("--repo-root", default=str(repo_default), help="Repository root path")

    parser.add_argument("--base-csv", default=str(base_csv_default), help="Base labeled scam CSV")
    parser.add_argument("--dataset-size", type=int, default=2000, help="Number of rows in expanded dataset")
    parser.add_argument("--scam-ratio", type=float, default=0.65, help="Fraction of scam rows in expanded dataset")
    parser.add_argument("--write-expanded-csv", default=None, help="Optional path to save expanded dataset CSV")

    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--engines", nargs="*", choices=ALL_ENGINES, default=list(ALL_ENGINES))
    parser.add_argument(
        "--pyibl5-python",
        default=None,
        help="Python executable that resolves pyibl==5.x for engine pyibl5_pip",
    )
    parser.add_argument(
        "--ibl4-python",
        default=None,
        help="Optional Python executable for ibl4_speedy_repo engine",
    )
    parser.add_argument(
        "--speedy-python",
        default=None,
        help="Optional Python executable for speedyibl_repo engine",
    )
    parser.add_argument(
        "--ibl6-python",
        default=None,
        help="Optional Python executable for ibl6_local engine",
    )

    parser.add_argument("--engine", choices=ALL_ENGINES, default=None, help="Single mode: engine to run")
    parser.add_argument("--artifact", default=None, help="Single mode: path to similarity artifact (.npz)")

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.mode == "compare":
        return _run_compare(args)

    if args.mode == "single":
        if not args.engine:
            raise SystemExit("--engine is required in single mode")
        if not args.artifact:
            raise SystemExit("--artifact is required in single mode")

        result = _run_engine_single(
            engine=args.engine,
            artifact_path=Path(args.artifact).resolve(),
            repo_root=Path(args.repo_root).resolve(),
            runs=args.runs,
            epochs=args.epochs,
            seed=args.seed,
        )
        print(json.dumps(result, sort_keys=True))
        return 0

    raise SystemExit(f"Unknown mode {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
