#!/usr/bin/env python3
"""Command-line interface for the public LEGO manipulation benchmark."""
from __future__ import annotations

import argparse
import os
import shutil
import json
from pathlib import Path

from .benchmark_core import (
    dump_json, dump_yaml, generate_episode, load_yaml, normalize_product,
    score_episode, validate_product,
)
from .scene_builder import BASE_DIR, build
from .executor_planner import plan_assembly


def normalized_from_path(path: str) -> dict:
    source = Path(path)
    return normalize_product(load_yaml(source), name=source.stem)


def generate(product_path: str, seed: int, output_dir: str) -> tuple[dict, Path]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    product = normalized_from_path(product_path)
    episode = generate_episode(product, seed=seed)
    product_out = output / "product.normalized.yaml"
    manifest = output / "episode_manifest.yaml"
    execution_plan = output / "execution_plan.yaml"
    scene = output / "scene.xml"
    shutil.copy2(Path(BASE_DIR) / "panda.xml", output / "panda.xml")
    shutil.copy2(Path(BASE_DIR) / "hand.xml", output / "hand.xml")
    shutil.copytree(
        Path(BASE_DIR) / "assets", output / "assets", dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("LEGO_Duplo_brick_*.stl"),
    )
    dump_yaml(product_out, product)
    dump_yaml(manifest, episode)
    dump_yaml(execution_plan, plan_assembly(product))
    build(template_xml=str(Path(BASE_DIR) / "scene_template.xml"), output_xml=str(scene),
          episode_manifest=str(manifest))
    return episode, scene


def main(argv=None):
    parser = argparse.ArgumentParser(prog="lego-bench")
    commands = parser.add_subparsers(dest="command", required=True)

    validate_cmd = commands.add_parser("validate", help="validate and normalize a product YAML")
    validate_cmd.add_argument("product")

    convert_cmd = commands.add_parser("convert", help="convert a legacy YAML to schema v1")
    convert_cmd.add_argument("product")
    convert_cmd.add_argument("output")

    generate_cmd = commands.add_parser("generate", help="generate a deterministic loose-parts scene")
    generate_cmd.add_argument("--product", required=True)
    generate_cmd.add_argument("--seed", type=int, default=0)
    generate_cmd.add_argument("--output-dir", default="runs/latest")

    run_cmd = commands.add_parser("run", help="generate and run one benchmark episode")
    run_cmd.add_argument("--product", required=True)
    run_cmd.add_argument("--seed", type=int, default=0)
    run_cmd.add_argument("--output-dir", default="runs/latest")
    run_cmd.add_argument("--headless", action="store_true")
    run_cmd.add_argument("--observation", choices=["oracle", "rgbd"], default="oracle")
    run_cmd.add_argument("--connection-mode", choices=["snap", "physics"], default="snap")

    score_cmd = commands.add_parser("score", help="score an exported actual_state YAML/JSON")
    score_cmd.add_argument("--manifest", required=True)
    score_cmd.add_argument("--actual", required=True)
    score_cmd.add_argument("--output")

    batch_cmd = commands.add_parser("batch-generate", help="generate a reproducible suite of episodes")
    batch_cmd.add_argument("--product", required=True)
    batch_cmd.add_argument("--first-seed", type=int, default=0)
    batch_cmd.add_argument("--count", type=int, required=True)
    batch_cmd.add_argument("--output-dir", default="runs/batch")

    summary_cmd = commands.add_parser("summarize", help="summarize result.json files under a run directory")
    summary_cmd.add_argument("run_dir")

    args = parser.parse_args(argv)
    if args.command == "validate":
        product = normalized_from_path(args.product)
        validate_product(product)
        print(f"VALID: {args.product} ({len(product['blocks'])} blocks)")
    elif args.command == "convert":
        product = normalized_from_path(args.product)
        dump_yaml(args.output, product)
        print(f"WROTE: {Path(args.output).resolve()}")
    elif args.command == "generate":
        episode, scene = generate(args.product, args.seed, args.output_dir)
        print(f"EPISODE: {episode['episode_id']}")
        print(f"SCENE: {scene}")
    elif args.command == "score":
        result = score_episode(load_yaml(args.manifest), load_yaml(args.actual))
        if args.output:
            dump_json(args.output, result)
        print(result)
    elif args.command == "batch-generate":
        suite = []
        for seed in range(args.first_seed, args.first_seed + args.count):
            directory = Path(args.output_dir) / f"seed-{seed}"
            episode, scene = generate(args.product, seed, str(directory))
            suite.append({"seed": seed, "episode_id": episode["episode_id"], "scene": str(scene)})
        dump_json(Path(args.output_dir) / "suite.json", {"episodes": suite})
        print(f"GENERATED: {len(suite)} episodes in {Path(args.output_dir).resolve()}")
    elif args.command == "summarize":
        results = [json.loads(p.read_text(encoding="utf-8")) for p in Path(args.run_dir).rglob("result.json")]
        summary = {
            "episodes": len(results),
            "successes": sum(bool(r.get("success")) for r in results),
            "success_rate": (sum(bool(r.get("success")) for r in results) / len(results)) if results else 0.0,
            "mean_completion": (sum(float(r.get("completion", 0.0)) for r in results) / len(results)) if results else 0.0,
        }
        print(json.dumps(summary, indent=2))
    elif args.command == "run":
        episode, scene = generate(args.product, args.seed, args.output_dir)
        output = Path(args.output_dir).resolve()
        os.environ["MJ_BRIDGE_MODEL"] = str(scene)
        os.environ["LEGO_BENCH_MANIFEST"] = str(output / "episode_manifest.yaml")
        os.environ["LEGO_BENCH_RUN_DIR"] = str(output)
        os.environ["LEGO_BENCH_OBSERVATION"] = args.observation
        os.environ["LEGO_BENCH_CONNECTION_MODE"] = args.connection_mode
        if args.headless:
            os.environ["MJ_BRIDGE_HEADLESS"] = "1"
            if args.observation == "rgbd":
                os.environ.setdefault("MUJOCO_GL", "egl")
        print(f"STARTING: {episode['episode_id']}")
        from .mj_bridge3 import main as bridge_main
        bridge_main()


if __name__ == "__main__":
    main()
