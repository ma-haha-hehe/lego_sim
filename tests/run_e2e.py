#!/usr/bin/env python3
"""Run isolated, headless ROS end-to-end benchmark episodes."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml


TASK_STATES = (
    "MOVE_TO_PREGRASP",
    "DESCEND_TO_GRASP",
    "CLOSE_GRIPPER",
    "GRASP_CONFIRMED",
    "LIFT",
    "ALIGN_TOOL_FOR_PLACE",
    "MOVE_TO_PREPLACE",
    "DESCEND_TO_TARGET",
    "OPEN_GRIPPER",
    "RETREAT",
)

MAX_XY_ERROR_M = 0.006
MAX_Z_ERROR_M = 0.004
MAX_YAW_ERROR_DEG = 8.0
MAX_TRANSLATION_SLIP_M = 0.002
MAX_ROTATION_SLIP_DEG = 3.0

CASES = {
    "oracle": {
        "script": "run_reference_pipeline.sh",
        "arguments": ("--observation", "oracle"),
        "observation_mode": "oracle",
    },
    "geometry": {
        "script": "run_visual_pipeline.sh",
        "arguments": ("--vision-backend", "geometry"),
        "observation_mode": "rgbd",
    },
}

PRODUCTS = {
    "single_block": "examples/products/single_block.yaml",
    "traffic_light": "examples/products/traffic_light.yaml",
    "bridge": "examples/products/bridge.yaml",
}


class EndToEndFailure(RuntimeError):
    """Raised when a benchmark episode violates the test contract."""


def tail(path: Path, line_count: int = 80) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-line_count:])


def stop_process_group(process: subprocess.Popen, grace_s: float = 12.0) -> int:
    """Stop a launch process and every child in its isolated process group."""
    if process.poll() is not None:
        return int(process.returncode)
    for requested_signal, wait_s in (
        (signal.SIGINT, grace_s),
        (signal.SIGTERM, 5.0),
        (signal.SIGKILL, 2.0),
    ):
        try:
            os.killpg(process.pid, requested_signal)
        except ProcessLookupError:
            break
        try:
            return int(process.wait(timeout=wait_s))
        except subprocess.TimeoutExpired:
            continue
    return int(process.poll() if process.poll() is not None else -1)


def wait_for_result(process: subprocess.Popen, path: Path, timeout_s: float) -> dict:
    deadline = time.monotonic() + timeout_s
    last_error = "result file was not created"
    while time.monotonic() < deadline:
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as error:
                last_error = f"result file is not complete: {error}"
        return_code = process.poll()
        if return_code is not None:
            raise EndToEndFailure(
                f"launch exited with code {return_code} before producing a result"
            )
        time.sleep(0.2)
    raise EndToEndFailure(f"episode timed out after {timeout_s:.1f}s: {last_error}")


def execution_order(run_dir: Path) -> list[str]:
    plan = yaml.safe_load(
        (run_dir / "execution_plan.yaml").read_text(encoding="utf-8")
    )
    block_ids = [str(step["id"]) for step in plan.get("steps", [])]
    if not block_ids:
        raise EndToEndFailure("execution plan contains no assembly steps")
    return block_ids


def assert_state_order(log_text: str, block_ids: list[str], visual: bool) -> None:
    """Verify every block follows the complete assembly state machine in plan order."""
    cursor = -1
    for block_id in block_ids:
        task_marker = f": {block_id}"
        position = log_text.find(task_marker, cursor + 1)
        if position < 0:
            raise EndToEndFailure(f"execution log is missing task: {block_id}")
        cursor = position
        if visual:
            for marker in (
                "MOVE_TO_OBSERVE",
                "Perception completed",
                f"[{block_id}] VISUAL_SOURCE",
            ):
                position = log_text.find(marker, cursor + 1)
                if position < 0:
                    raise EndToEndFailure(
                        f"execution log is missing {marker} for {block_id}"
                    )
                cursor = position
        for state in TASK_STATES:
            marker = f"[{block_id}] {state}"
            position = log_text.find(marker, cursor + 1)
            if position < 0:
                raise EndToEndFailure(
                    f"execution log is missing state {state} for {block_id}"
                )
            cursor = position
            if visual and state == "MOVE_TO_PREPLACE":
                placement_aligned = False
                skipped = log_text.find(
                    f"[{block_id}] VISUAL_PLACE_CORRECTION_SKIPPED", cursor + 1
                )
                descent = log_text.find(
                    f"[{block_id}] DESCEND_TO_TARGET", cursor + 1
                )
                if skipped >= 0 and (descent < 0 or skipped < descent):
                    cursor = skipped
                else:
                    position = log_text.find("Perception completed", cursor + 1)
                    if position < 0:
                        raise EndToEndFailure(
                            f"execution log is missing placement perception for {block_id}"
                        )
                    cursor = position
                    visual_source = log_text.find(
                        f"[{block_id}] VISUAL_PLACE_SOURCE", cursor + 1
                    )
                    skipped = log_text.find(
                        f"[{block_id}] VISUAL_PLACE_CORRECTION_SKIPPED", cursor + 1
                    )
                    candidates = [
                        value for value in (visual_source, skipped) if value >= 0
                    ]
                    if not candidates:
                        raise EndToEndFailure(
                            f"execution log has no placement-vision outcome for {block_id}"
                        )
                    cursor = min(candidates)
                    placement_aligned = cursor == visual_source
                if placement_aligned:
                    placement_marker = f"[{block_id}] VISUAL_ALIGN_CARRIED_BLOCK"
                    position = log_text.find(placement_marker, cursor + 1)
                    if position < 0:
                        raise EndToEndFailure(
                            f"execution log is missing {placement_marker} for {block_id}"
                        )
                    cursor = position
    if log_text.find("Benchmark result:", cursor + 1) < 0:
        raise EndToEndFailure("execution log is missing the final benchmark result")


def wait_for_final_log(process: subprocess.Popen, path: Path, timeout_s: float = 8.0) -> str:
    """Wait briefly for the executor's final message after result.json is flushed."""
    deadline = time.monotonic() + timeout_s
    text = ""
    while time.monotonic() < deadline:
        text = path.read_text(encoding="utf-8", errors="replace")
        if "Benchmark result:" in text:
            return text
        if process.poll() is not None:
            break
        time.sleep(0.1)
    return text


def validate_result(label: str, case: dict, result: dict, run_dir: Path,
                    block_ids: list[str]) -> None:
    block_count = len(block_ids)
    expected = {
        "success": True,
        "placed": block_count,
        "total": block_count,
        "completion": 1.0,
        "timeout": False,
        "collision_count": 0,
        "stability_violations": 0,
        "observation_mode": case["observation_mode"],
        "connection_mode": "physics",
        "grasp_stable": True,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise EndToEndFailure(
                f"{label}: expected {key}={value!r}, received {result.get(key)!r}"
            )

    blocks = result.get("blocks", [])
    if len(blocks) != block_count:
        raise EndToEndFailure(
            f"{label}: expected {block_count} scored blocks, received {len(blocks)}"
        )
    scored = {str(block.get("id")): block for block in blocks}
    for block_id in block_ids:
        block = scored.get(block_id)
        if block is None or block.get("success") is not True:
            raise EndToEndFailure(f"{label}: target block {block_id} was not placed")
        if float(block.get("xy_error_m", float("inf"))) > MAX_XY_ERROR_M:
            raise EndToEndFailure(
                f"{label}: {block_id} final XY error exceeded 6 mm"
            )
        if float(block.get("z_error_m", float("inf"))) > MAX_Z_ERROR_M:
            raise EndToEndFailure(
                f"{label}: {block_id} final Z error exceeded 4 mm"
            )
        if float(block.get("yaw_error_deg", float("inf"))) > MAX_YAW_ERROR_DEG:
            raise EndToEndFailure(f"{label}: {block_id} final yaw error exceeded 8 degrees")

    slip_by_block = result.get("grasp_slip_by_block", {})
    for block_id in block_ids:
        slip = slip_by_block.get(block_id)
        if slip is None:
            raise EndToEndFailure(f"{label}: no grasp-slip record for {block_id}")
        if float(slip.get("translation_m", float("inf"))) > MAX_TRANSLATION_SLIP_M:
            raise EndToEndFailure(f"{label}: {block_id} grasp translation exceeded 2 mm")
        if float(slip.get("rotation_deg", float("inf"))) > MAX_ROTATION_SLIP_DEG:
            raise EndToEndFailure(f"{label}: {block_id} grasp rotation exceeded 3 degrees")

    required_files = (
        "product.normalized.yaml",
        "episode_manifest.yaml",
        "execution_plan.yaml",
        "scene.xml",
        "actual_state.json",
        "result.json",
    )
    missing = [name for name in required_files if not (run_dir / name).is_file()]
    if missing:
        raise EndToEndFailure(f"{label}: missing run artifacts: {', '.join(missing)}")

    if case["observation_mode"] == "rgbd":
        vision_json = run_dir / "vision" / "latest.json"
        vision_image = run_dir / "vision" / "latest.png"
        if not vision_json.is_file() or not vision_image.is_file():
            raise EndToEndFailure(f"{label}: annotated RGB-D outputs were not written")
        captures = list((run_dir / "vision").glob("detection-*.json"))
        source_captures = []
        for path in captures:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("region") == "source":
                source_captures.append(payload)
        if len(source_captures) < block_count:
            raise EndToEndFailure(
                f"{label}: expected {block_count} source captures, "
                f"received {len(source_captures)}"
            )
        if any(
            payload.get("backend") != "rgbd_geometry"
            or not payload.get("detections")
            for payload in source_captures
        ):
            raise EndToEndFailure(f"{label}: a source RGB-D capture has no detections")


def run_case(repo: Path, session_dir: Path, case_name: str, product_name: str,
             timeout_s: float, domain_id: int) -> dict:
    case = CASES[case_name]
    label = f"{product_name}-{case_name}"
    run_dir = session_dir / label
    run_dir.mkdir(parents=True)
    log_path = run_dir / "launch.log"
    command = [
        str(repo / case["script"]),
        "--product", str(repo / PRODUCTS[product_name]),
        "--seed", "42",
        "--output-dir", str(run_dir),
        "--connection-mode", "physics",
        "--headless",
        *case["arguments"],
    ]
    environment = os.environ.copy()
    environment.update({
        "ROS_DOMAIN_ID": str(domain_id),
        "RCUTILS_COLORIZED_OUTPUT": "0",
        "PYTHONUNBUFFERED": "1",
    })

    started = time.monotonic()
    process = None
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=repo,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
            result = wait_for_result(process, run_dir / "result.json", timeout_s)
        log_text = wait_for_final_log(process, log_path)
        block_ids = execution_order(run_dir)
        assert_state_order(log_text, block_ids, case_name == "geometry")
        validate_result(label, case, result, run_dir, block_ids)
        return {
            "case": case_name,
            "product": product_name,
            "label": label,
            "passed": True,
            "wall_time_s": round(time.monotonic() - started, 3),
            "run_dir": str(run_dir),
            "log": str(log_path),
            "result": result,
        }
    except Exception as error:
        raise EndToEndFailure(
            f"{label} failed: {error}\n\nLast launch output:\n{tail(log_path)}"
        ) from error
    finally:
        if process is not None:
            stop_process_group(process)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("all", *CASES),
        default="all",
        help="episode to run (default: both)",
    )
    parser.add_argument(
        "--product",
        choices=("all", *PRODUCTS),
        default="all",
        help="product to assemble (default: all products)",
    )
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--domain-id", type=int)
    parser.add_argument("--output-root", default="runs/e2e")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    arguments = parse_arguments(argv)
    if arguments.timeout_s <= 0:
        raise SystemExit("--timeout-s must be positive")
    repo = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session_dir = (repo / arguments.output_root / timestamp).resolve()
    suffix = 1
    while session_dir.exists():
        session_dir = session_dir.with_name(f"{timestamp}-{suffix}")
        suffix += 1
    session_dir.mkdir(parents=True)

    domain_id = arguments.domain_id
    if domain_id is None:
        domain_id = 100 + os.getpid() % 100
    if not 0 <= domain_id <= 232:
        raise SystemExit("--domain-id must be between 0 and 232")

    selected_cases = list(CASES) if arguments.case == "all" else [arguments.case]
    selected_products = (
        list(PRODUCTS) if arguments.product == "all" else [arguments.product]
    )
    selected = [
        (case_name, product_name)
        for product_name in selected_products
        for case_name in selected_cases
    ]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository": str(repo),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "ros_domain_id": domain_id,
        "cases": [],
    }
    print(f"E2E output: {session_dir}", flush=True)
    print(f"ROS_DOMAIN_ID: {domain_id}", flush=True)
    summary_path = session_dir / "e2e_summary.json"
    for index, (case_name, product_name) in enumerate(selected):
        label = f"{product_name}-{case_name}"
        case_domain_id = (domain_id + index) % 233
        print(f"[{label}] starting (ROS_DOMAIN_ID={case_domain_id})", flush=True)
        try:
            case_report = run_case(
                repo, session_dir, case_name, product_name,
                arguments.timeout_s, case_domain_id
            )
        except EndToEndFailure as error:
            report["failure"] = {"case": label, "message": str(error)}
            summary_path.write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            raise
        report["cases"].append(case_report)
        result = case_report["result"]
        worst_xy = max(block["xy_error_m"] for block in result["blocks"])
        print(
            f"[{label}] passed in {case_report['wall_time_s']:.1f}s: "
            f"blocks={result['placed']}/{result['total']}, "
            f"worst_xy={worst_xy * 1000:.2f}mm, "
            f"slip={result['max_grasp_translation_slip_m'] * 1000:.2f}mm",
            flush=True,
        )

    summary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"All {len(selected)} end-to-end case(s) passed")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EndToEndFailure as error:
        print(error, file=sys.stderr)
        raise SystemExit(1)
