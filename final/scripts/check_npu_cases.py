"""File-driven NPU correctness checks; reference computation is test-only."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from RotaryPosEmb import impl

ROOT = Path(__file__).resolve().parents[1]


def compare_output(actual, expected):
    finite = np.isfinite(expected) & np.isfinite(actual)
    special_equal = ((np.isnan(expected) & np.isnan(actual)) |
                     (np.isposinf(expected) & np.isposinf(actual)) |
                     (np.isneginf(expected) & np.isneginf(actual)))
    categories_ok = bool(np.all(finite | special_equal))
    a, g = actual[finite].astype(np.float64), expected[finite].astype(np.float64)
    diff = np.abs(a - g)
    relative = diff / np.maximum(np.abs(g), np.finfo(np.float64).tiny)
    is_half = actual.dtype == np.float16
    atol, rtol, limit = (1.95e-3, 1.95e-3, 1e-1) if is_half else (9.77e-4, 1.53e-5, 1e-2)
    ratio = float((np.count_nonzero(diff <= atol + rtol * np.abs(g)) +
                   np.count_nonzero(special_equal)) / actual.size)
    max_abs = float(diff.max()) if diff.size else 0.0
    threshold = 1e-3 if is_half else 1e-4
    strict_abs = categories_ok and bool(np.all(diff < threshold))
    strict_rel = categories_ok and bool(np.all(relative < threshold))
    return {
        "nan_inf_categories_and_signs_equal": categories_ok,
        "special_category_mismatches": int(np.count_nonzero(~(finite | special_equal))),
        "table_rule": {"matched_ratio": ratio, "finite_max_abs_error": max_abs,
                       "pass": categories_ok and ratio >= 0.99 and max_abs <= limit},
        "second_description": {"all_abs_lt_threshold": strict_abs,
                               "all_relative_lt_threshold": strict_rel,
                               "all_both": strict_abs and strict_rel},
        "template_isclose": bool(np.all(np.isclose(actual, expected, rtol=threshold,
                                                   atol=threshold, equal_nan=True))),
    }


def make_input(dim, heads, tokens, dtype, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, (1, tokens, heads, dim)).astype(dtype)
    positions = rng.integers(0, 131073, (1, tokens), dtype=np.int32)
    positions[0, :5] = [0, 131072, 1, 8192, 0]
    positions[0, -2:] = [131071, 1]
    word = np.uint16 if dtype == np.float16 else np.uint32
    bits = ([0x0000, 0x8000, 0x7e11, 0xfe22, 0x7c00, 0xfc00, 0x7bff, 0xfbff]
            if dtype == np.float16 else
            [0x00000000, 0x80000000, 0x7fc00011, 0xffc00022,
             0x7f800000, 0xff800000, 0x477fe000, 0xc77fe000])
    x[0, 0, :, :8] = np.array(bits, dtype=word).view(dtype)
    x[0, 1, 0, :4] = [np.nan, np.inf, -np.inf, -0.0]
    x[0, 2, -1, 0] = 65504
    x[0, 2, -1, dim // 2] = 65504
    return x, positions


def run_runner(command, log_path, output_path, expected_bytes, timeout):
    if output_path.exists():
        output_path.unlink()
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"NPU runner failed with code {result.returncode}; see {log_path}")
    if not output_path.is_file() or output_path.stat().st_size != expected_bytes:
        raise RuntimeError(f"Missing or wrong-sized output file: {output_path}")
    return result.returncode


def bitwise_comparison(actual, baseline):
    word = np.uint16 if actual.dtype == np.float16 else np.uint32
    unequal = actual.view(word) != baseline.view(word)
    actual_nan, baseline_nan = np.isnan(actual), np.isnan(baseline)
    both_nan = actual_nan & baseline_nan
    neither_nan = ~actual_nan & ~baseline_nan
    both_finite = np.isfinite(actual) & np.isfinite(baseline)
    nan_categories_equal = bool(np.array_equal(actual_nan, baseline_nan))
    non_nan_mismatches = int(np.count_nonzero(unequal & neither_nan))
    return {
        "bitwise_equal": not bool(np.any(unequal)),
        "bitwise_mismatches": int(np.count_nonzero(unequal)),
        "finite_bitwise_mismatches": int(np.count_nonzero(unequal & both_finite)),
        "non_nan_bitwise_mismatches": non_nan_mismatches,
        "nan_categories_equal": nan_categories_equal,
        "nan_encoding_differences": int(np.count_nonzero(unequal & both_nan)),
        "non_nan_bitwise_equal_and_nan_categories_equal": nan_categories_equal and non_nan_mismatches == 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, default=ROOT / "bench/build/rotary_pos_emb_validate")
    parser.add_argument("--baseline-runner", type=Path, help="Optional saved NPU binary for the same input files")
    parser.add_argument("--output", type=Path, default=ROOT / "bench/npu-validation")
    parser.add_argument("--timeout", type=float, default=45.0, help="Per-process timeout in seconds")
    args = parser.parse_args()
    runner = args.runner.resolve()
    if not runner.is_file():
        parser.error(f"Build the NPU runner first: {runner}")
    baseline_runner = args.baseline_runner.resolve() if args.baseline_runner else None
    if baseline_runner is not None and not baseline_runner.is_file():
        parser.error(f"Missing baseline NPU runner: {baseline_runner}")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [(dim, heads, tokens, dtype)
             for dim in (18, 24, 30, 48, 80, 96, 130, 1022)
             for dtype in (np.float16, np.float32)
             for heads, tokens in ((7, 49), (9, 97))]
    cases += [(dim, heads, tokens, dtype)
              for dim in (24, 130) for dtype in (np.float16, np.float32)
              for heads, tokens in ((8, 49), (17, 97))]
    cases += [(dim, heads, tokens, dtype)
              for dim in (64, 128, 1024) for dtype in (np.float16, np.float32)
              for heads, tokens in ((8, 49), (9, 97))]
    report = {
        "kernel_sha256_lf": hashlib.sha256((ROOT / "kernel.asc").read_text().encode()).hexdigest(),
        "runner": str(runner), "runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
        "reference": "scripts/RotaryPosEmb.py impl; position-zero rows replaced with input per SPEC",
        "reference_sha256_lf": hashlib.sha256((ROOT / "scripts/RotaryPosEmb.py").read_text().encode()).hexdigest(),
        "cases_planned": len(cases), "cases": [],
        "case_suite": "Original 40 nonaligned cases, then 12 aligned D=64/128/1024 cases; earlier case IDs and seeds unchanged",
        "scope": "Independent local NPU correctness check. CPU golden is test-only; no performance timing or CANNJudge result. The two precision descriptions are reported separately.",
    }
    if baseline_runner is not None:
        report["baseline_runner"] = str(baseline_runner)
        report["baseline_runner_sha256"] = hashlib.sha256(baseline_runner.read_bytes()).hexdigest()
    runtime_failed = False
    for index, (dim, heads, tokens, dtype) in enumerate(cases):
        case_dir = args.output / f"case-{index:02d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        x, positions = make_input(dim, heads, tokens, dtype, 2026100200 + index)
        x_file, p_file, y_file = [case_dir / name for name in ("x.bin", "positions.bin", "y.bin")]
        x.tofile(x_file)
        positions.tofile(p_file)
        command = [str(runner), "1", str(tokens), str(heads), str(dim),
                   "fp16" if dtype == np.float16 else "fp32", "10000",
                   str(x_file.resolve()), str(p_file.resolve()), str(y_file.resolve())]
        item = {"id": index, "shape": list(x.shape), "dtype": str(np.dtype(dtype)),
                "theta": 10000.0, "seed": 2026100200 + index, "command": command}
        current_log = case_dir / "runner.log"
        try:
            item["returncode"] = run_runner(command, current_log, y_file, x.nbytes, args.timeout)
            actual = np.fromfile(y_file, dtype=dtype).reshape(x.shape)
            with np.errstate(invalid="ignore", over="ignore"):
                expected = impl(x, positions, np.float32(10000.0))
            zero = positions == 0
            expected[zero] = x[zero]
            item["position_zero_bitwise_equal"] = actual[zero].tobytes() == x[zero].tobytes()
            item.update(compare_output(actual, expected))
            if baseline_runner is not None:
                baseline_file = case_dir / "y-baseline.bin"
                baseline_command = [str(baseline_runner)] + command[1:-1] + [str(baseline_file.resolve())]
                current_log = case_dir / "baseline-runner.log"
                baseline_return = run_runner(baseline_command, current_log, baseline_file, x.nbytes, args.timeout)
                baseline = np.fromfile(baseline_file, dtype=dtype).reshape(x.shape)
                item["baseline"] = {"command": baseline_command, "returncode": baseline_return,
                                    "position_zero_bitwise_equal": baseline[zero].tobytes() == x[zero].tobytes(),
                                    **compare_output(baseline, expected)}
                item["vs_baseline"] = bitwise_comparison(actual, baseline)
        except subprocess.TimeoutExpired as error:
            item["error"] = f"Runner exceeded {args.timeout} seconds"
            output = (error.stdout or b"") + (error.stderr or b"")
            current_log.write_bytes(output)
            runtime_failed = True
        except (OSError, RuntimeError) as error:
            item["error"] = str(error)
            runtime_failed = True
        report["cases"].append(item)
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"case {index:02d} {item['shape']} {item['dtype']}: "
              f"{item.get('error', 'table=' + str(item.get('table_rule', {}).get('pass')) + ', strict=' + str(item.get('second_description', {}).get('all_both')))}",
              flush=True)
        if runtime_failed:
            break  # A device timeout/error needs investigation before another launch.
    checked = report["cases"]
    complete = len(checked) == len(cases) and not runtime_failed
    report["summary"] = {
        "runtime_complete": complete,
        "position_zero_and_special_values_pass": complete and all(
            item.get("position_zero_bitwise_equal") and item.get("nan_inf_categories_and_signs_equal") for item in checked),
        "table_rule_pass": complete and all(item.get("table_rule", {}).get("pass") for item in checked),
        "second_description_all_both_pass": complete and all(item.get("second_description", {}).get("all_both") for item in checked),
        "template_isclose_pass": complete and all(item.get("template_isclose") for item in checked),
    }
    if baseline_runner is not None:
        report["summary"]["npu_differential_bitwise_equal"] = complete and all(
            item.get("vs_baseline", {}).get("bitwise_equal") for item in checked)
        report["summary"]["npu_differential_non_nan_equal"] = complete and all(
            item.get("vs_baseline", {}).get("non_nan_bitwise_equal_and_nan_categories_equal") for item in checked)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Report: {args.output / 'report.json'}")
    # An exit of 1 can mean only a strict local precision description failed.
    # Inspect each summary field; this script never claims a platform pass.
    return 0 if all(report["summary"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
