from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = TESTS_DIR.parent / "outputs"


def discover_test_scripts() -> list[Path]:
    return sorted(
        path
        for path in TESTS_DIR.glob("*_test.py")
        if path.name != Path(__file__).name
    )


def discover_datasets() -> list[Path]:
    return sorted(path for path in OUTPUTS_DIR.glob("*.csv") if path.is_file())


def run_test_script(script_path: Path, dataset_path: Path) -> tuple[int, str]:
    env = os.environ.copy()
    env["DATASET_NAME"] = dataset_path.stem
    completed = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=TESTS_DIR.parent,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = completed.stdout
    if completed.stderr:
        if output and not output.endswith("\n"):
            output += "\n"
        output += completed.stderr
    return completed.returncode, output


def write_output(script_path: Path, dataset_path: Path, output: str) -> Path:
    output_path = TESTS_DIR / script_path.stem / f"{dataset_path.stem}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output, encoding="utf-8")
    return output_path


def main() -> int:
    scripts = discover_test_scripts()
    datasets = discover_datasets()
    if not scripts:
        print("No test scripts found in tests/.")
        return 1
    if not datasets:
        print("No datasets found in outputs/.")
        return 1

    overall_status = 0
    for script_path in scripts:
        for dataset_path in datasets:
            print(f"Running {script_path.name} on {dataset_path.name}...")
            return_code, output = run_test_script(script_path, dataset_path)
            output_path = write_output(script_path, dataset_path, output)
            print(f"Wrote {output_path.relative_to(TESTS_DIR)}")
            if return_code != 0:
                overall_status = return_code
                print(f"{script_path.name} on {dataset_path.name} exited with code {return_code}")

    return overall_status


if __name__ == "__main__":

    
    raise SystemExit(main())