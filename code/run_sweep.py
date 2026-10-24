"""Run the dataset-size experiment: train + evaluate every (condition, seed) pair (Part 3).

Safe to re-run: a run whose results/runs/<name>/eval_summary.json already
exists is skipped, and a run whose training was interrupted continues from its
last saved state. So after a Colab disconnect just run this script again.
Runs go seed by seed (all conditions for seed 0, then seed 1, ...), so even
an unfinished sweep has a full comparison.

With --push, each finished run's small result files are committed and pushed
right away (needs the HW2_PUSH_URL environment variable, set by the notebook).
If HW2_MIRROR_DIR is set (the notebook sets it to the Google Drive folder),
each finished run is also copied there.

Usage:
  python code/run_sweep.py [--sweep configs/sweep.yaml] [--only n100_s0 n25_s0] [--push]
  python code/run_sweep.py --set train.num_steps=500 --dry-run
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

import hw2  # noqa: F401
from hw2 import utils


def commit_and_push(paths, message):
  push_url = os.environ.get("HW2_PUSH_URL")
  if not push_url:
    print("  (--push was given but HW2_PUSH_URL is not set, so not pushing)")
    return
  repo = str(utils.REPO_ROOT)
  existing_paths = [str(path) for path in paths if path.exists()]
  subprocess.run(["git", "add", "-f", *existing_paths], cwd=repo, check=True)
  nothing_to_commit = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo).returncode == 0
  if nothing_to_commit:
    return
  subprocess.run(["git", "commit", "-q", "-m", message], cwd=repo, check=True)
  # The push URL contains the token, so never print it.
  result = subprocess.run(["git", "push", "-q", push_url, "HEAD"], cwd=repo, capture_output=True, text=True)
  if result.returncode == 0:
    print("  pushed")
  else:
    print("  push FAILED: " + result.stderr.replace(push_url, "<url>")[-300:])


def copy_run_to_mirror(run_name, results_dir):
  """Copy a finished run's results and videos to HW2_MIRROR_DIR (e.g. the Google Drive folder)."""
  mirror_dir = os.environ.get("HW2_MIRROR_DIR")
  if not mirror_dir:
    return
  shutil.copytree(results_dir, os.path.join(mirror_dir, "results", "runs", run_name), dirs_exist_ok=True)
  videos_dir = utils.REPO_ROOT / "videos" / "student" / run_name
  if videos_dir.exists():
    shutil.copytree(videos_dir, os.path.join(mirror_dir, "videos", "student", run_name), dirs_exist_ok=True)
  print(f"  copied to {mirror_dir}")


def run_python_script(arguments):
  print("  $", " ".join(arguments), flush=True)
  subprocess.run([sys.executable, *arguments], cwd=str(utils.REPO_ROOT), check=True)


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--sweep", default="configs/sweep.yaml")
  parser.add_argument("--only", nargs="*", default=None, help="only these runs, e.g. n100_s0")
  parser.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE",
                      help="extra config changes for every run (e.g. train.num_steps=...)")
  parser.add_argument("--push", action="store_true")
  parser.add_argument("--dry-run", action="store_true", help="only print what would run")
  args = parser.parse_args()

  sweep = utils.load_config(args.sweep)
  all_runs = [(f"{condition['name']}_s{seed}", condition, seed)
              for seed in sweep["seeds"] for condition in sweep["conditions"]]
  if args.only:
    all_runs = [run for run in all_runs if run[0] in args.only]

  for run_name, condition, seed in all_runs:
    results_dir = utils.REPO_ROOT / "results" / "runs" / run_name
    run_dir = utils.REPO_ROOT / "runs" / run_name
    if (results_dir / "eval_summary.json").exists():
      print(f"[skip] {run_name}: results already exist")
      continue
    run_overrides = [f"data.num_demos={condition['num_demos']}", f"seed={seed}", *args.overrides]
    print(f"[run] {run_name}: {run_overrides}", flush=True)
    if args.dry_run:
      continue

    start_time = time.time()
    if (run_dir / "checkpoint.pt").exists():
      print("  checkpoint already exists, skipping training")
    else:
      run_python_script(["code/train.py", "--config", sweep["base_config"], "--run-dir", str(run_dir),
                         "--set", *run_overrides])
    run_python_script(["code/evaluate.py", "--run-dir", str(run_dir), "--out-dir", str(results_dir),
                       "--video-dir", f"videos/student/{run_name}"])
    for file_name in ["loss.csv", "train_summary.json", "config.yaml", "loss_curve.png"]:
      shutil.copy2(run_dir / file_name, results_dir / file_name)
    print(f"[done] {run_name} in {(time.time() - start_time) / 60:.1f} min", flush=True)

    copy_run_to_mirror(run_name, results_dir)
    if args.push:
      commit_and_push([results_dir, utils.REPO_ROOT / "videos" / "student" / run_name], f"results: {run_name}")


if __name__ == "__main__":
  main()
