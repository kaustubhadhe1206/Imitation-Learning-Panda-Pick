"""Run the dataset-size ablation: train + evaluate every (condition, seed).

Resumable: a run whose results/runs/<name>/eval_summary.json already exists is
skipped, so after a Colab disconnect simply re-run this script. Runs are
ordered seed-major (all conditions for seed 0, then seed 1, ...), so a
partially finished sweep always contains a complete comparison.

With --push, the small result files of each finished run are committed and
pushed immediately (needs the HW2_PUSH_URL environment variable, see the
notebook), so nothing is lost if the session dies.

Usage:
  python code/run_sweep.py [--sweep configs/sweep.yaml] [--only n100_s0 n25_s0] [--push]
  python code/run_sweep.py --set train.num_steps=500 --dry-run
"""

import argparse
import shutil
import subprocess
import sys
import time

import hw2  # noqa: F401
from hw2 import utils

CODE = utils.REPO_ROOT / "code"


def git_sync(paths, message):
  import os
  url = os.environ.get("HW2_PUSH_URL")
  if not url:
    print("  (--push given but HW2_PUSH_URL is not set; skipping push)")
    return
  root = str(utils.REPO_ROOT)
  paths = [str(p) for p in paths if p.exists()]
  subprocess.run(["git", "add", "-f", *paths], cwd=root, check=True)
  if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=root).returncode == 0:
    return
  subprocess.run(["git", "commit", "-q", "-m", message], cwd=root, check=True)
  # The URL contains a token: never echo it.
  r = subprocess.run(["git", "push", "-q", url, "HEAD"], cwd=root, capture_output=True, text=True)
  print("  pushed" if r.returncode == 0 else "  push FAILED: " + r.stderr.replace(url, "<url>")[-300:])


def run(cmd):
  print("  $", " ".join(cmd), flush=True)
  subprocess.run([sys.executable, *cmd], cwd=str(utils.REPO_ROOT), check=True)


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--sweep", default="configs/sweep.yaml")
  parser.add_argument("--only", nargs="*", default=None, help="run names to include, e.g. n100_s0")
  parser.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE",
                      help="extra overrides applied to every run (e.g. train.num_steps=...)")
  parser.add_argument("--push", action="store_true")
  parser.add_argument("--dry-run", action="store_true")
  args = parser.parse_args()

  sweep = utils.load_config(args.sweep)
  plan = [(f"{c['name']}_s{seed}", c, seed) for seed in sweep["seeds"] for c in sweep["conditions"]]
  if args.only:
    plan = [p for p in plan if p[0] in args.only]

  for name, cond, seed in plan:
    res_dir = utils.REPO_ROOT / "results" / "runs" / name
    run_dir = utils.REPO_ROOT / "runs" / name
    if (res_dir / "eval_summary.json").exists():
      print(f"[skip] {name}: results exist")
      continue
    overrides = [f"data.num_demos={cond['num_demos']}", f"seed={seed}", *args.overrides]
    print(f"[run] {name}: {overrides}", flush=True)
    if args.dry_run:
      continue
    t0 = time.time()
    if not (run_dir / "checkpoint.pt").exists():
      run(["code/train.py", "--config", sweep["base_config"], "--run-dir", str(run_dir),
           "--set", *overrides])
    else:
      print("  checkpoint exists, skipping training")
    run(["code/evaluate.py", "--run-dir", str(run_dir), "--out-dir", str(res_dir),
         "--video-dir", f"videos/student/{name}"])
    for f in ["loss.csv", "train_summary.json", "config.yaml", "loss_curve.png"]:
      shutil.copy2(run_dir / f, res_dir / f)
    print(f"[done] {name} in {(time.time() - t0) / 60:.1f} min", flush=True)
    if args.push:
      git_sync([res_dir, utils.REPO_ROOT / "videos" / "student" / name], f"results: {name}")


if __name__ == "__main__":
  main()
