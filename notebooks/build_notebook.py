"""Generates notebooks/run_pipeline.ipynb (keeps the notebook diff-able as Python).

  python notebooks/build_notebook.py
"""

import json
from pathlib import Path

cells = []


def md(s):
  cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)})


def code(s):
  cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                "source": s.strip("\n").splitlines(keepends=True)})


md(r"""
# CSCI 645 HW2: Imitation learning on PandaPickCube (Diffusion Policy)

One notebook, two platforms. Run it **either** on Google Colab (T4) **or** on a CARC GPU node
(OnDemand JupyterLab with the `Python (hw2)` kernel from `slurm/setup_carc.sh`). Never mix
platforms inside one sweep, because GPU simulation differs slightly across GPU types.

Every cell only shells out to the scripts in `code/`, so the notebook does exactly what the
command-line / SLURM workflow does. Steps are **resumable**: finished results are committed to
GitHub, and re-running a cell skips work that is already done.

**Storage (Colab):** all outputs go to `My Drive/CSCI-645 Homework 2/`. That covers checkpoints and raw rollouts in `runs/`,
copies of `data/`, `results/` and `videos/`, and a log of every command in `logs/`. A disconnect therefore never
loses a trained model. Python packages and the git clone live on the VM and are reinstalled each session.

**Rules for edits:** change behaviour through `--set key=value` overrides where possible. If you
edit a `.py` file on the runtime, commit + push it (last cell) before the session ends.
""")

code(r"""
# ---- 0. Settings ----------------------------------------------------------------------------
REPO_URL = "https://github.com/kaustubhadhe1206/Imitation-Learning-Panda-Pick.git"   # private repo (no token here!)
BRANCH = "main"
GIT_NAME, GIT_EMAIL = "kashinadhe", "kaustubhadhe@gmail.com"
DRIVE_FOLDER = "CSCI-645 Homework 2"     # Colab: all outputs are stored in My Drive/<this folder>

import os, sys, subprocess, pathlib, shlex
IN_COLAB = os.path.isdir("/content") and not os.environ.get("SLURM_JOB_ID")
cwd = pathlib.Path.cwd()
if (cwd / "code" / "hw2").is_dir():
    REPO = cwd
elif (cwd.parent / "code" / "hw2").is_dir():
    REPO = cwd.parent                                  # notebook opened from notebooks/ (CARC)
else:
    REPO = pathlib.Path("/content" if IN_COLAB else pathlib.Path.home()) / pathlib.Path(REPO_URL).stem
print("platform:", "Colab" if IN_COLAB else "CARC/other", "| repo:", REPO)
""")

code(r"""
# ---- 1. GitHub access (token is typed, never saved) + clone / pull ---------------------------
# Use a fine-grained personal access token restricted to this one repo,
# permission "Contents: Read and write" (write is needed to push results back).
from getpass import getpass
if not os.environ.get("GH_TOKEN"):
    os.environ["GH_TOKEN"] = getpass("GitHub fine-grained token: ")
auth_url = REPO_URL.replace("https://", f"https://x-access-token:{os.environ['GH_TOKEN']}@")
os.environ["HW2_PUSH_URL"] = auth_url                  # used by run_sweep.py --push

def git(*args, check=True):
    r = subprocess.run(["git", *args], cwd=REPO if REPO.exists() else None, capture_output=True, text=True)
    out = (r.stdout + r.stderr).replace(os.environ["GH_TOKEN"], "***")
    if out.strip(): print(out.strip())
    if check and r.returncode: raise RuntimeError(f"git {args[0]} failed")
    return r

if not (REPO / ".git").exists():
    r = subprocess.run(["git", "clone", "-q", "-b", BRANCH, auth_url, str(REPO)], capture_output=True, text=True)
    if r.returncode: raise RuntimeError("clone failed: " + r.stderr.replace(os.environ["GH_TOKEN"], "***"))
    git("remote", "set-url", "origin", REPO_URL)       # keep the token out of .git/config
else:
    git("pull", "--ff-only", auth_url, BRANCH)
git("config", "user.name", GIT_NAME); git("config", "user.email", GIT_EMAIL)
os.chdir(REPO)
git("log", "--oneline", "-3")
""")

code(r"""
# ---- 1b. Google Drive storage (Colab only) --------------------------------------------------
# My Drive/CSCI-645 Homework 2/
#   runs/      checkpoints + raw rollouts (repo's runs/ is a symlink here, so they survive disconnects)
#   data/ results/ videos/   copies of everything the pipeline produces (refreshed after each step)
#   logs/      full console output of every pipeline command
# Packages and the git clone stay on the VM disk (Drive is too slow for imports / git).
import shutil, datetime
STORAGE = None
if IN_COLAB:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        STORAGE = pathlib.Path("/content/drive/MyDrive") / DRIVE_FOLDER
    except Exception as e:
        print("WARNING: Google Drive could not be mounted (" + repr(e)[:200] + ").\n"
              "Outputs stay on the VM; results are still pushed to GitHub after each step.")
if STORAGE is not None:
    for sub in ["runs", "logs", "data", "results", "videos"]:
        (STORAGE / sub).mkdir(parents=True, exist_ok=True)
    runs = REPO / "runs"
    if runs.exists() and not runs.is_symlink():          # move anything created before the mount
        shutil.copytree(runs, STORAGE / "runs", dirs_exist_ok=True)
        shutil.rmtree(runs)
    if not runs.is_symlink():
        runs.symlink_to(STORAGE / "runs", target_is_directory=True)
    os.environ["HW2_MIRROR_DIR"] = str(STORAGE)           # run_sweep.py copies each finished run here
    print("Drive storage:", STORAGE)
    print("existing checkpoints:", sorted(p.parent.name for p in (STORAGE / "runs").glob("*/checkpoint.pt")))

def save_to_drive():
    # Copy data/, results/, videos/ from the repo into the Drive folder (no-op without Drive).
    if STORAGE is None: return
    for sub in ["data", "results", "videos"]:
        if (REPO / sub).exists():
            shutil.copytree(REPO / sub, STORAGE / sub, dirs_exist_ok=True)
    print("saved to Drive:", STORAGE)
""")

code(r"""
# ---- 2. Install (Colab only; CARC uses the venv built by slurm/setup_carc.sh) ----------------
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
    # EGL vendor file for headless MuJoCo rendering on Colab GPUs (as in the Playground notebooks).
    icd = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    if not os.path.exists(icd):
        os.makedirs(os.path.dirname(icd), exist_ok=True)
        with open(icd, "w") as f:
            f.write('{"file_format_version" : "1.0.0", "ICD" : {"library_path" : "libEGL_nvidia.so.0"}}')
os.environ["MUJOCO_GL"] = "egl"
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"])

def sh(cmd):
    # Run a pipeline command from the repo root, streaming its output (and logging it to Drive); raise on failure.
    print("$", cmd, flush=True)
    log = None
    if STORAGE is not None:
        name = "-".join(pathlib.Path(a).stem for a in shlex.split(cmd)[1:3])
        log = open(STORAGE / "logs" / f"{datetime.datetime.now():%Y%m%d-%H%M%S}_{name}.log", "w")
        log.write(f"$ {cmd}\n")
    p = subprocess.Popen(shlex.split(cmd), cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="")
        if log: log.write(line); log.flush()
    code = p.wait()
    if log: log.close()
    if code: raise RuntimeError(f"command failed ({code}): {cmd}")

def push(paths, message):
    # Save outputs to Drive, then commit the given paths (if changed) and push to GitHub.
    save_to_drive()
    paths = [p for p in paths if (REPO / p).exists()]
    if not paths: return print("nothing to push")
    git("add", "-f", *paths)
    if git("diff", "--cached", "--quiet", check=False).returncode:
        git("commit", "-q", "-m", message)
        git("push", "-q", os.environ["HW2_PUSH_URL"], f"HEAD:{BRANCH}")
        print("pushed:", message)
    else:
        print("nothing to commit")

from IPython.display import Image, Video, display, Markdown
""")

md(r"""
## Smoke test
Versions, GPU, environment dimensions, expert success on 50 seeds, a rendered video and
training-speed estimates. **Paste the final `SMOKE TEST REPORT` block back to Claude.**
To tune the expert, re-run with overrides, e.g.
`sh("python code/smoke_test.py --sections env expert render --set expert.v_approach=0.025")`.
""")

code(r"""
sh("python tests/test_cpu.py")
sh("python code/smoke_test.py")
""")

code(r"""
for v in ["videos/smoke/expert_smoke.mp4", "videos/smoke/expert_smoke_failure.mp4"]:
    if (REPO / v).exists(): display(Markdown(f"`{v}`"), Video(str(REPO / v), embed=True, width=480))
""")

md(r"""
## Part 1: Collect and inspect demonstrations
Skipped automatically if `data/demos_baseline.npz` is already in the repo (set `FORCE = True` to recollect).
""")

code(r"""
FORCE = False
if FORCE or not (REPO / "data/demos_baseline.npz").exists():
    sh("python code/collect_demos.py")
sh("python code/inspect_demos.py")
for f in ["fig_frames", "fig_coverage", "fig_trajectories", "fig_actions", "fig_phases"]:
    p = REPO / f"results/demos/{f}.png"
    if p.exists(): display(Image(str(p)))
grid = REPO / "videos/expert/expert_demos_grid.mp4"
if grid.exists(): display(Video(str(grid), embed=True, width=640))
push(["data/demos_baseline.npz", "results/demos", "videos/expert"], "demos: baseline dataset + inspection")
""")

md(r"""
## Teacher evaluation on the held-out evaluation seeds
""")

code(r"""
if not (REPO / "results/teacher_eval/eval_summary.json").exists():
    sh("python code/evaluate.py --teacher --out-dir results/teacher_eval --video-dir videos/teacher_eval")
push(["results/teacher_eval", "videos/teacher_eval"], "results: teacher evaluation")
""")

md(r"""
## (Optional) Overfitting sanity check
Train on 5 demos and evaluate **on those same 5 seeds**. Near-perfect success here means the
data -> training -> execution path is wired correctly. Not part of the reported results.
""")

code(r"""
RUN_SANITY = True
if RUN_SANITY:
    sh("python code/train.py --run-dir runs/sanity_n5 --set data.num_demos=5 train.num_steps=3000 eval.batch_size=5")
    sh("python code/evaluate.py --run-dir runs/sanity_n5 --on-train-seeds --out-dir results/sanity_n5 --num-videos 1")
""")

md(r"""
## Parts 2-3: Train + evaluate the dataset-size sweep (resumable)
Runs `configs/sweep.yaml` (100 / 50 / 25 demos x seeds 0, 1, 2). Each finished run is copied to
Drive and pushed to GitHub immediately. Checkpoints are written straight to Drive, so a run whose training
finished is not retrained. If the session dies, reconnect, run cells 0-2 (incl. 1b), then this cell again.
""")

code(r"""
sh("python code/run_sweep.py --dry-run")
sh("python code/run_sweep.py --push")
""")

md(r"""
## Part 4: Aggregate, plot, push
""")

code(r"""
sh("python code/analyze.py")
display(Markdown((REPO / "results/summary_table.md").read_text()))
for f in ["fig_success_vs_demos", "fig_success_vs_threshold", "fig_failure_modes", "fig_loss_curves", "fig_smoothness"]:
    display(Image(str(REPO / f"results/{f}.png")))
push(["results", "videos"], "results: aggregate tables and figures")
""")

code(r"""
# ---- Commit any code edits made on this runtime (review the diff first!) ---------------------
git("status", "--short")
# push(["code", "configs"], "code: edits made on the runtime")
""")

nb = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"provenance": []}},
    "nbformat": 4, "nbformat_minor": 5,
}
for i, c in enumerate(nb["cells"]):
  c["id"] = f"cell-{i:02d}"
out = Path(__file__).with_name("run_pipeline.ipynb")
with open(out, "w", newline="\n") as f:
  json.dump(nb, f, indent=1)
print(f"wrote {out} ({len(cells)} cells)")
