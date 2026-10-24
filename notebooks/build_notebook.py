"""Generates notebooks/run_pipeline.ipynb (keeps the notebook diff-able as Python).

  python notebooks/build_notebook.py
"""

import json
from pathlib import Path

cells = []


def md(text):
  cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
  cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                "source": text.strip("\n").splitlines(keepends=True)})


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
REPO_URL = "https://github.com/kaustubhadhe1206/Imitation-Learning-Panda-Pick.git"   # never put a token here
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
# ---- 1. Clone / pull (public repo: no token needed) -----------------------------------------
# A token is OPTIONAL and only used to push results (CSVs, figures, videos) back to GitHub, so
# they show up in the repo without downloading them from Drive. Press Enter to skip it.
# If used: fine-grained token for this one repo, permission "Contents: Read and write".
from getpass import getpass
if "GH_TOKEN" not in os.environ:
    os.environ["GH_TOKEN"] = getpass("GitHub token for pushing results (Enter to skip): ").strip()
TOKEN = os.environ["GH_TOKEN"]
CAN_PUSH = bool(TOKEN)
if CAN_PUSH:
    os.environ["HW2_PUSH_URL"] = REPO_URL.replace("https://", f"https://x-access-token:{TOKEN}@")
else:
    os.environ.pop("HW2_PUSH_URL", None)               # run_sweep.py --push then skips pushing
print("GitHub push:", "enabled" if CAN_PUSH else "disabled (outputs are kept on Drive only)")

def run_git(*args, check=True):
    # Run a git command and print its output (with the token hidden).
    result = subprocess.run(["git", *args], cwd=REPO if REPO.exists() else None, capture_output=True, text=True)
    output = result.stdout + result.stderr
    if TOKEN: output = output.replace(TOKEN, "***")
    if output.strip(): print(output.strip())
    if check and result.returncode: raise RuntimeError(f"git {args[0]} failed")
    return result

if not (REPO / ".git").exists():
    run_git("clone", "-q", "-b", BRANCH, REPO_URL, str(REPO))
else:
    run_git("pull", "--ff-only", REPO_URL, BRANCH)
if CAN_PUSH:
    run_git("config", "user.name", GIT_NAME); run_git("config", "user.email", GIT_EMAIL)
os.chdir(REPO)
run_git("log", "--oneline", "-3")
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
    except Exception as error:
        print("WARNING: Google Drive could not be mounted (" + repr(error)[:200] + ").\n"
              "Outputs stay on the VM" + ("; results are still pushed to GitHub after each step." if CAN_PUSH
              else " and are LOST on disconnect. Re-run cell 1 with a GitHub token to push them."))
if STORAGE is not None:
    for folder in ["runs", "logs", "data", "results", "videos"]:
        (STORAGE / folder).mkdir(parents=True, exist_ok=True)
    runs_dir = REPO / "runs"
    if runs_dir.exists() and not runs_dir.is_symlink():   # move anything made before Drive was mounted
        shutil.copytree(runs_dir, STORAGE / "runs", dirs_exist_ok=True)
        shutil.rmtree(runs_dir)
    if not runs_dir.is_symlink():
        runs_dir.symlink_to(STORAGE / "runs", target_is_directory=True)
    os.environ["HW2_MIRROR_DIR"] = str(STORAGE)            # run_sweep.py copies each finished run here
    print("Drive storage:", STORAGE)
    saved_checkpoints = sorted(path.parent.name for path in (STORAGE / "runs").glob("*/checkpoint.pt"))
    print("existing checkpoints:", saved_checkpoints)

def copy_outputs_to_drive():
    # Copy data/, results/, videos/ from the repo into the Drive folder (does nothing without Drive).
    if STORAGE is None: return
    for folder in ["data", "results", "videos"]:
        if (REPO / folder).exists():
            shutil.copytree(REPO / folder, STORAGE / folder, dirs_exist_ok=True)
    print("saved to Drive:", STORAGE)
""")

code(r"""
# ---- 2. Install (Colab only; CARC uses the venv built by slurm/setup_carc.sh) ----------------
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
    # EGL vendor file for headless MuJoCo rendering on Colab GPUs (as in the Playground notebooks).
    egl_vendor_file = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    if not os.path.exists(egl_vendor_file):
        os.makedirs(os.path.dirname(egl_vendor_file), exist_ok=True)
        with open(egl_vendor_file, "w") as f:
            f.write('{"file_format_version" : "1.0.0", "ICD" : {"library_path" : "libEGL_nvidia.so.0"}}')
os.environ["MUJOCO_GL"] = "egl"
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"])

def run_command(command):
    # Run a pipeline command from the repo root, show its output live (and save it to a log on Drive).
    print("$", command, flush=True)
    log_file = None
    if STORAGE is not None:
        script_name = "-".join(pathlib.Path(arg).stem for arg in shlex.split(command)[1:3])
        log_file = open(STORAGE / "logs" / f"{datetime.datetime.now():%Y%m%d-%H%M%S}_{script_name}.log", "w")
        log_file.write(f"$ {command}\n")
    process = subprocess.Popen(shlex.split(command), cwd=REPO, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
        if log_file: log_file.write(line); log_file.flush()
    exit_code = process.wait()
    if log_file: log_file.close()
    if exit_code: raise RuntimeError(f"command failed ({exit_code}): {command}")

def save_and_push(paths, message):
    # Copy outputs to Drive, then (if a token was given) commit the given paths and push to GitHub.
    copy_outputs_to_drive()
    if not CAN_PUSH: return print("GitHub push skipped (no token)")
    paths = [path for path in paths if (REPO / path).exists()]
    if not paths: return print("nothing to push")
    run_git("add", "-f", *paths)
    if run_git("diff", "--cached", "--quiet", check=False).returncode:
        run_git("commit", "-q", "-m", message)
        run_git("push", "-q", os.environ["HW2_PUSH_URL"], f"HEAD:{BRANCH}")
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
`run_command("python code/smoke_test.py --checks env expert render --set expert.approach_speed=0.025")`.
""")

code(r"""
run_command("python tests/test_cpu.py")
run_command("python code/smoke_test.py")
""")

code(r"""
for video_path in ["videos/smoke/expert_smoke.mp4", "videos/smoke/expert_smoke_failure.mp4"]:
    if (REPO / video_path).exists():
        display(Markdown(f"`{video_path}`"), Video(str(REPO / video_path), embed=True, width=480))
""")

md(r"""
## Part 1: Collect and inspect demonstrations
Skipped automatically if `data/demos_baseline.npz` is already in the repo (set `FORCE = True` to recollect).
""")

code(r"""
FORCE = False
if FORCE or not (REPO / "data/demos_baseline.npz").exists():
    run_command("python code/collect_demos.py")
run_command("python code/inspect_demos.py")
for figure_name in ["fig_frames", "fig_coverage", "fig_trajectories", "fig_actions", "fig_phases"]:
    figure_path = REPO / f"results/demos/{figure_name}.png"
    if figure_path.exists(): display(Image(str(figure_path)))
grid_video = REPO / "videos/expert/expert_demos_grid.mp4"
if grid_video.exists(): display(Video(str(grid_video), embed=True, width=640))
save_and_push(["data/demos_baseline.npz", "results/demos", "videos/expert"], "demos: baseline dataset + inspection")
""")

md(r"""
## Teacher evaluation on the held-out evaluation seeds
""")

code(r"""
if not (REPO / "results/teacher_eval/eval_summary.json").exists():
    run_command("python code/evaluate.py --teacher --out-dir results/teacher_eval --video-dir videos/teacher_eval")
save_and_push(["results/teacher_eval", "videos/teacher_eval"], "results: teacher evaluation")
""")

md(r"""
## (Optional) Overfitting sanity check
Train on 5 demos and evaluate **on those same 5 seeds**. Near-perfect success here means the
data -> training -> execution path is wired correctly. Not part of the reported results.
""")

code(r"""
RUN_SANITY = True
if RUN_SANITY:
    run_command("python code/train.py --run-dir runs/sanity_n5 --set data.num_demos=5 train.num_steps=3000 eval.batch_size=5")
    run_command("python code/evaluate.py --run-dir runs/sanity_n5 --on-train-seeds --out-dir results/sanity_n5 --num-videos 1")
""")

md(r"""
## Parts 2-3: Train + evaluate the dataset-size sweep (resumable)
Runs `configs/sweep.yaml` (100 / 50 / 25 demos x seeds 0, 1, 2). Each finished run is copied to
Drive and pushed to GitHub immediately. Training writes a resumable state to Drive every 2000 steps
(`runs/<run>/train_state.pt`), and a finished run's checkpoint is never retrained. If the session dies,
reconnect, run cells 0-2 (incl. 1b), then this cell again: finished runs are skipped, and the interrupted run
continues from its last saved step.
""")

code(r"""
run_command("python code/run_sweep.py --dry-run")
run_command("python code/run_sweep.py --push")
""")

md(r"""
## Part 4: Aggregate, plot, push
""")

code(r"""
run_command("python code/analyze.py")
display(Markdown((REPO / "results/summary_table.md").read_text()))
for figure_name in ["fig_success_vs_demos", "fig_success_vs_threshold", "fig_failure_modes",
                    "fig_loss_curves", "fig_smoothness"]:
    display(Image(str(REPO / f"results/{figure_name}.png")))
save_and_push(["results", "videos"], "results: aggregate tables and figures")
""")

code(r"""
# ---- Commit any code edits made on this runtime (review the diff first!) ---------------------
run_git("status", "--short")
# save_and_push(["code", "configs"], "code: edits made on the runtime")
""")

notebook = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"provenance": []}},
    "nbformat": 4, "nbformat_minor": 5,
}
for index, cell in enumerate(notebook["cells"]):
  cell["id"] = f"cell-{index:02d}"
notebook_path = Path(__file__).with_name("run_pipeline.ipynb")
with open(notebook_path, "w", newline="\n") as f:
  json.dump(notebook, f, indent=1)
print(f"wrote {notebook_path} ({len(cells)} cells)")
