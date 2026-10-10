"""Agent 3 (seminar 5) R0: train one branch on the FROZEN public YOLO26-M trunk, with the fork's stage-2 recipe
(scripts/ds_yolo/recipe_yolo26m_stage2.yaml, the same fine-tune settings as the seminar-4 receipts: warm-up off).
The branch checkpoints come from build_branches.py; they carry the public head widths, which the stock trainer would
lose by rebuilding the model from YAML, so BranchTrainer.get_model returns the checkpoint's own module (fresh or resumed).
usage (pod): PYTHONPATH=/data/YOLO-Master python train_branch.py --init <b_init.pt|c_init.pt|yolo26m.pt> --name <run>
             --freeze <N> [--epochs 10] [--project runs_agent3] [--batch 64] [--extra k=v ...]
  B:  --init b_init.pt     --freeze 14   (trains public 19-23 as 14-18: P4 out, P5 path, P4/P5 heads)
  A*: --init yolo26m.pt    --freeze 18   (control: the same layers retrained WITH P3, same recipe, same epochs)
  C:  --init c_init.pt     --freeze 23   (trains the P2 leaf 23-25 and the whole four-level Detect)
Backups to /training_data/<project>/<name>/ every --every epochs; resumes from local last.pt, else from the backup."""
import argparse, os, shutil, sys
from pathlib import Path
import torch, yaml
sys.path.insert(0, "/data/YOLO-Master/scripts/ds_yolo")
from train_80ep import backup_run, RECIPE  # noqa: E402  (same backup rules as the 80-epoch runs)
import train_80ep  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.models.yolo.detect import DetectionTrainer  # noqa: E402

class BranchTrainer(DetectionTrainer):
    if os.environ.get("SMOKE"):  # CPU smoke env without polars
        def read_results_csv(self): return {}
    INIT = None
    def get_model(self, cfg=None, weights=None, verbose=True):
        if not isinstance(weights, torch.nn.Module):
            ck = torch.load(self.INIT, map_location="cpu", weights_only=False); weights = ck.get("ema") or ck["model"]
        m = weights.float()
        for p in m.parameters(): p.requires_grad_(True)
        return m

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--freeze", type=int, required=True); ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--every", type=int, default=5); ap.add_argument("--project", default="runs_agent3")
    ap.add_argument("--data", default="/data/datasets/coco/coco.yaml"); ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0"); ap.add_argument("--cache", default="disk")
    ap.add_argument("--mem-fraction", type=float, default=0.0); ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()
    LOCAL, BACKUP = ((Path(a.project), Path(a.project) / "_backup") if a.project.startswith("/")
                     else (Path("/data") / a.project, Path("/training_data") / a.project))
    train_80ep.BACKUP = BACKUP
    if a.mem_fraction > 0: torch.cuda.set_per_process_memory_fraction(a.mem_fraction, 0)
    BranchTrainer.INIT = a.init
    run_dir = LOCAL / a.name; local_last = run_dir / "weights" / "last.pt"; backup_last = BACKUP / a.name / "weights" / "last.pt"
    if not local_last.exists() and backup_last.exists():
        shutil.rmtree(run_dir, ignore_errors=True); shutil.copytree(BACKUP / a.name, run_dir, copy_function=shutil.copyfile)
        print(f"RESTORED {a.name}", flush=True)
    recipe = yaml.safe_load(open(RECIPE)); recipe["weight_decay"] = recipe["weight_decay"] * 128 / a.batch
    args = dict(data=a.data, epochs=a.epochs, batch=a.batch, imgsz=a.imgsz, cache=a.cache, workers=a.workers,
                device=a.device, project=str(LOCAL), name=a.name, exist_ok=True, seed=0, plots=False, **recipe)
    args.update(warmup_epochs=0, warmup_bias_lr=0.00038, close_mosaic=3)   # the seminar-4 fine-tune settings
    args["freeze"] = list(range(a.freeze))
    for kv in a.extra:
        k, v = kv.split("=", 1); args[k] = yaml.safe_load(v)
    def on_fit_epoch_end(tr):
        if (tr.epoch + 1) % a.every == 0: backup_run(Path(tr.save_dir), a.name, f"epoch {tr.epoch + 1}")
    def on_train_batch_start(tr):   # frozen layers keep their batch-norm statistics
        for layer in list(tr.model.model)[: a.freeze]: layer.eval()
    def on_train_start(tr):
        n_tr = sum(p.numel() for p in tr.model.parameters() if p.requires_grad)
        print(f"BRANCH {a.name}: trainable {n_tr:,} params, frozen layers 0-{a.freeze - 1}, Detect nl={tr.model.model[-1].nl}, "
              f"strides {tr.model.model[-1].stride.tolist()}", flush=True)
    if local_last.exists():
        model = YOLO(str(local_last)); train_args = {"resume": True, "cache": a.cache}
    else:
        model = YOLO(a.init); train_args = args
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end); model.add_callback("on_train_batch_start", on_train_batch_start)
    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_end", lambda tr: backup_run(Path(tr.save_dir), a.name, "final"))
    model.train(trainer=BranchTrainer, **train_args)
