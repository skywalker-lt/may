import os,sys,json; os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import croproute as C, contextlib, io
from pycocotools.cocoeval import COCOeval
f=sys.argv[1]; d=json.load(open(f)); sub=d['sub']; S=set(sub)
N=C.load(C.D+"dump_yolo26n_coco.json"); R=C.rects(N,0.05)
Nout=[r for i in sub for r in N.get(i,[]) if not C.inside(r,R[i])]
dense=d['dense']; crop=d['crop']
filt=[r for r in dense if C.inside(r,R[r['image_id']])]
cropin=[r for r in crop if C.inside(r,R[r['image_id']])]
def ev(dets,lab):
    with contextlib.redirect_stdout(io.StringIO()):
        E=COCOeval(C.coco,C.coco.loadRes(dets),"bbox"); E.params.imgIds=sub; E.evaluate(); E.accumulate(); E.summarize()
    s=E.stats; print(f"SUBSET {lab}: AP={s[0]:.4f} AP50={s[1]:.4f} AP75={s[2]:.4f} S/M/L={s[3]:.4f}/{s[4]:.4f}/{s[5]:.4f} n={len(dets)}",flush=True)
ev(dense,"dense (CPU fp32, single-label top300)")
ev(filt+Nout,"filter approx + N outside")
ev(crop+Nout,"real crop (all crop dets) + N outside")
ev(cropin+Nout,"real crop (centre in rect) + N outside")
print("pixel ratio",d['px'][1]/d['px'][0])
