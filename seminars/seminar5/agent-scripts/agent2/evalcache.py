# Run COCOeval.evaluate() once per dump and cache evalImgs (mixtures are then exact per-image selections + accumulate()).
import os, sys, json, pickle, time, io, contextlib
os.environ['OMP_NUM_THREADS']='1'
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
D='/data/tmp/ds-yolo/seminar5/inputs/dumps'; C='/data/tmp/ds-yolo/seminar5/work/agent2/cache'
with contextlib.redirect_stdout(io.StringIO()): gt=COCO(f'{D}/instances_val2017.json')
for name in sys.argv[1:]:
    out=f'{C}/{os.path.basename(name).replace(".json","")}.pkl'
    if os.path.exists(out): print(name,'cached'); continue
    t=time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        dt=gt.loadRes(name if name.endswith('.json') else f'{D}/{name}.json'); E=COCOeval(gt,dt,'bbox'); E.evaluate(); E.accumulate(); E.summarize()
    pickle.dump({'evalImgs':E.evalImgs,'imgIds':E.params.imgIds,'catIds':E.params.catIds,'stats':E.stats},open(out,'wb'),protocol=5)
    print(name,'AP=%.4f'%E.stats[0],'S/M/L=%.4f/%.4f/%.4f'%tuple(E.stats[3:6]),'%.0fs'%(time.time()-t),flush=True)
