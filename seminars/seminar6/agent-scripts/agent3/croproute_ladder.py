import os,sys,json; os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import croproute as C
N=C.load(C.D+"dump_yolo26n_coco.json"); R=C.rects(N,0.05); res=[]
for S in (512,576):
    d,c=C.cost('l',S,R); print(f"L@{S}: dense {d.mean():.2f} crop {c.mean():.2f}",flush=True)
    Ls=C.load(C.D2+f"dumpml_yolo26l_{S}_coco.json")
    r=C.evaluate(C.compose(Ls,R,N),f"L{S} crop tau0.05 + N outside"); r['crop_gmac']=float(c.mean()); res.append(r)
json.dump(res,open("croproute_ladder.json","w"),indent=1)
