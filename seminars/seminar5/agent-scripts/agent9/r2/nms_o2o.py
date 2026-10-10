# Attack on direction 8 with a SAME-BACKBONE head pair: YOLO26-M's shipped one-to-one output (the public multi-label dump)
# against the same output after class-wise greedy NMS (IoU thr t). If NMS changes nothing, the NMS-free head emits no
# duplicates that NMS could remove; any o2m gain would have to come from ranking, not suppression.
import json, sys, numpy as np, collections
D='/data/tmp/ds-yolo/seminar5/inputs/dumps'
dets=json.load(open(f'{D}/dumpml_yolo26m_coco.json'))
groups=collections.defaultdict(list)
for i,d in enumerate(dets): groups[(d['image_id'],d['category_id'])].append(i)
B=np.array([d['bbox'] for d in dets],float); S=np.array([d['score'] for d in dets])
def nms(idx,thr):
    idx=np.array(idx); idx=idx[np.argsort(-S[idx])]
    b=B[idx]; x1,y1=b[:,0],b[:,1]; x2,y2=x1+b[:,2],y1+b[:,3]; ar=b[:,2]*b[:,3]
    keep=[]; alive=np.ones(len(idx),bool)
    for j in range(len(idx)):
        if not alive[j]: continue
        keep.append(idx[j])
        xx1=np.maximum(x1[j],x1[j+1:]); yy1=np.maximum(y1[j],y1[j+1:]); xx2=np.minimum(x2[j],x2[j+1:]); yy2=np.minimum(y2[j],y2[j+1:])
        inter=np.clip(xx2-xx1,0,None)*np.clip(yy2-yy1,0,None); iou=inter/(ar[j]+ar[j+1:]-inter+1e-9)
        alive[j+1:]&=iou<=thr
    return keep
for thr in [float(t) for t in sys.argv[1:]]:
    keep=[]
    for k,idx in groups.items(): keep+=nms(idx,thr)
    keep=sorted(keep); print(thr,'kept',len(keep),'of',len(dets),flush=True)
    json.dump([dets[i] for i in keep],open(f'/data/tmp/ds-yolo/seminar5/work/agent9/r2/dumpml_yolo26m_nms{int(thr*100)}.json','w'))
