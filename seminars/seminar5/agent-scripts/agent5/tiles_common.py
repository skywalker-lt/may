import json, numpy as np, os
os.environ.setdefault('OMP_NUM_THREADS','1')
D='/data/tmp/ds-yolo/seminar5/inputs/dumps/'
GT=json.load(open(D+'instances_val2017.json'))
imgs={im['id']:im for im in GT['images']}
ids=sorted(imgs); idx={i:k for k,i in enumerate(ids)}; N=len(ids)
W=np.array([imgs[i]['width'] for i in ids],float); H=np.array([imgs[i]['height'] for i in ids],float)
R=np.minimum(640/W,640/H); PX=(640-np.round(W*R))/2; PY=(640-np.round(H*R))/2
def load(name):
    cache=f'/data/tmp/ds-yolo/seminar5/work/agent5/cache_{name}.npy'
    if os.path.exists(cache): return np.load(cache)
    d=json.load(open(D+name))
    a=np.array([[idx[x['image_id']],*x['bbox'],x['score'],x['category_id']] for x in d],float)
    np.save(cache,a); return a
def tile_of(a,G):
    """a: rows [imgidx,x,y,w,h,...] in original coords -> tile index on a GxG grid of the 640 letterbox"""
    i=a[:,0].astype(int); cx=(a[:,1]+a[:,3]/2)*R[i]+PX[i]; cy=(a[:,2]+a[:,4]/2)*R[i]+PY[i]
    T=640/G; tx=np.clip((cx//T).astype(int),0,G-1); ty=np.clip((cy//T).astype(int),0,G-1)
    return ty*G+tx
def gt_array():
    rows=[[idx[g['image_id']],*g['bbox'],g['area'],g['category_id'],g['iscrowd']] for g in GT['annotations']]
    return np.array(rows,float)
def content_mask(G):
    """tiles that overlap the image content (not pure letterbox padding)"""
    T=640/G; m=np.zeros((N,G*G),bool)
    for k in range(G*G):
        ty,tx=divmod(k,G); x0,x1,y0,y1=tx*T,(tx+1)*T,ty*T,(ty+1)*T
        m[:,k]=(x1>PX)&(x0<640-PX)&(y1>PY)&(y0<640-PY)
    return m
