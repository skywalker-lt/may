# Attack on direction 10 (saturation-routed precision): CPU PyTorch fake-quant simulation of "INT8 tail" on YOLO26-M.
# NOT TensorRT. Weights per-output-channel symmetric int8 (after BN fusion); activations at every conv input of layers
# 6-23 per-tensor symmetric int8 with amax calibrated on 128 train2017 images (two rules: max, 99.99th percentile).
# Kept in float, as agent 10 specifies: layer 10 (C2PSA) and the head's final output convs.
# Per image we record: layer-5 output saturation (max|x|/amax, fraction above amax) and mean clip fraction over all
# quantised tensors; outputs a COCO json per mode for 500 random val2017 images.
import sys, os, json, glob, time, numpy as np, torch, torch.nn as nn, cv2
sys.path.insert(0,'/data/YOLO-Master'); torch.set_num_threads(1)
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
OUT='/data/tmp/ds-yolo/seminar5/work/agent9/r2'
model=YOLO('/data/yolo-quant-work/weights/yolo26m.pt').model.float().eval().fuse()
head=model.model[23]
final=set()
for seq in list(head.one2one_cv2)+list(head.one2one_cv3): final.add(id(list(seq.modules())[-1]) if not isinstance(seq,nn.Conv2d) else id(seq))
for seq in list(head.one2one_cv2)+list(head.one2one_cv3):
    last=[m for m in seq.modules() if isinstance(m,nn.Conv2d)][-1]; final.add(id(last))
Q=[]  # (name, conv)
for li in range(6,24):
    if li==10: continue
    for nm,m in model.model[li].named_modules():
        if isinstance(m,nn.Conv2d) and id(m) not in final: Q.append((f'{li}.{nm}',m))
print('quantised convs',len(Q),flush=True)
for nm,m in Q:  # per-channel weight fake quant
    w=m.weight.data; s=w.abs().amax(dim=(1,2,3),keepdim=True).clamp(min=1e-12)/127; m.weight_q=(torch.round(w/s).clamp(-127,127)*s); m.weight_f=w.clone()
state={'mode':'calib','amax':{}, 'stats':[]}
CAL={nm:{'max':0.0,'samp':[]} for nm,_ in Q}
def mk_hook(nm,m):
    def pre(mod,inp):
        x=inp[0]; mode=state['mode']
        if mode=='calib':
            a=x.abs().flatten(); CAL[nm]['max']=max(CAL[nm]['max'],a.max().item())
            CAL[nm]['samp'].append(a[torch.randint(0,a.numel(),(20000,))].numpy()); return None
        if mode=='fp': 
            mod.weight.data=mod.weight_f; 
            if state['rec']:
                am=state['amax'][state['rule']][nm]; state['clip'].append((x.abs()>am).float().mean().item())
                if nm==state['l5name']: state['l5']=(x.abs().max().item()/am,(x.abs()>am).float().mean().item())
            return None
        mod.weight.data=mod.weight_q; am=state['amax'][mode][nm]; s=am/127
        return (torch.round(x/s).clamp(-127,127)*s,)
    return pre
for nm,m in Q: m.register_forward_pre_hook(mk_hook(nm,m))
l5name=Q[0][0]; state['l5name']=l5name; print('layer-5 output tensor read at',l5name,flush=True)
lb=LetterBox((640,640),auto=False)
def prep(path):
    im0=cv2.imread(path); h0,w0=im0.shape[:2]; im=lb(image=im0)
    r=min(640/h0,640/w0); pw=(640-round(w0*r))/2; ph=(640-round(h0*r))/2
    x=torch.from_numpy(im[:,:,::-1].copy()).permute(2,0,1).float()[None]/255; return x,(r,pw,ph,w0,h0)
rng=np.random.default_rng(0)
tr=sorted(glob.glob('/data/datasets/coco/images/train2017/*.jpg')); cal=[tr[i] for i in rng.choice(len(tr),int(sys.argv[2]) if len(sys.argv)>2 else 128,replace=False)]
t=time.time()
with torch.no_grad():
    for p in cal: model(prep(p)[0])
state['amax']={'max':{nm:CAL[nm]['max'] for nm,_ in Q},'p9999':{nm:float(np.percentile(np.concatenate(CAL[nm]['samp']),99.99)) for nm,_ in Q}}
print('calib done %.0fs'%(time.time()-t),flush=True)
gt=json.load(open('/data/tmp/ds-yolo/seminar5/inputs/dumps/instances_val2017.json'))
ids=sorted(im['id'] for im in gt['images']); fn={im['id']:im['file_name'] for im in gt['images']}
sub=[ids[i] for i in sorted(rng.choice(len(ids),int(sys.argv[1]) if len(sys.argv)>1 else 500,replace=False))]
cocoid=sorted(c['id'] for c in gt['categories'])
res={k:[] for k in ('fp','max','p9999')}; stats=[]
with torch.no_grad():
    for j,iid in enumerate(sub):
        x,(r,pw,ph,w0,h0)=prep(f'/data/datasets/coco/images/val2017/{fn[iid]}')
        st={'image_id':iid}
        for mode in ('fp','max','p9999'):
            state['mode']=mode; state['rec']=(mode=='fp')
            if mode=='fp':
                state['clip']=[]; state['rule']='p9999'; y=model(x)
            else:
                y=model(x)
            y=y[0] if isinstance(y,(tuple,list)) else y
            if mode=='fp': st['l5_ratio'],st['l5_frac']=state['l5']; st['clip_mean']=float(np.mean(state['clip'])); st['clip_max']=float(np.max(state['clip']))
            d=y[0].numpy()
            for x1,y1,x2,y2,s,c in d:
                if s<0.001: continue
                X1=(x1-pw)/r; Y1=(y1-ph)/r; X2=(x2-pw)/r; Y2=(y2-ph)/r
                res[mode].append({'image_id':iid,'category_id':cocoid[int(c)],'bbox':[float(X1),float(Y1),float(X2-X1),float(Y2-Y1)],'score':float(s)})
        stats.append(st)
        if j%50==0: print(j,'%.0fs'%(time.time()-t),flush=True)
for k,v in res.items(): json.dump(v,open(f'{OUT}/fq_{k}.json','w'))
json.dump(stats,open(f'{OUT}/fq_stats.json','w')); json.dump(sub,open(f'{OUT}/fq_ids.json','w'))
print('done %.0fs'%(time.time()-t))
