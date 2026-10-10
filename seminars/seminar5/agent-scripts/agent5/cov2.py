from mix_eval import *
M=load(MODELS['M']); rng=np.random.default_rng(0); G=10; tg=tile_of(gt,G)
for thr in (0.01,0.03,0.05,0.1,0.25):
    sc=scores(f'Mdet{thr}',G,M,None,rng); r=[]
    for s in (0.10,0.16,0.25):
        k=int(round(s*100)); sel=np.argsort(-sc,1)[:,:k]; S=np.zeros((N,100),bool); np.put_along_axis(S,sel,True,1)
        r.append(f'{s:.2f}:{S[gt[:,0].astype(int),tg].mean():.3f}')
    print('Mdet',thr,' '.join(r))
