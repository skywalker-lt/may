# T4 envelope bars for a query refiner added to a dense base (measured T4 points, T4-envelope.md). est. latencies.
pts=[(3.33,0.4937),(3.78,0.5063),(4.28,0.5117),(4.88,0.5262),(4.93,0.5203),(5.20,0.5234),(5.32,0.5329),(5.34,0.5261),(6.25,0.5368),(6.89,0.5417)]
def env(t):
    best=max(a for x,a in pts if x<=t)
    for (x1,a1) in pts:
        for (x2,a2) in pts:
            if x1<=t<=x2 and x2>x1: best=max(best,a1+(a2-a1)*(t-x1)/(x2-x1))
    return best
rows={'L544':(5.32,0.5329,{'L640':{16:0.5367,32:0.5380,64:0.5396,128:0.5407,300:0.5415},'X640':{16:0.5529,64:0.5628,128:0.5655,300:0.5674}},{'L640':{16:0.5343,64:0.5368},'X640':{16:0.5406,64:0.5508}}),
      'M640':(5.34,0.5261,{'L640':{32:0.5363,64:0.5394,128:0.5409,300:0.5419}},{'L640':{64:0.5345}})}
for inc in (0.20,0.28,0.40):
  for b,(t0,a0,R,Nl) in rows.items():
    t=t0+inc; bar=env(t)+0.003
    print(f'base {b} +{inc:.2f} ms -> {t:.2f} ms est.: envelope {env(t):.4f}, bar {bar:.4f}, need +{bar-a0:.4f} over base')
    for e,d in R.items():
      full=d.get(300); 
      for k,ap in sorted(d.items()):
        print(f'    exp {e} k={k:3d}: {ap:.4f} margin vs bar {ap-bar:+.4f}; retention of all-query gain {(ap-a0)/(full-a0):.2f}' + (f'; null {Nl[e][k]:.4f}, route-null {ap-Nl[e][k]:+.4f}' if k in Nl.get(e,{}) else ''))
