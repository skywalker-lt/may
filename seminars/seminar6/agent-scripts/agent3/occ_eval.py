import sys, json
sys.argv=[sys.argv[0]]
exec(open("occupancy.py").read().split('if __name__=="__main__":')[0])
import numpy as np
D2="/data/tmp/ds-yolo/seminar6/inputs/dumps_r2/"
M=load(D+"dumpml_yolo26m_coco.json"); N=load(D+"dump_yolo26n_coco.json"); L=load(D+"dump_yolo26l_coco.json")
L576=load(D2+"dumpml_yolo26l_576_coco.json"); L544=load(D2+"dumpml_yolo26l_544_coco.json")
print("loaded",flush=True)
R=[]
R.append(evaluate([d for i in ids for d in M.get(i,[])],"M dense"))
R.append(run(M,N,0,0.05,0,"rect","M rect tau0.05"))
R.append(run(M,N,0,0.1,0,"rect","M rect tau0.1"))
R.append(run(M,"gt",0,0,0,"rect","M rect GT (oracle)"))
R.append(run(M,N,8,0.05,1,"tile","M G8 tau0.05 halo1"))
R.append(evaluate([d for i in ids for d in L.get(i,[])],"L dense"))
R.append(run(L,N,0,0.05,0,"rect","L rect tau0.05"))
R.append(run(L,N,0,0.1,0,"rect","L rect tau0.1"))
R.append(run(L,"gt",0,0,0,"rect","L rect GT (oracle)"))
R.append(run(L576,N,0,0.05,0,"rect","L576 rect tau0.05"))
R.append(run(L544,N,0,0.05,0,"rect","L544 rect tau0.05"))
R.append(run(L,N,8,0.05,1,"tile","L G8 tau0.05 halo1"))
json.dump(R,open("occ_eval.json","w"),indent=1)
print("DONE",flush=True)
