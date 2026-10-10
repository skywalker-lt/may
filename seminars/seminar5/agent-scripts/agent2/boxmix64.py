# Proxy for level-allocated depth: the deep units on the P3 path mostly change small-object detections, those on the
# P4/P5 path medium and large ones. Synthesise per-image outputs that take L's boxes in one size band and M's elsewhere.
import json, os
D='/data/tmp/ds-yolo/seminar5/inputs/dumps'; W='/data/tmp/ds-yolo/seminar5/work/agent2/cache'
M=json.load(open(f'{D}/dumpml_yolo26m_coco.json')); L=json.load(open(f'{D}/dump_yolo26l_coco.json'))
T=64
small=lambda d: d['bbox'][2]*d['bbox'][3]<T**2
json.dump([d for d in L if small(d)]+[d for d in M if not small(d)],open(f'{W}/../boxmix64_Lsmall.json','w'))
json.dump([d for d in M if small(d)]+[d for d in L if not small(d)],open(f'{W}/../boxmix64_Lmedlarge.json','w'))
print('written')
