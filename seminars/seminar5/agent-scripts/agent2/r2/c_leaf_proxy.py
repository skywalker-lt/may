"""C' proxy for agent 3's P2 leaf: M@768's detections under 16 px (the P2 leaf's job) + M@640's detections at >= 16 px
(P3-P5 unchanged inside C). Writes a dump; evalcache.py then caches it."""
import json
D = '/data/tmp/ds-yolo/seminar5/inputs/dumps'
m = json.load(open(f'{D}/dumpml_yolo26m_coco.json')); h = json.load(open(f'{D}/dumpml_yolo26m_768_coco.json'))
s = lambda d: (max(d['bbox'][2] * d['bbox'][3], 0)) ** 0.5
out = [d for d in m if s(d) >= 16] + [d for d in h if s(d) < 16]
json.dump(out, open('/data/tmp/ds-yolo/seminar5/work/agent2/r2/cleaf_proxy.json', 'w')); print(len(out))
