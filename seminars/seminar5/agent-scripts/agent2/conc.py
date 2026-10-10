import json,numpy as np,collections
a=json.load(open('/data/tmp/ds-yolo/seminar5/inputs/dumps/instances_val2017.json'))
c=collections.Counter(x['image_id'] for x in a['annotations'] if not x['iscrowd']); n=np.array(sorted([c.get(i['id'],0) for i in a['images']],reverse=True))
cs=np.array([sum(1 for x in a['annotations'] if not x['iscrowd'] and x['area']<32**2)])
for s in (0.16,0.31,0.42): k=int(round(s*len(n))); print(f'share {s}: top images by GT count hold {n[:k].sum()/n.sum():.3f} of instances')
