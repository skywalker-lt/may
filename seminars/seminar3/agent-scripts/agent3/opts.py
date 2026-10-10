"""Agent 3: option-level MACs/params from recs.json (written by flops.py)."""
import json
R = json.load(open('recs.json'))
tot = sum(r[7] for r in R); P = 20411132
r11 = [r for r in R if 6 <= r[0] <= 22 and r[2] == 1 and r[5] == 1]
outs = [r for r in r11 if r[1].endswith('.cv2.conv') and r[1].count('.') == 3]
print('block-output convs', len(outs), [r[1] for r in outs], 'channels', sum(r[4] for r in outs), 'out elems', sum(r[8] for r in outs))
el39 = sum(r[8] for r in r11); el8 = sum(r[8] for r in outs); ch39 = sum(r[4] for r in r11); ch8 = sum(r[4] for r in outs)
w39 = sum(r[3] * r[4] for r in r11)
stemP, midP, detP, rtr = 3580416, 15373696, 1457020, 33092
def row(n, held, read, macs): print(f'{n:34s} held {held/1e6:7.3f}M read {read/1e6:7.3f}M  MACs {macs/1e9:8.4f}G GFLOPs {2*macs/1e9:7.3f} ({macs/tot:.4f}x)  routed-param frac of read {0:.0f}')
rm = 512 * 64 + 64 * 4
row('YOLO26-M', P, P, tot)
row('A full (4x layers 6-23)', stemP + 4 * (midP + detP) + rtr, P + rtr, tot + rm)
row('A tied (4x 39 1x1 only)', P + 3 * 6959616 + rtr, P + rtr, tot + rm)
row('A P5 (4x layers 11-23)', P + 3 * 10069992 + rtr, P + rtr, tot + rm)
row('A6 tied (6 pair branches)', P + 5 * 6959616 + rtr, P + rtr, tot + rm)
row('B scale+shift 8 outs', P + 4 * 2 * ch8 + rtr, P + 2 * ch8 + rtr, tot + rm + el8)
row('C shift 39', P + 4 * ch39 + rtr, P + ch39 + rtr, tot + rm)
row('C scale+shift 39', P + 4 * 2 * ch39 + rtr, P + 2 * ch39 + rtr, tot + rm + el39)
lr = sum(16 * (r[3] + r[4]) for r in r11); lrm = sum(r[8] / r[4] * 16 * (r[3] + r[4]) for r in r11)
row('C rank-16 39', P + 4 * lr + rtr, P + lr + rtr, tot + rm + lrm)
row('D top-1', P + 3 * w39 + rtr, P + rtr, tot + rm)
row('D top-2', P + 3 * w39 + rtr, P + w39 + rtr, tot + rm + 2 * w39)
row('D soft', P + 3 * w39 + rtr, P + 3 * w39 + rtr, tot + rm + 4 * w39)
print('routed values per expert: B', 2 * ch8, 'C shift', ch39, 'C s+s', 2 * ch39, 'C r16', lr, 'D/A-tied', w39, 'A full', midP + detP, 'A P5', 10069992)
print('1x1 MAC share', sum(r[7] for r in r11) / tot, 'layers11-23 params', 1640448+541696+590080+1509376+2359808+1971584+1457020)
for L, n in [(5.181, 'A'), (5.721, 'B'), (5.922, 'C shift'), (6.044, 'C s+s'), (6.759, 'C r16'), (5.953, 'D top1'), (6.130, 'D top2')]:
    print(n, L, 'F(L)=%.4f bar=%.4f' % (0.5261 + 0.0102 * (L - 5.36), 0.5291 + 0.0102 * (L - 5.36)))
