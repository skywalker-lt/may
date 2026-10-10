"""CPU sanity (onnxruntime, 1 thread): (1) wrapper base and 'off' stubs reproduce the reference yolo26m.onnx output on
real val2017 images; (2) gather and onehot lowerings give the same output; (3) CPU median ms (not a device number)."""
import sys, glob, time, numpy as np, onnxruntime as ort, cv2
so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
def sess(f): return ort.InferenceSession(f, so, providers=['CPUExecutionProvider'])
def letterbox(p):
    im = cv2.imread(p); h, w = im.shape[:2]; r = min(640/h, 640/w); nh, nw = round(h*r), round(w*r)
    im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR); out = np.full((640, 640, 3), 114, np.uint8)
    t, l = (640-nh)//2, (640-nw)//2; out[t:t+nh, l:l+nw] = im
    return np.ascontiguousarray(out[..., ::-1].transpose(2, 0, 1)[None], np.float32) / 255

def main():
  ims = [letterbox(p) for p in sorted(glob.glob('/data/datasets/coco/images/val2017/*.jpg'))[:4]]
  S = {n: sess(f) for n, f in [('ref', '/data/tmp/l4-row0/onnx/yolo26m.onnx'), ('base', 'sim_base.onnx'),
       ('goff', 'sim_gather_off.onnx'), ('ooff', 'sim_onehot_off.onnx'), ('g', 'sim_gather.onnx'), ('o', 'sim_onehot.onnx'),
       ('gh', 'sim_gather_halo2.onnx')]}
  run = lambda n, x: S[n].run(None, {'images': x})[0]
  for j, x in enumerate(ims):
      r = run('ref', x); out = {n: run(n, x) for n in S}
      msg = ' '.join(f'{n}:{np.abs(out[n]-r).max():.2e}' for n in ['base', 'goff', 'ooff'])
      print(f'img{j} max|diff| vs ref  {msg}  | gather vs onehot {np.abs(out["g"]-out["o"]).max():.2e}'
            f' | stub top-300 rows from expert (score!=M): g {int((np.abs(out["g"]-r).sum(-1)>1e-3).sum())}')
  x = ims[0]
  for n in S:
      for _ in range(3): run(n, x)
      t = []
      for _ in range(15): t0 = time.perf_counter(); run(n, x); t.append(time.perf_counter() - t0)
      print(f'CPU-1thread median {n}: {1000*np.median(t):.1f} ms')

if __name__ == '__main__': main()
