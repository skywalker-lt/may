#!/usr/bin/env python3
"""Build an fp16 engine from an ONNX with the TensorRT Python API (micro graphs only; full models use the bundle CLI)."""
import sys, tensorrt as trt
src, dst = sys.argv[1], sys.argv[2]
lg = trt.Logger(trt.Logger.WARNING); b = trt.Builder(lg); net = b.create_network(0); p = trt.OnnxParser(net, lg)
assert p.parse_from_file(src), [p.get_error(i) for i in range(p.num_errors)]
cfg = b.create_builder_config(); cfg.set_flag(trt.BuilderFlag.FP16)
open(dst, "wb").write(b.build_serialized_network(net, cfg)); print("built", dst)
