"""Agent 3: does the router receive gradient when the current module trains at top_k=1? And with a straight-through gate?"""
import torch
torch.set_num_threads(2); torch.manual_seed(0)
from ultralytics.nn.modules.moe.weight_bank import GateState, WeightBankRouter, BankConv2d
for k in (2, 1):
    st = GateState(num_experts=4, top_k=k); r = WeightBankRouter(16, 4, k); c = BankConv2d(16, 8, st, router=r); c.train()
    x = torch.randn(8, 16, 10, 10); c(x).pow(2).mean().backward()
    print(f'top_k={k}: router fc2 weight grad abs-sum = {r.fc2.weight.grad.abs().sum().item():.3e}; expert grads nonzero: {[bool(e.grad is not None and e.grad.abs().sum() > 0) for e in c.experts]}')
# straight-through top-1: forward one-hot, backward softmax
st = GateState(4, 1); r = WeightBankRouter(16, 4, 1); c = BankConv2d(16, 8, st, router=r); c.train()
x = torch.randn(8, 16, 10, 10); lg = r.logits(x); p = lg.softmax(1)
hard = torch.zeros_like(p).scatter(1, p.argmax(1, keepdim=True), 1.0); g = hard + p - p.detach()
bank = torch.stack(tuple(c.experts)).flatten(1); kern = (g @ bank).view(8, 8, 16)
y = torch.bmm(kern, x.flatten(2)); y.pow(2).mean().backward()
print(f'straight-through top-1: router fc2 grad abs-sum = {r.fc2.weight.grad.abs().sum().item():.3e}; forward gates one-hot: {bool((g.detach().max(1).values == 1).all())}')
