"""Muon burst on a tiny LLM. Model, Muon and the Muon/AdamW parameter split are imported unchanged from llm-research-kit.

Each sequence: task token, 6 letters, '|', then the 6 answer letters the model generates.
Tasks: C = copy, S = shift each letter by +1, R = reverse. Phase 1 trains C and S; phase 2 adds R at FREQ_NEW of each batch.
Accuracy = exact match of the greedy answer on fresh random sequences (= every teacher-forced argmax right).
Arms, all from the same phase-1 model: adam (AdamW on every weight), burst (BURST steps of Muon on the kit's Muon weights, then a
fresh AdamW), muon (kit's Muon + AdamW throughout).

usage: python3 tinylm.py SEED [ARM ...]      env: P1, P2, LR, MUON_LR, BURST, WARM, WD, FREQ_NEW, OUT
"""
import copy, json, os, sys, time
import torch, torch.nn.functional as F

KIT = "/Users/vukrosic/my-life/research-repos/llm-research-kit"
sys.path.insert(0, KIT)
from configs.llm_config import LLMConfig
from models.llm import MinimalLLM
from optimizers.muon import Muon
from training.trainer import setup_muon_optimizer

torch.set_num_threads(int(os.environ.get("THREADS", 1)))
E = lambda k, d: type(d)(os.environ.get(k, d))
P1, P2, LR, MUON_LR, BURST = E("P1", 1000), E("P2", 1000), E("LR", 0.003), E("MUON_LR", 0.024), E("BURST", 100)
WARM, WD, FREQ_NEW, BS, EVERY = E("WARM", 100), E("WD", 0.2), E("FREQ_NEW", 0.12), 64, 50
L, A = 6, 10                                                 # letters per sequence, alphabet size
TASKS = "CSR"; BAR = A + len(TASKS); V = BAR + 1             # tokens: letters 0..9, task tokens, '|'
T = 2 * L + 2                                                # task, L letters, '|', L answers
CFG = LLMConfig(d_model=64, n_heads=4, n_layers=2, d_ff=256, n_kv_heads=4, max_seq_len=T, vocab_size=V,
                device="cpu", compile_model=False, muon_lr=MUON_LR, adamw_lr=LR, weight_decay=WD, dropout=0.0)


def batch(n, task, g):
    x = torch.randint(A, (n, L), generator=g)
    y = {"C": x, "S": (x + 1) % A, "R": x.flip(1)}[task]
    return torch.cat([torch.full((n, 1), A + TASKS.index(task)), x, torch.full((n, 1), BAR), y], 1)


def mixed(n, tasks, freqs, g):
    k = torch.multinomial(torch.tensor(freqs), n, replacement=True, generator=g)
    return torch.cat([batch(int((k == i).sum()), t, g) for i, t in enumerate(tasks) if (k == i).any()])


def answer_logits(m, s): return m(s[:, :-1])[:, L + 1:]     # logits that predict the L answer tokens


@torch.no_grad()
def evaluate(m, g):
    out, pre = {}, []
    hooks = [b.feed_forward.up_proj.register_forward_hook(lambda mod, i, o: pre.append(o)) for b in m.transformer_blocks]
    for t in TASKS:
        s = batch(256, t, g); pre.clear()
        out[t] = round((answer_logits(m, s).argmax(-1) == s[:, L + 2:]).all(1).float().mean().item(), 3)
        if t == "R":                                         # feed-forward units with pre-activation <= 0 on every R answer position
            out["off"] = round(sum((p[:, L + 1:].amax((0, 1)) <= 0).float().mean().item() for p in pre) / len(pre), 3)
    for h in hooks: h.remove()
    return out


def train(m, steps, tasks, freqs, g, opt, log, t0, eg):
    muon_opt, adamw = setup_muon_optimizer(m, CFG)          # kit's split: 2-D non-embedding weights -> Muon, rest -> AdamW
    if opt == "adam":                                        # AdamW on every weight, the kit's AdamW settings
        adamw = torch.optim.AdamW(list(m.parameters()), lr=LR, weight_decay=WD)
        opts, base = [adamw], [LR]
    else:
        opts, base = [muon_opt, adamw], [MUON_LR, LR]
    for i in range(steps):
        if i % EVERY == 0: log.append(dict(step=t0 + i, **evaluate(m, eg)))
        sc = min(1, 2 * (1 - i / steps)) * (min(1, (i + 1) / WARM) if opt == "adam" and WARM else 1)
        loss = F.cross_entropy(answer_logits(m, s := mixed(BS, tasks, freqs, g)).reshape(-1, V), s[:, L + 2:].reshape(-1))
        for o in opts: o.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), CFG.grad_clip)
        for o, lr in zip(opts, base):
            for gr in o.param_groups: gr["lr"] = lr * sc
            o.step()
    log.append(dict(step=t0 + steps, **evaluate(m, eg)))


if __name__ == "__main__":
    seed, arms = int(sys.argv[1]), sys.argv[2:] or ["adam", "burst", "muon"]
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    t = time.time(); m1 = MinimalLLM(CFG); log1 = []
    train(m1, P1, "CS", [0.5, 0.5], g, "adam", log1, 0, torch.Generator().manual_seed(999))
    print(f"seed {seed} phase 1 {time.time() - t:.1f}s  {log1[-1]}", flush=True)
    new = ["CSR", [(1 - FREQ_NEW) / 2, (1 - FREQ_NEW) / 2, FREQ_NEW]]
    for arm in arms:
        t = time.time(); m = copy.deepcopy(m1); g2 = torch.Generator().manual_seed(seed + 100); eg = torch.Generator().manual_seed(999); log = []
        if arm == "burst":
            train(m, BURST, *new, g2, "muon", log, 0, eg); log.pop()
            train(m, P2, *new, g2, "adam", log, BURST, eg)
        else:
            train(m, P2, *new, g2, arm, log, 0, eg)
        print(f"seed {seed} {arm:>6} {time.time() - t:.1f}s  {log[-1]}", flush=True)
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get("OUT", "runs.jsonl")), "a") as f:
            f.write(json.dumps(dict(seed=seed, arm=arm, p1=P1, p2=P2, lr=LR, muon_lr=MUON_LR, burst=BURST, warm=WARM, wd=WD,
                                    freq_new=FREQ_NEW, phase1=log1[-1], log=log)) + "\n")
