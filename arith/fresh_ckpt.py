"""Save an untrained model with the same train/test split as a saved model (for fair phase-2 controls).
Usage: python3 fresh_ckpt.py SRC.pt DST.pt"""
import sys, torch
import arith as A
ck = torch.load(sys.argv[1])
torch.manual_seed(ck["seed"])
torch.save(dict(model=A.Net().state_dict(), tr=ck["tr"], te=ck["te"], order=ck["order"], seed=ck["seed"]), sys.argv[2])
