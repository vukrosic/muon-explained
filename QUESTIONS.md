# Good questions to look at

Open research questions for this repo. Each has a test that says when it is answered. When one is answered, link the README section with the result.

## 1. The forgetting–plasticity curve: is Muon on a better trade-off, or the same one shifted?

**Why.** In the arithmetic task, Muon learns a new op more easily than Adam ([arith: can a trained model learn a new op](arith/README.md#can-a-trained-model-learn-a-new-op-adam-no-muon-yes)). It also forgets a removed op 2–10× faster ([arith: does the partner protect an op](arith/README.md#does-the-partner-protect-an-op-from-being-forgotten)). Both could come from one cause: Muon moves every direction each step, which keeps units alive and also overwrites old knowledge. If so, Adam with a larger learning rate or more weight decay might land at the same point.

**Setup.** Arithmetic mod 97, ops trained one after another ([cl.py](arith/cl.py)). Sweep learning rate and weight decay for both optimizers. For each setting, measure two numbers:
- plasticity: held-out accuracy on the new op at the end of its phase;
- retention: held-out accuracy on the previous op at the end of the new phase.

**Answered when.** One chart of retention against plasticity, one point per setting, Adam and Muon in different colors, over 3 seeds. Either Muon's points lie beyond Adam's frontier (a better trade-off) or on it (the same trade-off, shifted).
