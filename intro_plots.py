"""Intro images: Muon timeline and Muon-vs-AdamW numbers as reported by the sources in the README."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTE, MUON, ADAM = "#222222", "#888888", "#b5532f", "#9aa5b1"
plt.rcParams.update({"font.size": 13})

# ---------- timeline ----------
events = [
    (0.0, "1933", "Schulz publishes an iteration\nthat inverts a matrix using\nonly multiplications"),
    (1.0, "Oct 2024", "Muon breaks the NanoGPT\nspeedrun record:\ntraining 1.35× faster"),
    (2.0, "Dec 2024", "Keller Jordan's blog post\nnames it: MomentUm\nOrthogonalized by Newton–Schulz"),
    (3.0, "Feb 2025", "Moonshot AI trains Moonlight\n(16B parameters, 5.7T tokens)\nwith Muon"),
    (4.0, "Jul 2025", "Kimi K2: 1 trillion parameters,\n15.5T tokens, trained with\nMuonClip, a Muon variant"),
]
fig, ax = plt.subplots(figsize=(16, 4.6), dpi=130)
ax.plot([-0.3, 4.3], [0, 0], color=INK, lw=2)
ax.plot([0.35, 0.65], [0, 0], color="white", lw=4)
ax.text(0.5, 0.06, "91 years", ha="center", color=MUTE, fontsize=12)
for i, (x, date, text) in enumerate(events):
    up = i % 2 == 0
    ax.plot(x, 0, "o", ms=13, color=MUON if i else INK, zorder=3)
    ax.text(x, -0.13, date, ha="center", va="top", fontsize=15, fontweight="bold")
    ax.text(x, 0.55 if up else -0.42, text, ha="center", va="center", fontsize=12.5,
            bbox=dict(boxstyle="round,pad=0.5", fc="#f6f1e9", ec=MUON if i else INK, lw=1.5))
    ax.plot([x, x], [0.07, 0.33] if up else [-0.24, -0.31], color=MUTE, lw=1)
ax.set_xlim(-0.6, 4.6); ax.set_ylim(-0.75, 0.85); ax.axis("off")
ax.set_title("From a 1933 matrix trick to trillion-parameter models", fontsize=17, fontweight="bold")
fig.tight_layout(); fig.savefig("images/muon_timeline.png"); plt.close(fig)

# ---------- numbers ----------
fig, ax = plt.subplots(1, 4, figsize=(17, 4.8), dpi=130)


def pair(a, title, vals, unit, fmt="{:g}", names=("AdamW", "Muon")):
    bars = a.bar(list(names), vals, color=[ADAM, MUON], width=0.6)
    for b, v in zip(bars, vals):
        a.text(b.get_x() + b.get_width() / 2, v, fmt.format(v) + unit, ha="center", va="bottom", fontsize=14, fontweight="bold")
    a.set_ylim(0, max(vals) * 1.25); a.set_yticks([])
    for s in ["top", "right", "left"]: a.spines[s].set_visible(False)
    a.set_title(title, fontsize=13.5)


pair(ax[0], "CIFAR-10 to 94% accuracy\n(record, A100-seconds)", [3.3, 2.6], " s", names=("old record", "Muon"))
pair(ax[1], "1.5B model to GPT-2 XL level\n(8×H100-hours)", [13.3, 10], " h")
pair(ax[2], "Compute to reach the same loss\n(Moonshot AI scaling laws)", [1.0, 0.5], "", "{:.0%}")
pair(ax[3], "Numbers stored per weight\n(optimizer memory)", [2, 1], "")
ax[3].text(0.5, -0.17, "AdamW: average + average square\nMuon: average only", transform=ax[3].transAxes, ha="center", va="top", fontsize=11.5, color=MUTE)
ax[2].text(0.5, -0.17, "about 2× more efficient", transform=ax[2].transAxes, ha="center", va="top", fontsize=11.5, color=MUTE)
fig.suptitle("Muon in numbers, as reported. Price: the 5 rounds add about 0.5–0.7% extra compute.", fontsize=16, fontweight="bold")
fig.tight_layout(); fig.savefig("images/muon_numbers.png"); plt.close(fig)
print("saved images/muon_timeline.png images/muon_numbers.png")
