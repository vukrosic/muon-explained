"""Approximate Muon orthogonalization (Newton–Schulz) on a matrix, quintic vs cubic.

Same coefficients as the clip (src/lib.ts):
  quintic: 3.4445·X − 4.775·(XXᵀ)X + 2.0315·(XXᵀ)²X   (Muon default, Keller Jordan)
  cubic:   1.5·X − 0.5·(XXᵀ)X                          (textbook Newton–Schulz)
Each round applies the polynomial to every singular value of X.
"""
import numpy as np

QUINTIC = (3.4445, -4.775, 2.0315)
CUBIC = (1.5, -0.5)


def muon_orth(G: np.ndarray, steps: int = 5, kind: str = "quintic", eps: float = 1e-7) -> np.ndarray:
    """Push every singular value of G toward 1 with `steps` Newton–Schulz rounds."""
    X = G / (np.linalg.norm(G) + eps)  # Frobenius norm ≥ largest singular value, so all strengths ≤ 1
    tall = X.shape[0] > X.shape[1]
    if tall:
        X = X.T  # keep XXᵀ the smaller square matrix
    for _ in range(steps):
        A = X @ X.T
        if kind == "quintic":
            a, b, c = QUINTIC
            X = a * X + (b * A + c * A @ A) @ X
        elif kind == "cubic":
            a, b = CUBIC
            X = a * X + b * A @ X
        else:
            raise ValueError(f"unknown kind {kind!r}")
    return X.T if tall else X


def main() -> None:
    np.seterr(all="ignore")  # macOS Accelerate matmul raises false overflow warnings
    rng = np.random.default_rng(0)
    # 8×8 matrix with the clip's strengths: singular values 1, .6, .3, … .01
    s0 = np.array([1.0, 0.6, 0.3, 0.15, 0.08, 0.04, 0.02, 0.01])
    U, _ = np.linalg.qr(rng.standard_normal((8, 8)))
    V, _ = np.linalg.qr(rng.standard_normal((8, 8)))
    G = U @ np.diag(s0) @ V.T

    exact = U @ V.T  # true orthogonalization: every strength exactly 1
    print("before        ", np.round(np.linalg.svd(G / np.linalg.norm(G), compute_uv=False), 3))
    for kind, steps in [("quintic", 4), ("quintic", 5), ("cubic", 5)]:
        O = muon_orth(G, steps, kind)
        sv = np.linalg.svd(O, compute_uv=False)
        err = np.linalg.norm(O - exact) / np.linalg.norm(exact)
        print(f"{kind:7s} × {steps}   ", np.round(sv, 3), f" range {sv.min():.2f}–{sv.max():.2f}  err vs exact {err:.3f}")


if __name__ == "__main__":
    main()
