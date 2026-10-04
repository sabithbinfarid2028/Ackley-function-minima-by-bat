"""
Ackley function minimization using the Bat Algorithm (NumPy only).

f(x) = -a*exp(-b*sqrt(mean(x^2))) - exp(mean(cos(c*x))) + a + e
Global minimum: f(0, ..., 0) = 0, search domain: [-32.768, 32.768]^d
"""

import numpy as np


def ackley(x, a=20.0, b=0.2, c=2 * np.pi):
    x = np.asarray(x, dtype=float)
    d = x.size
    term1 = -a * np.exp(-b * np.sqrt(np.sum(x ** 2) / d))
    term2 = -np.exp(np.sum(np.cos(c * x)) / d)
    return term1 + term2 + a + np.e


def bat_algorithm(func, dim=2, bounds=(-32.768, 32.768), n_bats=40, max_iter=500,
                  f_min=0.0, f_max=2.0, A0=1.0, r0=0.5, alpha=0.9, gamma=0.9,
                  tol=1e-10, walk_scale=0.1, seed=None):
    rng = np.random.default_rng(seed)
    lo, hi = bounds

    x = rng.uniform(lo, hi, (n_bats, dim))      # positions
    v = np.zeros((n_bats, dim))                 # velocities
    A = np.full(n_bats, A0)                     # loudness
    r = np.full(n_bats, r0)                     # pulse emission rate
    fit = np.array([func(p) for p in x])

    best_idx = fit.argmin()
    x_best, f_best = x[best_idx].copy(), fit[best_idx]
    history = []

    for t in range(1, max_iter + 1):
        for i in range(n_bats):
            # frequency, velocity and position update
            freq = f_min + (f_max - f_min) * rng.random()
            v[i] += (x[i] - x_best) * freq
            x_new = np.clip(x[i] + v[i], lo, hi)

            # local random walk around the best solution
            if rng.random() > r[i]:
                x_new = np.clip(x_best + walk_scale * (hi - lo) * (1 - t / max_iter) ** 3 * rng.normal(size=dim),
                                lo, hi)

            f_new = func(x_new)

            # accept if better and loudness allows
            if f_new <= fit[i] and rng.random() < A[i]:
                x[i], fit[i] = x_new, f_new
                A[i] *= alpha
                r[i] = r0 * (1 - np.exp(-gamma * t))

            if f_new < f_best:
                x_best, f_best = x_new.copy(), f_new

        history.append(f_best)
        if f_best < tol:
            break

    return x_best, f_best, history


if __name__ == "__main__":
    DIM = 5
    x_best, f_best, hist = bat_algorithm(ackley, dim=DIM, seed=42)

    print(f"Dimension        : {DIM}")
    print(f"Iterations run   : {len(hist)}")
    print(f"Best x           : {np.round(x_best, 6)}")
    print(f"Best f(x)        : {f_best:.3e}")
    print("True minimum     : f(0,...,0) = 0")

    # Optional convergence plot (needs matplotlib)
    try:
        import matplotlib.pyplot as plt

        plt.semilogy(hist, color="black")
        plt.xlabel("Iteration")
        plt.ylabel("Best f(x)")
        plt.title("Ackley - Bat Algorithm convergence")
        plt.grid(True, alpha=0.3)
        plt.show()
    except ImportError:
        pass
