"""
CAGL (Consensus-Agreement Gate Learning) minimal implementation.

Based on Phase 1.4d Stage 3 Target C finding:
- Weight (GT signal) + Gate (consensus signal) + Multiplicative integration
- Load-bearing novelty: consensus-agreement gate update rule
"""

import numpy as np


class CAGL:
    """Consensus-Agreement Gate Learning."""

    def __init__(self, n_topologies, n_labels,
                 success_rate=0.1, failure_rate=0.7,
                 gate_mode="consensus"):
        """
        Parameters
        ----------
        gate_mode : str
            "consensus" (CAGL) | "gt" (Ablation: both GT) | "none" (weight only)
        """
        self.n_topologies = n_topologies
        self.n_labels = n_labels
        self.success_rate = success_rate
        self.failure_rate = failure_rate
        self.gate_mode = gate_mode
        self.reset()

    def reset(self):
        self.w = np.full((self.n_topologies, self.n_labels), 0.5)
        self.g = np.full((self.n_topologies, self.n_labels), 0.5)

    def predict(self, pred, weight_mode="multiplicative"):
        """
        Produce integrated prediction for a single data point.

        Parameters
        ----------
        pred : ndarray, shape (n_topologies, n_labels), binary
        weight_mode : str
            "multiplicative" (w*g) | "weight_only" (w) | "gate_only" (0.5*g)
        """
        if weight_mode == "weight_only":
            effective = self.w.copy()
        elif weight_mode == "gate_only":
            effective = 0.5 * self.g
        else:  # multiplicative
            effective = self.w * self.g

        final = np.zeros(self.n_labels, dtype=int)
        for label in range(self.n_labels):
            total = effective[:, label].sum()
            if total < 1e-9:
                final[label] = 0
            else:
                score = (effective[:, label] * pred[:, label]).sum() / total
                final[label] = 1 if score >= 0.5 else 0
        return final

    def update(self, pred, gt, final):
        """Update weights and gates for a single data point."""
        # Weight update (GT signal)
        for t in range(self.n_topologies):
            for l in range(self.n_labels):
                if pred[t, l] == gt[l]:
                    self.w[t, l] += self.success_rate * (1 - self.w[t, l])
                else:
                    self.w[t, l] *= self.failure_rate

        # Gate update
        if self.gate_mode == "consensus":
            for t in range(self.n_topologies):
                for l in range(self.n_labels):
                    if pred[t, l] == final[l]:
                        self.g[t, l] += self.success_rate * (1 - self.g[t, l])
                    else:
                        self.g[t, l] *= self.failure_rate
        elif self.gate_mode == "gt":
            for t in range(self.n_topologies):
                for l in range(self.n_labels):
                    if pred[t, l] == gt[l]:
                        self.g[t, l] += self.success_rate * (1 - self.g[t, l])
                    else:
                        self.g[t, l] *= self.failure_rate
        # "none" → no gate update
