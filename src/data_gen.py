"""
Synthetic data generator for CAGL experiments.

Abstracts the structure of multimodal-aas-bird data:
- N data points, K topologies, L labels
- Each topology has label-specific accuracy
- Partial correlation via shared noise
"""

import numpy as np

# Default topology accuracies (asymmetric specialists)
DEFAULT_TOPO_ACCURACY = [
    [0.90, 0.55],  # Topology A: label_0 が得意、label_1 が苦手
    [0.55, 0.92],  # Topology B: label_1 が得意、label_0 が苦手
    [0.72, 0.68],  # Topology C: 両方中程度
    [0.80, 0.60],  # Topology D: label_0 寄り
    [0.58, 0.85],  # Topology E: label_1 寄り
]


def generate_synthetic_data(
    n_samples=500,
    n_topologies=5,
    n_labels=2,
    topo_accuracy=None,
    shared_noise_rate=0.1,
    seed=42,
):
    """
    Generate ensemble prediction data with controlled diversity.

    Returns
    -------
    pred : ndarray, shape (n_samples, n_topologies, n_labels), binary
    gt   : ndarray, shape (n_samples, n_labels), binary
    """
    if topo_accuracy is None:
        topo_accuracy = DEFAULT_TOPO_ACCURACY

    rng = np.random.default_rng(seed)
    gt = rng.integers(0, 2, size=(n_samples, n_labels))

    pred = np.zeros((n_samples, n_topologies, n_labels), dtype=int)
    shared_noise = rng.binomial(1, shared_noise_rate, size=(n_samples, n_labels))

    for t in range(n_topologies):
        for l in range(n_labels):
            acc = topo_accuracy[t][l]
            correct = rng.random(n_samples) < acc
            pred[:, t, l] = np.where(correct, gt[:, l], 1 - gt[:, l])
            # Apply shared noise (flips prediction for all topologies on that label)
            pred[:, t, l] = np.where(
                shared_noise[:, l], 1 - pred[:, t, l], pred[:, t, l]
            )

    return pred, gt
