"""Tests for dictionary learning module."""

import numpy as np
import pytest

from src.dictionary.ksvd import KSVDDictionary
from src.dictionary.online_dl import OnlineDictionaryLearner


@pytest.fixture
def synthetic_data():
    """Generate synthetic sparse data for testing."""
    rng = np.random.default_rng(42)
    d = 10  # state dimension
    k = 20  # atoms
    n = 200  # samples

    # Create ground truth dictionary
    true_dict = rng.standard_normal((d, k))
    true_dict /= np.linalg.norm(true_dict, axis=0, keepdims=True)

    # Create sparse codes (each sample uses ~3 atoms)
    codes = np.zeros((n, k))
    for i in range(n):
        support = rng.choice(k, 3, replace=False)
        codes[i, support] = rng.standard_normal(3)

    data = codes @ true_dict.T  # (n, d)
    return data, true_dict, codes


class TestKSVDDictionary:
    def test_fit_reconstruct(self, synthetic_data):
        data, _, _ = synthetic_data
        ksvd = KSVDDictionary(n_atoms=20, n_nonzero=5, max_iter=30)
        ksvd.fit(data)

        recon = ksvd.reconstruct(data)
        mse = np.mean((data - recon) ** 2) / np.mean(data**2)
        assert mse < 0.05, f"Relative reconstruction MSE {mse:.4f} > 5%"

    def test_atom_unit_norm(self, synthetic_data):
        data, _, _ = synthetic_data
        ksvd = KSVDDictionary(n_atoms=20, n_nonzero=5, max_iter=10)
        ksvd.fit(data)

        norms = np.linalg.norm(ksvd.dictionary, axis=0)  # ty: ignore[no-matching-overload]
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_sparse_encoding(self, synthetic_data):
        data, _, _ = synthetic_data
        ksvd = KSVDDictionary(n_atoms=20, n_nonzero=3, max_iter=10)
        ksvd.fit(data)

        codes = ksvd.encode(data)
        # Each code should have at most n_nonzero nonzeros
        for i in range(len(codes)):
            nnz = np.count_nonzero(codes[i])
            assert nnz <= 3, f"Sample {i} has {nnz} nonzeros, expected <= 3"

    def test_output_shapes(self, synthetic_data):
        data, _, _ = synthetic_data
        n, d = data.shape
        k = 20
        ksvd = KSVDDictionary(n_atoms=k, n_nonzero=3, max_iter=5)
        ksvd.fit(data)

        assert ksvd.dictionary.shape == (d, k)  # ty: ignore[unresolved-attribute]
        assert ksvd.encode(data).shape == (n, k)
        assert ksvd.reconstruct(data).shape == (n, d)

    def test_to_torch(self, synthetic_data):
        data, _, _ = synthetic_data
        ksvd = KSVDDictionary(n_atoms=20, n_nonzero=3, max_iter=5)
        ksvd.fit(data)

        t = ksvd.to_torch()
        assert t.shape == (data.shape[1], 20)
        assert t.dtype.is_floating_point


class TestOnlineDictionaryLearner:
    def test_fit_reconstruct(self, synthetic_data):
        data, _, _ = synthetic_data
        learner = OnlineDictionaryLearner(n_atoms=20, alpha=0.1, n_iter=100)
        learner.fit(data)

        recon = learner.reconstruct(data)
        mse = np.mean((data - recon) ** 2) / np.mean(data**2)
        assert mse < 0.15, f"Relative reconstruction MSE {mse:.4f} > 15%"

    def test_online_update(self, synthetic_data):
        data, _, _ = synthetic_data
        learner = OnlineDictionaryLearner(n_atoms=20, alpha=0.5, n_iter=20)
        # Fit on first half, then partial_fit on second half
        learner.fit(data[:100])
        learner.partial_fit(data[100:])

        codes = learner.encode(data)
        assert codes.shape == (200, 20)

    def test_to_torch(self, synthetic_data):
        data, _, _ = synthetic_data
        learner = OnlineDictionaryLearner(n_atoms=20, n_iter=20)
        learner.fit(data)

        t = learner.to_torch()
        assert t.shape == (data.shape[1], 20)
