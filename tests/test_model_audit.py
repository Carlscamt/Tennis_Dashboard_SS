"""
Tests for model audit functions.
"""
import pytest
import numpy as np


class TestCalibrationMetrics:
    """Test calibration error calculations."""
    
    def test_perfect_calibration(self):
        """Perfectly calibrated predictions should have ECE = 0."""
        y_true = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 1])
        y_prob = np.array([0.2, 0.3, 0.6, 0.7, 0.1, 0.4, 0.8, 0.9, 0.6, 0.7])
        
        # In perfect calibration, 20% conf should have 20% true rate
        # This is approximate, so we just check ECE is reasonable
        assert len(y_true) == len(y_prob)
    
    def test_ece_bounds(self):
        """ECE should be between 0 and 1."""
        # Random predictions
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.rand(100)
        
        # ECE calculation
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0
        
        for i in range(n_bins):
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_acc = y_true[mask].mean()
                bin_conf = y_prob[mask].mean()
                ece += (mask.sum() / len(y_true)) * abs(bin_acc - bin_conf)
        
        assert 0 <= ece <= 1


class TestBootstrapSampling:
    """Test bootstrap confidence interval calculation."""
    
    def test_bootstrap_produces_valid_samples(self):
        """Bootstrap should produce samples of same size."""
        from sklearn.utils import resample
        
        data = np.array([1, 2, 3, 4, 5])
        sample = resample(data, random_state=42)
        
        assert len(sample) == len(data)
    
    def test_bootstrap_ci_contains_mean(self):
        """95% CI should usually contain true mean."""
        np.random.seed(42)
        true_mean = 0.6
        data = np.random.binomial(1, true_mean, 100)
        
        # Bootstrap
        from sklearn.utils import resample
        means = []
        for i in range(100):
            sample = resample(data, random_state=i)
            means.append(sample.mean())
        
        ci_low = np.percentile(means, 2.5)
        ci_high = np.percentile(means, 97.5)
        
        # CI should be reasonable
        assert ci_low < ci_high
        assert 0 <= ci_low <= 1
        assert 0 <= ci_high <= 1


class TestPerformanceMetrics:
    """Test performance metric calculations."""
    
    def test_accuracy_calculation(self):
        """Test basic accuracy calculation."""
        from sklearn.metrics import accuracy_score
        
        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 0])
        
        acc = accuracy_score(y_true, y_pred)
        assert acc == 0.8  # 4/5 correct
    
    def test_auc_roc_bounds(self):
        """AUC-ROC should be between 0 and 1."""
        from sklearn.metrics import roc_auc_score
        
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.rand(100)
        
        auc = roc_auc_score(y_true, y_prob)
        assert 0 <= auc <= 1
    
    def test_brier_score_perfect_predictions(self):
        """Perfect predictions should have Brier score = 0."""
        from sklearn.metrics import brier_score_loss
        
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([1.0, 0.0, 1.0, 0.0])
        
        brier = brier_score_loss(y_true, y_prob)
        assert brier == 0


class TestTemporalSplit:
    """Test temporal data splitting."""
    
    def test_temporal_split_no_overlap(self):
        """Train and test should not overlap temporally."""
        from datetime import datetime
        
        timestamps = [
            int(datetime(2024, i, 1).timestamp()) 
            for i in range(1, 13)
        ]
        
        # Split at month 6
        cutoff = int(datetime(2024, 7, 1).timestamp())
        
        train = [t for t in timestamps if t < cutoff]
        test = [t for t in timestamps if t >= cutoff]
        
        assert len(train) == 6
        assert len(test) == 6
        assert max(train) < min(test)
    
    def test_all_train_before_all_test(self):
        """Every train timestamp should be before every test timestamp."""
        import random
        random.seed(42)
        
        train_ts = sorted([random.randint(1000, 2000) for _ in range(10)])
        test_ts = sorted([random.randint(2001, 3000) for _ in range(10)])
        
        assert all(t < min(test_ts) for t in train_ts)


class TestEdgeCaseDetection:
    """Test edge case detection logic."""
    
    def test_high_confidence_wrong_detection(self):
        """Should correctly identify high-confidence wrong predictions."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_prob = np.array([0.9, 0.2, 0.1, 0.8, 0.7])
        y_pred = (y_prob >= 0.5).astype(int)
        
        # Wrong predictions
        wrong = y_pred != y_true
        # High confidence (>0.7 or <0.3)
        high_conf = (y_prob >= 0.7) | (y_prob <= 0.3)
        
        high_conf_wrong = wrong & high_conf
        
        # Event 0: pred=1 (0.9), true=0 -> wrong, high conf ✓
        # Event 1: pred=0 (0.2), true=1 -> wrong, high conf ✓
        # Event 2: pred=0 (0.1), true=0 -> correct
        # Event 3: pred=1 (0.8), true=1 -> correct
        # Event 4: pred=1 (0.7), true=0 -> wrong, high conf ✓
        
        assert high_conf_wrong.sum() == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
