"""Two fixed models; all learned transforms live in sklearn pipelines."""
import numpy as np
from scipy.signal import welch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP


class BandPower(TransformerMixin, BaseEstimator):
    def __init__(self, sfreq=160):
        self.sfreq = sfreq

    def fit(self, x, y=None):
        self.n_features_in_ = x.shape[1]
        return self

    def transform(self, x):
        frequencies, psd = welch(x, fs=self.sfreq, nperseg=self.sfreq,
                                 noverlap=self.sfreq // 2, axis=-1)
        features = []
        for low, high in [(8, 13), (13, 30)]:
            mask = (frequencies >= low) & (frequencies < high)
            power = psd[..., mask].sum(axis=-1) * (frequencies[1] - frequencies[0])
            features.append(np.log(np.maximum(power, 1e-30)))
        return np.concatenate(features, axis=1)


def spectral_classifier():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000, random_state=20260911))


def make_model(name):
    if name == "spectral":
        return make_pipeline(BandPower(), StandardScaler(),
                             LogisticRegression(C=1, max_iter=2000, random_state=20260911))
    if name == "csp":
        return make_pipeline(CSP(n_components=4, reg="ledoit_wolf", log=True, norm_trace=False),
                             LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    raise ValueError(f"Unknown model: {name}")
