"""
baselines_clean.py
==================
Clean, honestly named sequential change-point detectors, empirical matched-FAR
calibration, empirical ARL0 estimation (including fused Tri-Track ARL0),
symmetric OR-triggering Tri-Track fusion with causal 3-way structural fault
attribution, fair learned GNN ablation, and timeout-free downstream scheduler
benchmarking.

Upgrades over v1:
  1. TriTrackFusion uses symmetric OR-triggering across Tracks 1, 2, and 3
     (each calibrated to target_alpha / 3 so the fused detector respects the
     global false-alarm budget), with non-truncating 10-step confirmation
     logging and causal 3-way structural attribution:
       - node_disruption   (active |V(t)| changes -> aggressive response)
       - link_degradation  (stable |V(t)|, dropping active edge density -> conservative)
       - topology_reconfig (stable |V(t)|, non-decreasing/rewired density -> aggressive)
  2. FairSTGNNEncoder augments node inputs with normalized local degree and
     uses permutation-invariant [mean, sum / M0] readout + LayerNorm so the
     learned GNN baseline can genuinely observe churn and link loss.
  3. evaluate_downstream_schedulers removes the artificial hold_steps=15
     timeout, starts in the pre-change regime's optimal mode, and benchmarks
     four policies: Static-PreRegime, Scalar-Aggressive, Scalar-Conservative,
     and Attributed-TriTrack (using strictly causal attribution timestamps).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from scipy.special import gammaln, logsumexp

from fog_sim_clean import FogGraphGenerator, REGIMES


# =========================================================================
# 1. ADAPTIVE EWMA-CUSUM (TRACK 1 / TRACK 2 / TRACK 3 / GNN)
# =========================================================================

class CUSUMDetector:
    """Adaptive EWMA-CUSUM on Euclidean deviation ||x_t - mu_{t-1}||_2."""

    def __init__(
        self,
        kappa=0.05,
        threshold=1.0,
        alpha=0.05,
        refractory=10,
        warmup=20,
    ):
        self.kappa = float(kappa)
        self.threshold = float(threshold)
        self.alpha = float(alpha)
        self.refractory = int(refractory)
        self.warmup = int(warmup)
        self.reset()

    def reset(self):
        self.S = 0.0
        self.mu = None
        self.last_alarm = -10**9
        self.alarms = []

    def step(self, x, t):
        x = np.asarray(x, dtype=np.float32)
        if self.mu is None:
            self.mu = x.copy()
            return 0.0
        dist = float(np.linalg.norm(x - self.mu))
        self.S = max(0.0, self.S + dist - self.kappa)
        alarm = (
            self.S >= self.threshold
            and t >= self.warmup
            and (t - self.last_alarm > self.refractory)
        )
        if alarm:
            self.last_alarm = t
            self.S = 0.0
            self.mu = x.copy()
            self.alarms.append(int(t))
        else:
            self.mu = (1.0 - self.alpha) * self.mu + self.alpha * x
        return self.S


# =========================================================================
# 2. MULTIVARIATE EWMA CONTROL CHART
# =========================================================================

class EWMADetector:
    """EWMA control chart with scalar variance tracking."""

    def __init__(
        self,
        lambda_=0.1,
        threshold=6.0,
        refractory=10,
        warmup=20,
    ):
        self.lambda_ = float(lambda_)
        self.threshold = float(threshold)
        self.refractory = int(refractory)
        self.warmup = int(warmup)
        self.reset()

    def reset(self):
        self.z = None
        self.mu = None
        self.var = 1e-3
        self.last_alarm = -10**9
        self.alarms = []

    def step(self, x, t):
        x = np.asarray(x, dtype=np.float32)
        if self.z is None:
            self.z = x.copy()
            self.mu = x.copy()
            return 0.0
        self.z = self.lambda_ * x + (1.0 - self.lambda_) * self.z
        diff = self.z - self.mu
        v = float(np.mean(diff ** 2))
        self.var = 0.9 * self.var + 0.1 * max(v, 1e-6)
        dev = float(np.linalg.norm(diff))
        scale = np.sqrt(max(self.var * self.lambda_ / (2.0 - self.lambda_), 1e-8))
        score = dev / scale
        alarm = (
            score >= self.threshold
            and t >= self.warmup
            and (t - self.last_alarm > self.refractory)
        )
        if alarm:
            self.last_alarm = t
            self.z = x.copy()
            self.mu = x.copy()
            self.alarms.append(int(t))
        else:
            self.mu = (1.0 - self.lambda_) * self.mu + self.lambda_ * x
        return score


# =========================================================================
# 3. AUTHENTIC ADAMS & MACKAY (2007) BOCPD
# =========================================================================

class BOCPDDetector:
    """Bayesian Online Changepoint Detection (Adams & MacKay, 2007)."""

    def __init__(
        self,
        hazard=1.0 / 50.0,
        threshold=0.5,
        cp_window=3,
        max_rl=100,
        refractory=10,
        warmup=20,
        mean_h0=None,
        std_h0=None,
    ):
        self.hazard = float(hazard)
        self.threshold = float(threshold)
        self.cp_window = int(cp_window)
        self.max_rl = int(max_rl)
        self.refractory = int(refractory)
        self.warmup = int(warmup)
        self.mean_h0 = (
            None if mean_h0 is None else np.asarray(mean_h0, dtype=np.float64)
        )
        self.std_h0 = (
            None
            if std_h0 is None
            else np.maximum(np.asarray(std_h0, dtype=np.float64), 1e-3)
        )
        self.kappa0 = 1.0
        self.alpha0 = 2.0
        self.beta0 = 1.0
        self.reset()

    def reset(self):
        self.log_R = None
        self.mu = None
        self.kappa = None
        self.alpha = None
        self.beta = None
        self.last_alarm = -10**9
        self.alarms = []

    def _init_posterior(self, d):
        self.log_R = np.array([0.0], dtype=np.float64)
        self.mu = np.zeros((1, d), dtype=np.float64)
        self.kappa = np.array([self.kappa0], dtype=np.float64)
        self.alpha = np.array([self.alpha0], dtype=np.float64)
        self.beta = np.full((1, d), self.beta0, dtype=np.float64)

    def _student_t_logpdf(self, x):
        nu = 2.0 * self.alpha[:, None]
        scale2 = (
            self.beta
            * (self.kappa[:, None] + 1.0)
            / (self.alpha[:, None] * self.kappa[:, None])
        )
        scale2 = np.maximum(scale2, 1e-8)
        diff2 = (x[None, :] - self.mu) ** 2
        log_p = (
            gammaln(0.5 * (nu + 1.0))
            - gammaln(0.5 * nu)
            - 0.5 * np.log(np.pi * nu * scale2)
            - 0.5 * (nu + 1.0) * np.log1p(diff2 / (nu * scale2))
        )
        return np.sum(log_p, axis=1)

    def step(self, x, t):
        x = np.asarray(x, dtype=np.float64)
        if self.mean_h0 is not None and self.std_h0 is not None:
            x = (x - self.mean_h0) / self.std_h0

        d = x.shape[0]
        if self.log_R is None:
            self._init_posterior(d)

        log_pred = self._student_t_logpdf(x)
        log_joint = self.log_R + log_pred

        log_growth = log_joint + np.log(1.0 - self.hazard)
        log_cp = float(logsumexp(log_joint)) + np.log(self.hazard)

        log_R_new = np.concatenate([[log_cp], log_growth])
        log_R_new -= float(logsumexp(log_R_new))

        k_col = self.kappa[:, None]
        mu_tail = (k_col * self.mu + x[None, :]) / (k_col + 1.0)
        beta_tail = self.beta + 0.5 * (k_col / (k_col + 1.0)) * (
            x[None, :] - self.mu
        ) ** 2

        self.mu = np.vstack([np.zeros((1, d), dtype=np.float64), mu_tail])
        self.kappa = np.concatenate([[self.kappa0], self.kappa + 1.0])
        self.alpha = np.concatenate([[self.alpha0], self.alpha + 0.5])
        self.beta = np.vstack([np.full((1, d), self.beta0), beta_tail])
        self.log_R = log_R_new

        if len(self.log_R) > self.max_rl:
            self.log_R = self.log_R[: self.max_rl]
            self.log_R -= float(logsumexp(self.log_R))
            self.mu = self.mu[: self.max_rl]
            self.kappa = self.kappa[: self.max_rl]
            self.alpha = self.alpha[: self.max_rl]
            self.beta = self.beta[: self.max_rl]

        w = min(self.cp_window, len(self.log_R))
        cp_mass = float(np.sum(np.exp(self.log_R[:w])))

        alarm = (
            cp_mass >= self.threshold
            and t >= self.warmup
            and (t - self.last_alarm > self.refractory)
        )
        if alarm:
            self.last_alarm = t
            self.alarms.append(int(t))
            self._init_posterior(d)
        return cp_mass


# =========================================================================
# 4. TWO-WINDOW RBF MMD DETECTOR
# =========================================================================

class MMDDetector:
    """Sliding two-window RBF-kernel MMD^2 detector."""

    def __init__(
        self,
        window=15,
        bandwidth=0.2,
        threshold=0.05,
        refractory=10,
        warmup=30,
    ):
        self.window = int(window)
        self.bandwidth = float(max(bandwidth, 1e-3))
        self.threshold = float(threshold)
        self.refractory = int(refractory)
        self.warmup = int(warmup)
        self.reset()

    def reset(self):
        self.buffer = []
        self.last_alarm = -10**9
        self.alarms = []

    def _mmd2(self, X, Y):
        gamma = 1.0 / (2.0 * self.bandwidth ** 2)
        Kxx = np.exp(-gamma * cdist(X, X, "sqeuclidean")).mean()
        Kyy = np.exp(-gamma * cdist(Y, Y, "sqeuclidean")).mean()
        Kxy = np.exp(-gamma * cdist(X, Y, "sqeuclidean")).mean()
        return max(0.0, float(Kxx + Kyy - 2.0 * Kxy))

    def step(self, x, t):
        x = np.asarray(x, dtype=np.float32)
        self.buffer.append(x)
        if len(self.buffer) < 2 * self.window:
            return 0.0
        X = np.asarray(self.buffer[-2 * self.window : -self.window])
        Y = np.asarray(self.buffer[-self.window :])
        score = self._mmd2(X, Y)
        alarm = (
            score >= self.threshold
            and t >= self.warmup
            and (t - self.last_alarm > self.refractory)
        )
        if alarm:
            self.last_alarm = t
            self.alarms.append(int(t))
            self.buffer = self.buffer[-self.window :]
        return score


# =========================================================================
# 5. SWORD (MULTI-PROBE CHEBYSHEV SPECTRAL MOMENTS)
# =========================================================================

class SWORDDetector:
    """Spectral Chebyshev-moment detector with multi-probe Hutchinson estimator."""

    def __init__(
        self,
        K=10,
        window=15,
        n_probes=16,
        threshold=0.5,
        refractory=10,
        warmup=30,
        seed=0,
    ):
        self.K = int(K)
        self.window = int(window)
        self.n_probes = int(n_probes)
        self.threshold = float(threshold)
        self.refractory = int(refractory)
        self.warmup = int(warmup)
        self.seed = int(seed)
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.seed)
        self.moments_buffer = []
        self.last_alarm = -10**9
        self.alarms = []

    def _chebyshev_moments(self, A):
        n = A.shape[0]
        if n < 2:
            return np.zeros(self.K, dtype=np.float64)
        d = A.sum(axis=1)
        d_safe = np.maximum(d, 1e-8)
        d_inv_sqrt = np.where(d > 1e-8, 1.0 / np.sqrt(d_safe), 0.0)
        L_norm = np.eye(n) - (d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :])
        L_scaled = L_norm - np.eye(n)

        Z = self.rng.choice([-1.0, 1.0], size=(n, self.n_probes))
        moments = np.zeros(self.K, dtype=np.float64)
        T_prev = Z
        moments[0] = float(np.sum(Z * T_prev)) / (n * self.n_probes)
        if self.K > 1:
            T_curr = L_scaled @ T_prev
            moments[1] = float(np.sum(Z * T_curr)) / (n * self.n_probes)
            for k in range(2, self.K):
                T_next = 2.0 * (L_scaled @ T_curr) - T_prev
                moments[k] = float(np.sum(Z * T_next)) / (n * self.n_probes)
                T_prev, T_curr = T_curr, T_next
        return moments

    def step(self, A, t):
        moments = self._chebyshev_moments(A)
        self.moments_buffer.append(moments)
        if len(self.moments_buffer) < 2 * self.window:
            return 0.0
        M_ref = np.mean(
            self.moments_buffer[-2 * self.window : -self.window], axis=0
        )
        M_curr = np.mean(self.moments_buffer[-self.window :], axis=0)
        dist = float(np.sum(np.abs(M_curr - M_ref)))
        alarm = (
            dist >= self.threshold
            and t >= self.warmup
            and (t - self.last_alarm > self.refractory)
        )
        if alarm:
            self.last_alarm = t
            self.alarms.append(int(t))
            self.moments_buffer = self.moments_buffer[-self.window :]
        return dist


# =========================================================================
# 6. AUTOENCODER RECONSTRUCTION DETECTOR (MLP-AE)
# =========================================================================

class _MLPAE(nn.Module):
    def __init__(self, input_dim, hidden=32, latent=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AutoencoderDetector:
    """Reconstruction-error detector trained on standardized H0 features."""

    def __init__(
        self,
        model,
        mean_h0,
        std_h0,
        threshold=0.5,
        refractory=10,
        warmup=20,
    ):
        self.model = model
        self.mean_h0 = np.asarray(mean_h0, dtype=np.float32)
        self.std_h0 = np.maximum(np.asarray(std_h0, dtype=np.float32), 1e-3)
        self.threshold = float(threshold)
        self.refractory = int(refractory)
        self.warmup = int(warmup)
        self.reset()

    @staticmethod
    def fit_model(X_train, hidden=32, latent=8, epochs=80, lr=1e-3, seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        X_arr = np.asarray(X_train, dtype=np.float32)
        mean_h0 = X_arr.mean(axis=0)
        std_h0 = np.maximum(X_arr.std(axis=0), 1e-3)
        X_norm = (X_arr - mean_h0) / std_h0
        X_t = torch.tensor(X_norm, dtype=torch.float32)
        model = _MLPAE(X_arr.shape[1], hidden=hidden, latent=latent)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            recon = model(X_t)
            loss = F.mse_loss(recon, X_t)
            loss.backward()
            opt.step()
        model.eval()
        return model, mean_h0, std_h0

    def reset(self):
        self.last_alarm = -10**9
        self.alarms = []

    def step(self, x, t):
        x_norm = (np.asarray(x, dtype=np.float32) - self.mean_h0) / self.std_h0
        x_t = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            recon = self.model(x_t)
            err = float(((recon - x_t) ** 2).mean().item())
        alarm = (
            err >= self.threshold
            and t >= self.warmup
            and (t - self.last_alarm > self.refractory)
        )
        if alarm:
            self.last_alarm = t
            self.alarms.append(int(t))
        return err


# =========================================================================
# 7. WINDOWED TWO-SAMPLE MEAN-SHIFT GLR
# =========================================================================

class WindowedGLRDetector:
    """Sliding two-window Welch-style mean-shift GLR detector."""

    def __init__(
        self,
        window=15,
        threshold=10.0,
        refractory=10,
        warmup=30,
        min_var=1e-4,
    ):
        self.window = int(window)
        self.threshold = float(threshold)
        self.refractory = int(refractory)
        self.warmup = int(warmup)
        self.min_var = float(min_var)
        self.reset()

    def reset(self):
        self.buffer = []
        self.last_alarm = -10**9
        self.alarms = []

    def step(self, x, t):
        x = np.asarray(x, dtype=np.float32)
        self.buffer.append(x)
        if len(self.buffer) < 2 * self.window:
            return 0.0
        X_pre = np.asarray(self.buffer[-2 * self.window : -self.window])
        X_post = np.asarray(self.buffer[-self.window :])
        mu_pre = X_pre.mean(axis=0)
        mu_post = X_post.mean(axis=0)
        var_pre = np.maximum(X_pre.var(axis=0), self.min_var)
        var_post = np.maximum(X_post.var(axis=0), self.min_var)
        score = float(np.sum((mu_post - mu_pre) ** 2 / (var_pre + var_post)))
        alarm = (
            score >= self.threshold
            and t >= self.warmup
            and (t - self.last_alarm > self.refractory)
        )
        if alarm:
            self.last_alarm = t
            self.alarms.append(int(t))
            self.buffer = self.buffer[-self.window :]
        return score


# =========================================================================
# 8. FAIR LEARNED GNN ABLATION (DEGREE FEATURE + [MEAN, SUM/M0] READOUT)
# =========================================================================

class _FairSpatialMP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.Linear(dim, dim)
        self.ls = nn.Linear(dim, dim)

    def forward(self, x, A):
        deg = A.sum(1, keepdim=True).clamp(min=1.0)
        return F.relu(self.ln(A @ x / deg) + self.ls(x))


class FairSTGNNEncoder(nn.Module):
    """Permutation-invariant, variable-size GNN encoder with structural readout.

    Fixes two blind spots of naive mean-pooled GraphSAGE:
      - Appends normalized node degree deg_i / (M0 - 1) to node features so
        local connectivity changes (S3 link degradation, S5 reconfig) are visible.
      - Uses dual [mean_pool, sum_pool / M0] readout + LayerNorm before the GRU
        so node-count shifts (S2 churn, S4 cascading) are preserved while keeping
        output dimension strictly independent of |V(t)|.
    """

    def __init__(self, feat_dim=4, hidden=32, latent=16, layers=2, M0=32):
        super().__init__()
        self.M0 = int(M0)
        self.proj = nn.Linear(feat_dim + 1, hidden)
        self.layers = nn.ModuleList(
            [_FairSpatialMP(hidden) for _ in range(layers)]
        )
        self.readout_norm = nn.LayerNorm(2 * hidden)
        self.gru = nn.GRUCell(2 * hidden, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent),
        )
        self.latent = int(latent)
        self._h = None

    def reset(self):
        self._h = None

    def forward(self, A, X):
        A_t = torch.tensor(A, dtype=torch.float32)
        X_t = torch.tensor(X, dtype=torch.float32)
        deg_norm = A_t.sum(1, keepdim=True) / max(self.M0 - 1.0, 1.0)
        x_in = torch.cat([X_t, deg_norm], dim=1)
        h = F.relu(self.proj(x_in))
        for layer in self.layers:
            h = layer(h, A_t)
        g_mean = h.mean(0, keepdim=True)
        g_sum = h.sum(0, keepdim=True) / max(float(self.M0), 1.0)
        g = self.readout_norm(torch.cat([g_mean, g_sum], dim=1))
        if self._h is None:
            self._h = torch.zeros(1, h.shape[1], dtype=torch.float32)
        self._h = self.gru(g, self._h)
        return self.head(self._h).squeeze(0)


def train_fair_gnn(
    seed=0,
    M=32,
    n_episodes=80,
    seq_len=30,
    epochs=5,
    lr=2e-3,
):
    """Train FairSTGNNEncoder over continuous sequences (GRU reset once/episode)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    enc = FairSTGNNEncoder(feat_dim=4, hidden=32, latent=16, layers=2, M0=M)
    clf = nn.Linear(enc.latent, len(REGIMES))
    opt = torch.optim.Adam(
        list(enc.parameters()) + list(clf.parameters()), lr=lr
    )
    rng = np.random.default_rng(seed)

    enc.train()
    for _ in range(epochs):
        for _ in range(n_episodes):
            gen = FogGraphGenerator(M, seed=int(rng.integers(1, 10**9)))
            regime = str(rng.choice(REGIMES))
            target = torch.tensor([REGIMES.index(regime)], dtype=torch.long)
            enc.reset()
            opt.zero_grad()
            losses = []
            # Brief 5-step S1 prefix on half the episodes so transitions are seen
            if regime != "S1_nominal" and rng.random() < 0.5:
                for t in range(5):
                    A, X = gen.step("S1_nominal", t)
                    _ = enc(A, X)
            for t in range(seq_len):
                A, X = gen.step(regime, t)
                z = enc(A, X)
                losses.append(F.cross_entropy(clf(z).unsqueeze(0), target))
            loss = torch.stack(losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            opt.step()

    enc.eval()
    return enc


# =========================================================================
# 9. SYMMETRIC OR-TRIGGERING TRI-TRACK FUSION + FAULT ATTRIBUTION
# =========================================================================

SCENARIO_TRUE_ATTRIBUTION = {
    "S1->S2": "node_disruption",
    "S1->S3": "link_degradation",
    "S1->S4": "node_disruption",
    "S1->S5": "topology_reconfig",
    "S2->S3": "link_degradation",
}

ATTRIBUTION_CLASSES = [
    "node_disruption",
    "link_degradation",
    "topology_reconfig",
]


class TriTrackFusion:
    """Symmetric OR-triggering tri-track detector with causal fault attribution.

    Any of the three tracks (each calibrated at target_alpha / 3 so the union
    respects the global false-alarm budget) raises an alarm immediately. A
    non-truncating `confirm_window` records complementary track confirmations
    independently and updates a causal 3-way structural attribution:
      - 'node_disruption':   active node ratio changes in [t_d - 1, t]
      - 'link_degradation':  node ratio stable & active edge density drops
      - 'topology_reconfig': node ratio stable & active edge density rises/rewires
    """

    def __init__(self, v1, v2, v3, confirm_window=10, refractory=10):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.confirm_window = int(confirm_window)
        self.refractory = int(refractory)
        self.reset()

    def reset(self):
        self.v1.reset()
        self.v2.reset()
        self.v3.reset()
        self.events = []
        self._open_events = []
        self.alarms = []
        self.last_alarm = -10**9
        self._n_hist = []
        self._dens_hist = []

    @staticmethod
    def _classify_attribution(n_min, n_max, active_dens, ref_dens):
        if (n_max - n_min) > 1e-3:
            return "node_disruption"
        if active_dens < ref_dens - 0.005:
            return "link_degradation"
        return "topology_reconfig"

    def step(self, stats_vec, ph_vec, spec_vec, t):
        n_ratio = float(stats_vec[0])
        active_dens = float(stats_vec[1]) / max(n_ratio * n_ratio, 1e-6)
        self._n_hist.append(n_ratio)
        self._dens_hist.append(active_dens)

        self.v1.step(stats_vec, t)
        self.v2.step(ph_vec, t)
        self.v3.step(spec_vec, t)

        v1_fired = bool(self.v1.alarms and self.v1.alarms[-1] == t)
        v2_fired = bool(self.v2.alarms and self.v2.alarms[-1] == t)
        v3_fired = bool(self.v3.alarms and self.v3.alarms[-1] == t)
        any_fired = v1_fired or v2_fired or v3_fired

        triggered_now = False
        if any_fired and (t - self.last_alarm > self.refractory):
            triggered_now = True
            self.last_alarm = t
            self.alarms.append(int(t))

            triggers = []
            if v1_fired:
                triggers.append("Track1")
            if v2_fired:
                triggers.append("Track2")
            if v3_fired:
                triggers.append("Track3")

            # Causal pre-alarm active-density reference (lagged to avoid shift)
            hist_len = len(self._dens_hist)
            ref_lo = max(0, hist_len - 21)
            ref_hi = max(1, hist_len - 6)
            if ref_hi > ref_lo:
                ref_dens = float(np.mean(self._dens_hist[ref_lo:ref_hi]))
            else:
                ref_dens = self._dens_hist[max(0, hist_len - 2)]

            n_pre = self._n_hist[max(0, hist_len - 2)]
            n_min = min(n_pre, n_ratio)
            n_max = max(n_pre, n_ratio)
            init_attr = self._classify_attribution(
                n_min, n_max, active_dens, ref_dens
            )

            ev = {
                "t": int(t),
                "triggers": triggers,
                "t_v1": int(t) if v1_fired else None,
                "t_v2": int(t) if v2_fired else None,
                "t_v3": int(t) if v3_fired else None,
                "n_tracks": len(triggers),
                "t_confirm": int(t) if len(triggers) >= 2 else None,
                "confirm_latency": 0 if len(triggers) >= 2 else None,
                "status": "confirmed" if len(triggers) >= 2 else "pending",
                "attribution": init_attr,
                "attribution_history": [(int(t), init_attr)],
                "_ref_dens": ref_dens,
                "_n_min": n_min,
                "_n_max": n_max,
            }
            self.events.append(ev)
            self._open_events.append(ev)

        # Update all open confirmation windows without premature closing
        still_open = []
        for ev in self._open_events:
            if v1_fired and ev["t_v1"] is None:
                ev["t_v1"] = int(t)
            if v2_fired and ev["t_v2"] is None:
                ev["t_v2"] = int(t)
            if v3_fired and ev["t_v3"] is None:
                ev["t_v3"] = int(t)

            n_active = sum(
                1 for k in ("t_v1", "t_v2", "t_v3") if ev[k] is not None
            )
            ev["n_tracks"] = n_active
            if ev["t_confirm"] is None and n_active >= 2:
                ev["t_confirm"] = int(t)
                ev["confirm_latency"] = int(t - ev["t"])
                ev["status"] = "confirmed"

            # Causal attribution refinement as post-alarm steps arrive
            ev["_n_min"] = min(ev["_n_min"], n_ratio)
            ev["_n_max"] = max(ev["_n_max"], n_ratio)
            new_attr = self._classify_attribution(
                ev["_n_min"], ev["_n_max"], active_dens, ev["_ref_dens"]
            )
            if new_attr != ev["attribution"]:
                ev["attribution"] = new_attr
                ev["attribution_history"].append((int(t), new_attr))

            if t - ev["t"] >= self.confirm_window:
                if ev["status"] == "pending":
                    ev["status"] = "single_track"
            else:
                still_open.append(ev)

        self._open_events = still_open
        return triggered_now

    def finalize(self):
        for ev in self._open_events:
            if ev["status"] == "pending":
                ev["status"] = "single_track"
        self._open_events = []

    @staticmethod
    def summarize_events(events, ct, true_attribution=None):
        post = [ev for ev in events if ev["t"] >= ct]

        def _tally(ev_list):
            trig_t1 = sum(1 for e in ev_list if "Track1" in e["triggers"])
            trig_t2 = sum(1 for e in ev_list if "Track2" in e["triggers"])
            trig_t3 = sum(1 for e in ev_list if "Track3" in e["triggers"])

            active_t1 = sum(1 for e in ev_list if e["t_v1"] is not None)
            active_t2 = sum(1 for e in ev_list if e["t_v2"] is not None)
            active_t3 = sum(1 for e in ev_list if e["t_v3"] is not None)

            single = sum(1 for e in ev_list if e["n_tracks"] == 1)
            two = sum(1 for e in ev_list if e["n_tracks"] == 2)
            all3 = sum(1 for e in ev_list if e["n_tracks"] == 3)

            lats = [
                e["confirm_latency"]
                for e in ev_list
                if e["confirm_latency"] is not None
            ]
            attr_counts = {c: 0 for c in ATTRIBUTION_CLASSES}
            for e in ev_list:
                attr_counts[e["attribution"]] = (
                    attr_counts.get(e["attribution"], 0) + 1
                )
            n_correct = (
                attr_counts.get(true_attribution, 0)
                if true_attribution is not None
                else 0
            )
            return {
                "n_total": len(ev_list),
                "n_confirmed": two + all3,
                "single_track": single,
                "two_tracks": two,
                "all_three": all3,
                "trigger_t1": trig_t1,
                "trigger_t2": trig_t2,
                "trigger_t3": trig_t3,
                "active_t1": active_t1,
                "active_t2": active_t2,
                "active_t3": active_t3,
                "confirm_latencies": lats,
                "attr_counts": attr_counts,
                "attr_correct": n_correct,
            }

        first_list = [post[0]] if post else []
        return {
            "first_alarm": _tally(first_list),
            "all_post_change": _tally(post),
        }


# =========================================================================
# 10. EMPIRICAL CALIBRATION & ARL0 ESTIMATION (INCLUDING FUSED TRI-TRACK)
# =========================================================================

def calibrate_cusum_kappa(stream_list, alpha=0.05, warmup=10, k_sigma=1.0):
    """Estimate CUSUM drift allowance kappa from held-out H0 EWMA deviations."""
    devs = []
    for seq in stream_list:
        mu = None
        for t, x in enumerate(seq):
            x = np.asarray(x, dtype=np.float32)
            if mu is None:
                mu = x.copy()
                continue
            d = float(np.linalg.norm(x - mu))
            if t >= warmup:
                devs.append(d)
            mu = (1.0 - alpha) * mu + alpha * x
    arr = np.asarray(devs, dtype=float)
    kappa = float(arr.mean() + k_sigma * arr.std())
    return max(kappa, 1e-4)


def calibrate_threshold_empirical(
    det_factory,
    stream_list,
    target_alpha=0.005,
    warmup=20,
    lo=1e-4,
    hi=10.0,
    max_hi=1e5,
    n_iter=18,
):
    """Find tightest threshold achieving empirical alpha_step <= target_alpha."""

    def _eval_alpha(thresh):
        total_alarms = 0
        total_active = 0
        for seq in stream_list:
            det = det_factory(thresh)
            for t, x in enumerate(seq):
                det.step(x, t)
            total_alarms += sum(1 for a in det.alarms if a >= warmup)
            total_active += max(len(seq) - warmup, 0)
        return total_alarms / max(total_active, 1)

    cur_hi = float(hi)
    alpha_hi = _eval_alpha(cur_hi)
    while alpha_hi > target_alpha and cur_hi < max_hi:
        cur_hi = min(cur_hi * 2.0, max_hi)
        alpha_hi = _eval_alpha(cur_hi)

    cur_lo = float(lo)
    best_th = cur_hi
    best_alpha = alpha_hi
    for _ in range(n_iter):
        mid = 0.5 * (cur_lo + cur_hi)
        a_mid = _eval_alpha(mid)
        if a_mid <= target_alpha:
            best_th = mid
            best_alpha = a_mid
            cur_hi = mid
        else:
            cur_lo = mid
    return float(best_th), float(best_alpha)


def estimate_empirical_arl0(det_factory, long_streams, warmup=20):
    """Estimate empirical ARL0 and per-step FAR on long stationary H0 runs."""
    total_alarms = 0
    total_active = 0
    first_alarm_rl = []
    for seq in long_streams:
        det = det_factory()
        for t, x in enumerate(seq):
            det.step(x, t)
        active = [a for a in det.alarms if a >= warmup]
        total_alarms += len(active)
        active_len = max(len(seq) - warmup, 0)
        total_active += active_len
        if active:
            first_alarm_rl.append(float(active[0] - warmup))
        else:
            first_alarm_rl.append(float(active_len))

    alpha_step = total_alarms / max(total_active, 1)
    arl0_mle = total_active / max(total_alarms, 1)
    return {
        "total_active_steps": int(total_active),
        "total_alarms": int(total_alarms),
        "alpha_step": float(alpha_step),
        "arl0_mle": float(arl0_mle),
        "arl0_censored": bool(total_alarms == 0),
        "mean_first_rl_censored": float(np.mean(first_alarm_rl)),
    }


def estimate_tritrack_arl0(fusion_factory, streams_dict, warmup=20):
    """Estimate empirical ARL0 directly on the OR-fused TriTrackFusion."""
    total_alarms = 0
    total_active = 0
    first_alarm_rl = []
    n_eps = len(streams_dict["stats"])
    for ep in range(n_eps):
        fusion = fusion_factory()
        s_seq = streams_dict["stats"][ep]
        p_seq = streams_dict["ph"][ep]
        sp_seq = streams_dict["spec"][ep]
        for t in range(len(s_seq)):
            fusion.step(s_seq[t], p_seq[t], sp_seq[t], t)
        fusion.finalize()
        active = [a for a in fusion.alarms if a >= warmup]
        total_alarms += len(active)
        active_len = max(len(s_seq) - warmup, 0)
        total_active += active_len
        if active:
            first_alarm_rl.append(float(active[0] - warmup))
        else:
            first_alarm_rl.append(float(active_len))

    alpha_step = total_alarms / max(total_active, 1)
    arl0_mle = total_active / max(total_alarms, 1)
    return {
        "total_active_steps": int(total_active),
        "total_alarms": int(total_alarms),
        "alpha_step": float(alpha_step),
        "arl0_mle": float(arl0_mle),
        "arl0_censored": bool(total_alarms == 0),
        "mean_first_rl_censored": float(np.mean(first_alarm_rl)),
    }


# =========================================================================
# 11. TIMEOUT-FREE DOWNSTREAM SCHEDULER BENCHMARK
# =========================================================================

SCHEDULER_COST_TABLE = {
    "S1_nominal": {"nominal": 0.00, "conservative": 0.15, "aggressive": 0.60},
    "S2_churn": {"nominal": 1.00, "conservative": 0.55, "aggressive": 0.15},
    "S3_link_degradation": {
        "nominal": 0.45,
        "conservative": 0.10,
        "aggressive": 0.55,
    },
    "S4_cascading": {"nominal": 1.00, "conservative": 0.60, "aggressive": 0.15},
    "S5_reconfig": {"nominal": 0.85, "conservative": 0.45, "aggressive": 0.15},
}

ATTRIBUTION_TO_MODE = {
    "node_disruption": "aggressive",
    "link_degradation": "conservative",
    "topology_reconfig": "aggressive",
}


def _optimal_mode_for_regime(regime):
    costs = SCHEDULER_COST_TABLE[regime]
    return min(costs, key=costs.get)


def _optimal_step_cost(regime):
    return min(SCHEDULER_COST_TABLE[regime].values())


def evaluate_downstream_schedulers(regime_at, scalar_alarms, tri_events):
    """Timeout-free downstream scheduling regret across four causal policies:
      1. Static-PreRegime: remains in the pre-change regime's optimal mode.
      2. Scalar-Aggressive (Track 3 scalar alarm -> always aggressive).
      3. Scalar-Conservative (Track 3 scalar alarm -> always conservative).
      4. Attributed-TriTrack (uses strictly causal attribution_history from
         Tri-track to route node_disruption/topology_reconfig -> aggressive
         and link_degradation -> conservative, with no artificial timeout).
    """
    T = len(regime_at)
    start_mode = _optimal_mode_for_regime(regime_at[0])
    scalar_set = set(int(a) for a in scalar_alarms)

    # Causal single-transition latch: lock onto the first event that moves
    # the policy away from start_mode (allowing causal updates within that
    # event's confirmation window), matching the single-switch scalar policies.
    attr_updates = {}
    latched_event_id = None
    for idx, ev in enumerate(tri_events):
        for t_upd, attr_lbl in ev["attribution_history"]:
            mode = ATTRIBUTION_TO_MODE[attr_lbl]
            if latched_event_id is None:
                if mode != start_mode:
                    latched_event_id = idx
                    attr_updates[int(t_upd)] = mode
            elif latched_event_id == idx:
                attr_updates[int(t_upd)] = mode
        if latched_event_id is not None:
            break

    regret = {
        "Static-PreRegime": 0.0,
        "Scalar-Aggressive": 0.0,
        "Scalar-Conservative": 0.0,
        "Attributed-TriTrack": 0.0,
    }

    mode_s_agg = start_mode
    mode_s_con = start_mode
    mode_attr = start_mode

    for t in range(T):
        reg = regime_at[t]
        opt_c = _optimal_step_cost(reg)

        if t in scalar_set:
            mode_s_agg = "aggressive"
            mode_s_con = "conservative"
        if t in attr_updates:
            mode_attr = attr_updates[t]

        regret["Static-PreRegime"] += (
            SCHEDULER_COST_TABLE[reg][start_mode] - opt_c
        )
        regret["Scalar-Aggressive"] += (
            SCHEDULER_COST_TABLE[reg][mode_s_agg] - opt_c
        )
        regret["Scalar-Conservative"] += (
            SCHEDULER_COST_TABLE[reg][mode_s_con] - opt_c
        )
        regret["Attributed-TriTrack"] += (
            SCHEDULER_COST_TABLE[reg][mode_attr] - opt_c
        )

    return regret