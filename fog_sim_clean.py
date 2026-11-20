"""
fog_sim_clean.py
================
Bug-fixed Fog-Cloud-Edge dynamic attributed graph simulator, deterministic
tri-track feature extractors, fair sequential GNN ablation encoder, and
rigorous statistical evaluation utilities.

Updates in v2:
  - Fair STGNNEncoder: appends normalized node degree to node features and
    combines mean pooling with normalized sum pooling ([mean, sum / M0]),
    preserving permutation invariance and size-independent latent dimension d
    while remaining sensitive to churn and link loss.
  - Added `compute_attribution_metrics` for formal multi-class fault-type
    isolation evaluation (accuracy, macro-F1, and confusion counts).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as scipy_stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh

try:
    import gudhi
    HAS_TDA = True
except ImportError:
    HAS_TDA = False


REGIMES = [
    "S1_nominal",
    "S2_churn",
    "S3_link_degradation",
    "S4_cascading",
    "S5_reconfig",
]


# =========================================================================
# 1. DYNAMIC FOG GRAPH GENERATOR (STATIONARY H0 + DISTINCT S2-S5)
# =========================================================================

class FogGraphGenerator:
    """Time-varying attributed Fog graph G(t) = (V(t), E(t), X(t), A(t))."""

    def __init__(
        self,
        num_nodes=32,
        feat_dim=4,
        p0=0.10,
        p_flicker=0.01,
        phi=0.85,
        feat_sigma=0.02,
        seed=0,
    ):
        self.M0 = num_nodes
        self.feat_dim = feat_dim
        self.p0 = p0
        self.p_flicker = p_flicker
        self.p_on = p_flicker * p0 / max(1.0 - p0, 1e-6)
        self.phi = phi
        self.feat_sigma = feat_sigma
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.seed)
        self.adj = self._erdos_renyi(self.M0, self.p0)
        self.mu_feat = self.rng.uniform(
            0.2, 0.6, size=(self.M0, self.feat_dim)
        ).astype(np.float32)
        self.features = self.mu_feat.copy()
        self.alive = np.ones(self.M0, dtype=bool)
        self._cascade_next = set()

    def _erdos_renyi(self, n, p):
        A = (self.rng.random((n, n)) < p).astype(np.float32)
        A = np.triu(A, 1)
        return A + A.T

    def _nominal_edge_flicker(self):
        """Stationary Markov link flicker preserving edge density p0."""
        if self.p_flicker <= 0.0:
            return
        iu, ju = np.triu_indices(self.M0, k=1)
        alive_pair = self.alive[iu] & self.alive[ju]
        if not np.any(alive_pair):
            return
        u = iu[alive_pair]
        v = ju[alive_pair]
        present = self.adj[u, v] > 0.5
        rand_vals = self.rng.random(len(u))
        drop_mask = present & (rand_vals < self.p_flicker)
        add_mask = (~present) & (rand_vals < self.p_on)
        if np.any(drop_mask):
            self.adj[u[drop_mask], v[drop_mask]] = 0.0
            self.adj[v[drop_mask], u[drop_mask]] = 0.0
        if np.any(add_mask):
            self.adj[u[add_mask], v[add_mask]] = 1.0
            self.adj[v[add_mask], u[add_mask]] = 1.0

    def _revive_node(self, node_idx):
        self.alive[node_idx] = True
        self.features[node_idx] = self.mu_feat[node_idx].copy()
        other_alive = np.where(self.alive)[0]
        other_alive = other_alive[other_alive != node_idx]
        if len(other_alive) > 0:
            links = (self.rng.random(len(other_alive)) < self.p0).astype(
                np.float32
            )
            self.adj[node_idx, other_alive] = links
            self.adj[other_alive, node_idx] = links

    def step(self, regime, t):
        # Stationary mean-reverting AR(1) node feature dynamics
        noise = self.rng.normal(
            0.0, self.feat_sigma, size=self.features.shape
        ).astype(np.float32)
        self.features = np.clip(
            self.mu_feat + self.phi * (self.features - self.mu_feat) + noise,
            0.0,
            1.0,
        )

        if regime == "S1_nominal":
            self._cascade_next.clear()
            self._nominal_edge_flicker()

        elif regime == "S2_churn":
            self._cascade_next.clear()
            n_toggle = max(1, int(self.rng.uniform(0.05, 0.15) * self.M0))
            idx = self.rng.choice(self.M0, size=n_toggle, replace=False)
            for i in idx:
                if self.alive[i]:
                    self.alive[i] = False
                    self.adj[i, :] = 0.0
                    self.adj[:, i] = 0.0
                else:
                    self._revive_node(int(i))
            if int(self.alive.sum()) < self.M0 // 2:
                dead = np.where(~self.alive)[0]
                need = (self.M0 // 2) - int(self.alive.sum())
                revive_ids = self.rng.choice(dead, size=need, replace=False)
                for i in revive_ids:
                    self._revive_node(int(i))

        elif regime == "S3_link_degradation":
            self._cascade_next.clear()
            iu, ju = np.triu_indices(self.M0, k=1)
            existing = np.where(
                (self.adj[iu, ju] > 0.5) & self.alive[iu] & self.alive[ju]
            )[0]
            if len(existing) > 0:
                n_drop = max(1, int(round(0.06 * len(existing))))
                n_drop = min(n_drop, len(existing))
                chosen = self.rng.choice(existing, size=n_drop, replace=False)
                for c in chosen:
                    u, v = iu[c], ju[c]
                    self.adj[u, v] = 0.0
                    self.adj[v, u] = 0.0

        elif regime == "S4_cascading":
            alive_idx = np.where(self.alive)[0]
            min_alive = max(4, self.M0 // 4)
            budget = max(0, len(alive_idx) - min_alive)
            if budget > 0:
                target_fail = min(
                    budget, max(1, int(self.rng.uniform(0.05, 0.15) * self.M0))
                )
                candidates = [i for i in self._cascade_next if self.alive[i]]
                failed_now = []
                if candidates:
                    n_pick = min(target_fail, len(candidates))
                    chosen = self.rng.choice(
                        candidates, size=n_pick, replace=False
                    )
                    failed_now.extend(chosen.tolist())

                remaining = target_fail - len(failed_now)
                if remaining > 0:
                    pool = np.array(
                        [i for i in alive_idx if i not in failed_now], dtype=int
                    )
                    if len(pool) > 0:
                        deg = self.adj[pool][:, alive_idx].sum(axis=1) + 1.0
                        probs = deg / deg.sum()
                        n_pick = min(remaining, len(pool))
                        chosen = self.rng.choice(
                            pool, size=n_pick, replace=False, p=probs
                        )
                        failed_now.extend(chosen.tolist())

                next_q = set()
                for i in failed_now:
                    nbrs = np.where((self.adj[i] > 0.5) & self.alive)[0]
                    for nb in nbrs:
                        if nb not in failed_now:
                            self.features[nb, 0] = min(
                                1.0, float(self.features[nb, 0]) + 0.10
                            )
                            next_q.add(int(nb))
                for i in failed_now:
                    self.alive[i] = False
                    self.adj[i, :] = 0.0
                    self.adj[:, i] = 0.0
                self._cascade_next = next_q

        elif regime == "S5_reconfig":
            self._cascade_next.clear()
            self.alive = np.ones(self.M0, dtype=bool)
            self.adj = self._erdos_renyi(self.M0, 0.20)

        return self.get_graph()

    def get_graph(self):
        idx = np.where(self.alive)[0]
        return self.adj[np.ix_(idx, idx)].copy(), self.features[idx].copy()


# =========================================================================
# 2. TRACK 1: GRAPH STATISTICS
# =========================================================================

def compute_graph_stats(A, X, M0=32):
    """6D Track 1 feature vector:
      [0] alive node ratio: |V(t)| / M0
      [1] normalized edge count: |E(t)| / (M0 * (M0 - 1) / 2)
      [2] normalized mean degree: (2 |E(t)| / |V(t)|) / (M0 - 1)
      [3] mean CPU utilization
      [4] mean memory utilization
      [5] mean battery level
    """
    n = A.shape[0]
    if n == 0:
        return np.zeros(6, dtype=np.float32)
    n_e = float(A.sum()) / 2.0
    max_edges = max(M0 * (M0 - 1) / 2.0, 1.0)
    mean_deg = (2.0 * n_e) / max(n, 1)
    return np.array(
        [
            float(n) / max(M0, 1),
            n_e / max_edges,
            mean_deg / max(M0 - 1, 1),
            float(X[:, 0].mean()),
            float(X[:, 1].mean()),
            float(X[:, 2].mean()),
        ],
        dtype=np.float32,
    )


# =========================================================================
# 3. TRACK 2: PERSISTENT HOMOLOGY (2-SIMPLEX EXPANSION, SINGLE PASS)
# =========================================================================

def compute_persistence(A, X, max_dim=1):
    """Clique-complex filtration on cosine feature dissimilarity over edges."""
    n = A.shape[0]
    if n == 0:
        return {d: np.zeros((0, 2), dtype=np.float32) for d in range(max_dim + 1)}

    if not HAS_TDA:
        _, labels = connected_components(A, directed=False)
        n_c = int(labels.max() + 1) if len(labels) > 0 else 0
        out = {0: np.array([[0.0, n_c / max(n, 1)]], dtype=np.float32)}
        for d in range(1, max_dim + 1):
            out[d] = np.zeros((0, 2), dtype=np.float32)
        return out

    feats = X.astype(np.float64) + 1e-8
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    feats_n = feats / np.maximum(norms, 1e-8)

    st = gudhi.SimplexTree()
    for i in range(n):
        st.insert([i], filtration=0.0)
    iu, ju = np.triu_indices(n, k=1)
    edge_idx = np.where(A[iu, ju] > 0.5)[0]
    if len(edge_idx) > 0:
        u = iu[edge_idx]
        v = ju[edge_idx]
        cos_sim = np.sum(feats_n[u] * feats_n[v], axis=1)
        dists = np.clip(1.0 - cos_sim, 0.0, 1.0)
        for ui, vi, di in zip(u, v, dists):
            st.insert([int(ui), int(vi)], filtration=float(di))

    if max_dim >= 1:
        st.expansion(max_dim + 1)

    st.initialize_filtration()
    st.compute_persistence(homology_coeff_field=2)

    diags = {}
    for dim in range(max_dim + 1):
        pairs = st.persistence_intervals_in_dimension(dim)
        if len(pairs) > 0:
            arr = np.array(pairs, dtype=np.float32)
            arr[~np.isfinite(arr[:, 1]), 1] = 1.0
            diags[dim] = arr
        else:
            diags[dim] = np.zeros((0, 2), dtype=np.float32)
    return diags


def compute_ph_signature(A, X, max_dim=1, pers_threshold=0.005):
    """Single-pass PH signature returning 6D feature vector and 4D summary."""
    diags = compute_persistence(A, X, max_dim=max_dim)
    feats = []
    summary = np.zeros(2 * (max_dim + 1), dtype=np.float32)
    for dim in range(max_dim + 1):
        d = diags.get(dim, np.zeros((0, 2), dtype=np.float32))
        if len(d) == 0:
            feats.extend([0.0, 0.0, 0.0])
            continue
        pers = d[:, 1] - d[:, 0]
        summary[dim * 2] = float(len(d))
        summary[dim * 2 + 1] = float(pers.max())
        pers_filt = pers[pers > pers_threshold]
        if len(pers_filt) == 0:
            feats.extend([0.0, 0.0, 0.0])
        else:
            feats.extend(
                [
                    float(len(pers_filt)),
                    float(pers_filt.sum()),
                    float(pers_filt.max()),
                ]
            )
    return {
        "features": np.array(feats, dtype=np.float32),
        "summary": summary,
        "diagrams": diags,
    }


# =========================================================================
# 4. TRACK 3: NORMALIZED LAPLACIAN SPECTRAL SIGNATURE
# =========================================================================

def compute_spectral_signature(A, k=10, normalized=True, use_sparse=False):
    """Extract k smallest non-trivial eigenvalues of the normalized Laplacian."""
    n = A.shape[0]
    if n <= 1:
        return np.zeros(k, dtype=np.float32)

    d = A.sum(axis=1)
    if normalized:
        d_safe = np.maximum(d, 1e-8)
        d_inv_sqrt = np.where(d > 1e-8, 1.0 / np.sqrt(d_safe), 0.0)
        L = np.eye(n, dtype=np.float64) - (
            d_inv_sqrt[:, None] * A.astype(np.float64) * d_inv_sqrt[None, :]
        )
    else:
        L = np.diag(d.astype(np.float64)) - A.astype(np.float64)

    k_actual = min(k, n - 1)
    try:
        if use_sparse and n > k_actual + 4:
            L_sp = csr_matrix(L)
            vals = eigsh(
                L_sp,
                k=k_actual + 1,
                which="SM",
                return_eigenvectors=False,
                tol=1e-4,
                maxiter=500,
            )
            vals = np.sort(np.real(vals))
        else:
            vals = np.linalg.eigvalsh(L)
        sig = np.clip(vals[1 : k_actual + 1], 0.0, None).astype(np.float32)
        if len(sig) < k:
            sig = np.concatenate(
                [sig, np.zeros(k - len(sig), dtype=np.float32)]
            )
        return sig
    except Exception:
        vals = np.linalg.eigvalsh(L)
        sig = np.clip(vals[1 : k_actual + 1], 0.0, None).astype(np.float32)
        if len(sig) < k:
            sig = np.concatenate(
                [sig, np.zeros(k - len(sig), dtype=np.float32)]
            )
        return sig


# =========================================================================
# 5. FAIR LEARNED GNN ABLATION (DEGREE FEATURE + MEAN & SUM POOLING)
# =========================================================================

class SpatialMP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.Linear(dim, dim)
        self.ls = nn.Linear(dim, dim)

    def forward(self, x, A):
        deg = A.sum(1, keepdim=True).clamp(min=1.0)
        return F.relu(self.ln(A @ x / deg) + self.ls(x))


class STGNNEncoder(nn.Module):
    """2-layer GraphSAGE + [mean, sum/M0] permutation-invariant pooling + GRU.

    Appends each node's normalized degree deg_i / (M0 - 1) to its attribute
    vector and concatenates mean pooling with normalized sum pooling so the
    learned encoder can observe structural churn and link degradation while
    preserving exact permutation invariance and fixed output dimension `latent`.
    """

    def __init__(self, feat_dim=4, hidden=32, latent=16, layers=2, M0=32):
        super().__init__()
        self.M0 = max(int(M0), 2)
        self.proj = nn.Linear(feat_dim + 1, hidden)
        self.layers = nn.ModuleList([SpatialMP(hidden) for _ in range(layers)])
        self.gru = nn.GRUCell(2 * hidden, hidden)
        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, latent)
        )
        self.latent = latent
        self._h = None

    def reset(self):
        self._h = None

    def forward(self, A, X):
        A_t = torch.tensor(A, dtype=torch.float32)
        X_t = torch.tensor(X, dtype=torch.float32)
        deg = A_t.sum(dim=1, keepdim=True) / float(self.M0 - 1)
        x_in = torch.cat([X_t, deg], dim=1)
        h = F.relu(self.proj(x_in))
        for layer in self.layers:
            h = layer(h, A_t)
        g_mean = h.mean(dim=0, keepdim=True)
        g_sum = h.sum(dim=0, keepdim=True) / float(self.M0)
        g = torch.cat([g_mean, g_sum], dim=1)
        if self._h is None:
            self._h = torch.zeros(1, h.shape[1], dtype=torch.float32)
        self._h = self.gru(g, self._h)
        return self.readout(self._h).squeeze(0)


def train_gnn(
    seed=0,
    M=32,
    n_episodes=60,
    seq_len=25,
    epochs=6,
    lr=2e-3,
):
    """Train STGNNEncoder sequentially across balanced regimes (GRU reset per episode)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    enc = STGNNEncoder(feat_dim=4, hidden=32, latent=16, layers=2, M0=M)
    clf = nn.Linear(enc.latent, len(REGIMES))
    opt = torch.optim.Adam(
        list(enc.parameters()) + list(clf.parameters()), lr=lr
    )
    rng = np.random.default_rng(seed)

    enc.train()
    for _ in range(epochs):
        for ep_idx in range(n_episodes):
            gen = FogGraphGenerator(M, seed=int(rng.integers(1, 10**9)))
            regime = REGIMES[ep_idx % len(REGIMES)]
            target = torch.tensor([REGIMES.index(regime)], dtype=torch.long)
            enc.reset()
            opt.zero_grad()
            losses = []
            # Mix pure-regime and transition-from-S1 sequences so GRU sees shifts
            if regime != "S1_nominal" and (ep_idx % 2 == 1):
                split = seq_len // 2
                for t in range(seq_len):
                    r_cur = "S1_nominal" if t < split else regime
                    t_cur = torch.tensor(
                        [REGIMES.index(r_cur)], dtype=torch.long
                    )
                    A, X = gen.step(r_cur, t)
                    z = enc(A, X)
                    losses.append(F.cross_entropy(clf(z).unsqueeze(0), t_cur))
            else:
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
# 6. STATISTICAL METRICS, TESTS & FAULT-ATTRIBUTION EVALUATION
# =========================================================================

def metrics_from_alarms(alarms, ct, T, warmup=0):
    """Compute delay, miss indicator, and active pre-change false alarms."""
    early = [a for a in alarms if warmup <= a < ct]
    valid = [a for a in alarms if a >= ct]
    delay = int(valid[0] - ct) if valid else None
    return {
        "delay": delay,
        "missed": int(delay is None),
        "fa": len(early),
        "active_pre_steps": max(ct - warmup, 1),
        "n_alarms": len(alarms),
    }


def hierarchical_bootstrap_ci(
    values_by_seed, n_boot=2000, alpha=0.05, seed=42
):
    """True 95% hierarchical bootstrap CI: resample seeds, then replications."""
    vals = [np.asarray(v, dtype=float) for v in values_by_seed if len(v) > 0]
    if not vals:
        return (float("nan"), float("nan"))
    n_seeds = len(vals)
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        s_idx = rng.integers(0, n_seeds, size=n_seeds)
        means = []
        for i in s_idx:
            arr = vals[i]
            sub = rng.choice(arr, size=len(arr), replace=True)
            means.append(sub.mean())
        boot_means[b] = np.mean(means)
    return (
        float(np.quantile(boot_means, alpha / 2.0)),
        float(np.quantile(boot_means, 1.0 - alpha / 2.0)),
    )


def student_t_ci(per_seed_means, alpha=0.05):
    """Exact Student-t 95% confidence interval over seed means (ddof=1)."""
    arr = np.asarray(per_seed_means, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    if n < 2:
        return (mean, mean)
    sd = float(arr.std(ddof=1))
    if sd == 0.0:
        return (mean, mean)
    t_crit = float(scipy_stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
    half = t_crit * sd / np.sqrt(n)
    return (mean - half, mean + half)


def paired_permutation_pvalue(ref_flat, other_flat, n_perm=20000, seed=42):
    """Two-sided paired sign-flip permutation test across matched episodes."""
    ref = np.asarray(ref_flat, dtype=float)
    other = np.asarray(other_flat, dtype=float)
    if len(ref) == 0 or len(ref) != len(other):
        return 1.0
    diffs = other - ref
    obs = abs(float(diffs.mean()))
    if obs == 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    chunk = 2000
    n_extreme = 0
    done = 0
    while done < n_perm:
        m = min(chunk, n_perm - done)
        signs = rng.choice([-1.0, 1.0], size=(m, len(diffs)))
        perm_means = np.abs((signs * diffs[None, :]).mean(axis=1))
        n_extreme += int(np.sum(perm_means >= obs - 1e-12))
        done += m
    return float((n_extreme + 1) / (n_perm + 1))


def compute_worst_case(delays_by_seed, T):
    """Compute max, p95, and median across all runs (missed -> T)."""
    all_d = [
        (d if d is not None else T)
        for seed_list in delays_by_seed
        for d in seed_list
    ]
    arr = np.asarray(all_d, dtype=float)
    if len(arr) == 0:
        return {"max": float(T), "p95": float(T), "median": float(T)}
    return {
        "max": float(arr.max()),
        "p95": float(np.percentile(arr, 95)),
        "median": float(np.median(arr)),
    }


def compute_attribution_metrics(
    y_true,
    y_pred,
    classes=("node_disruption", "link_degradation", "topology_reconfig"),
):
    """Compute overall accuracy, macro-F1, and confusion matrix for fault attribution."""
    n = len(y_true)
    if n == 0:
        return {"n": 0, "accuracy": 0.0, "macro_f1": 0.0, "confusion": {}}
    acc = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / float(n)
    confusion = {c: {c2: 0 for c2 in classes} for c in classes}
    for yt, yp in zip(y_true, y_pred):
        if yt in confusion and yp in confusion[yt]:
            confusion[yt][yp] += 1

    f1s = []
    per_class = {}
    for c in classes:
        tp = confusion[c][c]
        fp = sum(confusion[other][c] for other in classes if other != c)
        fn = sum(confusion[c][other] for other in classes if other != c)
        prec = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class[c] = {
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "support": int(tp + fn),
        }
        if tp + fn > 0:
            f1s.append(f1)
    return {
        "n": int(n),
        "accuracy": float(acc),
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "per_class": per_class,
        "confusion": confusion,
    }