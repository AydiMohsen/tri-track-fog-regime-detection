"""
run_benchmark_clean.py
======================
End-to-end, reproducible Q1-grade evaluation pipeline for symmetric OR-triggering
Tri-Track structural regime detection & fault attribution on Fog-Cloud-Edge
graphs.

Upgrades over v1:
  - Calibrates Tri-Track's three component CUSUMs at target_alpha / 3 each and
    verifies the fused Tri-Track's empirical ARL0 directly on long S1 runs.
  - Evaluates causal 3-way structural attribution (per-scenario accuracy,
    confusion matrix, and overall macro-F1 across node_disruption,
    link_degradation, and topology_reconfig).
  - Runs the timeout-free 4-policy downstream scheduler benchmark
    (Static-PreRegime vs Scalar-Aggressive vs Scalar-Conservative vs
    Attributed-TriTrack) with paired permutation tests against both scalar
    policies.
  - Trains and per-seed calibrates FairSTGNNEncoder (degree feature +
    [mean, sum/M0] readout) as a fair, non-strawman learned GNN ablation.
  - Generates clean English vector PDF figures (no embedded titles, no French
    labels, no outdated "certificate" terminology).

Usage:
  Quick smoke test (~2-3 min):
      python run_benchmark_clean.py --quick
  Full evaluation (10 seeds x 10 reps):
      python run_benchmark_clean.py --seeds 10 --reps 10 --output results_clean
"""

import argparse
import json
import os
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.distance import pdist

from fog_sim_clean import (
    HAS_TDA,
    FogGraphGenerator,
    compute_graph_stats,
    compute_persistence,
    compute_ph_signature,
    compute_spectral_signature,
    compute_worst_case,
    hierarchical_bootstrap_ci,
    metrics_from_alarms,
    paired_permutation_pvalue,
    student_t_ci,
)
from baselines_clean import (
    ATTRIBUTION_CLASSES,
    SCENARIO_TRUE_ATTRIBUTION,
    AutoencoderDetector,
    BOCPDDetector,
    CUSUMDetector,
    EWMADetector,
    MMDDetector,
    SWORDDetector,
    TriTrackFusion,
    WindowedGLRDetector,
    calibrate_cusum_kappa,
    calibrate_threshold_empirical,
    estimate_empirical_arl0,
    estimate_tritrack_arl0,
    evaluate_downstream_schedulers,
    train_fair_gnn,
)


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


SCENARIOS = {
    "S1->S2": ("S1_nominal", "S2_churn", 60, 100),
    "S1->S3": ("S1_nominal", "S3_link_degradation", 60, 100),
    "S1->S4": ("S1_nominal", "S4_cascading", 60, 100),
    "S1->S5": ("S1_nominal", "S5_reconfig", 60, 100),
    "S2->S3": ("S2_churn", "S3_link_degradation", 60, 100),
}

METHODS_TRACK1 = [
    "CUSUM-T1",
    "EWMA-T1",
    "BOCPD-T1",
    "MMD-T1",
    "AE-T1",
    "WGLR-T1",
]
METHODS_SPEC = [
    "SWORD",
    "EWMA-Spec",
    "BOCPD-Spec",
    "MMD-Spec",
    "AE-Spec",
    "WGLR-Spec",
    "CUSUM-Spec",
    "Tri-track",
]
ALL_DET_METHODS = METHODS_TRACK1 + METHODS_SPEC


# =========================================================================
# 1. PRECOMPUTE CALIBRATION & ARL0 STREAMS
# =========================================================================

def burn_in_generator(gen, regime, burn_in=30):
    """Advance non-nominal initial regimes (e.g. S2_churn) to equilibrium."""
    if regime != "S1_nominal" and burn_in > 0:
        for tb in range(burn_in):
            gen.step(regime, -burn_in + tb)


def generate_stationary_streams(
    regime, n_eps, T, M=32, seed_base=50000, k_spec=10
):
    """Precompute (A, stats, ph, spec) streams once for fast calibration."""
    streams = {"A": [], "stats": [], "ph": [], "spec": []}
    for ep in range(n_eps):
        gen = FogGraphGenerator(M, seed=seed_base + ep)
        burn_in_generator(gen, regime)
        a_seq, s_seq, p_seq, sp_seq = [], [], [], []
        for t in range(T):
            A, X = gen.step(regime, t)
            a_seq.append(A)
            s_seq.append(compute_graph_stats(A, X, M0=M))
            p_seq.append(compute_ph_signature(A, X, max_dim=1)["features"])
            sp_seq.append(
                compute_spectral_signature(A, k=k_spec, normalized=True)
            )
        streams["A"].append(a_seq)
        streams["stats"].append(s_seq)
        streams["ph"].append(p_seq)
        streams["spec"].append(sp_seq)
    return streams


def median_heuristic_bandwidth(stream_list, max_points=600, seed=0):
    rng = np.random.default_rng(seed)
    pts = np.vstack([np.asarray(seq) for seq in stream_list])
    if len(pts) > max_points:
        idx = rng.choice(len(pts), size=max_points, replace=False)
        pts = pts[idx]
    dists = pdist(pts, metric="euclidean")
    pos = dists[dists > 1e-6]
    return float(np.median(pos)) if len(pos) > 0 else 0.2


def calibrate_suite_on_regime(
    regime,
    n_cal_eps,
    T_cal,
    M,
    target_alpha,
    seed_base,
    k_spec=10,
):
    """Calibrate baselines at target_alpha and Tri-track tracks at target_alpha/3."""
    print(f"  [Calibrating on {regime}: {n_cal_eps} eps x {T_cal} steps]")
    streams = generate_stationary_streams(
        regime, n_cal_eps, T_cal, M=M, seed_base=seed_base, k_spec=k_spec
    )
    ae_train_streams = generate_stationary_streams(
        regime,
        max(6, n_cal_eps // 2),
        T_cal,
        M=M,
        seed_base=seed_base + 5000,
        k_spec=k_spec,
    )

    stats_flat = np.vstack([np.asarray(s) for s in streams["stats"]])
    spec_flat = np.vstack([np.asarray(s) for s in streams["spec"]])
    stats_mean, stats_std = stats_flat.mean(axis=0), stats_flat.std(axis=0)
    spec_mean, spec_std = spec_flat.mean(axis=0), spec_flat.std(axis=0)

    bw_stats = median_heuristic_bandwidth(streams["stats"], seed=seed_base)
    bw_spec = median_heuristic_bandwidth(streams["spec"], seed=seed_base + 1)

    ae_t1_model, ae_t1_mu, ae_t1_sd = AutoencoderDetector.fit_model(
        np.vstack([np.asarray(s) for s in ae_train_streams["stats"]]),
        seed=seed_base,
    )
    ae_sp_model, ae_sp_mu, ae_sp_sd = AutoencoderDetector.fit_model(
        np.vstack([np.asarray(s) for s in ae_train_streams["spec"]]),
        seed=seed_base + 1,
    )

    k_t1 = calibrate_cusum_kappa(streams["stats"], alpha=0.05, warmup=10, k_sigma=1.0)
    k_ph = calibrate_cusum_kappa(streams["ph"], alpha=0.05, warmup=10, k_sigma=1.0)
    k_sp = calibrate_cusum_kappa(streams["spec"], alpha=0.05, warmup=10, k_sigma=1.0)

    cfg = {
        "regime": regime,
        "k_t1": k_t1,
        "k_ph": k_ph,
        "k_sp": k_sp,
        "stats_mean": stats_mean,
        "stats_std": stats_std,
        "spec_mean": spec_mean,
        "spec_std": spec_std,
        "bw_stats": bw_stats,
        "bw_spec": bw_spec,
        "ae_t1": (ae_t1_model, ae_t1_mu, ae_t1_sd),
        "ae_sp": (ae_sp_model, ae_sp_mu, ae_sp_sd),
        "thresholds": {},
        "cal_alphas": {},
    }

    alpha_tri = target_alpha / 3.0

    # (name, stream_list, warmup, factory, lo, hi, max_hi, target_a)
    specs = [
        (
            "CUSUM-T1",
            streams["stats"],
            20,
            lambda th: CUSUMDetector(kappa=k_t1, threshold=th, alpha=0.05, warmup=20),
            1e-3, 5.0, 1e4, target_alpha,
        ),
        (
            "CUSUM-Spec",
            streams["spec"],
            20,
            lambda th: CUSUMDetector(kappa=k_sp, threshold=th, alpha=0.05, warmup=20),
            1e-3, 5.0, 1e4, target_alpha,
        ),
        # Bonferroni-split (target_alpha / 3) thresholds for symmetric Tri-track
        (
            "Tri-v1",
            streams["stats"],
            20,
            lambda th: CUSUMDetector(kappa=k_t1, threshold=th, alpha=0.05, warmup=20),
            1e-3, 5.0, 1e4, alpha_tri,
        ),
        (
            "Tri-v2",
            streams["ph"],
            20,
            lambda th: CUSUMDetector(kappa=k_ph, threshold=th, alpha=0.05, warmup=20),
            1e-3, 20.0, 1e4, alpha_tri,
        ),
        (
            "Tri-v3",
            streams["spec"],
            20,
            lambda th: CUSUMDetector(kappa=k_sp, threshold=th, alpha=0.05, warmup=20),
            1e-3, 5.0, 1e4, alpha_tri,
        ),
        (
            "EWMA-T1",
            streams["stats"],
            20,
            lambda th: EWMADetector(lambda_=0.1, threshold=th, warmup=20),
            0.5, 20.0, 1e3, target_alpha,
        ),
        (
            "EWMA-Spec",
            streams["spec"],
            20,
            lambda th: EWMADetector(lambda_=0.1, threshold=th, warmup=20),
            0.5, 20.0, 1e3, target_alpha,
        ),
        (
            "BOCPD-T1",
            streams["stats"],
            20,
            lambda th: BOCPDDetector(
                hazard=1.0 / 50.0,
                threshold=th,
                warmup=20,
                mean_h0=stats_mean,
                std_h0=stats_std,
            ),
            0.05, 0.99, 1.0, target_alpha,
        ),
        (
            "BOCPD-Spec",
            streams["spec"],
            20,
            lambda th: BOCPDDetector(
                hazard=1.0 / 50.0,
                threshold=th,
                warmup=20,
                mean_h0=spec_mean,
                std_h0=spec_std,
            ),
            0.05, 0.99, 1.0, target_alpha,
        ),
        (
            "MMD-T1",
            streams["stats"],
            30,
            lambda th: MMDDetector(
                window=15, bandwidth=bw_stats, threshold=th, warmup=30
            ),
            1e-4, 1.0, 10.0, target_alpha,
        ),
        (
            "MMD-Spec",
            streams["spec"],
            30,
            lambda th: MMDDetector(
                window=15, bandwidth=bw_spec, threshold=th, warmup=30
            ),
            1e-4, 1.0, 10.0, target_alpha,
        ),
        (
            "SWORD",
            streams["A"],
            30,
            lambda th: SWORDDetector(
                K=10, window=15, n_probes=16, threshold=th, warmup=30, seed=0
            ),
            1e-3, 2.0, 1e3, target_alpha,
        ),
        (
            "AE-T1",
            streams["stats"],
            20,
            lambda th: AutoencoderDetector(
                ae_t1_model, ae_t1_mu, ae_t1_sd, threshold=th, warmup=20
            ),
            1e-3, 10.0, 1e4, target_alpha,
        ),
        (
            "AE-Spec",
            streams["spec"],
            20,
            lambda th: AutoencoderDetector(
                ae_sp_model, ae_sp_mu, ae_sp_sd, threshold=th, warmup=20
            ),
            1e-3, 10.0, 1e4, target_alpha,
        ),
        (
            "WGLR-T1",
            streams["stats"],
            30,
            lambda th: WindowedGLRDetector(window=15, threshold=th, warmup=30),
            0.1, 30.0, 1e4, target_alpha,
        ),
        (
            "WGLR-Spec",
            streams["spec"],
            30,
            lambda th: WindowedGLRDetector(window=15, threshold=th, warmup=30),
            0.1, 30.0, 1e4, target_alpha,
        ),
    ]

    for name, s_list, wu, factory, lo, hi, max_hi, t_alpha in specs:
        th, a_cal = calibrate_threshold_empirical(
            factory,
            s_list,
            target_alpha=t_alpha,
            warmup=wu,
            lo=lo,
            hi=hi,
            max_hi=max_hi,
        )
        cfg["thresholds"][name] = th
        cfg["cal_alphas"][name] = a_cal
        print(
            f"    {name:<12} th={th:>8.4f}  cal_alpha={a_cal:.5f} (tgt={t_alpha:.5f})"
        )

    return cfg


def build_detectors(cfg, seed=0):
    th = cfg["thresholds"]
    ae_t1_model, ae_t1_mu, ae_t1_sd = cfg["ae_t1"]
    ae_sp_model, ae_sp_mu, ae_sp_sd = cfg["ae_sp"]

    dets = {
        "CUSUM-T1": CUSUMDetector(
            kappa=cfg["k_t1"], threshold=th["CUSUM-T1"], alpha=0.05, warmup=20
        ),
        "EWMA-T1": EWMADetector(
            lambda_=0.1, threshold=th["EWMA-T1"], warmup=20
        ),
        "BOCPD-T1": BOCPDDetector(
            hazard=1.0 / 50.0,
            threshold=th["BOCPD-T1"],
            warmup=20,
            mean_h0=cfg["stats_mean"],
            std_h0=cfg["stats_std"],
        ),
        "MMD-T1": MMDDetector(
            window=15,
            bandwidth=cfg["bw_stats"],
            threshold=th["MMD-T1"],
            warmup=30,
        ),
        "AE-T1": AutoencoderDetector(
            ae_t1_model, ae_t1_mu, ae_t1_sd, threshold=th["AE-T1"], warmup=20
        ),
        "WGLR-T1": WindowedGLRDetector(
            window=15, threshold=th["WGLR-T1"], warmup=30
        ),
        "SWORD": SWORDDetector(
            K=10,
            window=15,
            n_probes=16,
            threshold=th["SWORD"],
            warmup=30,
            seed=seed,
        ),
        "EWMA-Spec": EWMADetector(
            lambda_=0.1, threshold=th["EWMA-Spec"], warmup=20
        ),
        "BOCPD-Spec": BOCPDDetector(
            hazard=1.0 / 50.0,
            threshold=th["BOCPD-Spec"],
            warmup=20,
            mean_h0=cfg["spec_mean"],
            std_h0=cfg["spec_std"],
        ),
        "MMD-Spec": MMDDetector(
            window=15,
            bandwidth=cfg["bw_spec"],
            threshold=th["MMD-Spec"],
            warmup=30,
        ),
        "AE-Spec": AutoencoderDetector(
            ae_sp_model, ae_sp_mu, ae_sp_sd, threshold=th["AE-Spec"], warmup=20
        ),
        "WGLR-Spec": WindowedGLRDetector(
            window=15, threshold=th["WGLR-Spec"], warmup=30
        ),
        "CUSUM-Spec": CUSUMDetector(
            kappa=cfg["k_sp"], threshold=th["CUSUM-Spec"], alpha=0.05, warmup=20
        ),
    }

    # Symmetric Tri-track uses Bonferroni-split (target_alpha / 3) tracks
    v1 = CUSUMDetector(
        kappa=cfg["k_t1"], threshold=th["Tri-v1"], alpha=0.05, warmup=20
    )
    v2 = CUSUMDetector(
        kappa=cfg["k_ph"], threshold=th["Tri-v2"], alpha=0.05, warmup=20
    )
    v3 = CUSUMDetector(
        kappa=cfg["k_sp"], threshold=th["Tri-v3"], alpha=0.05, warmup=20
    )
    fusion = TriTrackFusion(v1, v2, v3, confirm_window=10, refractory=10)
    return dets, fusion


def get_warmup(method_name):
    if method_name in ("MMD-T1", "MMD-Spec", "SWORD", "WGLR-T1", "WGLR-Spec"):
        return 30
    return 20


# =========================================================================
# 2. EPISODE RUNNER
# =========================================================================

def run_single_episode(
    scen_name,
    r0,
    r1,
    ct,
    T,
    cfg,
    M=32,
    seed=0,
    k_spec=10,
    gnn_entry=None,
    record_trace=False,
):
    gen = FogGraphGenerator(M, seed=seed)
    burn_in_generator(gen, r0)
    regime_at = [r0] * ct + [r1] * (T - ct)
    dets, fusion = build_detectors(cfg, seed=seed)

    gnn_det = None
    gnn_enc = None
    if gnn_entry is not None:
        gnn_enc, gnn_kappa, gnn_th = gnn_entry
        gnn_enc.reset()
        gnn_det = CUSUMDetector(
            kappa=gnn_kappa, threshold=gnn_th, alpha=0.05, warmup=20
        )

    trace = {"S1": [], "S2": [], "S3": []} if record_trace else None

    for t in range(T):
        A, X = gen.step(regime_at[t], t)
        stats_vec = compute_graph_stats(A, X, M0=M)
        ph_vec = compute_ph_signature(A, X, max_dim=1)["features"]
        spec_vec = compute_spectral_signature(A, k=k_spec, normalized=True)

        dets["CUSUM-T1"].step(stats_vec, t)
        dets["EWMA-T1"].step(stats_vec, t)
        dets["BOCPD-T1"].step(stats_vec, t)
        dets["MMD-T1"].step(stats_vec, t)
        dets["AE-T1"].step(stats_vec, t)
        dets["WGLR-T1"].step(stats_vec, t)

        dets["SWORD"].step(A, t)
        dets["EWMA-Spec"].step(spec_vec, t)
        dets["BOCPD-Spec"].step(spec_vec, t)
        dets["MMD-Spec"].step(spec_vec, t)
        dets["AE-Spec"].step(spec_vec, t)
        dets["WGLR-Spec"].step(spec_vec, t)
        dets["CUSUM-Spec"].step(spec_vec, t)

        fusion.step(stats_vec, ph_vec, spec_vec, t)

        if record_trace:
            trace["S1"].append(float(fusion.v1.S))
            trace["S2"].append(float(fusion.v2.S))
            trace["S3"].append(float(fusion.v3.S))

        if gnn_det is not None:
            with torch.no_grad():
                z = gnn_enc(A, X).numpy()
            gnn_det.step(z, t)

    fusion.finalize()

    alarms = {m: list(dets[m].alarms) for m in dets}
    alarms["Tri-track"] = list(fusion.alarms)
    if gnn_det is not None:
        alarms["GNN-Encoder"] = list(gnn_det.alarms)

    true_attr = SCENARIO_TRUE_ATTRIBUTION[scen_name]
    qual = TriTrackFusion.summarize_events(
        fusion.events, ct, true_attribution=true_attr
    )
    regret = evaluate_downstream_schedulers(
        regime_at, alarms["CUSUM-Spec"], fusion.events
    )

    return {
        "alarms": alarms,
        "qual": qual,
        "regret": regret,
        "events": fusion.events if record_trace else None,
        "trace": trace,
    }


# =========================================================================
# 3. ATTRIBUTION MACRO-F1 HELPER
# =========================================================================

def compute_attribution_metrics(qual_by_scen, bucket="first_alarm"):
    """Compute overall accuracy, confusion matrix, and macro-F1 across 3 classes."""
    conf = {
        true_c: {pred_c: 0 for pred_c in ATTRIBUTION_CLASSES}
        for true_c in ATTRIBUTION_CLASSES
    }
    for scen, true_c in SCENARIO_TRUE_ATTRIBUTION.items():
        ac = qual_by_scen[scen][bucket]["attr_counts"]
        for pred_c in ATTRIBUTION_CLASSES:
            conf[true_c][pred_c] += int(ac.get(pred_c, 0))

    per_class = {}
    f1s = []
    tot_correct = 0
    tot_n = 0
    for c in ATTRIBUTION_CLASSES:
        tp = conf[c][c]
        fp = sum(conf[other][c] for other in ATTRIBUTION_CLASSES if other != c)
        fn = sum(conf[c][other] for other in ATTRIBUTION_CLASSES if other != c)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class[c] = {
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "support": int(tp + fn),
        }
        f1s.append(f1)
        tot_correct += tp
        tot_n += tp + fn

    return {
        "overall_accuracy": float(tot_correct / max(tot_n, 1)),
        "macro_f1": float(np.mean(f1s)),
        "per_class": per_class,
        "confusion_matrix": conf,
    }


# =========================================================================
# 4. SCALABILITY BENCHMARK
# =========================================================================

def run_scalability_benchmark(scales=(16, 32, 64, 128), n_trials=20):
    print("\n[SCALABILITY BENCHMARK]")
    out = {}
    for M in scales:
        gen = FogGraphGenerator(M, seed=7)
        A, X = gen.step("S1_nominal", 0)

        t0 = time.perf_counter()
        for _ in range(n_trials * 5):
            compute_graph_stats(A, X, M0=M)
        t1_ms = (time.perf_counter() - t0) / (n_trials * 5) * 1000.0

        t0 = time.perf_counter()
        for _ in range(n_trials):
            compute_persistence(A, X, max_dim=0)
        t2_h0_ms = (time.perf_counter() - t0) / n_trials * 1000.0

        t0 = time.perf_counter()
        for _ in range(max(5, n_trials // 2)):
            compute_persistence(A, X, max_dim=1)
        t2_h01_ms = (
            (time.perf_counter() - t0) / max(5, n_trials // 2) * 1000.0
        )

        t0 = time.perf_counter()
        for _ in range(n_trials):
            compute_spectral_signature(A, k=10, normalized=True, use_sparse=False)
        t3_dense_ms = (time.perf_counter() - t0) / n_trials * 1000.0

        t0 = time.perf_counter()
        for _ in range(n_trials):
            compute_spectral_signature(A, k=10, normalized=True, use_sparse=True)
        t3_sparse_ms = (time.perf_counter() - t0) / n_trials * 1000.0

        out[str(M)] = {
            "track1_ms": float(t1_ms),
            "track2_h0_ms": float(t2_h0_ms),
            "track2_h01_ms": float(t2_h01_ms),
            "track3_dense_ms": float(t3_dense_ms),
            "track3_sparse_ms": float(t3_sparse_ms),
        }
        print(
            f"  |V|={M:>3d}: T1={t1_ms:.3f}ms  T2(H0)={t2_h0_ms:.3f}ms  "
            f"T2(H0+H1)={t2_h01_ms:.3f}ms  T3(dense)={t3_dense_ms:.3f}ms  "
            f"T3(Lanczos)={t3_sparse_ms:.3f}ms"
        )
    return out


# =========================================================================
# 5. CLEAN ENGLISH VECTOR PDF FIGURES (IEEE STYLE, NO EMBEDDED TITLES)
# =========================================================================

def generate_all_figures(results, sample_trace, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    scen_list = list(SCENARIOS.keys())
    x = np.arange(len(scen_list))

    # --- Fig 1: Scalability latency ---
    scal = results["scalability"]
    scales = [int(k) for k in scal.keys()]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(scales, [scal[str(m)]["track1_ms"] for m in scales], "o-", label="Track 1 (graph stats)", linewidth=1.8, markersize=5)
    ax.plot(scales, [scal[str(m)]["track2_h0_ms"] for m in scales], "s--", label="Track 2 (PH, $H_0$)", linewidth=1.8, markersize=5)
    ax.plot(scales, [scal[str(m)]["track2_h01_ms"] for m in scales], "D-.", label="Track 2 (PH, $H_0+H_1$)", linewidth=1.8, markersize=5)
    ax.plot(scales, [scal[str(m)]["track3_dense_ms"] for m in scales], "^:", label="Track 3 (spectral, dense)", linewidth=1.8, markersize=5)
    ax.plot(scales, [scal[str(m)]["track3_sparse_ms"] for m in scales], "v-", label="Track 3 (spectral, Lanczos)", linewidth=1.8, markersize=5)
    ax.set_xlabel("Number of nodes $|\\mathcal{V}|$")
    ax.set_ylabel("Latency per step (ms)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25, linestyle=":")
    ax.legend(frameon=True, loc="upper left", ncol=2)
    fig.savefig(f"{out_dir}/fig_latency.pdf")
    plt.close(fig)

    # --- Fig 2: Detection delay across scenarios ---
    plot_methods = [
        "CUSUM-T1",
        "EWMA-T1",
        "BOCPD-T1",
        "MMD-T1",
        "SWORD",
        "CUSUM-Spec",
        "AE-Spec",
        "WGLR-Spec",
        "Tri-track",
    ]
    w = 0.088
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    for i, m in enumerate(plot_methods):
        means = [results["scenarios"][s][m]["mean_delay"] for s in scen_list]
        lo = [
            max(0.0, results["scenarios"][s][m]["mean_delay"] - results["scenarios"][s][m]["ci95_boot"][0])
            for s in scen_list
        ]
        hi = [
            max(0.0, results["scenarios"][s][m]["ci95_boot"][1] - results["scenarios"][s][m]["mean_delay"])
            for s in scen_list
        ]
        ax.bar(
            x + (i - len(plot_methods) / 2.0 + 0.5) * w,
            means,
            w,
            yerr=[lo, hi],
            capsize=1.8,
            label=m,
            edgecolor="white",
            linewidth=0.4,
            error_kw={"elinewidth": 0.8},
        )
    ax.axvline(3.5, color="#555555", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(3.55, 95, "Re-baselined\non S2", fontsize=7, color="#444444", va="top")
    ax.axhline(100, color="#888888", linestyle=":", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("->", "$\\rightarrow$") for s in scen_list])
    ax.set_ylabel("Detection delay (steps)")
    ax.set_ylim(0, 108)
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    ax.legend(loc="upper left", ncol=3, frameon=True)
    fig.savefig(f"{out_dir}/fig_detection_delay.pdf")
    plt.close(fig)

    # --- Fig 3 & 4: Reproducibility (sigma) and Tail Delay (Tri-track vs GNN) ---
    if "GNN-Encoder" in results["scenarios"][scen_list[0]]:
        fig, ax = plt.subplots(figsize=(6.2, 3.2))
        w = 0.25
        for idx, (m, col) in enumerate(
            [
                ("Tri-track", "#c62828"),
                ("CUSUM-Spec", "#2e7d32"),
                ("GNN-Encoder", "#6a1b9a"),
            ]
        ):
            stds = [results["scenarios"][s][m]["std_delay"] for s in scen_list]
            plot_vals = [max(v, 1e-2) for v in stds]
            bars = ax.bar(x + (idx - 1) * w, plot_vals, w, label=m, color=col, edgecolor="white")
            for bar, real_v in zip(bars, stds):
                if real_v < 1e-3:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        1.3e-2,
                        "0",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color=col,
                    )
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("->", "$\\rightarrow$") for s in scen_list])
        ax.set_ylabel("Inter-seed $\\sigma$ (steps)")
        ax.set_yscale("log")
        ax.set_ylim(8e-3, 120)
        ax.grid(True, axis="y", alpha=0.25, linestyle=":")
        ax.legend(loc="upper right", frameon=True)
        fig.savefig(f"{out_dir}/fig_variance_repro.pdf")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.2, 3.2))
        w = 0.34
        tri_max = [results["scenarios"][s]["Tri-track"]["worst_case"]["max"] for s in scen_list]
        tri_p95 = [results["scenarios"][s]["Tri-track"]["worst_case"]["p95"] for s in scen_list]
        gnn_max = [results["scenarios"][s]["GNN-Encoder"]["worst_case"]["max"] for s in scen_list]
        gnn_p95 = [results["scenarios"][s]["GNN-Encoder"]["worst_case"]["p95"] for s in scen_list]

        ax.bar(x - w / 2, tri_max, w, color="#c62828", alpha=0.35, edgecolor="#c62828", label="Tri-track (max)")
        ax.bar(x - w / 2, tri_p95, w, color="#c62828", alpha=0.90, label="Tri-track (p95)")
        ax.bar(x + w / 2, gnn_max, w, color="#6a1b9a", alpha=0.35, edgecolor="#6a1b9a", label="GNN encoder (max)")
        ax.bar(x + w / 2, gnn_p95, w, color="#6a1b9a", alpha=0.90, label="GNN encoder (p95)")
        ax.axhline(100, color="#888888", linestyle=":", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("->", "$\\rightarrow$") for s in scen_list])
        ax.set_ylabel("Tail detection delay (steps)")
        ax.set_ylim(0, 112)
        ax.grid(True, axis="y", alpha=0.25, linestyle=":")
        ax.legend(loc="upper right", ncol=2, frameon=True)
        fig.savefig(f"{out_dir}/fig_worst_case.pdf")
        plt.close(fig)

    # --- Fig 5: Multi-track agreement & Structural Attribution Accuracy ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.3))
    all3_pct, two_pct, one_pct, attr_acc = [], [], [], []
    for s in scen_list:
        q = results["qualification"][s]["first_alarm"]
        tot = max(q["n_total"], 1)
        all3_pct.append(100.0 * q["all_three"] / tot)
        two_pct.append(100.0 * q["two_tracks"] / tot)
        one_pct.append(100.0 * q["single_track"] / tot)
        attr_acc.append(100.0 * q["attr_correct"] / tot)

    b1, b2, b3 = np.array(all3_pct), np.array(two_pct), np.array(one_pct)
    ax1.bar(x, b1, 0.55, label="3/3 tracks agree", color="#1b5e20")
    ax1.bar(x, b2, 0.55, bottom=b1, label="2/3 tracks agree", color="#66bb6a")
    ax1.bar(x, b3, 0.55, bottom=b1 + b2, label="1/3 track only", color="#ef5350")
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.replace("->", "$\\rightarrow$") for s in scen_list])
    ax1.set_ylabel("First post-change alarms (%)")
    ax1.set_ylim(0, 115)
    ax1.grid(True, axis="y", alpha=0.25, linestyle=":")
    ax1.legend(loc="lower right", frameon=True)

    bars = ax2.bar(x, attr_acc, 0.55, color="#1565c0", edgecolor="white")
    for bar, val in zip(bars, attr_acc):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            min(val + 2.5, 104),
            f"{val:.0f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels([s.replace("->", "$\\rightarrow$") for s in scen_list])
    ax2.set_ylabel("Fault attribution accuracy (%)")
    ax2.set_ylim(0, 115)
    ax2.grid(True, axis="y", alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig_qualification_attribution.pdf")
    plt.close(fig)

    # --- Fig 6: Timeout-free Downstream Scheduler Regret (4 policies) ---
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    w = 0.19
    policies = [
        ("Static-PreRegime", "#757575"),
        ("Scalar-Aggressive", "#ef6c00"),
        ("Scalar-Conservative", "#8e24aa"),
        ("Attributed-TriTrack", "#1565c0"),
    ]
    for idx, (pol, col) in enumerate(policies):
        means = [results["downstream"][s][pol]["mean"] for s in scen_list]
        lo = [
            max(0.0, results["downstream"][s][pol]["mean"] - results["downstream"][s][pol]["ci95_boot"][0])
            for s in scen_list
        ]
        hi = [
            max(0.0, results["downstream"][s][pol]["ci95_boot"][1] - results["downstream"][s][pol]["mean"])
            for s in scen_list
        ]
        ax.bar(
            x + (idx - 1.5) * w,
            means,
            w,
            yerr=[lo, hi],
            capsize=1.8,
            label=pol,
            color=col,
            edgecolor="white",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("->", "$\\rightarrow$") for s in scen_list])
    ax.set_ylabel("Cumulative scheduling regret")
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    ax.legend(loc="upper right", ncol=2, frameon=True)
    fig.savefig(f"{out_dir}/fig_downstream_regret.pdf")
    plt.close(fig)

    # --- Fig 7: Single integer-step episode timeline on S1->S2 ---
    if sample_trace is not None:
        tr = sample_trace["trace"]
        evs = [e for e in sample_trace["events"] if e["t"] >= 60]
        t_axis = np.arange(len(tr["S3"]))
        fig, ax = plt.subplots(figsize=(6.5, 3.0))
        ax.axvspan(0, 60, color="#e3f2fd", alpha=0.4, label="Nominal (S1)")
        ax.axvspan(60, len(t_axis), color="#ffebee", alpha=0.35, label="Churn (S2)")
        ax.axvline(60, color="#333333", linestyle="--", linewidth=1.0)
        for key, lbl, col in [
            ("S1", "Track 1 (stats CUSUM)", "#1565c0"),
            ("S2", "Track 2 (PH CUSUM)", "#2e7d32"),
            ("S3", "Track 3 (spectral CUSUM)", "#c62828"),
        ]:
            arr = np.asarray(tr[key], dtype=float)
            denom = max(float(arr.max()), 1e-6)
            ax.plot(t_axis, arr / denom, label=lbl, color=col, linewidth=1.3)
        if evs:
            ev0 = evs[0]
            ax.scatter(
                [ev0["t"]], [1.03], color="#c62828", zorder=5, s=35,
                label=f"OR alarm (t={ev0['t']}, {ev0['attribution']})"
            )
            if ev0["t_confirm"] is not None:
                ax.scatter(
                    [ev0["t_confirm"]], [1.03], color="#1b5e20", marker="s",
                    zorder=5, s=35,
                    label=f"Multi-track confirmed (t={ev0['t_confirm']})"
                )
        ax.set_xlabel("Simulation step $t$")
        ax.set_ylabel("Normalized CUSUM statistic")
        ax.set_xlim(0, len(t_axis) - 1)
        ax.set_ylim(-0.05, 1.18)
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.legend(loc="upper left", ncol=2, fontsize=7, frameon=True)
        fig.savefig(f"{out_dir}/fig_timeline_example.pdf")
        plt.close(fig)


# =========================================================================
# 6. MAIN BENCHMARK ORCHESTRATOR
# =========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Fast smoke test")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--M", type=int, default=32)
    parser.add_argument("--target-alpha", type=float, default=0.005)
    parser.add_argument("--n-perm", type=int, default=20000)
    parser.add_argument("--output", default="results_clean")
    parser.add_argument("--skip-gnn", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    if args.quick:
        n_seeds, n_reps = 3, 4
        n_cal_eps, T_cal = 10, 150
        n_arl_eps, T_arl = 5, 500
        n_perm = 2000
        gnn_eps, gnn_epochs = 40, 4
    else:
        n_seeds, n_reps = args.seeds, args.reps
        n_cal_eps, T_cal = 20, 200
        n_arl_eps, T_arl = 15, 1000
        n_perm = args.n_perm
        gnn_eps, gnn_epochs = 80, 5

    os.makedirs(args.output, exist_ok=True)
    print("=" * 74)
    print("Clean Symmetric Tri-Track & Attribution Benchmark (v2)")
    print("=" * 74)
    print(
        f"Platform: {platform.platform()} | CPU: {platform.processor() or 'N/A'}"
    )
    print(
        f"HAS_TDA={HAS_TDA} | M={args.M} | seeds={n_seeds} x reps={n_reps} "
        f"| target_alpha={args.target_alpha} | perms={n_perm}"
    )

    # 1. Held-out calibration on S1_nominal and S2_churn
    print("\n[STEP 1] Empirical Matched-FAR Calibration")
    cfg_s1 = calibrate_suite_on_regime(
        "S1_nominal",
        n_cal_eps,
        T_cal,
        args.M,
        args.target_alpha,
        seed_base=50000,
    )
    cfg_s2 = calibrate_suite_on_regime(
        "S2_churn",
        n_cal_eps,
        T_cal,
        args.M,
        args.target_alpha,
        seed_base=60000,
    )

    # 2. Long-horizon empirical ARL0 verification on S1_nominal
    print(f"\n[STEP 2] Empirical ARL0 on S1 ({n_arl_eps} x {T_arl} steps)")
    arl_streams = generate_stationary_streams(
        "S1_nominal", n_arl_eps, T_arl, M=args.M, seed_base=70000
    )
    arl0_report = {}
    for m in ALL_DET_METHODS:
        if m == "Tri-track":
            arl_res = estimate_tritrack_arl0(
                lambda: build_detectors(cfg_s1, seed=0)[1],
                arl_streams,
                warmup=20,
            )
        else:
            wu = get_warmup(m)
            if m == "SWORD":
                s_key = "A"
            elif m.endswith("-T1"):
                s_key = "stats"
            else:
                s_key = "spec"
            arl_res = estimate_empirical_arl0(
                lambda m_=m: build_detectors(cfg_s1, seed=0)[0][m_],
                arl_streams[s_key],
                warmup=wu,
            )
        arl0_report[m] = arl_res
        tag = ">=" if arl_res["arl0_censored"] else "  "
        print(
            f"  {m:<12} ARL0={tag}{arl_res['arl0_mle']:>7.1f}  "
            f"alpha_step={arl_res['alpha_step']:.5f}  "
            f"alarms={arl_res['total_alarms']}"
        )

    # 3. Train & per-seed calibrate fair GNN ablation
    gnn_entries_s1 = [None] * n_seeds
    gnn_entries_s2 = [None] * n_seeds
    active_methods = list(ALL_DET_METHODS)
    if not args.skip_gnn:
        active_methods.append("GNN-Encoder")
        print(
            f"\n[STEP 3] Training & Per-Seed Calibration of Fair GNN ({n_seeds} seeds)"
        )

        def _gen_ax(regime, n_ep, T_len, sbase):
            out = []
            for e in range(n_ep):
                g = FogGraphGenerator(args.M, seed=sbase + e)
                burn_in_generator(g, regime)
                out.append([g.step(regime, t) for t in range(T_len)])
            return out

        ax_s1 = _gen_ax("S1_nominal", n_cal_eps, T_cal, 80000)
        ax_s2 = _gen_ax("S2_churn", n_cal_eps, T_cal, 85000)

        for s in range(n_seeds):
            enc = train_fair_gnn(
                seed=s,
                M=args.M,
                n_episodes=gnn_eps,
                seq_len=30,
                epochs=gnn_epochs,
            )
            for ax_list, store in [
                (ax_s1, gnn_entries_s1),
                (ax_s2, gnn_entries_s2),
            ]:
                z_streams = []
                for ep_ax in ax_list:
                    enc.reset()
                    with torch.no_grad():
                        z_streams.append([enc(A, X).numpy() for A, X in ep_ax])
                kap = calibrate_cusum_kappa(
                    z_streams, alpha=0.05, warmup=10, k_sigma=1.0
                )
                th, _ = calibrate_threshold_empirical(
                    lambda h_, k_=kap: CUSUMDetector(
                        kappa=k_, threshold=h_, alpha=0.05, warmup=20
                    ),
                    z_streams,
                    target_alpha=args.target_alpha,
                    warmup=20,
                    lo=1e-3,
                    hi=10.0,
                    max_hi=1e4,
                )
                store[s] = (enc, kap, th)
            print(f"  seed {s}: trained & calibrated on S1/S2")

    # 4. Multi-scenario evaluation
    print(f"\n[STEP 4] Multi-Scenario Evaluation ({n_seeds} seeds x {n_reps} reps)")
    results = {
        "metadata": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "has_tda": HAS_TDA,
            "M": args.M,
            "n_seeds": n_seeds,
            "n_reps": n_reps,
            "target_alpha": args.target_alpha,
            "n_perm": n_perm,
        },
        "calibration": {
            "S1_thresholds": cfg_s1["thresholds"],
            "S1_cal_alphas": cfg_s1["cal_alphas"],
            "S2_thresholds": cfg_s2["thresholds"],
            "S2_cal_alphas": cfg_s2["cal_alphas"],
        },
        "arl0_s1": arl0_report,
        "scenarios": {},
        "qualification": {},
        "attribution_summary": {},
        "downstream": {},
        "significance": {},
    }

    sample_trace = None
    delays_flat_all = {s: {m: [] for m in active_methods} for s in SCENARIOS}
    policy_names = (
        "Static-PreRegime",
        "Scalar-Aggressive",
        "Scalar-Conservative",
        "Attributed-TriTrack",
    )

    for scen, (r0, r1, ct, T) in SCENARIOS.items():
        print(f"\n  === {scen} ({r0} -> {r1}) ===")
        cfg = cfg_s1 if r0 == "S1_nominal" else cfg_s2
        gnn_list = gnn_entries_s1 if r0 == "S1_nominal" else gnn_entries_s2

        delays_by_seed = {m: [] for m in active_methods}
        fa_by_seed = {m: [] for m in active_methods}
        active_steps_by_seed = {m: [] for m in active_methods}
        missed_by_seed = {m: [] for m in active_methods}

        def _empty_bucket():
            return {
                "n_total": 0,
                "n_confirmed": 0,
                "single_track": 0,
                "two_tracks": 0,
                "all_three": 0,
                "trigger_t1": 0,
                "trigger_t2": 0,
                "trigger_t3": 0,
                "active_t1": 0,
                "active_t2": 0,
                "active_t3": 0,
                "confirm_latencies": [],
                "attr_counts": {c: 0 for c in ATTRIBUTION_CLASSES},
                "attr_correct": 0,
            }

        qual_accum = {
            "first_alarm": _empty_bucket(),
            "all_post_change": _empty_bucket(),
        }
        regret_by_seed = {p: [] for p in policy_names}

        for s in range(n_seeds):
            s_del = {m: [] for m in active_methods}
            s_fa = {m: [] for m in active_methods}
            s_act = {m: [] for m in active_methods}
            s_mis = {m: [] for m in active_methods}
            s_reg = {p: [] for p in policy_names}

            for r in range(n_reps):
                ep_seed = 1000 + s * 100 + r
                want_trace = scen == "S1->S2" and s == 0 and r == 0
                out = run_single_episode(
                    scen,
                    r0,
                    r1,
                    ct,
                    T,
                    cfg,
                    M=args.M,
                    seed=ep_seed,
                    gnn_entry=gnn_list[s] if not args.skip_gnn else None,
                    record_trace=want_trace,
                )
                if want_trace:
                    sample_trace = out

                for m in active_methods:
                    wu = get_warmup(m)
                    met = metrics_from_alarms(out["alarms"][m], ct, T, warmup=wu)
                    d = met["delay"] if met["delay"] is not None else T
                    s_del[m].append(d)
                    s_fa[m].append(met["fa"])
                    s_act[m].append(met["active_pre_steps"])
                    s_mis[m].append(met["missed"])

                for bucket in ("first_alarm", "all_post_change"):
                    qb = out["qual"][bucket]
                    acc = qual_accum[bucket]
                    for k_int in (
                        "n_total",
                        "n_confirmed",
                        "single_track",
                        "two_tracks",
                        "all_three",
                        "trigger_t1",
                        "trigger_t2",
                        "trigger_t3",
                        "active_t1",
                        "active_t2",
                        "active_t3",
                        "attr_correct",
                    ):
                        acc[k_int] += qb[k_int]
                    acc["confirm_latencies"].extend(qb["confirm_latencies"])
                    for c in ATTRIBUTION_CLASSES:
                        acc["attr_counts"][c] += qb["attr_counts"].get(c, 0)

                for p in policy_names:
                    s_reg[p].append(out["regret"][p])

            for m in active_methods:
                delays_by_seed[m].append(s_del[m])
                fa_by_seed[m].append(s_fa[m])
                active_steps_by_seed[m].append(s_act[m])
                missed_by_seed[m].append(s_mis[m])
                delays_flat_all[scen][m].extend(s_del[m])
            for p in policy_names:
                regret_by_seed[p].append(s_reg[p])

        # Aggregate detection metrics
        results["scenarios"][scen] = {}
        n_total_runs = n_seeds * n_reps
        for m in active_methods:
            seed_means = [float(np.mean(x)) for x in delays_by_seed[m]]
            mean_d = float(np.mean(seed_means))
            std_d = (
                float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0
            )
            ci_boot = hierarchical_bootstrap_ci(delays_by_seed[m])
            ci_t = student_t_ci(seed_means)
            wc = compute_worst_case(delays_by_seed[m], T)

            tot_fa = sum(sum(x) for x in fa_by_seed[m])
            tot_act = sum(sum(x) for x in active_steps_by_seed[m])
            ep_fa = sum(1 for sl in fa_by_seed[m] for c in sl if c > 0)
            tot_mis = sum(sum(x) for x in missed_by_seed[m])

            results["scenarios"][scen][m] = {
                "mean_delay": mean_d,
                "std_delay": std_d,
                "ci95_boot": list(ci_boot),
                "ci95_t": list(ci_t),
                "per_seed_means": seed_means,
                "worst_case": wc,
                "alpha_step_active": float(tot_fa / max(tot_act, 1)),
                "alpha_episode": float(ep_fa / max(n_total_runs, 1)),
                "beta": float(tot_mis / max(n_total_runs, 1)),
                "total_fa": int(tot_fa),
                "total_missed": int(tot_mis),
            }
            print(
                f"    {m:<12} delay={mean_d:>5.1f} (sigma={std_d:>4.2f}) "
                f"CI=[{ci_boot[0]:>4.1f},{ci_boot[1]:>4.1f}] "
                f"p95={wc['p95']:>4.1f} a_step={tot_fa/max(tot_act,1):.4f} "
                f"beta={tot_mis/n_total_runs:.2f}"
            )

        # Summarize qualification & attribution
        results["qualification"][scen] = {}
        for bucket in ("first_alarm", "all_post_change"):
            qb = qual_accum[bucket]
            lats = qb["confirm_latencies"]
            tot = max(qb["n_total"], 1)
            results["qualification"][scen][bucket] = {
                "n_total": qb["n_total"],
                "n_confirmed": qb["n_confirmed"],
                "fraction_confirmed": float(qb["n_confirmed"] / tot),
                "single_track": qb["single_track"],
                "two_tracks": qb["two_tracks"],
                "all_three": qb["all_three"],
                "trigger_t1": qb["trigger_t1"],
                "trigger_t2": qb["trigger_t2"],
                "trigger_t3": qb["trigger_t3"],
                "active_t1": qb["active_t1"],
                "active_t2": qb["active_t2"],
                "active_t3": qb["active_t3"],
                "mean_confirm_latency": float(np.mean(lats)) if lats else None,
                "median_confirm_latency": float(np.median(lats)) if lats else None,
                "attr_counts": qb["attr_counts"],
                "attr_correct": qb["attr_correct"],
                "attr_accuracy": float(qb["attr_correct"] / tot),
            }
        fa_q = results["qualification"][scen]["first_alarm"]
        print(
            f"    [First Alarm] confirmed={fa_q['n_confirmed']}/{fa_q['n_total']} "
            f"(3/3={fa_q['all_three']}, 2/3={fa_q['two_tracks']}, 1/3={fa_q['single_track']}) | "
            f"triggers(T1/T2/T3)={fa_q['trigger_t1']}/{fa_q['trigger_t2']}/{fa_q['trigger_t3']} | "
            f"attr_acc={fa_q['attr_accuracy']*100:.1f}%"
        )

        # Summarize timeout-free downstream regret
        results["downstream"][scen] = {}
        for p in policy_names:
            sm = [float(np.mean(x)) for x in regret_by_seed[p]]
            results["downstream"][scen][p] = {
                "mean": float(np.mean(sm)),
                "std": float(np.std(sm, ddof=1)) if len(sm) > 1 else 0.0,
                "ci95_boot": list(hierarchical_bootstrap_ci(regret_by_seed[p])),
            }
        attr_flat = [v for sl in regret_by_seed["Attributed-TriTrack"] for v in sl]
        s_agg_flat = [v for sl in regret_by_seed["Scalar-Aggressive"] for v in sl]
        s_con_flat = [v for sl in regret_by_seed["Scalar-Conservative"] for v in sl]
        p_vs_agg = paired_permutation_pvalue(attr_flat, s_agg_flat, n_perm=n_perm, seed=42)
        p_vs_con = paired_permutation_pvalue(attr_flat, s_con_flat, n_perm=n_perm, seed=43)
        results["downstream"][scen]["p_attr_vs_scalar_agg"] = p_vs_agg
        results["downstream"][scen]["p_attr_vs_scalar_con"] = p_vs_con
        print(
            f"    [Regret] Static={results['downstream'][scen]['Static-PreRegime']['mean']:.2f} | "
            f"Scalar-Agg={results['downstream'][scen]['Scalar-Aggressive']['mean']:.2f} | "
            f"Scalar-Con={results['downstream'][scen]['Scalar-Conservative']['mean']:.2f} | "
            f"Attributed={results['downstream'][scen]['Attributed-TriTrack']['mean']:.2f} "
            f"(p_agg={p_vs_agg:.4g}, p_con={p_vs_con:.4g})"
        )

    # Overall fault attribution macro-F1
    results["attribution_summary"]["first_alarm"] = compute_attribution_metrics(
        results["qualification"], bucket="first_alarm"
    )
    results["attribution_summary"]["all_post_change"] = compute_attribution_metrics(
        results["qualification"], bucket="all_post_change"
    )
    fa_attr = results["attribution_summary"]["first_alarm"]
    print(
        f"\n[ATTRIBUTION SUMMARY - First Alarm] "
        f"Overall Acc = {fa_attr['overall_accuracy']*100:.1f}% | "
        f"Macro-F1 = {fa_attr['macro_f1']:.3f}"
    )

    # 5. Paired permutation tests (Tri-track vs every baseline, incl. CUSUM-Spec)
    print(f"\n[STEP 5] Paired Permutation Tests (B={n_perm}, Bonferroni-corrected)")
    comp_methods = [m for m in active_methods if m != "Tri-track"]
    n_tests = len(SCENARIOS) * len(comp_methods)
    for scen in SCENARIOS:
        ref = delays_flat_all[scen]["Tri-track"]
        for m in comp_methods:
            other = delays_flat_all[scen][m]
            p_raw = paired_permutation_pvalue(ref, other, n_perm=n_perm, seed=42)
            p_bonf = min(1.0, p_raw * n_tests)
            results["significance"][f"{scen}|{m}"] = {
                "p_raw": p_raw,
                "p_bonf": p_bonf,
                "significant": bool(p_bonf < 0.05),
                "mean_diff_other_minus_tritrack": float(
                    np.mean(other) - np.mean(ref)
                ),
            }
    n_sig = sum(1 for r in results["significance"].values() if r["significant"])
    print(f"  Significant comparisons (Bonferroni alpha=0.05): {n_sig}/{n_tests}")

    # 6. Scalability latency benchmark
    results["scalability"] = run_scalability_benchmark()

    # 7. Save JSON
    json_path = os.path.join(args.output, "all_results_clean.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {json_path}")

    # 8. Generate clean English vector PDF figures
    if not args.skip_plots:
        fig_dir = os.path.join(args.output, "figures")
        generate_all_figures(results, sample_trace, fig_dir)
        print(f"Saved vector PDF figures to: {fig_dir}/")


if __name__ == "__main__":
    main()