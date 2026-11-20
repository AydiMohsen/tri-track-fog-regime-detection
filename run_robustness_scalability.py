"""
run_robustness_scalability.py
=============================
Expérimentations complémentaires locales pour renforcer la soumission Q1 :
  1. Passage à l'échelle du délai de détection, de l'ARL0, de l'attribution
     causale et du regret aval pour M0 in {64, 128} nœuds.
  2. Robustesse à la perte de télémétrie (p_loss in {0%, 5%, 10%, 20%}) à
     M0 = 32 nœuds en mode zero-shot (calibré sur canal propre) et en mode
     calibré sur site (site_calibrated).

Dépendances locales : fog_sim_clean.py et baselines_clean.py dans le même
répertoire (aucune modification requise de ces deux fichiers).

Usage :
  Test rapide (~1-2 min) :
      python run_robustness_scalability.py --quick
  Campagne complète (10 seeds x 10 reps, ~6-10 min sur CPU Intel) :
      python run_robustness_scalability.py --seeds 10 --reps 10 --output results_extra
  Exécuter une seule expérience :
      python run_robustness_scalability.py --exp scalability
      python run_robustness_scalability.py --exp telemetry
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

from fog_sim_clean import (
    HAS_TDA,
    FogGraphGenerator,
    compute_graph_stats,
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
    CUSUMDetector,
    EWMADetector,
    SWORDDetector,
    TriTrackFusion,
    calibrate_cusum_kappa,
    calibrate_threshold_empirical,
    estimate_empirical_arl0,
    estimate_tritrack_arl0,
    evaluate_downstream_schedulers,
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

# Sous-ensemble représentatif : 2 baselines Track 1, 3 baselines spectrales, Tri-track
CORE_METHODS = [
    "CUSUM-T1",
    "EWMA-T1",
    "SWORD",
    "AE-Spec",
    "CUSUM-Spec",
    "Tri-track",
]

POLICY_NAMES = (
    "Static-PreRegime",
    "Scalar-Aggressive",
    "Scalar-Conservative",
    "Attributed-TriTrack",
)


# =========================================================================
# 1. MODÈLE DE CANAL DE TÉLÉMÉTRIE AVEC PERTE DE PAQUETS
# =========================================================================

class TelemetryLossChannel:
    """Canal d'observation Fog/Edge soumis à la perte de télémétrie.

    Si p_loss == 0.0 : canal idéal sans perte (copie exacte de l'état réel).
    Si p_loss > 0.0 : à chaque pas t, chaque nœud physiquement actif perd son
    paquet de télémétrie/battement indépendamment avec probabilité p_loss :
      - Si reçu : les 4 métriques sont actualisées, missed_hb = 0, le nœud est
        marqué actif, et ses liens incidents sont mis à jour.
      - Si perdu (ou nœud physiquement en panne) : les métriques CPU/mémoire/
        batterie restent figées à la dernière valeur reçue (LOCF), l'indicateur
        de fraîcheur du battement (colonne 3) passe à 0.0, et missed_hb += 1.
      - Une arête (u, v) entre deux nœuds observés actifs est actualisée dès
        qu'au moins une de ses deux extrémités transmet son paquet ; sinon
        elle reste à son dernier état connu (LOCF).
      - Dès que missed_hb[i] >= timeout_k (défaut : 3 battements consécutifs),
        le superviseur déclare le nœud i hors ligne et coupe ses liens observés.
    """

    def __init__(self, M0, p_loss=0.0, timeout_k=3, seed=0):
        self.M0 = int(M0)
        self.p_loss = float(p_loss)
        self.timeout_k = 1 if self.p_loss <= 0.0 else int(timeout_k)
        self.rng = np.random.default_rng(seed)
        self.obs_alive = None
        self.obs_adj = None
        self.obs_features = None
        self.missed_hb = None

    def sync_initial(self, gen):
        self.obs_alive = gen.alive.copy()
        self.obs_adj = gen.adj.copy()
        self.obs_features = gen.features.copy()
        self.missed_hb = np.where(gen.alive, 0, self.timeout_k).astype(int)

    def observe(self, gen):
        if self.p_loss <= 0.0:
            return gen.get_graph()

        if self.obs_alive is None:
            self.sync_initial(gen)

        delivered = gen.alive & (self.rng.random(self.M0) >= self.p_loss)

        # Nœuds ayant transmis ce pas
        if np.any(delivered):
            self.obs_alive[delivered] = True
            self.missed_hb[delivered] = 0
            self.obs_features[delivered] = gen.features[delivered]

        # Nœuds n'ayant pas transmis (paquet perdu ou panne physique)
        missing = ~delivered
        if np.any(missing):
            self.missed_hb[missing] += 1
            still_up_but_stale = missing & self.obs_alive
            if np.any(still_up_but_stale):
                # Indicateur de fraîcheur du heartbeat mis à 0 (LOCF sur 0..2)
                self.obs_features[still_up_but_stale, 3] = 0.0
            timed_out = self.missed_hb >= self.timeout_k
            if np.any(timed_out):
                self.obs_alive[timed_out] = False

        # Mise à jour symétrique des arêtes observées
        either_deliv = delivered[:, None] | delivered[None, :]
        both_obs = self.obs_alive[:, None] & self.obs_alive[None, :]
        upd = either_deliv & both_obs
        self.obs_adj[upd] = gen.adj[upd]

        dead_obs = ~self.obs_alive
        if np.any(dead_obs):
            self.obs_adj[dead_obs, :] = 0.0
            self.obs_adj[:, dead_obs] = 0.0
        np.fill_diagonal(self.obs_adj, 0.0)

        idx = np.where(self.obs_alive)[0]
        return self.obs_adj[np.ix_(idx, idx)].copy(), self.obs_features[idx].copy()


# =========================================================================
# 2. UTILITAIRES DE CALIBRATION RAPIDE (SWORD PRÉCALCULÉ)
# =========================================================================

class _SWORDFromMoments:
    """Identique à SWORDDetector mais consomme des moments Chebyshev précalculés
    afin d'accélérer la recherche dichotomique des seuils à M0 = 64/128."""

    def __init__(self, window=15, threshold=0.5, refractory=10, warmup=30):
        self.window = int(window)
        self.threshold = float(threshold)
        self.refractory = int(refractory)
        self.warmup = int(warmup)
        self.reset()

    def reset(self):
        self.moments_buffer = []
        self.last_alarm = -10**9
        self.alarms = []

    def step(self, moments, t):
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


def burn_in_generator(gen, regime, burn_in=30):
    if regime != "S1_nominal" and burn_in > 0:
        for tb in range(burn_in):
            gen.step(regime, -burn_in + tb)


def generate_stationary_streams(
    regime,
    n_eps,
    T,
    M=32,
    p_loss=0.0,
    timeout_k=3,
    seed_base=50000,
    k_spec=10,
):
    """Précalcule (stats, ph, spec, sword_moments) sur des flux stationnaires."""
    streams = {"stats": [], "ph": [], "spec": [], "sword_moments": []}
    for ep in range(n_eps):
        ep_seed = seed_base + ep
        gen = FogGraphGenerator(M, seed=ep_seed)
        burn_in_generator(gen, regime)
        chan = TelemetryLossChannel(
            M, p_loss=p_loss, timeout_k=timeout_k, seed=ep_seed + 77777
        )
        chan.sync_initial(gen)
        sw_helper = SWORDDetector(
            K=10, window=15, n_probes=16, threshold=1e9, warmup=30, seed=0
        )

        s_seq, p_seq, sp_seq, m_seq = [], [], [], []
        for t in range(T):
            gen.step(regime, t)
            A, X = chan.observe(gen)
            s_seq.append(compute_graph_stats(A, X, M0=M))
            p_seq.append(compute_ph_signature(A, X, max_dim=1)["features"])
            sp_seq.append(
                compute_spectral_signature(A, k=k_spec, normalized=True)
            )
            m_seq.append(sw_helper._chebyshev_moments(A))

        streams["stats"].append(s_seq)
        streams["ph"].append(p_seq)
        streams["spec"].append(sp_seq)
        streams["sword_moments"].append(m_seq)
    return streams


def calibrate_core_suite(
    regime,
    n_cal_eps,
    T_cal,
    M,
    target_alpha,
    seed_base,
    p_loss=0.0,
    timeout_k=3,
    k_spec=10,
):
    """Calibre empiriquement les 6 méthodes clés à taux de fausses alarmes apparié."""
    t0 = time.time()
    streams = generate_stationary_streams(
        regime,
        n_cal_eps,
        T_cal,
        M=M,
        p_loss=p_loss,
        timeout_k=timeout_k,
        seed_base=seed_base,
        k_spec=k_spec,
    )
    ae_train_streams = generate_stationary_streams(
        regime,
        max(5, n_cal_eps // 2),
        T_cal,
        M=M,
        p_loss=p_loss,
        timeout_k=timeout_k,
        seed_base=seed_base + 5000,
        k_spec=k_spec,
    )

    ae_sp_model, ae_sp_mu, ae_sp_sd = AutoencoderDetector.fit_model(
        np.vstack([np.asarray(s) for s in ae_train_streams["spec"]]),
        seed=seed_base + 1,
    )

    k_t1 = calibrate_cusum_kappa(
        streams["stats"], alpha=0.05, warmup=10, k_sigma=1.0
    )
    k_ph = calibrate_cusum_kappa(
        streams["ph"], alpha=0.05, warmup=10, k_sigma=1.0
    )
    k_sp = calibrate_cusum_kappa(
        streams["spec"], alpha=0.05, warmup=10, k_sigma=1.0
    )

    cfg = {
        "regime": regime,
        "M": M,
        "p_loss_cal": p_loss,
        "k_t1": k_t1,
        "k_ph": k_ph,
        "k_sp": k_sp,
        "ae_sp": (ae_sp_model, ae_sp_mu, ae_sp_sd),
        "thresholds": {},
        "cal_alphas": {},
    }

    alpha_tri = target_alpha / 3.0
    specs = [
        (
            "CUSUM-T1",
            streams["stats"],
            20,
            lambda th: CUSUMDetector(
                kappa=k_t1, threshold=th, alpha=0.05, warmup=20
            ),
            1e-3, 5.0, 1e4, target_alpha,
        ),
        (
            "EWMA-T1",
            streams["stats"],
            20,
            lambda th: EWMADetector(lambda_=0.1, threshold=th, warmup=20),
            0.5, 20.0, 1e3, target_alpha,
        ),
        (
            "SWORD",
            streams["sword_moments"],
            30,
            lambda th: _SWORDFromMoments(window=15, threshold=th, warmup=30),
            1e-3, 2.0, 1e3, target_alpha,
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
            "CUSUM-Spec",
            streams["spec"],
            20,
            lambda th: CUSUMDetector(
                kappa=k_sp, threshold=th, alpha=0.05, warmup=20
            ),
            1e-3, 5.0, 1e4, target_alpha,
        ),
        (
            "Tri-v1",
            streams["stats"],
            20,
            lambda th: CUSUMDetector(
                kappa=k_t1, threshold=th, alpha=0.05, warmup=20
            ),
            1e-3, 5.0, 1e4, alpha_tri,
        ),
        (
            "Tri-v2",
            streams["ph"],
            20,
            lambda th: CUSUMDetector(
                kappa=k_ph, threshold=th, alpha=0.05, warmup=20
            ),
            1e-3, 20.0, 1e4, alpha_tri,
        ),
        (
            "Tri-v3",
            streams["spec"],
            20,
            lambda th: CUSUMDetector(
                kappa=k_sp, threshold=th, alpha=0.05, warmup=20
            ),
            1e-3, 5.0, 1e4, alpha_tri,
        ),
    ]

    for name, s_list, wu, factory, lo, hi, max_hi, t_a in specs:
        th, a_cal = calibrate_threshold_empirical(
            factory,
            s_list,
            target_alpha=t_a,
            warmup=wu,
            lo=lo,
            hi=hi,
            max_hi=max_hi,
        )
        cfg["thresholds"][name] = th
        cfg["cal_alphas"][name] = a_cal

    print(
        f"  [Cal {regime} | M={M}, p_loss={p_loss:.0%}] done in {time.time()-t0:.1f}s "
        f"(Tri th: v1={cfg['thresholds']['Tri-v1']:.3f}, "
        f"v2={cfg['thresholds']['Tri-v2']:.2f}, v3={cfg['thresholds']['Tri-v3']:.3f})"
    )
    return cfg


def build_core_detectors(cfg, seed=0):
    th = cfg["thresholds"]
    ae_sp_model, ae_sp_mu, ae_sp_sd = cfg["ae_sp"]
    dets = {
        "CUSUM-T1": CUSUMDetector(
            kappa=cfg["k_t1"], threshold=th["CUSUM-T1"], alpha=0.05, warmup=20
        ),
        "EWMA-T1": EWMADetector(
            lambda_=0.1, threshold=th["EWMA-T1"], warmup=20
        ),
        "SWORD": SWORDDetector(
            K=10,
            window=15,
            n_probes=16,
            threshold=th["SWORD"],
            warmup=30,
            seed=seed,
        ),
        "AE-Spec": AutoencoderDetector(
            ae_sp_model, ae_sp_mu, ae_sp_sd, threshold=th["AE-Spec"], warmup=20
        ),
        "CUSUM-Spec": CUSUMDetector(
            kappa=cfg["k_sp"], threshold=th["CUSUM-Spec"], alpha=0.05, warmup=20
        ),
    }
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
    return 30 if method_name == "SWORD" else 20


# =========================================================================
# 3. ÉVALUATION D'UNE CONFIGURATION (M0, p_loss) SUR LES 5 SCÉNARIOS
# =========================================================================

def compute_attribution_summary(qual_by_scen, bucket="first_alarm"):
    conf = {
        tc: {pc: 0 for pc in ATTRIBUTION_CLASSES} for tc in ATTRIBUTION_CLASSES
    }
    for scen, tc in SCENARIO_TRUE_ATTRIBUTION.items():
        ac = qual_by_scen[scen][bucket]["attr_counts"]
        for pc in ATTRIBUTION_CLASSES:
            conf[tc][pc] += int(ac.get(pc, 0))

    per_class = {}
    f1s = []
    tot_correct, tot_n = 0, 0
    for c in ATTRIBUTION_CLASSES:
        tp = conf[c][c]
        fp = sum(conf[o][c] for o in ATTRIBUTION_CLASSES if o != c)
        fn = sum(conf[c][o] for o in ATTRIBUTION_CLASSES if o != c)
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


def evaluate_configuration(
    label,
    M,
    p_loss,
    timeout_k,
    cfg_s1,
    cfg_s2,
    n_seeds,
    n_reps,
    n_perm,
    arl_eps=0,
    arl_T=500,
    k_spec=10,
):
    """Exécute les 5 scénarios (+ ARL0 optionnel) pour une configuration donnée."""
    t_start = time.time()
    print(f"\n--- Évaluation : {label} (M={M}, p_loss={p_loss:.0%}, "
          f"{n_seeds}x{n_reps} runs/scénario) ---")

    res = {
        "label": label,
        "M": M,
        "p_loss_eval": p_loss,
        "p_loss_cal": cfg_s1["p_loss_cal"],
        "timeout_k": 1 if p_loss <= 0.0 else timeout_k,
        "thresholds_s1": cfg_s1["thresholds"],
        "thresholds_s2": cfg_s2["thresholds"],
        "arl0_s1": {},
        "scenarios": {},
        "five_scen_mean_delay": {},
        "qualification": {},
        "attribution_summary": {},
        "downstream": {},
        "five_scen_mean_regret": {},
        "significance": {},
    }

    # ARL0 stationnaire sur S1 (optionnel)
    if arl_eps > 0:
        arl_streams = generate_stationary_streams(
            "S1_nominal",
            arl_eps,
            arl_T,
            M=M,
            p_loss=p_loss,
            timeout_k=timeout_k,
            seed_base=70000 + M * 10 + int(round(p_loss * 100)),
            k_spec=k_spec,
        )
        for m in CORE_METHODS:
            if m == "Tri-track":
                arl_r = estimate_tritrack_arl0(
                    lambda: build_core_detectors(cfg_s1, seed=0)[1],
                    arl_streams,
                    warmup=20,
                )
            elif m == "SWORD":
                arl_r = estimate_empirical_arl0(
                    lambda: _SWORDFromMoments(
                        window=15,
                        threshold=cfg_s1["thresholds"]["SWORD"],
                        warmup=30,
                    ),
                    arl_streams["sword_moments"],
                    warmup=30,
                )
            else:
                s_key = "stats" if m.endswith("-T1") else "spec"
                arl_r = estimate_empirical_arl0(
                    lambda m_=m: build_core_detectors(cfg_s1, seed=0)[0][m_],
                    arl_streams[s_key],
                    warmup=get_warmup(m),
                )
            res["arl0_s1"][m] = arl_r
        print(
            f"  [ARL0 S1] Tri-track={res['arl0_s1']['Tri-track']['arl0_mle']:.1f} "
            f"(alpha={res['arl0_s1']['Tri-track']['alpha_step']:.4f}) | "
            f"CUSUM-Spec={res['arl0_s1']['CUSUM-Spec']['arl0_mle']:.1f} | "
            f"CUSUM-T1={res['arl0_s1']['CUSUM-T1']['arl0_mle']:.1f}"
        )

    delays_flat_all = {s: {m: [] for m in CORE_METHODS} for s in SCENARIOS}

    for scen, (r0, r1, ct, T) in SCENARIOS.items():
        cfg = cfg_s1 if r0 == "S1_nominal" else cfg_s2
        true_attr = SCENARIO_TRUE_ATTRIBUTION[scen]

        delays_by_seed = {m: [] for m in CORE_METHODS}
        fa_by_seed = {m: [] for m in CORE_METHODS}
        act_by_seed = {m: [] for m in CORE_METHODS}
        mis_by_seed = {m: [] for m in CORE_METHODS}
        regret_by_seed = {p: [] for p in POLICY_NAMES}

        def _empty_q():
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

        qual_acc = {
            "first_alarm": _empty_q(),
            "all_post_change": _empty_q(),
        }

        for s in range(n_seeds):
            s_del = {m: [] for m in CORE_METHODS}
            s_fa = {m: [] for m in CORE_METHODS}
            s_act = {m: [] for m in CORE_METHODS}
            s_mis = {m: [] for m in CORE_METHODS}
            s_reg = {p: [] for p in POLICY_NAMES}

            for r in range(n_reps):
                ep_seed = 1000 + s * 100 + r
                gen = FogGraphGenerator(M, seed=ep_seed)
                burn_in_generator(gen, r0)
                chan = TelemetryLossChannel(
                    M, p_loss=p_loss, timeout_k=timeout_k, seed=ep_seed + 77777
                )
                chan.sync_initial(gen)

                regime_at = [r0] * ct + [r1] * (T - ct)
                dets, fusion = build_core_detectors(cfg, seed=ep_seed)

                for t in range(T):
                    gen.step(regime_at[t], t)
                    A, X = chan.observe(gen)
                    stats_vec = compute_graph_stats(A, X, M0=M)
                    ph_vec = compute_ph_signature(A, X, max_dim=1)["features"]
                    spec_vec = compute_spectral_signature(
                        A, k=k_spec, normalized=True
                    )

                    dets["CUSUM-T1"].step(stats_vec, t)
                    dets["EWMA-T1"].step(stats_vec, t)
                    dets["SWORD"].step(A, t)
                    dets["AE-Spec"].step(spec_vec, t)
                    dets["CUSUM-Spec"].step(spec_vec, t)
                    fusion.step(stats_vec, ph_vec, spec_vec, t)

                fusion.finalize()

                alarms = {m: list(dets[m].alarms) for m in dets}
                alarms["Tri-track"] = list(fusion.alarms)

                for m in CORE_METHODS:
                    wu = get_warmup(m)
                    met = metrics_from_alarms(alarms[m], ct, T, warmup=wu)
                    d = met["delay"] if met["delay"] is not None else T
                    s_del[m].append(d)
                    s_fa[m].append(met["fa"])
                    s_act[m].append(met["active_pre_steps"])
                    s_mis[m].append(met["missed"])

                q_ep = TriTrackFusion.summarize_events(
                    fusion.events, ct, true_attribution=true_attr
                )
                for b in ("first_alarm", "all_post_change"):
                    qb = q_ep[b]
                    acc = qual_acc[b]
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

                reg_ep = evaluate_downstream_schedulers(
                    regime_at, alarms["CUSUM-Spec"], fusion.events
                )
                for p in POLICY_NAMES:
                    s_reg[p].append(reg_ep[p])

            for m in CORE_METHODS:
                delays_by_seed[m].append(s_del[m])
                fa_by_seed[m].append(s_fa[m])
                act_by_seed[m].append(s_act[m])
                mis_by_seed[m].append(s_mis[m])
                delays_flat_all[scen][m].extend(s_del[m])
            for p in POLICY_NAMES:
                regret_by_seed[p].append(s_reg[p])

        # Agrégation scénario
        res["scenarios"][scen] = {}
        n_runs = n_seeds * n_reps
        for m in CORE_METHODS:
            seed_means = [float(np.mean(x)) for x in delays_by_seed[m]]
            mean_d = float(np.mean(seed_means))
            std_d = (
                float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0
            )
            ci_boot = hierarchical_bootstrap_ci(delays_by_seed[m])
            ci_t = student_t_ci(seed_means)
            wc = compute_worst_case(delays_by_seed[m], T)
            tot_fa = sum(sum(x) for x in fa_by_seed[m])
            tot_act = sum(sum(x) for x in act_by_seed[m])
            ep_fa = sum(1 for sl in fa_by_seed[m] for c in sl if c > 0)
            tot_mis = sum(sum(x) for x in mis_by_seed[m])

            res["scenarios"][scen][m] = {
                "mean_delay": mean_d,
                "std_delay": std_d,
                "ci95_boot": list(ci_boot),
                "ci95_t": list(ci_t),
                "worst_case": wc,
                "alpha_step_active": float(tot_fa / max(tot_act, 1)),
                "alpha_episode": float(ep_fa / max(n_runs, 1)),
                "beta": float(tot_mis / max(n_runs, 1)),
                "total_fa": int(tot_fa),
                "total_missed": int(tot_mis),
            }

        res["qualification"][scen] = {}
        for b in ("first_alarm", "all_post_change"):
            qb = qual_acc[b]
            lats = qb["confirm_latencies"]
            tot = max(qb["n_total"], 1)
            res["qualification"][scen][b] = {
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

        res["downstream"][scen] = {}
        for p in POLICY_NAMES:
            sm = [float(np.mean(x)) for x in regret_by_seed[p]]
            res["downstream"][scen][p] = {
                "mean": float(np.mean(sm)),
                "std": float(np.std(sm, ddof=1)) if len(sm) > 1 else 0.0,
                "ci95_boot": list(hierarchical_bootstrap_ci(regret_by_seed[p])),
            }
        attr_flat = [v for sl in regret_by_seed["Attributed-TriTrack"] for v in sl]
        s_agg_flat = [v for sl in regret_by_seed["Scalar-Aggressive"] for v in sl]
        s_con_flat = [v for sl in regret_by_seed["Scalar-Conservative"] for v in sl]
        res["downstream"][scen]["p_attr_vs_scalar_agg"] = paired_permutation_pvalue(
            attr_flat, s_agg_flat, n_perm=n_perm, seed=42
        )
        res["downstream"][scen]["p_attr_vs_scalar_con"] = paired_permutation_pvalue(
            attr_flat, s_con_flat, n_perm=n_perm, seed=43
        )

        tr = res["scenarios"][scen]["Tri-track"]
        cs = res["scenarios"][scen]["CUSUM-Spec"]
        c1 = res["scenarios"][scen]["CUSUM-T1"]
        qa = res["qualification"][scen]["first_alarm"]
        print(
            f"  {scen:<6} | Tri={tr['mean_delay']:>5.2f} (a={tr['alpha_step_active']:.4f}, b={tr['beta']:.2f}) "
            f"| Spec={cs['mean_delay']:>5.2f} | T1={c1['mean_delay']:>5.2f} "
            f"| Conf={qa['fraction_confirmed']*100:>5.1f}% Attr={qa['attr_accuracy']*100:>5.1f}%"
        )

    # Moyennes sur les 5 scénarios
    for m in CORE_METHODS:
        res["five_scen_mean_delay"][m] = float(
            np.mean([res["scenarios"][s][m]["mean_delay"] for s in SCENARIOS])
        )
    for p in POLICY_NAMES:
        res["five_scen_mean_regret"][p] = float(
            np.mean([res["downstream"][s][p]["mean"] for s in SCENARIOS])
        )

    res["attribution_summary"]["first_alarm"] = compute_attribution_summary(
        res["qualification"], "first_alarm"
    )
    res["attribution_summary"]["all_post_change"] = compute_attribution_summary(
        res["qualification"], "all_post_change"
    )

    # Tests de permutation appariés (Tri-track vs 5 baselines)
    comp = [m for m in CORE_METHODS if m != "Tri-track"]
    n_tests = len(SCENARIOS) * len(comp)
    for scen in SCENARIOS:
        ref = delays_flat_all[scen]["Tri-track"]
        for m in comp:
            other = delays_flat_all[scen][m]
            p_raw = paired_permutation_pvalue(ref, other, n_perm=n_perm, seed=42)
            p_bonf = min(1.0, p_raw * n_tests)
            res["significance"][f"{scen}|{m}"] = {
                "p_raw": p_raw,
                "p_bonf": p_bonf,
                "significant": bool(p_bonf < 0.05),
                "mean_diff_other_minus_tritrack": float(
                    np.mean(other) - np.mean(ref)
                ),
            }

    fa_a = res["attribution_summary"]["first_alarm"]
    print(
        f"  => 5-Scen Mean Delay: Tri={res['five_scen_mean_delay']['Tri-track']:.2f} "
        f"vs Spec={res['five_scen_mean_delay']['CUSUM-Spec']:.2f} "
        f"vs T1={res['five_scen_mean_delay']['CUSUM-T1']:.2f} | "
        f"1st-Alarm Macro-F1={fa_a['macro_f1']:.3f} | "
        f"Mean Regret={res['five_scen_mean_regret']['Attributed-TriTrack']:.2f} "
        f"({time.time()-t_start:.1f}s)"
    )
    return res


# =========================================================================
# 4. FIGURES VECTORIELLES (ANGLAIS, STYLE IEEE, SANS TITRE INTÉGRÉ)
# =========================================================================

def plot_scalability_results(scal_results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    scales = sorted(int(k) for k in scal_results["by_scale"].keys())
    if not scales:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 3.2))
    x = np.arange(len(scales))
    methods = ["CUSUM-T1", "SWORD", "AE-Spec", "CUSUM-Spec", "Tri-track"]
    colors = ["#757575", "#7b1fa2", "#ef6c00", "#2e7d32", "#c62828"]
    w = 0.15

    # Panneau gauche : délai moyen sur 5 scénarios
    for i, (m, col) in enumerate(zip(methods, colors)):
        vals = [
            scal_results["by_scale"][str(M)]["five_scen_mean_delay"][m]
            for M in scales
        ]
        ax1.bar(
            x + (i - 2) * w, vals, w, label=m, color=col, edgecolor="white"
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"$M_0={M}$" for M in scales])
    ax1.set_ylabel("5-scenario mean delay (steps)")
    ax1.grid(True, axis="y", alpha=0.25, linestyle=":")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, frameon=True)

    # Panneau droit : délai par scénario pour Tri-track à travers les échelles
    scen_list = list(SCENARIOS.keys())
    xs = np.arange(len(scen_list))
    ws = 0.72 / len(scales)
    scale_cols = ["#1565c0", "#00897b", "#c62828"]
    for idx, M in enumerate(scales):
        means = [
            scal_results["by_scale"][str(M)]["scenarios"][s]["Tri-track"]["mean_delay"]
            for s in scen_list
        ]
        lo = [
            max(
                0.0,
                means[j]
                - scal_results["by_scale"][str(M)]["scenarios"][s]["Tri-track"]["ci95_boot"][0],
            )
            for j, s in enumerate(scen_list)
        ]
        hi = [
            max(
                0.0,
                scal_results["by_scale"][str(M)]["scenarios"][s]["Tri-track"]["ci95_boot"][1]
                - means[j],
            )
            for j, s in enumerate(scen_list)
        ]
        ax2.bar(
            xs + (idx - (len(scales) - 1) / 2.0) * ws,
            means,
            ws,
            yerr=[lo, hi],
            capsize=2.0,
            label=f"Tri-track ($M_0={M}$)",
            color=scale_cols[idx % len(scale_cols)],
            edgecolor="white",
        )
    ax2.set_xticks(xs)
    ax2.set_xticklabels([s.replace("->", "$\\rightarrow$") for s in scen_list])
    ax2.set_ylabel("Tri-track detection delay (steps)")
    ax2.grid(True, axis="y", alpha=0.25, linestyle=":")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=len(scales), frameon=True)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_scalability_detection.pdf"))
    plt.close(fig)


def plot_telemetry_results(tel_results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rates = sorted(float(k) for k in tel_results["zero_shot"].keys())
    pct = [r * 100.0 for r in rates]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))

    # (a) 5-scenario mean delay
    for mode, style, col, lbl in [
        ("zero_shot", "o--", "#ef6c00", "Tri-track (zero-shot)"),
        ("site_calibrated", "s-", "#c62828", "Tri-track (site-calibrated)"),
    ]:
        ys = [
            tel_results[mode][f"{r:.2f}"]["five_scen_mean_delay"]["Tri-track"]
            for r in rates
        ]
        axes[0].plot(pct, ys, style, color=col, linewidth=1.8, markersize=5, label=lbl)
    ys_sp = [
        tel_results["site_calibrated"][f"{r:.2f}"]["five_scen_mean_delay"]["CUSUM-Spec"]
        for r in rates
    ]
    axes[0].plot(pct, ys_sp, "^:", color="#2e7d32", linewidth=1.6, markersize=5, label="CUSUM-Spec (calibrated)")
    axes[0].set_xlabel("Telemetry loss rate $p_{\\mathrm{loss}}$ (%)")
    axes[0].set_ylabel("5-scenario mean delay (steps)")
    axes[0].grid(True, alpha=0.25, linestyle=":")
    axes[0].legend(loc="upper left", frameon=True, fontsize=7)

    # (b) Realized active pre-change alpha_step on S1
    for mode, style, col, lbl in [
        ("zero_shot", "o--", "#ef6c00", "Tri-track (zero-shot)"),
        ("site_calibrated", "s-", "#c62828", "Tri-track (site-calibrated)"),
    ]:
        ys = [
            tel_results[mode][f"{r:.2f}"]["scenarios"]["S1->S2"]["Tri-track"]["alpha_step_active"]
            for r in rates
        ]
        axes[1].plot(pct, ys, style, color=col, linewidth=1.8, markersize=5, label=lbl)
    axes[1].axhline(0.005, color="#555555", linestyle=":", linewidth=1.0, label="Target $\\alpha=0.005$")
    axes[1].set_xlabel("Telemetry loss rate $p_{\\mathrm{loss}}$ (%)")
    axes[1].set_ylabel("Realized S1 $\\alpha_{\\mathrm{step}}$")
    axes[1].grid(True, alpha=0.25, linestyle=":")
    axes[1].legend(loc="upper left", frameon=True, fontsize=7)

    # (c) First-alarm causal attribution Macro-F1
    for mode, style, col, lbl in [
        ("zero_shot", "o--", "#ef6c00", "Tri-track (zero-shot)"),
        ("site_calibrated", "s-", "#1565c0", "Tri-track (site-calibrated)"),
    ]:
        ys = [
            100.0 * tel_results[mode][f"{r:.2f}"]["attribution_summary"]["first_alarm"]["macro_f1"]
            for r in rates
        ]
        axes[2].plot(pct, ys, style, color=col, linewidth=1.8, markersize=5, label=lbl)
    axes[2].set_xlabel("Telemetry loss rate $p_{\\mathrm{loss}}$ (%)")
    axes[2].set_ylabel("First-alarm attribution Macro-F1 (%)")
    axes[2].set_ylim(0, 108)
    axes[2].grid(True, alpha=0.25, linestyle=":")
    axes[2].legend(loc="lower left", frameon=True, fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_telemetry_robustness.pdf"))
    plt.close(fig)


# =========================================================================
# 5. ORCHESTRATEUR PRINCIPAL
# =========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Test rapide (~1-2 min)")
    parser.add_argument("--exp", choices=["both", "scalability", "telemetry"], default="both")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--scales", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--loss-rates", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.20])
    parser.add_argument("--timeout-k", type=int, default=3)
    parser.add_argument("--target-alpha", type=float, default=0.005)
    parser.add_argument("--n-perm", type=int, default=20000)
    parser.add_argument("--output", default="results_extra")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    if args.quick:
        n_seeds, n_reps = 3, 4
        n_cal_eps, T_cal = 8, 140
        arl_eps, arl_T = 4, 300
        n_perm = 2000
    else:
        n_seeds, n_reps = args.seeds, args.reps
        n_cal_eps, T_cal = 15, 200
        arl_eps, arl_T = 10, 500
        n_perm = args.n_perm

    os.makedirs(args.output, exist_ok=True)
    fig_dir = os.path.join(args.output, "figures")

    print("=" * 76)
    print("Expérimentations Q1 : Scalabilité (M0=64/128) & Perte de Télémétrie")
    print("=" * 76)
    print(f"Plateforme : {platform.platform()} | CPU : {platform.processor() or 'N/A'}")
    print(f"HAS_TDA={HAS_TDA} | seeds={n_seeds} x reps={n_reps} | alpha={args.target_alpha}")

    t_global = time.time()

    # ---------------------------------------------------------------------
    # EXPÉRIENCE 1 : SCALABILITÉ DE LA DÉTECTION (M0 = 64, 128)
    # ---------------------------------------------------------------------
    if args.exp in ("both", "scalability"):
        print("\n" + "=" * 76)
        print(f"[EXP 1] Passage à l'échelle de la détection : M0 in {args.scales}")
        print("=" * 76)
        scal_payload = {
            "metadata": {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "has_tda": HAS_TDA,
                "scales": args.scales,
                "n_seeds": n_seeds,
                "n_reps": n_reps,
                "target_alpha": args.target_alpha,
                "n_perm": n_perm,
            },
            "by_scale": {},
        }

        for M in args.scales:
            cfg_s1 = calibrate_core_suite(
                "S1_nominal",
                n_cal_eps,
                T_cal,
                M=M,
                target_alpha=args.target_alpha,
                seed_base=50000 + M * 100,
                p_loss=0.0,
            )
            cfg_s2 = calibrate_core_suite(
                "S2_churn",
                n_cal_eps,
                T_cal,
                M=M,
                target_alpha=args.target_alpha,
                seed_base=60000 + M * 100,
                p_loss=0.0,
            )
            scal_payload["by_scale"][str(M)] = evaluate_configuration(
                label=f"Scale_M{M}",
                M=M,
                p_loss=0.0,
                timeout_k=1,
                cfg_s1=cfg_s1,
                cfg_s2=cfg_s2,
                n_seeds=n_seeds,
                n_reps=n_reps,
                n_perm=n_perm,
                arl_eps=arl_eps,
                arl_T=arl_T,
            )

        scal_path = os.path.join(args.output, "results_scalability.json")
        with open(scal_path, "w", encoding="utf-8") as f:
            json.dump(scal_payload, f, indent=2)
        print(f"\n[OK] Sauvegardé : {scal_path}")

        if not args.skip_plots:
            plot_scalability_results(scal_payload, fig_dir)
            print(f"[OK] Figure générée : {os.path.join(fig_dir, 'fig_scalability_detection.pdf')}")

    # ---------------------------------------------------------------------
    # EXPÉRIENCE 2 : ROBUSTESSE À LA PERTE DE TÉLÉMÉTRIE (M0 = 32)
    # ---------------------------------------------------------------------
    if args.exp in ("both", "telemetry"):
        print("\n" + "=" * 76)
        print(
            f"[EXP 2] Robustesse à la perte de télémétrie (M0=32, "
            f"p_loss={args.loss_rates}, timeout_k={args.timeout_k})"
        )
        print("=" * 76)
        M_tel = 32
        tel_payload = {
            "metadata": {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "has_tda": HAS_TDA,
                "M": M_tel,
                "loss_rates": args.loss_rates,
                "timeout_k": args.timeout_k,
                "n_seeds": n_seeds,
                "n_reps": n_reps,
                "target_alpha": args.target_alpha,
                "n_perm": n_perm,
            },
            "zero_shot": {},
            "site_calibrated": {},
        }

        # Calibration de référence sur canal propre (p_loss = 0.0)
        cfg_clean_s1 = calibrate_core_suite(
            "S1_nominal",
            n_cal_eps,
            T_cal,
            M=M_tel,
            target_alpha=args.target_alpha,
            seed_base=50000,
            p_loss=0.0,
        )
        cfg_clean_s2 = calibrate_core_suite(
            "S2_churn",
            n_cal_eps,
            T_cal,
            M=M_tel,
            target_alpha=args.target_alpha,
            seed_base=60000,
            p_loss=0.0,
        )

        for p_loss in args.loss_rates:
            key = f"{p_loss:.2f}"
            # 2a. Mode zero-shot (seuils du canal propre)
            res_zs = evaluate_configuration(
                label=f"Telemetry_ZeroShot_p{int(round(p_loss*100))}",
                M=M_tel,
                p_loss=p_loss,
                timeout_k=args.timeout_k,
                cfg_s1=cfg_clean_s1,
                cfg_s2=cfg_clean_s2,
                n_seeds=n_seeds,
                n_reps=n_reps,
                n_perm=n_perm,
                arl_eps=arl_eps if p_loss > 0.0 else 0,
                arl_T=arl_T,
            )
            tel_payload["zero_shot"][key] = res_zs

            # 2b. Mode calibré sur le taux de perte du site
            if p_loss <= 0.0:
                tel_payload["site_calibrated"][key] = res_zs
            else:
                offset = int(round(p_loss * 1000))
                cfg_noisy_s1 = calibrate_core_suite(
                    "S1_nominal",
                    n_cal_eps,
                    T_cal,
                    M=M_tel,
                    target_alpha=args.target_alpha,
                    seed_base=50000 + offset,
                    p_loss=p_loss,
                    timeout_k=args.timeout_k,
                )
                cfg_noisy_s2 = calibrate_core_suite(
                    "S2_churn",
                    n_cal_eps,
                    T_cal,
                    M=M_tel,
                    target_alpha=args.target_alpha,
                    seed_base=60000 + offset,
                    p_loss=p_loss,
                    timeout_k=args.timeout_k,
                )
                res_cal = evaluate_configuration(
                    label=f"Telemetry_SiteCal_p{int(round(p_loss*100))}",
                    M=M_tel,
                    p_loss=p_loss,
                    timeout_k=args.timeout_k,
                    cfg_s1=cfg_noisy_s1,
                    cfg_s2=cfg_noisy_s2,
                    n_seeds=n_seeds,
                    n_reps=n_reps,
                    n_perm=n_perm,
                    arl_eps=arl_eps,
                    arl_T=arl_T,
                )
                tel_payload["site_calibrated"][key] = res_cal

        tel_path = os.path.join(args.output, "results_telemetry_loss.json")
        with open(tel_path, "w", encoding="utf-8") as f:
            json.dump(tel_payload, f, indent=2)
        print(f"\n[OK] Sauvegardé : {tel_path}")

        if not args.skip_plots:
            plot_telemetry_results(tel_payload, fig_dir)
            print(f"[OK] Figure générée : {os.path.join(fig_dir, 'fig_telemetry_robustness.pdf')}")

    print(f"\nTerminé en {(time.time() - t_global) / 60.0:.1f} min.")


if __name__ == "__main__":
    main()