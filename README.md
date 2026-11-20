# Symmetric Tri-Track Structural Regime Detection and Fault Attribution for Non-Stationary Fog–Cloud–Edge Topologies

Official simulation and benchmark codebase for:

> **Symmetric Tri-Track Structural Regime Detection and Fault Attribution for Non-Stationary Fog–Cloud–Edge Topologies in Industry 4.0**  
> Mohsen Aydi, Houssem Eddine Nouri, and Olfa Belkahla Driss  
> *LARIA, UR22ES01, ENSI, Université de la Manouba, Tunisia*

## Repository Structure

- `fog_sim_clean.py` — Stationary dynamic attributed Fog graph simulator (`S1`–`S5`), deterministic tri-track extractors (Track 1: normalized graph statistics; Track 2: clique-complex $H_0+H_1$ persistent homology; Track 3: normalized Laplacian spectrum), and statistical test utilities.
- `baselines_clean.py` — Empirically matched-FAR sequential detectors (`CUSUM`, `EWMA`, Adams & MacKay `BOCPD`, `MMD`, `SWORD`, `AE`, `Windowed-GLR`), degree-augmented spatial-temporal GNN ablation (`FairSTGNNEncoder`), symmetric OR-triggering `TriTrackFusion` with 3-way causal fault attribution, and timeout-free downstream scheduler regret evaluator.
- `run_benchmark_clean.py` — Full 10-seed × 10-replication benchmark at $M_0 = 32$ (`ARL0`, detection delay, per-track ablation, attribution Macro-F1, downstream regret, paired permutation tests, and per-step latency scaling).
- `run_robustness_scalability.py` — Extended scalability benchmark ($M_0 \in \{64, 128\}$) and telemetry packet-loss robustness benchmark ($p_{\mathrm{loss}} \in \{0\%, 5\%, 10\%, 20\%\}$ in zero-shot and site-calibrated modes).
- `results/` — Saved JSON outputs from the 10-seed × 10-replication experiments reported in the paper.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducing the Experiments

**1. Main Benchmark ($M_0 = 32$, 10 seeds × 10 replications):**
```bash
python run_benchmark_clean.py --seeds 10 --reps 10 --output results_final
```
*(Use `--quick` for a 2-minute smoke test.)*

**2. Scalability ($M_0 \in \{64, 128\}$) and Telemetry Loss ($p_{\mathrm{loss}} \in \{0\%, 5\%, 10\%, 20\%\}$):**
```bash
python run_robustness_scalability.py --seeds 10 --reps 10 --output results_extra
```

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.