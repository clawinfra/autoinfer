#!/usr/bin/env python3
"""
Bayesian optimizer for Phase 12 — uses Optuna TPE to warm-start from
existing Phase 11/12 IQ2_XXS data and predict optimal configs for IQ1_M/Q2_K.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("Optuna not installed, running manual grid analysis instead.")


PHASES_DIR = Path("/tmp/qwen35-moe-offload")
TOTAL_VRAM_GB = 42.0  # RTX 3090 (24) + 3080 (10) + 2070S (8)

# Known OOM boundaries from experiments
OOM_CONFIGS = {
    "IQ2_XXS": {"n_gpu_max": 27},  # 27 crashes, 26 works
    "Q3_K_M": {"n_gpu_max": 17},
    "IQ2_M": {"n_gpu_max": 24},
}


def load_phase11_12_data():
    """Load Phase 11/12 IQ2_XXS experiments for warm-starting."""
    rows = []
    
    for phase_file, phase in [
        (PHASES_DIR / "results_phase11.tsv", 11),
        (PHASES_DIR / "results_phase12.tsv", 12),
    ]:
        if not phase_file.exists():
            continue
        
        with open(phase_file) as f:
            lines = f.readlines()
        
        for line in lines[1:]:
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue
            try:
                exp_id = parts[0]
                tok_s = float(parts[1])
                vram_mb = int(float(parts[2]))
                n_ctx = int(float(parts[3]))
                type_k = parts[4]
                type_v = parts[5]
                flash = parts[6].lower() in ("true", "1")
                n_gpu = int(float(parts[7]))
                n_batch = int(float(parts[8]))
                n_ubatch = int(float(parts[9])) if len(parts) > 9 else 16
                label = parts[10] if len(parts) > 10 else ""
                status = parts[11] if len(parts) > 11 else "ok"
                
                if status not in ("ok",) or tok_s <= 0:
                    continue
                
                # Only include IQ2_XXS configs
                if "iq2xxs" not in label.lower() and phase == 11:
                    if "iq2_xxs" not in label.lower():
                        continue
                
                # KV type as numeric (q4_0=4, q8_0=8, f16=16)
                kv_map = {"q4_0": 4, "q8_0": 8, "f16": 16, "f32": 32}
                kv_bits = kv_map.get(type_k, 8)
                
                rows.append({
                    "phase": phase,
                    "exp_id": exp_id,
                    "tok_s": tok_s,
                    "vram_mb": vram_mb,
                    "n_ctx": n_ctx,
                    "kv_bits": kv_bits,
                    "flash": flash,
                    "n_gpu": n_gpu,
                    "n_batch": n_batch,
                    "n_ubatch": n_ubatch,
                    "label": label,
                })
            except (ValueError, IndexError):
                continue
    
    return rows


def build_surrogate_model(rows):
    """Build a simple empirical model from the data for IQ2_XXS."""
    if not rows:
        return None
    
    # Find the relationships
    # 1. n_gpu → tok_s (holding others constant around optimal)
    ngpu_data = defaultdict(list)
    for r in rows:
        ngpu_data[r["n_gpu"]].append(r["tok_s"])
    
    ngpu_means = {k: sum(v)/len(v) for k, v in ngpu_data.items()}
    
    # 2. n_ctx → tok_s
    ctx_data = defaultdict(list)
    for r in rows:
        ctx_data[r["n_ctx"]].append(r["tok_s"])
    ctx_means = {k: sum(v)/len(v) for k, v in ctx_data.items()}
    
    # 3. kv_bits → tok_s
    kv_data = defaultdict(list)
    for r in rows:
        kv_data[r["kv_bits"]].append(r["tok_s"])
    kv_means = {k: sum(v)/len(v) for k, v in kv_data.items()}
    
    # 4. n_batch → tok_s (small batches seem better for IQ2_XXS)
    batch_data = defaultdict(list)
    for r in rows:
        batch_key = r["n_batch"] // 16 * 16  # bin to 16s
        batch_data[batch_key].append(r["tok_s"])
    batch_means = {k: sum(v)/len(v) for k, v in batch_data.items() if len(v) >= 1}
    
    return {
        "ngpu": ngpu_means,
        "ctx": ctx_means,
        "kv_bits": kv_means,
        "batch": batch_means,
        "best_observed": max(r["tok_s"] for r in rows),
        "best_config": max(rows, key=lambda r: r["tok_s"]),
    }


def optuna_optimization(rows, model_quant="IQ2_XXS", n_trials=200):
    """Run Optuna TPE to predict optimal config."""
    if not HAS_OPTUNA:
        return None
    
    # Build lookup from observations
    # We'll use a polynomial interpolation approach
    # Since we can't run new benchmarks, we predict based on the surrogate
    
    surrogate = build_surrogate_model(rows)
    if not surrogate:
        return None
    
    ngpu_means = surrogate["ngpu"]
    ctx_means = surrogate["ctx"]
    kv_means = surrogate["kv_bits"]
    batch_means = surrogate["batch"]
    
    # Global mean
    all_tok_s = [r["tok_s"] for r in rows]
    global_mean = sum(all_tok_s) / len(all_tok_s)
    
    def predict_tok_s(n_gpu, n_ctx, kv_bits, n_batch, n_ubatch):
        """Predict tok_s for a config using additive effects model."""
        # Baseline: global mean
        base = global_mean
        
        # Additive effects (log-scale multiplicative)
        def effect(lookup, key, default_key):
            sorted_keys = sorted(lookup.keys())
            if not sorted_keys:
                return 1.0
            # Find nearest observed value
            nearest = min(sorted_keys, key=lambda k: abs(k - key))
            ref_key = min(sorted_keys, key=lambda k: abs(k - default_key))
            if lookup.get(ref_key, 0) == 0:
                return 1.0
            return lookup.get(nearest, global_mean) / lookup.get(ref_key, global_mean)
        
        # Reference config
        ref_ngpu = max(ngpu_means.keys(), key=lambda k: ngpu_means[k])
        ref_ctx = min(ctx_means.keys()) if ctx_means else 256
        ref_kv = 8  # q8_0
        ref_batch = 32
        
        ngpu_eff = effect(ngpu_means, n_gpu, ref_ngpu)
        ctx_eff = effect(ctx_means, n_ctx, ref_ctx)
        kv_eff = effect(kv_means, kv_bits, ref_kv)
        
        predicted = base * ngpu_eff * ctx_eff * kv_eff
        
        # Ubatch ratio penalty
        if n_ubatch > n_batch:
            predicted *= 0.7  # severe penalty
        elif n_ubatch > n_batch // 2:
            predicted *= 0.95
        
        return max(0, predicted)
    
    def objective(trial):
        # Sample config
        if model_quant == "IQ2_XXS":
            n_gpu = trial.suggest_int("n_gpu", 22, 27)
        elif model_quant == "IQ1_M":
            n_gpu = trial.suggest_int("n_gpu", 24, 30)
        elif model_quant == "Q2_K":
            n_gpu = trial.suggest_int("n_gpu", 20, 26)
        else:
            n_gpu = trial.suggest_int("n_gpu", 16, 28)
        
        n_ctx = trial.suggest_categorical("n_ctx", [128, 256, 512, 1024])
        kv_bits = trial.suggest_categorical("kv_bits", [4, 8])
        n_batch = trial.suggest_categorical("n_batch", [16, 24, 32, 48, 64, 96, 128, 252])
        n_ubatch_ratio = trial.suggest_float("ubatch_ratio", 0.25, 0.75)
        n_ubatch = max(1, int(n_batch * n_ubatch_ratio))
        
        return predict_tok_s(n_gpu, n_ctx, kv_bits, n_batch, n_ubatch)
    
    # Warm-start: add known good configs as initial trials
    good_configs = sorted(rows, key=lambda r: r["tok_s"], reverse=True)[:5]
    
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    
    # Enqueue best known configs
    for r in good_configs:
        kv_map = {4: 4, 8: 8, 16: 16}
        try:
            study.enqueue_trial({
                "n_gpu": r["n_gpu"],
                "n_ctx": r["n_ctx"],
                "kv_bits": r["kv_bits"],
                "n_batch": r["n_batch"] if r["n_batch"] in [16, 24, 32, 48, 64, 96, 128, 252] else 32,
                "ubatch_ratio": r["n_ubatch"] / max(1, r["n_batch"]),
            })
        except Exception:
            pass
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    best = study.best_params
    best_value = study.best_value
    
    # Top 5 unique configs
    top_trials = sorted(study.trials, key=lambda t: t.value or 0, reverse=True)
    top_configs = []
    seen = set()
    for t in top_trials:
        if t.params:
            key = (t.params.get("n_gpu"), t.params.get("n_ctx"), 
                   t.params.get("kv_bits"), t.params.get("n_batch"))
            if key not in seen:
                seen.add(key)
                ubatch = max(1, int(t.params.get("n_batch", 32) * t.params.get("ubatch_ratio", 0.5)))
                top_configs.append({
                    "n_gpu": t.params.get("n_gpu"),
                    "n_ctx": t.params.get("n_ctx"),
                    "kv_bits": t.params.get("kv_bits"),
                    "n_batch": t.params.get("n_batch"),
                    "n_ubatch": ubatch,
                    "predicted_tok_s": round(t.value or 0, 3),
                })
            if len(top_configs) >= 5:
                break
    
    return {
        "best_predicted": round(best_value, 3),
        "best_config": {
            "n_gpu": best.get("n_gpu"),
            "n_ctx": best.get("n_ctx"),
            "kv_bits": best.get("kv_bits"),
            "kv_type": "q8_0" if best.get("kv_bits") == 8 else "q4_0",
            "n_batch": best.get("n_batch"),
            "n_ubatch": max(1, int(best.get("n_batch", 32) * best.get("ubatch_ratio", 0.5))),
        },
        "top_5_configs": top_configs,
    }


def main():
    print("=" * 70)
    print("Bayesian Optimizer — Phase 12 Config Prediction")
    print("=" * 70)
    
    rows = load_phase11_12_data()
    print(f"\nLoaded {len(rows)} valid IQ2_XXS experiments from Phase 11-12")
    
    if not rows:
        print("No data found!")
        return
    
    # Surrogate model analysis
    surrogate = build_surrogate_model(rows)
    
    print(f"\nEmpirical analysis:")
    print(f"  Best observed: {surrogate['best_observed']:.3f} tok/s")
    best_cfg = surrogate["best_config"]
    print(f"  Best config: n_gpu={best_cfg['n_gpu']}, ctx={best_cfg['n_ctx']}, "
          f"kv={best_cfg['kv_bits']}bit, batch={best_cfg['n_batch']}/{best_cfg['n_ubatch']}")
    
    print(f"\nn_gpu effect (mean tok/s by GPU layers):")
    for ngpu in sorted(surrogate["ngpu"].keys()):
        print(f"  n_gpu={ngpu:3d}: {surrogate['ngpu'][ngpu]:.3f} tok/s")
    
    print(f"\nn_ctx effect:")
    for ctx in sorted(surrogate["ctx"].keys()):
        print(f"  n_ctx={ctx:6d}: {surrogate['ctx'][ctx]:.3f} tok/s")
    
    print(f"\nKV bits effect:")
    for kv in sorted(surrogate["kv_bits"].keys()):
        kv_name = {4: "q4_0", 8: "q8_0", 16: "f16"}.get(kv, str(kv))
        print(f"  {kv_name} ({kv}bit): {surrogate['kv_bits'][kv]:.3f} tok/s")
    
    print(f"\nBatch size effect:")
    for batch in sorted(surrogate["batch"].keys()):
        print(f"  n_batch≈{batch:4d}: {surrogate['batch'][batch]:.3f} tok/s")
    
    if HAS_OPTUNA:
        print(f"\n{'='*70}")
        print("Running Optuna TPE optimization (200 trials)...")
        
        result = optuna_optimization(rows, model_quant="IQ2_XXS", n_trials=200)
        
        if result:
            print(f"\nBest predicted IQ2_XXS config:")
            print(f"  Predicted tok/s: {result['best_predicted']:.3f}")
            cfg = result["best_config"]
            print(f"  n_gpu={cfg['n_gpu']}, n_ctx={cfg['n_ctx']}, "
                  f"kv={cfg['kv_type']}, batch={cfg['n_batch']}/{cfg['n_ubatch']}")
            
            print(f"\nTop 5 recommended configs to try:")
            for i, cfg in enumerate(result["top_5_configs"], 1):
                kv = {4: "q4_0", 8: "q8_0"}.get(cfg["kv_bits"], "q8_0")
                print(f"  {i}. n_gpu={cfg['n_gpu']}, ctx={cfg['n_ctx']}, "
                      f"kv={kv}, batch={cfg['n_batch']}/{cfg['n_ubatch']} "
                      f"→ {cfg['predicted_tok_s']:.3f} tok/s pred")
        
        # Now predict for IQ1_M and Q2_K (scale from IQ2_XXS model)
        print(f"\n{'='*70}")
        print("Phase 12 IQ1_M and Q2_K predictions (scaled from power law):")
        
        # Power law: tok_s = C × (VRAM_free)^4.25
        # IQ2_XXS best: 27.59 tok/s, free VRAM: 33 GB
        # IQ1_M: free VRAM: 34.2 GB
        # Q2_K: free VRAM: 30.8 GB
        
        alpha = 4.247
        free_iq2xxs = 42.0 - 9.0
        best_iq2xxs = 27.589
        
        C = best_iq2xxs / (free_iq2xxs ** alpha)
        
        print(f"\nPower law: tok_s = {C:.6e} × VRAM_free^{alpha:.3f}")
        
        quants = {
            "IQ1_M":  {"size_gb": 7.8,  "quality": 0.620, "ngpu_est": 29},
            "Q2_K":   {"size_gb": 11.2, "quality": 0.880, "ngpu_est": 24},
            "IQ2_XXS":{"size_gb": 9.0,  "quality": 0.750, "ngpu_est": 27},
        }
        
        print(f"\n{'Quant':12s} {'Pred tok/s':11s} {'Quality':8s} {'Q-Adj':8s} {'Recommended n_gpu'}")
        print("-" * 60)
        for quant, meta in quants.items():
            free = 42.0 - meta["size_gb"]
            pred = C * (free ** alpha)
            qa = pred * meta["quality"]
            print(f"{quant:12s} {pred:11.1f} {meta['quality']:8.3f} {qa:8.1f} n_gpu~{meta['ngpu_est']}")
        
        print(f"""
KEY PHASE 12 RECOMMENDATIONS:
1. IQ1_M (~{C*(34.2**alpha):.0f} tok/s pred): Try n_gpu=28-30, ctx=256, q8_0 KV, batch=32/16
   Risk: Quality cliff. Run perplexity check immediately after first benchmark.
   
2. Q2_K (~{C*(30.8**alpha):.0f} tok/s pred): Try n_gpu=22-24, ctx=512, q8_0 KV, batch=32/16  
   More predictable quality. Expected quality-adj ≈ 18.1 (near IQ2_M territory).

3. Don't leave on table: IQ2_XXS at ctx=128 with n_gpu=28 might squeeze another 2-3%
   Current best is ctx=512 gen400 — try ctx=256 gen200 for purer throughput.

4. Perplexity measurement priority: IQ2_XXS quality_score=0.75 is a guess.
   Real measurement could change quality-adj rankings significantly.
""")
    else:
        print("\nInstall optuna for full Bayesian optimization:")
        print("  uv pip install optuna")
    
    # Save predictions
    output = {
        "surrogate": {
            "ngpu_effects": {str(k): v for k, v in surrogate["ngpu"].items()},
            "ctx_effects": {str(k): v for k, v in surrogate["ctx"].items()},
            "kv_effects": {str(k): v for k, v in surrogate["kv_bits"].items()},
            "batch_effects": {str(k): v for k, v in surrogate["batch"].items()},
        },
        "phase12_predictions": {
            "IQ1_M": {"predicted_tok_s": round(C * (34.2**alpha), 1), "ngpu_recommended": 29},
            "Q2_K": {"predicted_tok_s": round(C * (30.8**alpha), 1), "ngpu_recommended": 24},
        },
    }
    
    with open("/tmp/autoinfer/scripts/bayesian_predictions.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Predictions saved to: /tmp/autoinfer/scripts/bayesian_predictions.json")


if __name__ == "__main__":
    main()
