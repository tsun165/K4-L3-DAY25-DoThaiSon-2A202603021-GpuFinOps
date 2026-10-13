"""M1 — Efficiency Audit: MFU/MBU, the GPU-Util lie, and idle waste (deck §5).

Run: python missions/m1_efficiency_audit.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num, catalog_by_type
from finops import metrics


def run(verbose: bool = True) -> dict:
    tel = load_csv("gpu_telemetry.csv")
    cat = catalog_by_type()

    # per-row MFU/MBU, then aggregate per GPU
    agg = defaultdict(lambda: {"util": [], "mfu": [], "mbu": [], "type": None, "idle_hours": 0})
    for r in tel:
        gtype = r["gpu_type"]
        peak_fp16 = num(cat[gtype]["peak_tflops_fp16"])
        peak_bw = num(cat[gtype]["peak_bw_tbs"])
        mfu = metrics.compute_mfu(num(r["achieved_tflops"]), peak_fp16)
        mbu = metrics.compute_mbu(num(r["achieved_bw_tbs"]), peak_bw)
        a = agg[r["gpu_id"]]
        a["type"] = gtype
        a["util"].append(num(r["gpu_util_pct"]))
        a["mfu"].append(mfu)
        a["mbu"].append(mbu)
        if num(r["gpu_util_pct"]) < 10:  # effectively idle this interval (1h)
            a["idle_hours"] += 1

    summary = []
    for gid, a in agg.items():
        summary.append({
            "gpu_id": gid, "gpu_type": a["type"],
            "gpu_util_pct": round(sum(a["util"]) / len(a["util"]), 1),
            "mfu": round(sum(a["mfu"]) / len(a["mfu"]), 3),
            "mbu": round(sum(a["mbu"]) / len(a["mbu"]), 3),
            "idle_hours": a["idle_hours"],
        })

    lies = metrics.flag_util_lies(summary)
    idle_waste = 0.0
    for s in summary:
        on_demand = num(catalog_by_type()[s["gpu_type"]]["on_demand_hr"])
        idle_waste += metrics.idle_waste_usd(s["idle_hours"], on_demand)

    # --- Extension 2: Right-sizing by MBU and $/GB-VRAM economics ---
    vram_pricing = {}
    for gtype, cat_info in catalog_by_type().items():
        vram = num(cat_info["hbm_gb"])
        od = num(cat_info["on_demand_hr"])
        bw = num(cat_info["peak_bw_tbs"])
        vram_pricing[gtype] = {
            "vram_gb": vram,
            "cost_per_gb_vram": round(od / vram, 4) if vram > 0 else 0.0,
            "peak_bw_tbs": bw,
            "on_demand_hr": od,
        }

    # Find right-sizing opportunities for memory-bound GPUs (e.g. low MBU or low MFU with high headroom)
    rightsizing_recs = []
    total_rightsize_monthly_savings = 0.0
    for s in summary:
        cur_type = s["gpu_type"]
        cur_bw = vram_pricing[cur_type]["peak_bw_tbs"]
        cur_cost = vram_pricing[cur_type]["on_demand_hr"]
        # Check if GPU achieves low MBU (< 0.25) or low MFU (< 0.30)
        if s["mbu"] < 0.25 and s["mfu"] < 0.30:
            # Look for a cheaper GPU in catalog with sufficient bandwidth headroom for achieved bw
            achieved_bw = s["mbu"] * cur_bw
            candidates = [
                (gt, info) for gt, info in vram_pricing.items()
                if info["on_demand_hr"] < cur_cost and info["peak_bw_tbs"] >= achieved_bw * 1.5
            ]
            if candidates:
                # pick lowest cost candidate with sufficient bandwidth
                best_gt, best_info = min(candidates, key=lambda x: x[1]["on_demand_hr"])
                hourly_saving = cur_cost - best_info["on_demand_hr"]
                monthly_saving = hourly_saving * 24 * 30
                total_rightsize_monthly_savings += monthly_saving
                rightsizing_recs.append({
                    "gpu_id": s["gpu_id"],
                    "current_gpu": cur_type,
                    "target_gpu": best_gt,
                    "achieved_bw_tbs": round(achieved_bw, 3),
                    "target_peak_bw": best_info["peak_bw_tbs"],
                    "hourly_saving": round(hourly_saving, 2),
                    "monthly_saving": round(monthly_saving, 2),
                })

    if verbose:
        print("== M1 Efficiency Audit ==")
        print(f"{'GPU':14}{'type':7}{'util%':>7}{'MFU':>7}{'MBU':>7}{'idle_h':>8}")
        for s in sorted(summary, key=lambda x: x["mfu"]):
            print(f"{s['gpu_id']:14}{s['gpu_type']:7}{s['gpu_util_pct']:>7}{s['mfu']:>7}{s['mbu']:>7}{s['idle_hours']:>8}")
        print(f"\nGPU-Util LIES (util>=90% but MFU<30%): {[l['gpu_id'] for l in lies]}")
        print(f"Idle waste (1 day): ${idle_waste:,.2f}  ->  ${idle_waste*30:,.0f}/month")
        if rightsizing_recs:
            print("\n-- Extension 2: MBU-Guided Right-Sizing Analysis --")
            print(f"{'GPU ID':14}{'Current':8}{'Target':8}{'Achieved BW':>12}{'Target BW':>11}{'Save/mo':>10}")
            for r in rightsizing_recs:
                print(f"{r['gpu_id']:14}{r['current_gpu']:8}{r['target_gpu']:8}{r['achieved_bw_tbs']:>10.2f} TB/s{r['target_peak_bw']:>9.2f} TB/s ${r['monthly_saving']:>8,.0f}")
            print(f"Total potential monthly right-sizing savings: ${total_rightsize_monthly_savings:,.0f}")

    return {
        "summary": summary,
        "lies": lies,
        "idle_waste_daily": round(idle_waste, 2),
        "vram_pricing": vram_pricing,
        "rightsizing_recs": rightsizing_recs,
        "rightsize_monthly_savings": round(total_rightsize_monthly_savings, 2),
    }


if __name__ == "__main__":
    run()
