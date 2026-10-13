"""Unit tests for Your Turn extensions (Extension 1, 2, 3)."""
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from finops import pricing
from missions import m1_efficiency_audit, m2_inference_levers, m3_purchasing


def test_extension1_recommend_tier_factors():
    # Interruptible workload with low duty cycle should recommend spot
    assert pricing.recommend_tier(6, interruptible=True, gpu_type="H100", job_days=30) == "spot"
    # Steady 24/7 non-interruptible workload should recommend reserved
    assert pricing.recommend_tier(24, interruptible=False, gpu_type="A100", job_days=365) == "reserved"
    # Spiky non-interruptible workload (< break-even duty) should recommend on_demand
    assert pricing.recommend_tier(4, interruptible=False, gpu_type="L4", job_days=30) == "on_demand"
    # Spot risk dictionary check
    assert pricing.GPU_SPOT_INTERRUPT_RISK["H100"] < pricing.GPU_SPOT_INTERRUPT_RISK["A10G"]


def test_extension2_rightsizing_by_mbu():
    res = m1_efficiency_audit.run(verbose=False)
    assert "rightsizing_recs" in res
    assert "rightsize_monthly_savings" in res
    assert res["rightsize_monthly_savings"] > 0
    # gpu-h100-4 has low MFU/MBU and should have right-sizing recommendation
    recs = res["rightsizing_recs"]
    assert any(r["gpu_id"] == "gpu-h100-4" for r in recs)


def test_extension3_cache_is_worth_it():
    # Breakeven with read_discount = 0.10 is 1 / (1 - 0.10) = 1.111 reads
    assert pricing.cache_is_worth_it(avg_cache_reads=1.5, write_cost_per_m=3.0, read_discount=0.10) is True
    assert pricing.cache_is_worth_it(avg_cache_reads=1.0, write_cost_per_m=3.0, read_discount=0.10) is False
    assert pricing.cache_is_worth_it(avg_cache_reads=0.0, write_cost_per_m=3.0, read_discount=0.10) is False
    # Explicit base price and write price
    # Saving per read = 3.0 * 0.9 = 2.7. For 2 reads: 5.4 > 3.0 -> True
    assert pricing.cache_is_worth_it(
        avg_cache_reads=2.0, write_cost_per_m=3.0, read_discount=0.10, base_read_price_per_m=3.0
    ) is True
