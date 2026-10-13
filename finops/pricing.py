"""Pricing & purchasing economics — measure in $/1M-token, not $/GPU-hr.

Figures are June-2026 as-of snapshots from the deck's RESEARCH dossier; treat
live prices as fast-moving (re-baseline before each cohort).
"""
from __future__ import annotations


def request_cost(
    input_tok: int,
    output_tok: int,
    price_in_per_m: float,
    price_out_per_m: float,
    cached_in: int = 0,
    cache_discount: float = 0.10,   # Anthropic cached-read ~0.1x (=-90%)
    batch: bool = False,
    batch_discount: float = 0.50,   # Batch API ~ -50%
) -> float:
    """USD cost of a single request. Cached input billed at cache_discount x price."""
    cached_in = min(max(0, cached_in), input_tok)
    uncached_in = input_tok - cached_in
    cost = (
        (uncached_in / 1e6) * price_in_per_m
        + (cached_in / 1e6) * price_in_per_m * cache_discount
        + (output_tok / 1e6) * price_out_per_m
    )
    if batch:
        cost *= batch_discount
    return cost


def dollars_per_million(total_cost_usd: float, total_tokens: int) -> float:
    """Aggregate unit economics: $ per 1,000,000 tokens served."""
    if total_tokens <= 0:
        return 0.0
    return total_cost_usd / (total_tokens / 1e6)


def discount_stack(
    batch: bool = False,
    cache_hit_frac: float = 0.0,
    batch_discount: float = 0.50,
    cache_discount: float = 0.10,
) -> float:
    """Effective fraction of the naive bill after stacking discounts (input-heavy view).

    Discounts MULTIPLY: cache applies to the cached share of input, batch to the
    whole bill. batch + 100% cache-hit -> 0.5 * 0.1 = 0.05 (~95% off).
    """
    cache_mult = cache_hit_frac * cache_discount + (1.0 - cache_hit_frac)
    batch_mult = batch_discount if batch else 1.0
    return cache_mult * batch_mult


def break_even_utilization(discount_frac: float) -> float:
    """Utilization at which a commitment pays off ~= 1 - discount.

    A 45% reserved discount needs ~55% utilization (~13.2h/day) to beat on-demand.
    """
    return max(0.0, min(1.0, 1.0 - discount_frac))


def cache_is_worth_it(
    avg_cache_reads: float,
    write_cost_per_m: float,
    read_discount: float = 0.10,
    base_read_price_per_m: float | None = None,
) -> bool:
    """Determine if caching pays off.

    Break-even economics:
    Saving per read = base_read_price * (1 - read_discount).
    Total saving over N reads = N * saving_per_read.
    Cache is worth it if total savings >= write cost.
    If base_read_price is not provided, assumes write_cost equals base read cost,
    meaning N * (1 - read_discount) >= 1.0  =>  N >= 1 / (1 - read_discount).
    """
    if avg_cache_reads <= 0:
        return False
    if base_read_price_per_m is None or base_read_price_per_m <= 0:
        unit_saving = max(0.0, 1.0 - read_discount)
        required_reads = 1.0 / unit_saving if unit_saving > 0 else float("inf")
    else:
        unit_saving = base_read_price_per_m * max(0.0, 1.0 - read_discount)
        required_reads = write_cost_per_m / unit_saving if unit_saving > 0 else float("inf")
    return avg_cache_reads >= required_reads


# Typical interruption likelihood across GPU families
GPU_SPOT_INTERRUPT_RISK = {
    "H100": 0.03,    # Low-medium volatility in modern dedicated pools
    "H200": 0.04,
    "B200": 0.05,
    "A100": 0.08,
    "MI300X": 0.08,
    "A10G": 0.15,    # High churn on commodity instances
    "L4": 0.12,
}


def recommend_tier(
    hours_per_day: float,
    interruptible: bool,
    reserved_discount: float = 0.45,
    gpu_type: str | None = None,
    job_days: int | float | None = None,
) -> str:
    """Pick a purchasing tier from duty cycle, interruptibility, GPU risk, and duration.

    Multi-factor policy:
      - If interruptible:
          - If spot preemption risk is acceptable or duty cycle is non-continuous (< 24h),
            recommend 'spot'.
          - Exception: If job is critical 24/7 and interrupt risk is prohibitive (>20%),
            fallback to on_demand / reserved.
      - If non-interruptible:
          - If duty cycle >= break-even (duty >= 1 - reserved_discount):
              - For long-term workloads (job_days >= 365 or recurring long-term),
                recommend 'reserved'.
              - Otherwise recommend 'reserved' if steady utilization pays off commitments.
          - Otherwise: 'on_demand' (spiky or low duty cycle).
    """
    duty = max(0.0, hours_per_day) / 24.0
    be = break_even_utilization(reserved_discount)

    if interruptible:
        risk = GPU_SPOT_INTERRUPT_RISK.get(gpu_type, 0.08) if gpu_type else 0.08
        # If job is 24/7 or very high churn risk, spot might incur excessive checkpoint overhead
        if hours_per_day >= 24 and risk > 0.20:
            return "reserved" if duty >= be else "on_demand"
        return "spot"

    if duty >= be:
        return "reserved"

    return "on_demand"


def spot_checkpoint_cost(
    job_hours: float,
    spot_hr: float,
    on_demand_hr: float,
    interrupt_rate: float = 0.05,      # per-hour chance (H100 spot ~<5%)
    ckpt_overhead_frac: float = 0.03,  # steady cost of writing checkpoints
    rework_hours_per_interrupt: float = 0.5,
) -> dict:
    """Effective cost of running a checkpointable job on spot vs on-demand.

    Interruptions waste the compute since the last checkpoint (rework); checkpointing
    adds a small steady overhead. Spot still wins for interruptible jobs.
    """
    expected_interrupts = job_hours * interrupt_rate
    rework_hours = expected_interrupts * rework_hours_per_interrupt
    effective_hours = job_hours * (1.0 + ckpt_overhead_frac) + rework_hours
    spot_cost = effective_hours * spot_hr
    on_demand_cost = job_hours * on_demand_hr
    savings_pct = (1.0 - spot_cost / on_demand_cost) * 100.0 if on_demand_cost > 0 else 0.0
    return {
        "spot_effective_hours": round(effective_hours, 2),
        "spot_cost": round(spot_cost, 2),
        "on_demand_cost": round(on_demand_cost, 2),
        "savings_pct": round(savings_pct, 1),
    }
