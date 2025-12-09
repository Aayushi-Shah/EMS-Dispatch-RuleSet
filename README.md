## **0. Hard constraints (non-negotiable)**
1. Never send BLS to a call that requires ALS.
2. Policy logic only chooses which feasible unit, never violates level-of-care feasibility.

---

## **1. Baseline policy and scenarios**

**Baseline policy P0 = pure nearest ETA**

1. Build feasible candidate set for a call:
    - Available units only.
    - Filter by ALS/BLS requirement.
2. Dispatch the unit with minimum ETA to scene.

**Scenarios we will always test on:**
- S₀: Normal load
    - Current call volume, current unit fleet.
- S₁: Stress by demand
    - Increased call density in a selected time window (e.g., 1.5–2× calls in peak hours).
- S₂: Stress by supply
    - Reduced number of units (e.g., 75% and 50% of current fleet).

(We can combine S₁+S₂ later if needed, but these three are the core.)

---

## **2. Define “critical zones” from historical data**

1. Spatial binning:
    - Partition service area into zones (beats / townships / grid cells — one scheme only).
2. Per-zone stats from historical CAD:
    - total_calls(z)
    - high_priority_calls(z)
    - pw_calls(z) = priority-weighted calls
        - Example weights: P1=3, P2=2, others=1.
3. Zone risk score:
    - risk_score(z) = pw_calls(z) (or a simple tweak if needed).
4. Critical zones:
    - Sort zones by risk_score(z) descending.
    - Pick top N (≈ 3–5) zones with:
        - Sufficient volume, and
        - Clear geographic meaning.

These become the **critical demand zones** used in rules R1/R2.

---

## **3. Baseline sim and ΔETA analysis (to get K)**

Run P0 (nearest ETA) on S₀/S₁/S₂ and log:

1. For each call where multiple units are feasible:
    - Find the **nearest** unit u_near.
    - Find the “best alternative” v_alt that:
        - Would preserve coverage in a different way (e.g., from another zone / not last unit in a critical zone).
    - Compute ΔETA = ETA(v_alt) − ETA(u_near) wherever v_alt exists.
2. Look at the ΔETA distribution in these “alternative exists” cases, especially when:
    - u_near is in a critical zone, or
    - u_near is ALS and the call does not require ALS.
3. Pick a **candidate K**:
    - Based on median / 75th percentile of ΔETA in those relevant cases.
    - Initial working range: K ≈ 2–3 minutes.

K is the maximum ETA penalty we’re willing to pay to protect coverage/ALS in those specific situations.

---

## **4. Define rule templates (no fairness)**

We use **nearest ETA as default** and add only small exception rules.

### **R1 – ALS Zone Protection (for non-ALS calls)**

Trigger conditions (all must hold):

1. Call does **not** require ALS.
2. u_near (nearest unit by ETA) is ALS.
3. u_near is the **last idle ALS** in its critical zone.
4. There exists another feasible unit v (BLS or ALS) such that:
    - ETA(v) ≤ ETA(u_near) + K.

Action:

- Dispatch v instead of u_near.
- Otherwise, stick with u_near.

R1 never downgrades an ALS-required call.

---

### **R2 – Critical Zone Coverage (any level)**

Trigger conditions (all must hold):

1. u_near is the **last idle unit** (any level) in a critical zone.
2. The incident location is **outside** that critical zone.
3. There exists another feasible unit v such that:
    - ETA(v) ≤ ETA(u_near) + K.

Action:

- Dispatch v instead of u_near.
- Otherwise, stick with u_near.

---

## **5. Policy variants to evaluate**

We keep the set minimal:

- **P0 – Baseline**
    - Pure nearest ETA (already defined).
- **P1 – Nearest ETA + ALS Protection (R1 only)**
    1. Build feasible candidates (respect ALS/BLS).
    2. Find u_near by ETA.
    3. Apply R1.
    4. Dispatch final chosen unit.
- **P2 – Nearest ETA + ALS Protection + Critical Zone Coverage (R1 + R2)**
    1. Build feasible candidates (respect ALS/BLS).
    2. Find u_near by ETA.
    3. Apply R1. If R1 triggers, done.
    4. If R1 did not trigger, apply R2.
    5. Dispatch final chosen unit.

No fairness rules. No weight soup. Just these stacked if/elses.

---

## **6. KPIs we will track**

We limit to the smallest set that can kill or validate a policy.

1. **Response time (by priority)**
    - For high-priority:
        - Mean
        - P90
        - P95
        - % high-priority calls with response time > T (e.g., T = 15 or 20 minutes; we pick one).
2. **Coverage in critical zones**
    - For each critical zone:
        - % of sim time with no idle unit within Z minutes (Z ≈ 10–12).
    - Optionally aggregated across critical zones.
3. (Optional, only if cheap)
    - Overall mean/P90 response for non-critical priorities just to ensure we’re not blowing them up.

Fairness metrics are out of scope for now.

---

## **7. Evaluation protocol**

For each policy P0, P1, P2:

1. Run on S₀ (normal) + S₁/S₂ (stress) with same seeds as far as possible.
2. Compute the KPIs above.
3. Use these decision rules:
- **Calm-mode constraint:**
    - In S₀, P1/P2 must be numerically very close to P0 on high-priority response times.
    - If a policy hurts calm-mode high-priority performance noticeably → discard it.
- **Stress-mode benefit requirement:**
    - In S₁/S₂, P1/P2 must show clear improvement vs P0 on at least one of:
        - High-priority tail metrics (P90/P95, % > T), or
        - Coverage metrics in critical zones (% time uncovered).
    - If a policy does **not** deliver clear benefit under stress → discard it.

We keep only policies that satisfy both constraints.

---

## **8. Final deliverable: rule-based frontier**

Whatever survives (likely P0 + 1–2 policies from P1/P2, with a chosen K) becomes:

1. Final **rule lists** (plain English “policy cards”).
2. Formal if/else definitions (for documentation + reproducibility).
3. KPI tables/plots showing:
    - P0 vs P1/P2 under normal and stress scenarios.
    - Clear narrative:
        - “Nearest ETA is baseline.”
        - “Policy X protects ALS/coverage with ≤ Y impact in calm mode and Z% better tails / coverage under stress.”

That’s the methodology we execute.