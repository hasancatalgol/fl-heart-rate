# Fuzzy Heart Rate — Visual Guide

## What is Fuzzy Logic?
Classical logic says a heart rate is either “normal” or “not.” Fuzzy logic says it can be **0.7 normal** and **0.2 high** at the same time.  
That nuance flows through **IF–THEN rules**, producing a **smooth** decision that reflects reality better than rigid thresholds.

![Components of a fuzzy logic controller.](docs/Components-of-a-fuzzy-logic-controller.png)


1. **Inputs**: read numbers from sensors.  
2. **Fuzzification**: convert each number to degrees (0–1) for terms like *Low*, *Normal*, *High*.  
3. **Rules & Inference**: fire IF–THEN rules and combine their effects.  
4. **Defuzzification**: turn the fuzzy result back into **one output number**.

---

## What this project models
An **age-aware** fuzzy system that maps **Heart Rate (bpm)** and **Symptoms (0–10)** to a **Risk** score (0–10). It’s a learning/demo artifact, **not** a medical device.

- **Antecedents (inputs)**  
  - **Heart Rate (HR)**: 30–200 bpm. The *Normal* band is adjusted from the helper function based on age (and athlete flag).  
  - **Symptoms**: self-reported severity 0–10.

- **Consequent (output)**  
  - **Risk**: 0–10 (illustrative scale).

### Membership functions used
- **HR**:  
  - *Low*: Z-shaped function that falls from 1 → 0 at the lower edge of the age-specific normal band.  
  - *Normal*: triangular, centered on the age-specific band.  
  - *High*: S-shaped function that rises from 0 → 1 at the upper edge of the band.
- **Symptoms**: low (Z), medium (triangle), high (S).
- **Risk**: low / medium / high (triangles).

**Figure — Memberships (age-adjusted)**  
![Memberships](docs/membership_functions_age_adjusted.png)

Why these shapes? Z/S functions give **soft shoulders** at the band edges; triangles keep the output interpretable and make the inference surface piecewise smooth.

---

## Rule base

<table>
  <thead>
    <tr>
      <th style="text-align:center;"></th>
      <th style="text-align:center;" colspan="3"><strong>Symptoms</strong></th>
    </tr>
    <tr>
      <th style="text-align:center;"><strong>HR</strong></th>
      <th style="text-align:center;">Low (0–3)</th>
      <th style="text-align:center;">Medium (3–7)</th>
      <th style="text-align:center;">High (≥7)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th style="text-align:center;">Low</th>
      <td style="text-align:center;">🟠 Medium <em>(M1)</em></td>
      <td style="text-align:center;">🔴 High <em>(H1)</em></td>
      <td style="text-align:center;">🔴 High <em>(H2)</em></td>
    </tr>
    <tr>
      <th style="text-align:center;">Normal</th>
      <td style="text-align:center;">🟢 Low <em>(L1)</em></td>
      <td style="text-align:center;">🟠 Medium <em>(M3)</em></td>
      <td style="text-align:center;">🔴 High <em>(H5)</em></td>
    </tr>
    <tr>
      <th style="text-align:center;">High</th>
      <td style="text-align:center;">🟠 Medium <em>(M2)</em></td>
      <td style="text-align:center;">🔴 High <em>(H3)</em></td>
      <td style="text-align:center;">🔴 High <em>(H4)</em></td>
    </tr>
  </tbody>
</table>

**Baseline (M4):** when HR is **extremely low or high**, add a small **Medium** risk even if symptoms are low (conservative safety net).


## Mamdani inference — how the engine works
This repository uses **Mamdani** (a.k.a. Max–Min) inference with **centroid** defuzzification.

1. **Fuzzification**: look up membership grades for the crisp inputs (e.g., HR=75 → μ_low, μ_normal, μ_high).  
2. **Rule evaluation**: combine antecedents with a **t-norm**; here **AND = min**, **OR = max**.  
3. **Implication**: each rule **clips** its consequent set by the rule’s firing strength (**min**).  
4. **Aggregation**: take the **max** across all clipped consequents to get one fuzzy output curve.  
5. **Defuzzification**: use method **Centroid (a.k.a. center-of-area, COA)**. of that aggregated curve to produce the crisp risk. There are many other defuzzification methods such as **MOM(mean of maxima)**, **SOM/LOM(smallest/largest of maxima)**, **bisector—splits area in half.** 

[Check this link for all methods.](https://en.wikipedia.org/wiki/Defuzzification)

**Figure — One Mamdani pass (aggregated output + centroid)**  
![One Mamdani Example](docs/age_adjusted_mamdani_example.png),


Notes:
- With overlapping membership functions and the **centroid** method, the crisp mapping is **continuous** but can have **kinks** where different rules dominate. **“Kinks”** means your Mamdani surface = places where the output is still continuous but the slope abruptly changes (continuous-but-not-smooth).
- True discontinuities more often appear with MOM/SOM/LOM defuzzification or with non-overlapping/gapped sets.

---

## Global behavior — the risk surface
We can evaluate the Mamdani system over a grid of inputs (HR × Symptoms) and defuzzify at each point. The result is a surface like this:

![Mamdani Risk Surface](docs/mamdani_risk_surface.png)

How to read it:
- **Valley** at **HR ≈ normal** and **Symptoms low** → “normal & low → low risk” dominates.  
- **Ridge** as symptoms move to **medium** → “normal & medium → medium risk”.  
- **Wide high plateau** once **Symptoms high** (≥7) → many **→ high** rules fire strongly.  
- **Side ledges** for **very low or very high HR** even with few symptoms → baseline “extremes carry some risk” rule.

---

## Sugeno (Takagi–Sugeno–Kang) — why & how we use it
A Sugeno system makes each rule output a **number**, then blends those numbers with a **weighted average**.  
Perfect when different symptoms should carry **different importance** or when you want parameters that can be **learned**.

### Two flavors
- **Zero-order (constant)**: each rule returns a fixed number (e.g., 2, 5, 8.5).  
- **First-order (linear)**: each rule returns a small linear function of inputs (e.g., `risk = a0 + a1·syncope + a2·chest_pain + …`).  
  This is where **symptom importance** lives (the coefficients).

### How it contrasts with Mamdani
| Aspect | **Mamdani** | **Sugeno** |
|---|---|---|
| Rule output | Fuzzy set (Low/Med/High) | **Number** (constant or linear) |
| Combine rules | Max of clipped fuzzy sets | **Weighted average** of numbers |
| Final crisp value | **Defuzzify** (e.g., centroid) | **Already crisp** (no defuzz) |
| Smoothness | Piecewise-smooth; has **kinks** | **Very smooth**; few kinks/plateaus |
| Personalization | Harder to tune from data | **Easy to learn** weights/coeffs (ANFIS) |
| Interpretability | Very human-readable rules | Clear math; coefficients carry meaning |

### 📊 Figure — Rule influence at a single point
*(Example: HR = 40, Symptoms = 7)*

- **What you see:** which rules matter most (bar = firing strength/weight).  
- **Annotations:** `z` (rule’s output value) and `w·z` (that rule’s contribution).  
- **Crisp risk:** title shows final Sugeno result = **weighted average** of all rule outputs.  
- **Why it’s useful:** reveals whether the decision is driven by, say, *“HR Low & Symptoms High → High”* vs *“Normal & Medium”*—making **symptom importance and rule impact transparent**.

![Sugeno rule weights](docs/sugeno_rule_weights_hr40_sym7.png)

---

### 🌊 Figure — Sugeno Risk Surface (zero-order)

- **Smoother** map than Mamdani: no defuzzification; just a weighted average of rule outputs.  
- **Tuning symptom impact:**  
  - *Zero-order:* choose the constants per rule (e.g., 2/5/8.5) to set target risk levels.  
  - *First-order:* give **red-flag symptoms** bigger coefficients in `risk = a0 + Σ a_i·symptom_i (+ a_HR·HR)`.  
- **What to check:**  
  - High risk appears **where it should** (e.g., red flags or HR extremes).  
  - Transitions between Low/Med/High feel **gradual and realistic**.

![Sugeno Risk Surface](docs/sugeno_risk_surface.png)

---

### 🧭 Practical tips
- Start **zero-order** to match your Mamdani levels; then move to **first-order** to weight individual symptoms.  
- Keep a few **“red-flag” rules** (e.g., syncope at rest) with higher targets or coefficients.  
- Validate with **spot checks** (e.g., 75/1 → low, 40/7 → high, 110/2 → medium-ish) and with contour/heatmaps for readability.

---

## Tuning guide (what to adjust)
- **Age & athlete profile**: changes the HR normal band.  
- **MF steepness**: widen or narrow Z/S shoulders (±12 bpm in this demo). Softer shoulders reduce abrupt changes.  
- **Symptoms thresholds**: shift 3/5/7 to change when medium/high kick in.  
- **Rule weights**: lower the influence of the “→ High” rules to reduce the high plateau; or split *High* into *Moderately High* vs *Very High*.  
- **Defuzzification**: centroid is smooth and robust; MOM/SOM/LOM give more abrupt behavior.

---

## Sanity checks (expected outputs)
- **HR=75, Symptoms=1** → low risk (in the valley).  
- **HR=40, Symptoms=7** → high risk (on the top plateau).  
- **HR=110, Symptoms=2** → medium-ish (edge ledge).

---
