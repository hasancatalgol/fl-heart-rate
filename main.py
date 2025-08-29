import os
import numpy as np
import skfuzzy as fuzz
import matplotlib

# Use a non-interactive backend if no display is available (helps on servers/CI)
if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# -------------------------------
# Age-aware resting heart-rate bands (illustrative)
# -------------------------------
def hr_normal_band_by_age(age_years, athlete=False):
    a = float(age_years)
    if a < 0.42:
        lo, hi = 100, 160  # newborn–5 mo
    elif a < 1.0:
        lo, hi = 80, 140  # 6–12 mo
    elif a < 3.0:
        lo, hi = 80, 130  # 1–3 y
    elif a < 6.0:
        lo, hi = 80, 120  # 3–5 y
    elif a < 11.0:
        lo, hi = 70, 110  # 6–10 y
    elif a < 15.0:
        lo, hi = 60, 105  # 11–14 y
    else:
        lo, hi = 60, 100  # ≥15 y (adult)
    if athlete and a >= 15.0:
        lo = max(40, lo - 15)  # allow 40–60 in trained adults
    return lo, hi


# >>> Subject profile <<<
AGE_YEARS = 30
IS_ATHLETE = False
HR_LO, HR_HI = hr_normal_band_by_age(AGE_YEARS, athlete=IS_ATHLETE)

# 1) Universes
hr = np.linspace(30, 200, 400)  # Heart Rate (bpm)
sym = np.linspace(0, 10, 200)   # Symptoms (0–10)
risk = np.linspace(0, 10, 200)  # Risk (0–10)

# 2) Membership functions
hr_low = fuzz.zmf(hr, HR_LO - 12, HR_LO)
hr_normal = fuzz.trimf(hr, [HR_LO, (HR_LO + HR_HI) / 2, HR_HI])
hr_high = fuzz.smf(hr, HR_HI, HR_HI + 12)

sym_low = fuzz.zmf(sym, 2, 4)
sym_med = fuzz.trimf(sym, [3, 5, 7])
sym_high = fuzz.smf(sym, 6, 8)

risk_low = fuzz.trimf(risk, [0, 0, 4])
risk_med = fuzz.trimf(risk, [2, 5, 8])
risk_high = fuzz.trimf(risk, [6, 10, 10])


# 3) Plot (legends outside-right; suptitle above — no overlap)
def plot_memberships():
    fig, axs = plt.subplots(3, 1, figsize=(9, 9), constrained_layout=True)

    # Put the title ABOVE the layout area so it doesn't overlap the first axes
    fig.suptitle("Fuzzy Logic Knowledge Base — Membership Functions", y=1.04, fontsize=14)
    # Separate subtitle line
    fig.text(0.5, 1.00, f"(Rest HR; age={AGE_YEARS}, athlete={IS_ATHLETE})", ha="center", va="bottom", fontsize=11)

    # ---- INPUT 1: Heart Rate ----
    axs[0].plot(hr, hr_low, label="HR Low (age-adjusted)")
    axs[0].plot(hr, hr_normal, label="HR Normal (age-adjusted)")
    axs[0].plot(hr, hr_high, label="HR High (age-adjusted)")
    axs[0].axvspan(HR_LO, HR_HI, alpha=0.1)
    axs[0].axvline(HR_LO, linestyle="--", alpha=0.5)
    axs[0].axvline(HR_HI, linestyle="--", alpha=0.5)
    axs[0].set_title("1: Heart Rate — Memberships (Knowledge Base; used by Fuzzification)")
    axs[0].set_xlabel("Heart Rate (bpm) [RESTING]")
    axs[0].set_ylabel("Membership")
    axs[0].set_ylim(-0.05, 1.05)

    # ---- INPUT 2: Symptoms ----
    axs[1].plot(sym, sym_low, label="Symptoms Low")
    axs[1].plot(sym, sym_med, label="Symptoms Med")
    axs[1].plot(sym, sym_high, label="Symptoms High")
    axs[1].set_title("2: Symptoms — Memberships (Knowledge Base; used by Fuzzification)")
    axs[1].set_xlabel("Symptoms (0–10)")
    axs[1].set_ylabel("Membership")
    axs[1].set_ylim(-0.05, 1.05)

    # ---- OUTPUT: Risk ----
    axs[2].plot(risk, risk_low, label="Risk Low")
    axs[2].plot(risk, risk_med, label="Risk Medium")
    axs[2].plot(risk, risk_high, label="Risk High")
    axs[2].set_title("3: Risk — Consequent Memberships (Knowledge Base; used by Inference)")
    axs[2].set_xlabel("Risk (0–10)")
    axs[2].set_ylabel("Membership")
    axs[2].set_ylim(-0.05, 1.05)

    # Legends outside-right, without forcing an internal gutter
    for ax in axs:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, borderaxespad=0)

    # Save cleanly (includes outside legends and above-title)
    os.makedirs("docs", exist_ok=True)
    plt.savefig("docs/membership_functions_age_adjusted.png", dpi=150, bbox_inches="tight", pad_inches=0.05)
    print("Diagram saved to docs/membership_functions_age_adjusted.png")
    plt.show()


# --- Inference + Defuzzification (Mamdani min–max, centroid) ---
def grade(x, mf_x, val):
    return fuzz.interp_membership(x, mf_x, val)


def crisp_risk(hr_val, sym_val, show=False):
    μ_hr_low = grade(hr, hr_low, hr_val)
    μ_hr_norm = grade(hr, hr_normal, hr_val)
    μ_hr_high = grade(hr, hr_high, hr_val)

    μ_sym_low = grade(sym, sym_low, sym_val)
    μ_sym_med = grade(sym, sym_med, sym_val)
    μ_sym_high = grade(sym, sym_high, sym_val)

    # Rule strengths
    r_hi_1 = min(μ_hr_low,  μ_sym_med)
    r_hi_2 = min(μ_hr_low,  μ_sym_high)
    r_hi_3 = min(μ_hr_high, μ_sym_med)
    r_hi_4 = min(μ_hr_high, μ_sym_high)
    r_hi_5 = min(μ_hr_norm, μ_sym_high)

    r_med_1 = min(μ_hr_low,  μ_sym_low)
    r_med_2 = min(μ_hr_high, μ_sym_low)
    r_med_3 = min(μ_hr_norm, μ_sym_med)
    r_med_4 = 0.5 * max(μ_hr_low, μ_hr_high)  # baseline partial weight

    r_low_1 = min(μ_hr_norm, μ_sym_low)

    act_hi = np.fmax.reduce([
        np.fmin(risk_high, r_hi_1),
        np.fmin(risk_high, r_hi_2),
        np.fmin(risk_high, r_hi_3),
        np.fmin(risk_high, r_hi_4),
        np.fmin(risk_high, r_hi_5),
    ])

    act_med = np.fmax.reduce([
        np.fmin(risk_med, r_med_1),
        np.fmin(risk_med, r_med_2),
        np.fmin(risk_med, r_med_3),
        np.fmin(risk_med, r_med_4),
    ])

    act_low = np.fmin(risk_low, r_low_1)

    aggregated = np.fmax(act_low, np.fmax(act_med, act_hi))
    crisp = fuzz.defuzz(risk, aggregated, 'centroid')

    if show:
        plt.figure(figsize=(7.2, 3.6), constrained_layout=True)
        plt.plot(risk, risk_low, linestyle=':', label='Risk Low set')
        plt.plot(risk, risk_med, linestyle=':', label='Risk Med set')
        plt.plot(risk, risk_high, linestyle=':', label='Risk High set')
        plt.fill_between(risk, 0, aggregated, alpha=0.2, label='Aggregated')
        plt.axvline(crisp, linestyle='--', label=f'Crisp risk = {crisp:.2f}')
        plt.ylim(-0.05, 1.05)
        plt.xlabel("Risk (0–10)")
        plt.ylabel("Membership")
        plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
        os.makedirs("docs", exist_ok=True)
        plt.savefig("docs/age_adjusted_mamdani_example.png", dpi=150, bbox_inches="tight", pad_inches=0.05)
        print("Diagram saved to docs/age_adjusted_mamdani_example.png")
        plt.show()

    return crisp


# ============================================================
# Mamdani risk surface (centroid defuzz)
# ============================================================
def mamdani_risk_surface(hr_pts=151, sym_pts=101, hr_min=30, hr_max=200, sym_min=0, sym_max=10,
                         save_path="docs/mamdani_risk_surface.png"):
    """
    Evaluate crisp risk over an HR×Symptoms grid using our Mamdani engine (crisp_risk) and plot a 3D surface.
    """
    H = np.linspace(hr_min, hr_max, hr_pts)
    S = np.linspace(sym_min, sym_max, sym_pts)
    Z = np.empty((sym_pts, hr_pts), dtype=float)

    # Note: crisp_risk uses centroid and our current rule base.
    for i, s in enumerate(S):
        for j, h in enumerate(H):
            Z[i, j] = crisp_risk(h, s, show=False)

    HH, SS = np.meshgrid(H, S)
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(HH, SS, Z, cmap='viridis', linewidth=0, antialiased=True, alpha=0.95)
    ax.set_xlabel("Heart Rate (bpm)")
    ax.set_ylabel("Symptoms (0–10)")
    ax.set_zlabel("Defuzzified Risk")
    ax.set_zlim(0, 10)
    ax.set_title("Mamdani Risk Surface (centroid defuzz)")
    cbar = fig.colorbar(surf, shrink=0.6, pad=0.1)
    cbar.set_label("Risk")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=160, bbox_inches="tight", pad_inches=0.05)
    print(f"Surface saved to {save_path}")
    plt.show()


# ============================================================
# Sugeno (zero-order) — crisp inference + surface + weight plot
# ============================================================

# Representative constants for the three consequents (tune as you like)
SUGENO_Z = {"LOW": 2.0, "MED": 5.0, "HIGH": 8.5}


def sugeno_infer(hr_val, sym_val, and_op="prod", z_vals=SUGENO_Z, verbose=False):
    """
    Zero-order Sugeno inference using the SAME antecedents and rule structure
    as the Mamdani engine. Returns (crisp, weights_info).

    - and_op: 'prod' (product t-norm, default) or 'min'
    - z_vals: dict with keys 'LOW', 'MED', 'HIGH' giving rule constants

    weights_info: list of dicts: {"rule","weight","z","contrib"}
    """
    # Antecedent degrees
    μ_hr_low  = grade(hr, hr_low,   hr_val)
    μ_hr_norm = grade(hr, hr_normal, hr_val)
    μ_hr_high = grade(hr, hr_high,  hr_val)

    μ_sym_low  = grade(sym, sym_low,  sym_val)
    μ_sym_med  = grade(sym, sym_med,  sym_val)
    μ_sym_high = grade(sym, sym_high, sym_val)

    # t-norm for AND
    def AND(a, b):
        return (a * b) if and_op == "prod" else min(a, b)

    # ---- Rule weights (mirror Mamdani rules) ----
    labels = [
        "H1: HR Low ∧ Sym Med → High",
        "H2: HR Low ∧ Sym High → High",
        "H3: HR High ∧ Sym Med → High",
        "H4: HR High ∧ Sym High → High",
        "H5: HR Normal ∧ Sym High → High",
        "M1: HR Low ∧ Sym Low → Med",
        "M2: HR High ∧ Sym Low → Med",
        "M3: HR Normal ∧ Sym Med → Med",
        "M4: 0.5·max(HR Low, HR High) → Med",
        "L1: HR Normal ∧ Sym Low → Low",
    ]

    w_hi_1 = AND(μ_hr_low,  μ_sym_med)
    w_hi_2 = AND(μ_hr_low,  μ_sym_high)
    w_hi_3 = AND(μ_hr_high, μ_sym_med)
    w_hi_4 = AND(μ_hr_high, μ_sym_high)
    w_hi_5 = AND(μ_hr_norm, μ_sym_high)

    w_med_1 = AND(μ_hr_low,  μ_sym_low)
    w_med_2 = AND(μ_hr_high, μ_sym_low)
    w_med_3 = AND(μ_hr_norm, μ_sym_med)
    w_med_4 = 0.5 * max(μ_hr_low, μ_hr_high)   # baseline partial rule

    w_low_1 = AND(μ_hr_norm, μ_sym_low)

    w = np.array([
        w_hi_1, w_hi_2, w_hi_3, w_hi_4, w_hi_5,
        w_med_1, w_med_2, w_med_3, w_med_4,
        w_low_1
    ], dtype=float)

    # Consequent constants per rule (zero-order)
    z = np.array([
        z_vals["HIGH"], z_vals["HIGH"], z_vals["HIGH"], z_vals["HIGH"], z_vals["HIGH"],
        z_vals["MED"],  z_vals["MED"],  z_vals["MED"],  z_vals["MED"],
        z_vals["LOW"]
    ], dtype=float)

    contrib = w * z
    denom = np.sum(w)
    crisp = float(np.sum(contrib) / denom) if denom > 1e-12 else 0.0
    crisp = float(np.clip(crisp, 0.0, 10.0))

    weights_info = [
        {"rule": labels[i], "weight": float(w[i]), "z": float(z[i]), "contrib": float(contrib[i])}
        for i in range(len(labels))
    ]

    if verbose:
        print(f"[Sugeno {and_op}] HR={hr_val:.1f}, Symptoms={sym_val:.1f} -> Risk={crisp:.2f}")
        for d in sorted(weights_info, key=lambda d: d["weight"], reverse=True):
            print(f"  {d['rule']}: w={d['weight']:.3f}, z={d['z']:.1f}, w*z={d['contrib']:.3f}")

    return crisp, weights_info


def sugeno_risk_surface(hr_pts=151, sym_pts=101,
                        hr_min=30, hr_max=200, sym_min=0, sym_max=10,
                        and_op="prod",
                        z_vals=SUGENO_Z,
                        save_path="docs/sugeno_risk_surface.png"):
    """
    Evaluate crisp risk over an HR×Symptoms grid using sugeno_infer()
    and plot a 3D surface.
    """
    H = np.linspace(hr_min, hr_max, hr_pts)
    S = np.linspace(sym_min, sym_max, sym_pts)
    Z = np.empty((sym_pts, hr_pts), dtype=float)

    for i, s in enumerate(S):
        for j, h in enumerate(H):
            Z[i, j], _ = sugeno_infer(h, s, and_op=and_op, z_vals=z_vals, verbose=False)

    HH, SS = np.meshgrid(H, S)
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(HH, SS, Z, cmap='viridis', linewidth=0, antialiased=True, alpha=0.95)
    ax.set_xlabel("Heart Rate (bpm)")
    ax.set_ylabel("Symptoms (0–10)")
    ax.set_zlabel("Sugeno Risk")
    ax.set_zlim(0, 10)
    ax.set_title(f"Sugeno Risk Surface (zero-order, AND='{and_op}')")
    cbar = fig.colorbar(surf, shrink=0.6, pad=0.1)
    cbar.set_label("Risk")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=160, bbox_inches="tight", pad_inches=0.05)
    print(f"Surface saved to {save_path}")
    plt.show()


def sugeno_weights_plot(hr_val, sym_val, and_op="prod", z_vals=SUGENO_Z,
                        save_path=None):
    """
    Plot a horizontal bar chart of rule firing strengths (weights)
    for a single input (hr_val, sym_val). Also annotates each bar with z and w*z.
    Returns the crisp Sugeno output.
    """
    crisp, info = sugeno_infer(hr_val, sym_val, and_op=and_op, z_vals=z_vals, verbose=False)

    labels   = [d["rule"] for d in info]
    weights  = np.array([d["weight"] for d in info], dtype=float)
    contribs = np.array([d["contrib"] for d in info], dtype=float)
    z_used   = np.array([d["z"] for d in info], dtype=float)

    order = np.argsort(weights)[::-1]  # show strongest rules on top

    fig = plt.figure(figsize=(11, 7), constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.barh(np.arange(len(order)), weights[order])
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([labels[i] for i in order])
    ax.invert_yaxis()
    ax.set_xlabel("Rule firing strength (weight)")
    ax.set_title(
        f"Sugeno Rule Weights (AND='{and_op}')\n"
        f"HR={hr_val}, Symptoms={sym_val} — Crisp Risk = {crisp:.2f}"
    )

    # Annotate each bar with z and w*z
    for k, i in enumerate(order):
        ax.text(weights[i] + 0.01, k,
                f"z={z_used[i]:.1f}, w·z={contribs[i]:.2f}",
                va="center")

    if save_path is None:
        os.makedirs("docs", exist_ok=True)
        save_path = f"docs/sugeno_rule_weights_hr{int(round(hr_val))}_sym{int(round(sym_val))}.png"

    plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    print(f"Weights diagram saved to {save_path}")
    plt.show()

    return crisp


# ============================================================
# Sugeno (first-order / TSK-1) — crisp inference + surface + weights
# ============================================================

# Coefficients for linear consequents per rule:
# Each rule outputs: z = a0 + a1*hrn + a2*symn, where hrn∈[0,1], symn∈[0,1]
# (These are illustrative and tuned to keep z mostly within 0..10; final result is clipped.)
DEFAULT_TSK1_COEFFS = {
    "H1": (7.0, 0.0, 1.8),  # HR Low & Sym Med  -> High-ish; driven by symptoms
    "H2": (7.2, 0.0, 2.0),  # HR Low & Sym High -> Higher; strongly by symptoms
    "H3": (7.0, 0.8, 1.8),  # HR High & Sym Med -> High; HR contributes too
    "H4": (7.5, 0.8, 1.7),  # HR High & Sym High-> Very high; both contribute
    "H5": (7.2, 0.5, 1.8),  # HR Normal & Sym High -> High; mostly symptoms
    "M1": (4.0, 0.0, 0.8),  # HR Low & Sym Low  -> Medium-ish low
    "M2": (4.4, 0.8, 0.6),  # HR High & Sym Low -> Medium-ish; HR contributes
    "M3": (4.8, 0.3, 1.2),  # HR Normal & Sym Med -> Mid; symptoms dominate
    "M4": (4.0, 1.0, 0.0),  # Baseline extremes (depends on HR only)
    "L1": (2.5, 0.2, -0.8), # HR Normal & Sym Low -> Low; higher symptoms lower this
}


def sugeno1_infer(hr_val, sym_val, and_op="prod", coeffs=None, verbose=False):
    """
    First-order (linear) Sugeno inference using SAME antecedents & rule structure.
    Returns (crisp, weights_info). Linear consequents use normalized inputs:
      hrn  = clamp((HR - HR_LO) / (HR_HI - HR_LO), 0..1)
      symn = clamp(Symptoms / 10, 0..1)
    """
    # Memberships
    μ_hr_low   = grade(hr, hr_low,   hr_val)
    μ_hr_norm  = grade(hr, hr_normal, hr_val)
    μ_hr_high  = grade(hr, hr_high,  hr_val)
    μ_sym_low  = grade(sym, sym_low,  sym_val)
    μ_sym_med  = grade(sym, sym_med,  sym_val)
    μ_sym_high = grade(sym, sym_high, sym_val)

    # AND t-norm
    def AND(a, b): return (a * b) if and_op == "prod" else min(a, b)

    # Rule weights (mirror Mamdani rule base)
    labels = [
        "H1: HR Low ∧ Sym Med → High",
        "H2: HR Low ∧ Sym High → High",
        "H3: HR High ∧ Sym Med → High",
        "H4: HR High ∧ Sym High → High",
        "H5: HR Normal ∧ Sym High → High",
        "M1: HR Low ∧ Sym Low → Med",
        "M2: HR High ∧ Sym Low → Med",
        "M3: HR Normal ∧ Sym Med → Med",
        "M4: 0.5·max(HR Low, HR High) → Med",
        "L1: HR Normal ∧ Sym Low → Low",
    ]
    w_hi_1 = AND(μ_hr_low,  μ_sym_med)
    w_hi_2 = AND(μ_hr_low,  μ_sym_high)
    w_hi_3 = AND(μ_hr_high, μ_sym_med)
    w_hi_4 = AND(μ_hr_high, μ_sym_high)
    w_hi_5 = AND(μ_hr_norm, μ_sym_high)
    w_med_1 = AND(μ_hr_low,  μ_sym_low)
    w_med_2 = AND(μ_hr_high, μ_sym_low)
    w_med_3 = AND(μ_hr_norm, μ_sym_med)
    w_med_4 = 0.5 * max(μ_hr_low, μ_hr_high)  # baseline partial rule
    w_low_1 = AND(μ_hr_norm, μ_sym_low)
    w = np.array([w_hi_1, w_hi_2, w_hi_3, w_hi_4, w_hi_5,
                  w_med_1, w_med_2, w_med_3, w_med_4,
                  w_low_1], dtype=float)

    # Normalized inputs for linear consequents
    hrn  = float(np.clip((hr_val - HR_LO) / max(1e-9, (HR_HI - HR_LO)), 0.0, 1.0))
    symn = float(np.clip(sym_val / 10.0, 0.0, 1.0))

    # Coefficients (a0, a1, a2) per rule
    C = (DEFAULT_TSK1_COEFFS if coeffs is None else coeffs)
    keys = ["H1","H2","H3","H4","H5","M1","M2","M3","M4","L1"]
    a = np.array([C[k] for k in keys], dtype=float)  # shape (10,3)

    # Linear outputs per rule: z_i = a0 + a1*hrn + a2*symn
    z = a[:, 0] + a[:, 1] * hrn + a[:, 2] * symn

    contrib = w * z
    denom = float(np.sum(w))
    crisp = float(np.sum(contrib) / denom) if denom > 1e-12 else 0.0
    crisp = float(np.clip(crisp, 0.0, 10.0))

    info = []
    for i, k in enumerate(keys):
        info.append({
            "rule": labels[i],
            "weight": float(w[i]),
            "a": {"a0": float(a[i,0]), "a1_hrn": float(a[i,1]), "a2_symn": float(a[i,2])},
            "z": float(z[i]),
            "contrib": float(contrib[i]),
        })

    if verbose:
        print(f"[Sugeno-1 {and_op}] HR={hr_val:.1f}, Symptoms={sym_val:.1f} "
              f"(hrn={hrn:.2f}, symn={symn:.2f}) -> Risk={crisp:.2f}")
        top = sorted(info, key=lambda d: d["weight"], reverse=True)[:5]
        for d in top:
            a0, a1, a2 = d["a"]["a0"], d["a"]["a1_hrn"], d["a"]["a2_symn"]
            print(f"  {d['rule']}: w={d['weight']:.3f}, z={d['z']:.2f} "
                  f"(a0+a1*hrn+a2*symn = {a0:.1f}+{a1:.1f}*{hrn:.2f}+{a2:.1f}*{symn:.2f}), w*z={d['contrib']:.2f}")

    return crisp, info


def sugeno1_risk_surface(hr_pts=151, sym_pts=101,
                         hr_min=30, hr_max=200, sym_min=0, sym_max=10,
                         and_op="prod", coeffs=None,
                         save_path="docs/sugeno1_risk_surface.png"):
    """3D surface for first-order Sugeno."""
    H = np.linspace(hr_min, hr_max, hr_pts)
    S = np.linspace(sym_min, sym_max, sym_pts)
    Z = np.empty((sym_pts, hr_pts), dtype=float)

    for i, s in enumerate(S):
        for j, h in enumerate(H):
            Z[i, j], _ = sugeno1_infer(h, s, and_op=and_op, coeffs=coeffs, verbose=False)

    HH, SS = np.meshgrid(H, S)
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(HH, SS, Z, cmap='viridis', linewidth=0, antialiased=True, alpha=0.95)
    ax.set_xlabel("Heart Rate (bpm)")
    ax.set_ylabel("Symptoms (0–10)")
    ax.set_zlabel("Sugeno-1 Risk")
    ax.set_zlim(0, 10)
    ax.set_title(f"Sugeno Risk Surface (first-order, AND='{and_op}')")
    cbar = fig.colorbar(surf, shrink=0.6, pad=0.1)
    cbar.set_label("Risk")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=160, bbox_inches="tight", pad_inches=0.05)
    print(f"Surface saved to {save_path}")
    plt.show()


def sugeno1_weights_plot(hr_val, sym_val, and_op="prod", coeffs=None, save_path=None):
    """
    Horizontal bar chart of rule weights (first-order). Annotates each with z and w·z,
    and shows the linear form used (a0 + a1*hrn + a2*symn).
    """
    crisp, info = sugeno1_infer(hr_val, sym_val, and_op=and_op, coeffs=coeffs, verbose=False)

    labels   = [d["rule"] for d in info]
    weights  = np.array([d["weight"] for d in info], dtype=float)
    contribs = np.array([d["contrib"] for d in info], dtype=float)
    zs       = np.array([d["z"] for d in info], dtype=float)
    forms    = [f"a0={d['a']['a0']:.1f}, a1={d['a']['a1_hrn']:.1f}, a2={d['a']['a2_symn']:.1f}" for d in info]

    order = np.argsort(weights)[::-1]
    fig = plt.figure(figsize=(12.5, 7.5), constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.barh(np.arange(len(order)), weights[order])
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([labels[i] for i in order])
    ax.invert_yaxis()
    ax.set_xlabel("Rule firing strength (weight)")
    ax.set_title(
        f"Sugeno-1 Rule Weights (AND='{and_op}')\n"
        f"HR={hr_val}, Symptoms={sym_val} — Crisp Risk = {crisp:.2f}"
    )

    for k, i in enumerate(order):
        ax.text(weights[i] + 0.01, k,
                f"z={zs[i]:.2f}, w·z={contribs[i]:.2f}  ({forms[i]})",
                va="center")

    if save_path is None:
        os.makedirs("docs", exist_ok=True)
        save_path = f"docs/sugeno1_rule_weights_hr{int(round(hr_val))}_sym{int(round(sym_val))}.png"

    plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    print(f"Weights diagram saved to {save_path}")
    plt.show()

    return crisp


# =========================
# Main
# =========================
def main():

    # Membership plots
    plot_memberships()

    # Quick sanity checks (Mamdani)
    cases = [(40, 2), (40, 7), (75, 1), (110, 2), (110, 7)]
    for h, s in cases:
        print(f"[Mamdani] HR={h:>3} bpm, Symptoms={s} -> Risk={crisp_risk(h, s):.2f}")

    # Single Mamdani visualization
    _ = crisp_risk(40, 7, show=True)

    # Mamdani surface
    mamdani_risk_surface()

    # --- Sugeno (zero-order) ---
    for h, s in cases:
        su, _ = sugeno_infer(h, s, and_op="prod")
        print(f"[Sugeno-0] HR={h:>3} bpm, Symptoms={s} -> Risk={su:.2f}")

    # Sugeno (zero-order) surface + weights
    sugeno_risk_surface(and_op="prod")
    _ = sugeno_weights_plot(40, 7, and_op="prod")

    # --- Sugeno (first-order / linear) ---
    for h, s in cases:
        su1, _ = sugeno1_infer(h, s, and_op="prod")
        print(f"[Sugeno-1] HR={h:>3} bpm, Symptoms={s} -> Risk={su1:.2f}")

    sugeno1_risk_surface(and_op="prod")
    _ = sugeno1_weights_plot(40, 7, and_op="prod")


if __name__ == "__main__":
    main()
