
import os
import numpy as np
import skfuzzy as fuzz
import matplotlib

# Use a non-interactive backend if no display is available (helps on servers/CI)
if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


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
# Mamdani risk surface (centroid defuzz) — append-only
# ============================================================
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


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


def main():
    # Generate the membership function plots
    plot_memberships()

    # Quick sanity checks
    cases = [(40, 2), (40, 7), (75, 1), (110, 2), (110, 7)]
    for h, s in cases:
        print(f"HR={h:>3} bpm, Symptoms={s} -> Risk={crisp_risk(h, s):.2f}")

    # Example visualization
    _ = crisp_risk(40, 7, show=True)

    # Generate the surface with reasonable resolution
    mamdani_risk_surface()


if __name__ == "__main__":
    main()
