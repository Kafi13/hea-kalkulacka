import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- KONFIGURACE APLIKACE ---
st.set_page_config(page_title="HEA-Thermo-H2 Expert", layout="wide", page_icon="⚛️")

# --- 1. DATABÁZE PRVKŮ (Data z 'Complete Data Reference' a 'Vývoj miniprogramu') ---
# r: Metallic Radius (Goldschmidt CN12) [Angstrom]
# VEC: Valence Electron Concentration
# Tm: Melting Temperature [K]
# H_inf: Enthalpy of solution at infinite dilution for Hydrogen [kJ/mol H] (Source: Griessen-Driessen table)
# H_f: Enthalpy of hydride formation [kJ/mol H] (Source: Griessen-Driessen table / Materials paper)
# AtomicWeight: [g/mol]

ELEMENTS_DB = {
    "Al": {"r": 1.43, "VEC": 3,  "Tm": 933,  "H_inf": 60,  "H_f": -6.0,  "M": 26.98},
    "Mg": {"r": 1.60, "VEC": 2,  "Tm": 923,  "H_inf": 21,  "H_f": -75.0, "M": 24.31},
    "Ti": {"r": 1.47, "VEC": 4,  "Tm": 1941, "H_inf": -52, "H_f": -130.0,"M": 47.87},
    "V":  {"r": 1.35, "VEC": 5,  "Tm": 2183, "H_inf": -30, "H_f": -40.0, "M": 50.94},
    "Cr": {"r": 1.29, "VEC": 6,  "Tm": 2180, "H_inf": 28,  "H_f": 6.0,   "M": 51.99},
    "Mn": {"r": 1.37, "VEC": 7,  "Tm": 1519, "H_inf": 1,   "H_f": 21.0,  "M": 54.94},
    "Fe": {"r": 1.26, "VEC": 8,  "Tm": 1811, "H_inf": 25,  "H_f": 20.0,  "M": 55.85},
    "Co": {"r": 1.25, "VEC": 9,  "Tm": 1768, "H_inf": 21,  "H_f": 31.0,  "M": 58.93},
    "Ni": {"r": 1.25, "VEC": 10, "Tm": 1728, "H_inf": 12,  "H_f": 15.0,  "M": 58.69},
    "Cu": {"r": 1.28, "VEC": 11, "Tm": 1358, "H_inf": 46,  "H_f": 45.0,  "M": 63.55},
    "Zn": {"r": 1.37, "VEC": 12, "Tm": 693,  "H_inf": 15,  "H_f": 0.0,   "M": 65.38},
    "Zr": {"r": 1.60, "VEC": 4,  "Tm": 2128, "H_inf": -58, "H_f": -164.0,"M": 91.22},
    "Nb": {"r": 1.47, "VEC": 5,  "Tm": 2750, "H_inf": -35, "H_f": -39.0, "M": 92.91},
    "Mo": {"r": 1.40, "VEC": 6,  "Tm": 2896, "H_inf": 25,  "H_f": 92.0,  "M": 95.95},
    "Pd": {"r": 1.37, "VEC": 10, "Tm": 1828, "H_inf": -10, "H_f": -20.0, "M": 106.42},
    "Ag": {"r": 1.44, "VEC": 11, "Tm": 1235, "H_inf": 63,  "H_f": 0.0,   "M": 107.87},
    "Hf": {"r": 1.59, "VEC": 4,  "Tm": 2506, "H_inf": -38, "H_f": -130.0,"M": 178.49},
    "Ta": {"r": 1.47, "VEC": 5,  "Tm": 3290, "H_inf": -36, "H_f": -78.0, "M": 180.95},
    "W":  {"r": 1.41, "VEC": 6,  "Tm": 3695, "H_inf": 96,  "H_f": 74.0,  "M": 183.84},
    "La": {"r": 1.88, "VEC": 3,  "Tm": 1193, "H_inf": -67, "H_f": -206.0,"M": 138.91},
    "Ce": {"r": 1.82, "VEC": 4,  "Tm": 1068, "H_inf": -74, "H_f": -200.0,"M": 140.12},
}

# --- 2. MIEDEMA BINARY MIXING ENTHALPIES (Source: Table on Page 5 & 6) ---
# Values in kJ/mol. Pair order doesn't matter (function handles symmetry).
MIEDEMA_PAIRS = {
    frozenset(["Ti", "V"]): -2, frozenset(["Ti", "Cr"]): -7, frozenset(["Ti", "Mn"]): -8, frozenset(["Ti", "Fe"]): -17,
    frozenset(["Ti", "Co"]): -28, frozenset(["Ti", "Ni"]): -35, frozenset(["Ti", "Cu"]): -9, frozenset(["Ti", "Al"]): -30,
    frozenset(["Ti", "Zr"]): 0, frozenset(["Ti", "Nb"]): 2, frozenset(["Ti", "Mo"]): -4,
    frozenset(["V", "Cr"]): -2, frozenset(["V", "Mn"]): -1, frozenset(["V", "Fe"]): -7,
    frozenset(["V", "Co"]): -14, frozenset(["V", "Ni"]): -18, frozenset(["V", "Cu"]): 5, frozenset(["V", "Al"]): -16,
    frozenset(["V", "Zr"]): -4, frozenset(["V", "Nb"]): -1, frozenset(["V", "Mo"]): 0,
    frozenset(["Cr", "Mn"]): 2, frozenset(["Cr", "Fe"]): -1, frozenset(["Cr", "Co"]): -4,
    frozenset(["Cr", "Ni"]): -7, frozenset(["Cr", "Cu"]): 12, frozenset(["Cr", "Al"]): -10, frozenset(["Cr", "Nb"]): -7,
    frozenset(["Mn", "Fe"]): 0, frozenset(["Mn", "Co"]): -5, frozenset(["Mn", "Ni"]): -8, frozenset(["Mn", "Cu"]): 4,
    frozenset(["Fe", "Co"]): -1, frozenset(["Fe", "Ni"]): -2, frozenset(["Fe", "Cu"]): 13, frozenset(["Fe", "Al"]): -11,
    frozenset(["Co", "Ni"]): 0, frozenset(["Co", "Cu"]): 6, frozenset(["Co", "Al"]): -19,
    frozenset(["Ni", "Cu"]): 4, frozenset(["Ni", "Al"]): -22, frozenset(["Ni", "Zr"]): -49, frozenset(["Ni", "Nb"]): -30,
    frozenset(["Zr", "Nb"]): 4, frozenset(["Zr", "Ni"]): -49, frozenset(["Nb", "Mo"]): -6, frozenset(["Mo", "W"]): 0,
    # Add assumptions for missing pairs or less common ones based on Miedema model rules or approximate 0 if solid solution predicted
}

def get_miedema_H(e1, e2):
    if e1 == e2: return 0
    key = frozenset([e1, e2])
    return MIEDEMA_PAIRS.get(key, 0) # Returns 0 if pair not found (Ideal mixing assumption fallback)

# --- 3. VÝPOČTOVÉ FUNKCE (THE "ENGINE") ---

def calculate_parameters(composition):
    """
    Vypočítá termodynamické parametry HEA.
    Vstup: Dictionary {Prvek: at.%} (např. {'Ti': 30, 'V': 30, 'Cr': 40})
    """
    # Normalizace na zlomky (0.0 - 1.0)
    total = sum(composition.values())
    c = {k: v/total for k, v in composition.items()}
    elements = list(c.keys())
    
    # 1. Entropy of Mixing (Delta S_mix)
    # Formula: -R * sum(c_i * ln(c_i))
    R = 8.314 # J/(mol*K)
    dS_mix = -R * sum([c[el] * np.log(c[el]) for el in elements])
    
    # 2. Enthalpy of Mixing (Delta H_mix)
    # Formula: sum(4 * H_ij * c_i * c_j) (Factor 4 converts binary to regular solution param)
    dH_mix = 0
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            el_i, el_j = elements[i], elements[j]
            H_ij = get_miedema_H(el_i, el_j)
            dH_mix += 4 * H_ij * c[el_i] * c[el_j] # kJ/mol
            
    # 3. Average Melting Point (Tm)
    Tm_avg = sum([c[el] * ELEMENTS_DB[el]["Tm"] for el in elements])
    
    # 4. Omega Parameter
    # Formula: (Tm * dS_mix) / |dH_mix * 1000| (dS is in J, dH is in kJ)
    if abs(dH_mix) < 1e-5:
        Omega = 999 # Infinite stability if H_mix is 0
    else:
        Omega = (Tm_avg * dS_mix) / (abs(dH_mix) * 1000)
        
    # 5. Atomic Size Difference (delta)
    # Formula: 100 * sqrt(sum(c_i * (1 - r_i/r_bar)^2))
    r_bar = sum([c[el] * ELEMENTS_DB[el]["r"] for el in elements])
    delta = 100 * np.sqrt(sum([c[el] * (1 - ELEMENTS_DB[el]["r"]/r_bar)**2 for el in elements]))
    
    # 6. Valence Electron Concentration (VEC)
    vec = sum([c[el] * ELEMENTS_DB[el]["VEC"] for el in elements])
    
    # 7. Hydrogen Affinity Parameters (Griessen-Driessen / Materials 2024 method)
    # Weighted average of enthalpies
    H_inf_alloy = sum([c[el] * ELEMENTS_DB[el]["H_inf"] for el in elements])
    H_f_alloy = sum([c[el] * ELEMENTS_DB[el]["H_f"] for el in elements])
    
    return {
        "dS_mix": dS_mix,
        "dH_mix": dH_mix,
        "Tm": Tm_avg,
        "Omega": Omega,
        "delta": delta,
        "VEC": vec,
        "H_inf": H_inf_alloy,
        "H_f": H_f_alloy
    }

def get_phase_prediction(params):
    res = []
    # Phase stability rules (Yang-Zhang)
    if params['Omega'] >= 1.1 and params['delta'] <= 6.6:
        res.append("✅ Stabilní tuhý roztok (Solid Solution)")
    elif params['delta'] > 6.6:
        res.append("⚠️ Riziko vzniku intermetalik / Lavesových fází (δ > 6.6%)")
    else:
        res.append("⚠️ Nízká stabilita (Ω < 1.1)")
        
    # Structure prediction (VEC rules from Guo)
    if params['VEC'] < 6.87:
        res.append("🟦 Struktura: BCC (Vhodné pro H2)")
    elif params['VEC'] >= 8.0:
        res.append("🟧 Struktura: FCC")
    else:
        res.append("🟪 Struktura: BCC + FCC")
        
    return res

def get_hydrogen_prediction(params):
    # Interpretation based on 'Materials 2024' & 'Vývoj miniprogramu'
    h_inf = params['H_inf']
    
    if h_inf < -40:
        return "🧽 Silný hydrid (Vysoká stabilita, nutná vyšší T pro desorpci)"
    elif -40 <= h_inf <= -10:
        return "⭐ Optimální pro skladování H2 (Reverzibilní za pokojové teploty)"
    elif -10 < h_inf < 10:
        return "⚖️ Nízká afinita / Intersticiální roztok"
    else:
        return "🛡️ Odolné vůči vodíku (Hydrogen Barrier / HEA4 type)"

# --- 4. UI APLIKACE ---

st.title("🧪 HEA-Thermo-H2 Calculator")
st.markdown("""
Tento nástroj slouží k predikci termodynamických parametrů a vodíkové afinity vysokoentropických slitin.
Založeno na modelech **Miedema**, **Griessen-Driessen** a datech z tvého výzkumu.
""")

# --- SIDEBAR: VSTUPY ---
st.sidebar.header("1. Složení slitiny")
available_elements = list(ELEMENTS_DB.keys())
selected_elements = st.sidebar.multiselect("Vyber prvky (3-6)", available_elements, default=["Ti", "V", "Cr", "Nb"])

composition = {}
if selected_elements:
    st.sidebar.subheader("2. Poměr prvků (at.%)")
    
    # Rovnoměrné rozdělení jako default
    default_val = 100.0 / len(selected_elements)
    
    current_total = 0
    for el in selected_elements:
        val = st.sidebar.number_input(f"{el} (at.%)", min_value=0.0, max_value=100.0, value=default_val, step=1.0)
        composition[el] = val
        current_total += val

    # Warning if not 100%
    if abs(current_total - 100.0) > 0.1:
        st.sidebar.warning(f"Součet je {current_total:.1f}%. Výpočet provede normalizaci automaticky.")

    # --- ACTION ---
    if st.button("Vypočítat parametry", type="primary"):
        results = calculate_parameters(composition)
        phase_pred = get_phase_prediction(results)
        h2_pred = get_hydrogen_prediction(results)
        
        # --- VÝSLEDKY ---
        st.divider()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Termodynamika")
            st.metric("Entropie míšení (ΔS)", f"{results['dS_mix']:.2f} J/mol·K", delta_color="normal")
            st.caption("Criterion: > 12.5 J/mol·K for HEA")
            st.metric("Entalpie míšení (ΔH)", f"{results['dH_mix']:.2f} kJ/mol")
            st.metric("Omega (Ω)", f"{results['Omega']:.2f}", delta="> 1.1 (Stable)" if results['Omega'] >= 1.1 else "< 1.1 (Unstable)")
            
        with col2:
            st.subheader("Struktura")
            st.metric("Atomová neshoda (δ)", f"{results['delta']:.2f} %", delta="< 6.6%" if results['delta'] <= 6.6 else "> 6.6%", delta_color="inverse")
            st.metric("VEC", f"{results['VEC']:.2f}")
            for p in phase_pred:
                st.info(p)
                
        with col3:
            st.subheader("Vodíková afinita")
            st.metric("ΔH (infinite dilution)", f"{results['H_inf']:.2f} kJ/mol H")
            st.metric("ΔH (hydride formation)", f"{results['H_f']:.2f} kJ/mol H")
            st.success(h2_pred)

        # --- GRAFICKÁ VIZUALIZACE (ASHBY PLOT) ---
        st.divider()
        st.subheader("Fázový diagram stability (Yang-Zhang)")
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Zones
        # Solid Solution zone: Delta < 6.6, Omega > 1.1
        rect_ss = patches.Rectangle((1.1, 0), 10, 6.6, linewidth=1, edgecolor='g', facecolor='green', alpha=0.1, label='Solid Solution')
        ax.add_patch(rect_ss)
        
        # Plot current point
        ax.scatter(results['Omega'], results['delta'], c='red', s=200, marker='*', edgecolors='black', label='Tvoje slitina', zorder=10)
        
        # Plot reference Alloys from 'Materials 2024' (Hardcoded for context)
        refs = {
            "HEA1 (CoNiMnCrFe)": (5.90, 3.01),
            "HEA4 (CoNiAlCrFe)": (1.82, 5.78),
            "TiVCr": (5.2, 4.5) # Hypothetical approx
        }
        for name, coords in refs.items():
            ax.scatter(coords[0], coords[1], c='gray', s=50, alpha=0.6)
            ax.text(coords[0], coords[1], f"  {name}", fontsize=8, color='gray')

        ax.set_xlabel('Omega Parameter (Ω)')
        ax.set_ylabel('Atomic Size Difference δ (%)')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axhline(6.6, color='black', linestyle='--')
        ax.axvline(1.1, color='black', linestyle='--')
        ax.text(8, 2, "Stable SS Phase", color='green', fontweight='bold')
        ax.text(2, 8, "Multiphase / Amorphous", color='orange')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # --- TABULKA DAT ---
        st.divider()
        st.subheader("Detailní data pro export")
        df_res = pd.DataFrame([results])
        st.dataframe(df_res)
        
else:
    st.info("👈 Začni výběrem prvků v levém menu.")

# --- FOOTER ---
st.markdown("---")
st.caption("Data sources: Miedema Model, Griessen-Driessen Model, Materials 2024 Article. Developed for HEA Hydrogen Research.")
