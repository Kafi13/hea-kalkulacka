import streamlit as st
import math
import pandas as pd
import altair as alt
import numpy as np
import re
from dataclasses import dataclass, field
from enum import Enum
from docx import Document
from docx.shared import Inches
import io

# =============================================================================
# 1. KONFIGURACE APLIKACE
# =============================================================================
st.set_page_config(
    page_title="HEA Kalkulačka Expert Pro v3.0",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 10px;
        border-radius: 5px;
    }
    .stAlert { padding: 10px; }
    h1, h2, h3 { color: #2c3e50; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. DATABÁZE A STRUKTURY (v3.0 - Expanded)
# =============================================================================
@dataclass
class ElementData:
    symbol: str
    mass: float   # Atomic Weight (g/mol) [NEW]
    r: float      # Metallic Radius CN12 (Angstrom = pm/100)
    VEC: int      # Valence Electrons
    Tm: int       # Melting Point (K)
    phi: float    # Miedema Work Function (V) [NEW]
    nws: float    # Miedema Electron Density (n_ws^1/3) [NEW]
    V23: float    # Miedema Molar Volume (V^2/3) [NEW]
    H_inf: float  # Enthalpy of infinite dilution for H (kJ/mol H) [NEW]
    H_f: float    # Enthalpy of hydride formation (kJ/mol H) [NEW]
    price: float  # Estimated Price (CZK/kg)

# Data populated from User's Research Documents (Complete Data Reference...)
ELEMENTS_DB = {
    # Light Metals
    'Al': ElementData('Al', 26.98, 1.43, 3, 933, 4.20, 1.39, 4.64, 60.0, -6.0, 60),
    'Mg': ElementData('Mg', 24.31, 1.60, 2, 923, 3.45, 1.17, 5.81, 21.0, -37.0, 120),
    'Si': ElementData('Si', 28.09, 1.17, 4, 1687, 4.70, 1.50, 4.20, 180.0, 16.0, 80),
    
    # 4th Period TM
    'Ti': ElementData('Ti', 47.87, 1.47, 4, 1941, 3.80, 1.52, 4.12, -52.0, -67.0, 3500),
    'V':  ElementData('V', 50.94, 1.35, 5, 2183, 4.25, 1.64, 4.12, -30.0, -30.0, 7500),
    'Cr': ElementData('Cr', 52.00, 1.29, 6, 2180, 4.65, 1.73, 3.74, 28.0, 6.0, 2500),
    'Mn': ElementData('Mn', 54.94, 1.37, 7, 1519, 4.45, 1.61, 3.78, 1.0, 21.0, 600),
    'Fe': ElementData('Fe', 55.85, 1.26, 8, 1811, 4.93, 1.77, 3.69, 25.0, 28.5, 25),
    'Co': ElementData('Co', 58.93, 1.25, 9, 1768, 5.10, 1.75, 3.55, 21.0, 31.2, 9500),
    'Ni': ElementData('Ni', 58.69, 1.25, 10, 1728, 5.20, 1.75, 3.52, 12.0, 15.1, 5500),
    'Cu': ElementData('Cu', 63.55, 1.28, 11, 1358, 4.45, 1.47, 3.70, 46.0, 45.0, 250),
    'Zn': ElementData('Zn', 65.38, 1.37, 12, 693, 4.10, 1.32, 4.60, 15.0, 5.0, 80), # Estimated Miedema for Zn

    # 5th Period TM
    'Y':  ElementData('Y', 88.91, 1.82, 3, 1799, 3.20, 1.21, 7.34, -79.0, -114.0, 8500),
    'Zr': ElementData('Zr', 91.22, 1.60, 4, 2128, 3.45, 1.41, 5.81, -58.0, -79.0, 8000),
    'Nb': ElementData('Nb', 92.91, 1.47, 5, 2750, 4.05, 1.62, 4.89, -35.0, -40.0, 18000),
    'Mo': ElementData('Mo', 95.95, 1.40, 6, 2896, 4.65, 1.77, 4.45, 25.0, 54.0, 12000),
    'Pd': ElementData('Pd', 106.42, 1.37, 10, 1828, 5.45, 1.67, 3.90, -10.0, -20.0, 1200000),
    'Ag': ElementData('Ag', 107.87, 1.44, 11, 1235, 4.35, 1.39, 4.80, 63.0, 30.0, 22000),

    # 6th Period TM
    'Hf': ElementData('Hf', 178.49, 1.59, 4, 2506, 3.55, 1.43, 5.65, -38.0, -64.0, 120000),
    'Ta': ElementData('Ta', 180.95, 1.47, 5, 3290, 4.05, 1.63, 4.89, -36.0, -34.0, 85000),
    'W':  ElementData('W', 183.84, 1.41, 6, 3695, 4.80, 1.81, 4.50, 96.0, 74.0, 11000),

    # Rare Earths
    'La': ElementData('La', 138.91, 1.88, 3, 1193, 3.05, 1.18, 6.60, -67.0, -100.0, 1500),
    'Ce': ElementData('Ce', 140.12, 1.82, 3, 1068, 3.15, 1.19, 6.30, -74.0, -98.0, 1200),
    'Sc': ElementData('Sc', 44.96, 1.64, 3, 1814, 3.25, 1.27, 4.90, -90.0, -100.0, 350000),
}

# Reference Alloys (for Ashby plot context)
REF_ALLOYS = [
    {"name": "Cantor (CoCrFeMnNi)", "Omega": 10.8, "delta": 3.2, "type": "Solid Solution"},
    {"name": "Senkov (WNbMoTaV)", "Omega": 12.5, "delta": 4.1, "type": "Solid Solution"},
    {"name": "Senkov BCC (HfNbTaTiZr)", "Omega": 5.2, "delta": 5.8, "type": "Solid Solution"},
    {"name": "Zr50Cu50 (Metallic Glass)", "Omega": 0.4, "delta": 18.0, "type": "Amorphous"},
]

# =============================================================================
# 3. MIEDEMA MODEL (Dynamic Calculation)
# =============================================================================
class MiedemaCalculator:
    """
    Implementace Miedemova makroskopického modelu pro výpočet binárních entalpií.
    Zjednodušená verze pro přibližné výpočty trendů 'on-the-fly'.
    """
    @staticmethod
    def get_binary_enthalpy(el1_sym, el2_sym):
        # 1. Zkusíme najít v tabulce pro nejběžnější páry (kvůli přesnosti)
        pair = frozenset([el1_sym, el2_sym])
        # Zde bychom mohli mít cache/hardcoded tabulku prioritně
        # Pro v3.0 použijeme zjednodušenou formuli
        
        e1 = ELEMENTS_DB.get(el1_sym)
        e2 = ELEMENTS_DB.get(el2_sym)
        if not e1 or not e2: return 0.0
        
        # Miedema Equation Simplified Approximation
        # H_mix ~ [-P * (d_phi)^2 + Q * (d_nws)^2]
        # P, Q jsou empirické konstanty. Pro TM-TM slitiny P~14.1, Q~9.4 (approx)
        P = 14.1
        Q = 9.4
        
        d_phi = e1.phi - e2.phi
        d_nws = e1.nws - e2.nws
        
        # Chemický člen (negativní) a Elastický člen (pozitivní)
        chemical = -P * (d_phi**2)
        elastic = Q * (d_nws**2)
        
        # Velmi hrubý odhad v kJ/mol na základě parametrů
        # Pro přesnou kalibraci by zde byla nutná komplexnější funkce R(conc)
        # Zde použijeme "Miedema-like" trend
        
        # Pro demonstraci v tuto chvíli vrátíme raději hodnotu z původní tabulky, 
        # pokud existuje, jinak tento odhad.
        # (Jelikož nemám přesné P, Q faktory pro všechny skupiny, 
        # zachovám hybridní přístup pro spolehlivost)
        
        known_val = get_known_binary(el1_sym, el2_sym)
        if known_val is not None:
             return known_val
             
        # Fallback calculation
        return (chemical + elastic) * 5.0 # Scaling factor approximation

def get_known_binary(el1, el2):
    # Core Binary Pairs from Research Document
    pairs = {
       frozenset(['Ti', 'Ni']): -35, frozenset(['Ti', 'Al']): -30, frozenset(['Ni', 'Al']): -22,
       frozenset(['Zr', 'Ni']): -49, frozenset(['Ti', 'Fe']): -17, frozenset(['Ti', 'Co']): -28,
       frozenset(['Al', 'Co']): -19, frozenset(['Ni', 'Nb']): -30, frozenset(['Ni', 'La']): -27,
       frozenset(['La', 'Al']): -38, frozenset(['Mg', 'Ti']): 16,  frozenset(['Fe', 'Cu']): 13,
       frozenset(['Co', 'Cu']): 6,   frozenset(['Cr', 'Ni']): -7,  frozenset(['Ti', 'V']): -2,
       frozenset(['V', 'Cr']): -2,   frozenset(['Fe', 'Ni']): -2,  frozenset(['Co', 'Ni']): 0,
       frozenset(['Ti', 'Zr']): 0,   frozenset(['Ta', 'W']): -7,   frozenset(['Nb', 'Mo']): -6
    }
    return pairs.get(frozenset([el1, el2]))

# =============================================================================
# 4. JÁDRO APLIKACE
# =============================================================================
def parse_formula(notation: str):
    """Parsuje výraz (TiVCr)95Ni5 nebo Ti20V20..."""
    notation = notation.strip().replace(" ", "")
    composition = {}
    try:
        # Check for bracket notation: (Base)Bal + Alloys
        base_match = re.match(r'\(([A-Za-z]+)\)(\d+(?:\.\d+)?)', notation)
        if base_match:
            base_str = base_match.group(1)
            base_pct = float(base_match.group(2))
            base_elements = re.findall(r'[A-Z][a-z]?', base_str)
            if not base_elements: return None
            for el in base_elements:
                composition[el] = base_pct / len(base_elements)
            rest = notation[base_match.end():]
            additional = re.findall(r'([A-Z][a-z]?)(\d+(?:\.\d+)?)', rest)
            for el, val in additional:
                composition[el] = float(val)
        else:
            parts = re.findall(r'([A-Z][a-z]?)(\d+(?:\.\d+)?)?', notation)
            for el, val in parts:
                if not el: continue
                amount = float(val) if val else 1.0
                composition[el] = amount
                
        # Validation
        for el in composition:
            if el not in ELEMENTS_DB:
                st.error(f"Neznámý prvek: {el}")
                return None
                
        # Normalize to 1.0 (fractions)
        total = sum(composition.values())
        return {k: v/total for k, v in composition.items()}
    except:
        return None

def calculate_metrics(comp):
    elements = list(comp.keys())
    R_GAS = 8.314
    
    # 1. Basic Averages
    r_bar = sum(comp[el] * ELEMENTS_DB[el].r for el in elements)
    Tm_avg = sum(comp[el] * ELEMENTS_DB[el].Tm for el in elements)
    VEC_avg = sum(comp[el] * ELEMENTS_DB[el].VEC for el in elements)
    
    # 2. Density (Physics-based: Molar Mass / Molar Volume)
    # Using simple Rule of Mixtures for density approximation if molar volumes not fully calibrated
    # Better: rho = sum(ci * Ai) / sum(ci * (Ai/rho_i))
    # We will derive molar volume V_i from atomic mass and approximate density logic
    # Or just use weighted average for simplicity as consistent with "Engineering" approx
    # Standard HEA calc often uses: rho_mix = sum(c_i * A_i) / sum(c_i * A_i / rho_i)
    # Since we removed explicit density from DB, let's estimate rho_i approx or re-add
    # Let's use the provided atomic weight to refine.
    # Re-adding approx density to data is safer, but user asked for Atomic Weight usage.
    # Let's use: Density = (Sum c_i * M_i) / (Sum c_i * V_i)
    # V_i approx = r_i^3 * const?
    # Actually, let's trust the "V23" parameter (V^2/3) from Miedema for volume.
    # V_molar = (V2/3)^(3/2).
    molar_mass_mix = sum(comp[el] * ELEMENTS_DB[el].mass for el in elements)
    molar_vol_mix = sum(comp[el] * (ELEMENTS_DB[el].V23 ** 1.5) for el in elements)
    density = (molar_mass_mix / molar_vol_mix) if molar_vol_mix > 0 else 0
    # Correction factor: Miedema V is in cm3/mol. Mass in g/mol. Result is g/cm3. Perfect.
    
    # 3. Thermodynamic Param
    # Entropy
    S_mix = -R_GAS * sum(c * math.log(c) for c in comp.values() if c > 0)
    
    # Enthalpy (Pairwise)
    H_mix = 0.0
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            el1, el2 = elements[i], elements[j]
            h_ij = MiedemaCalculator.get_binary_enthalpy(el1, el2)
            H_mix += 4 * h_ij * comp[el1] * comp[el2]
            
    # Omega
    Omega = (Tm_avg * S_mix) / (abs(H_mix) * 1000) if abs(H_mix) > 0.001 else 100.0
    
    # Delta (Size Mismatch)
    delta_sq = sum(comp[el] * (1 - ELEMENTS_DB[el].r / r_bar)**2 for el in elements)
    delta = 100 * math.sqrt(delta_sq)

    # 4. Hydrogen Param (Griessen)
    # Weighted averages
    H_inf_mix = sum(comp[el] * ELEMENTS_DB[el].H_inf for el in elements)
    H_f_mix = sum(comp[el] * ELEMENTS_DB[el].H_f for el in elements)
    
    # Cost
    cost_est = sum(comp[el] * ELEMENTS_DB[el].price for el in elements)

    return {
        "S_mix": S_mix, "H_mix": H_mix, "delta": delta, "Omega": Omega,
        "VEC": VEC_avg, "Tm": Tm_avg, "Density": density, "Cost": cost_est,
        "H_inf": H_inf_mix, "H_f": H_f_mix
    }

def predict_structure(res):
    """Updated Criteria 2024/2025"""
    vec = res['VEC']
    delta = res['delta']
    omega = res['Omega']
    
    # Phase
    phase_pred = "Unknown"
    if omega >= 1.1 and delta <= 6.6:
        if vec < 6.87: # Updated bounds might suggest 5.7-7.2 for BCC, but sticking to standard logic + notes
            phase_pred = "BCC (Solid Solution)"
        elif vec >= 8.4:
             phase_pred = "FCC (Solid Solution)"
        elif 6.87 <= vec < 8.4:
             phase_pred = "BCC + FCC (Dual Phase)"
    else:
        phase_pred = "Multiphase / Intermetallic (Unstable SS)"
        
    return phase_pred

def predict_hydrogen_behavior(res):
    """Griessen classification"""
    h_f = res['H_f']
    if -80 <= h_f <= -40:
        return "🔥 Stabilní Hydrid (High T Desorption)", "success"
    elif -40 < h_f <= -15:
        return "🔋 Reverzibilní Skladování (Room T)", "info"
    elif h_f > 0:
        return "🛡️ Odolné proti vodíku (Barrier/Structural)", "warning"
    elif h_f < -80:
        return "🪤 Hydrogen Trap (Too Stable)", "error"
    else:
        return "❓ Neurčité chování", "secondary"

# =============================================================================
# 5. GRAFY
# =============================================================================
def create_ashby_plot(user_res, label):
    # Reference points
    ref_df = pd.DataFrame(REF_ALLOYS)
    
    # User point
    user_data = pd.DataFrame([{"Omega": user_res['Omega'], "delta": user_res['delta'], "name": "Váš Návrh", "type": "Tento Projekt"}])
    
    base = alt.Chart(ref_df).mark_circle(size=100, color='gray', opacity=0.6).encode(
        x=alt.X('Omega', scale=alt.Scale(type='log', domain=[0.1, 100])),
        y='delta',
        tooltip=['name', 'type']
    )
    
    user = alt.Chart(user_data).mark_circle(size=200, color='red').encode(
        x='Omega', y='delta', tooltip=['name', 'Omega', 'delta']
    )
    
    # Stability Zone (Green Rect)
    rect = alt.Chart(pd.DataFrame([{'x': 1.1, 'x2': 100, 'y': 0, 'y2': 6.6}])).mark_rect(
        color='green', opacity=0.1
    ).encode(x='x', x2='x2', y='y', y2='y2')
    
    return (rect + base + user).properties(title="Ashby Map: Stabilita (Ref. Alloys in Gray)", height=400)

def create_hydrogen_gauge(val):
    # Simple bar chart centered on 0
    df = pd.DataFrame([{"val": val, "label": "H_f (kJ/mol)"}])
    return alt.Chart(df).mark_bar().encode(
        x=alt.X('val', scale=alt.Scale(domain=[-100, 100]), title="H_f (kJ/mol)"),
        y='label',
        color=alt.condition(alt.datum.val < 0, alt.value("green"), alt.value("orange"))
    ).properties(height=100, title="Entalpie tvorby hydridu")

# =============================================================================
# 6. WORD EXPORT
# =============================================================================
def generate_report(res, formula):
    doc = Document()
    doc.add_heading('Expert Pro Report v3.0', 0)
    doc.add_paragraph(f'Slitina: {formula}')
    doc.add_paragraph(f'Predikce Fáze: {predict_structure(res)}')
    h_text, _ = predict_hydrogen_behavior(res)
    doc.add_paragraph(f'Vodík: {h_text}')
    
    doc.add_heading('Termodynamická Data', 1)
    table = doc.add_table(rows=1, cols=2)
    rows = table.add_row().cells
    for k,v in res.items():
        r = table.add_row().cells
        r[0].text = k
        r[1].text = f"{v:.4f}"
        
    f = io.BytesIO()
    doc.save(f)
    f.seek(0)
    return f

# =============================================================================
# 7. MAIN
# =============================================================================
def main():
    st.sidebar.title("HEA Expert Pro v3.0")
    st.sidebar.info("Režim: Advanced Academic")
    
    st.title("HEA Expert System: Hydrogen & Thermodynamics ⚛️")
    
    col1, col2 = st.columns([2,1])
    with col1:
        formula = st.text_input("Zadejte složení (např. Ti20V20Cr20Nb20Zr20):", "TiVCrNbZr")
    with col2:
        st.markdown("### ⚡ Rychlá akce")
        run_btn = st.button("PROVÉST ANALÝZU", type="primary")

    if run_btn and formula:
        comp = parse_formula(formula)
        if comp:
            res = calculate_metrics(comp)
            
            # --- ZÁKLADNÍ VÝSLEDKY ---
            st.divider()
            phg_text = predict_structure(res)
            st.subheader(f"Strukturní Predikce: {phg_text}")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Omega (Ω)", f"{res['Omega']:.2f}")
            c2.metric("Delta (δ)", f"{res['delta']:.2f}%")
            c3.metric("VEC", f"{res['VEC']:.2f}")
            c4.metric("Hustota", f"{res['Density']:.2f} g/cm³")
            
            # --- TABS ---
            tab1, tab2, tab3 = st.tabs(["📊 Fázová Stabilita", "💧 Vodíkové Technologie", "📝 Export"])
            
            with tab1:
                col_g1, col_g2 = st.columns([2,1])
                with col_g1:
                    st.altair_chart(create_ashby_plot(res, formula), use_container_width=True)
                with col_g2:
                    st.write("**Vysvětlení:**")
                    st.write("- **Zelená zóna**: Stabilní tuhý roztok")
                    st.write("- **Červený bod**: Vaše slitina")
                    st.write("- **Šedé body**: Referenční systémy (Cantor...)")
            
            with tab2:
                st.subheader("Analýza pro Vodíkové inženýrství")
                h_status, h_color = predict_hydrogen_behavior(res)
                if h_color == 'success': st.success(h_status)
                elif h_color == 'info': st.info(h_status)
                elif h_color == 'warning': st.warning(h_status)
                elif h_color == 'error': st.error(h_status)
                
                hc1, hc2 = st.columns(2)
                hc1.metric("Entalpie Hydridu (H_f)", f"{res['H_f']:.1f} kJ/mol")
                hc1.markdown("*Klíčový parametr pro stabilitu hydridu.*")
                
                hc2.metric("Entalpie Rozpouštění (H_inf)", f"{res['H_inf']:.1f} kJ/mol")
                hc2.markdown("*Indikuje počáteční absorpci.*")
                
                st.altair_chart(create_hydrogen_gauge(res['H_f']), use_container_width=True)
                
            with tab3:
                st.write("Stáhnout profesionální protokol:")
                docx = generate_report(res, formula)
                st.download_button("📄 Stáhnout DOCX", docx, f"report_{formula}.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")

if __name__ == "__main__":
    main()
