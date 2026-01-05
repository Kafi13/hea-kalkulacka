import streamlit as st
import math
import pandas as pd
import altair as alt
import numpy as np
import re
from dataclasses import dataclass
from enum import Enum

# =============================================================================
# 1. KONFIGURACE APLIKACE
# =============================================================================
st.set_page_config(
    page_title="HEA Kalkulačka Expert",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS pro akademický vzhled
st.markdown("""
<style>
    .stMetric {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 10px;
        border-radius: 5px;
    }
    .stAlert {
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. DATABÁZE A STRUKTURY
# =============================================================================
class CrystalStructure(Enum):
    BCC = "BCC (Tělesně středěná)"
    FCC = "FCC (Plošně středěná)"
    MIXED = "Směs fází (BCC + FCC)"
    UNKNOWN = "Neurčeno / Intermetalická fáze"

@dataclass
class ElementData:
    symbol: str
    r: float      # Poloměr (Angstrom)
    VEC: int      # Valenční elektrony
    Tm: int       # Teplota tání (Kelvin)
    H_inf: float  # Rozpouštění vodíku (kJ/mol) - Griessen
    H_f: float    # Tvorba hydridu (kJ/mol)

# Databáze prvků (Zdroje: Griessen, Driessen, Takeuchi)
ELEMENTS_DB = {
    'Sc': ElementData('Sc', 1.64, 3, 1814, -90, -100),
    'Y':  ElementData('Y', 1.80, 3, 1799, -79, -110),
    'La': ElementData('La', 1.87, 3, 1193, -67, -150),
    'Ce': ElementData('Ce', 1.82, 3, 1068, -74, -140),
    'Ti': ElementData('Ti', 1.47, 4, 1941, -52, -68),
    'Zr': ElementData('Zr', 1.60, 4, 2128, -58, -82),
    'Hf': ElementData('Hf', 1.59, 4, 2506, -38, -70),
    'V':  ElementData('V', 1.31, 5, 2183, -30, -40),
    'Nb': ElementData('Nb', 1.43, 5, 2750, -35, -50),
    'Ta': ElementData('Ta', 1.43, 5, 3290, -36, -45),
    'Cr': ElementData('Cr', 1.25, 6, 2180, 28, -10),
    'Mo': ElementData('Mo', 1.39, 6, 2896, 25, 5),
    'W':  ElementData('W', 1.39, 6, 3695, 96, 10),
    'Mn': ElementData('Mn', 1.27, 7, 1519, 1, -8),
    'Fe': ElementData('Fe', 1.26, 8, 1811, 25, 15),
    'Co': ElementData('Co', 1.25, 9, 1768, 21, 18),
    'Ni': ElementData('Ni', 1.24, 10, 1728, 12, 5),
    'Pd': ElementData('Pd', 1.37, 10, 1828, -10, -20),
    'Cu': ElementData('Cu', 1.28, 11, 1358, 46, 25),
    'Ag': ElementData('Ag', 1.44, 11, 1234, 63, 30),
    'Zn': ElementData('Zn', 1.34, 12, 692, 15, 5),
    'Al': ElementData('Al', 1.43, 3, 933, 60, -6),
    'Mg': ElementData('Mg', 1.60, 2, 923, 21, -75),
    'Si': ElementData('Si', 1.32, 4, 1687, 180, 20),
    'Ca': ElementData('Ca', 1.97, 2, 1115, -94, -180),
    'Sn': ElementData('Sn', 1.62, 4, 505, 125, 40),
}

# Miedema model - Binární entalpie mísení
BINARY_ENTHALPIES = {
    frozenset(['Ti', 'V']): 2,   frozenset(['Ti', 'Cr']): -7,  frozenset(['Ti', 'Mn']): -22,
    frozenset(['Ti', 'Fe']): -17, frozenset(['Ti', 'Co']): -28, frozenset(['Ti', 'Ni']): -35,
    frozenset(['Ti', 'Cu']): -9,  frozenset(['Ti', 'Al']): -30, frozenset(['Ti', 'Zr']): 0,
    frozenset(['Ti', 'Nb']): 2,   frozenset(['Ti', 'Mo']): -4,  frozenset(['Ti', 'Hf']): 0,
    frozenset(['Ti', 'Ta']): 1,   frozenset(['Ti', 'W']): -6,   frozenset(['Ti', 'Mg']): 16,
    frozenset(['Ti', 'La']): 0,   frozenset(['Ti', 'Ce']): 0,   frozenset(['Ti', 'Sn']): -21,
    frozenset(['Ti', 'Y']): 0,    frozenset(['Ti', 'Sc']): 0,    frozenset(['Ti', 'Pd']): -55,
    frozenset(['V', 'Cr']): -2,   frozenset(['V', 'Mn']): -1,   frozenset(['V', 'Fe']): -7,
    frozenset(['V', 'Co']): -14,  frozenset(['V', 'Ni']): -18,  frozenset(['V', 'Cu']): 5,
    frozenset(['V', 'Al']): -16,  frozenset(['V', 'Zr']): -4,   frozenset(['V', 'Nb']): -1,
    frozenset(['V', 'Mo']): 0,    frozenset(['V', 'Hf']): -2,   frozenset(['V', 'Ta']): 0,
    frozenset(['V', 'W']): 0,     frozenset(['V', 'La']): 10,   frozenset(['V', 'Sn']): -9,
    frozenset(['V', 'Y']): 5,     frozenset(['V', 'Pd']): -30,
    frozenset(['Cr', 'Mn']): 2,   frozenset(['Cr', 'Fe']): -1,  frozenset(['Cr', 'Co']): -4,
    frozenset(['Cr', 'Ni']): -7,  frozenset(['Cr', 'Cu']): 12,  frozenset(['Cr', 'Al']): -10,
    frozenset(['Cr', 'Zr']): -12, frozenset(['Cr', 'Nb']): -7,  frozenset(['Cr', 'Mo']): 0,
    frozenset(['Cr', 'Hf']): -9,  frozenset(['Cr', 'Ta']): -7,  frozenset(['Cr', 'W']): 1,
    frozenset(['Cr', 'Y']): 5,    frozenset(['Cr', 'Pd']): -15,
    frozenset(['Mn', 'Fe']): 0,   frozenset(['Mn', 'Co']): -5,  frozenset(['Mn', 'Ni']): -8,
    frozenset(['Mn', 'Cu']): 4,   frozenset(['Mn', 'Al']): -19, frozenset(['Mn', 'Zr']): -26,
    frozenset(['Mn', 'Nb']): -11, frozenset(['Mn', 'La']): -5,
    frozenset(['Fe', 'Co']): -1,  frozenset(['Fe', 'Ni']): -2,  frozenset(['Fe', 'Cu']): 13,
    frozenset(['Fe', 'Al']): -11, frozenset(['Fe', 'Zr']): -25, frozenset(['Fe', 'Nb']): -16,
    frozenset(['Fe', 'Mo']): -2,  frozenset(['Fe', 'La']): 0,   frozenset(['Fe', 'Y']): 1,
    frozenset(['Co', 'Ni']): 0,   frozenset(['Co', 'Cu']): 6,   frozenset(['Co', 'Al']): -19,
    frozenset(['Co', 'Zr']): -41, frozenset(['Co', 'Nb']): -25, frozenset(['Co', 'Mo']): -5,
    frozenset(['Co', 'La']): -16, frozenset(['Co', 'Y']): -22,
    frozenset(['Ni', 'Cu']): 4,   frozenset(['Ni', 'Al']): -22, frozenset(['Ni', 'Zr']): -49,
    frozenset(['Ni', 'Nb']): -30, frozenset(['Ni', 'Mo']): -7,  frozenset(['Ni', 'Hf']): -43,
    frozenset(['Ni', 'Ta']): -29, frozenset(['Ni', 'W']): -8,   frozenset(['Ni', 'La']): -24,
    frozenset(['Ni', 'Ce']): -26, frozenset(['Ni', 'Sn']): -20, frozenset(['Ni', 'Y']): -31,
    frozenset(['Ni', 'Pd']): 0,   frozenset(['Ni', 'Sc']): -39,
    frozenset(['Nb', 'Mo']): -6,  frozenset(['Nb', 'Hf']): 4,   frozenset(['Nb', 'Ta']): 0,
    frozenset(['Nb', 'W']): -8,   frozenset(['Nb', 'Al']): -18, frozenset(['Nb', 'Cu']): 3,
    frozenset(['Nb', 'Zr']): 4,   frozenset(['Nb', 'Y']): 10,
    frozenset(['Zr', 'Mo']): -6,  frozenset(['Zr', 'Hf']): 0,   frozenset(['Zr', 'Ta']): 3,
    frozenset(['Zr', 'W']): -9,   frozenset(['Zr', 'Al']): -42, frozenset(['Zr', 'Cu']): -23,
    frozenset(['Zr', 'Sn']): -43, frozenset(['Zr', 'Y']): 0,    frozenset(['Zr', 'Pd']): -91,
    frozenset(['Hf', 'Ta']): 3,   frozenset(['Hf', 'Al']): -39, frozenset(['Hf', 'Y']): 0,
    frozenset(['Mo', 'Al']): -5,  frozenset(['Mo', 'Ta']): -5,  frozenset(['Mo', 'W']): 0,
    frozenset(['Ta', 'Al']): -19, frozenset(['Ta', 'W']): -7,
    frozenset(['W', 'Al']): -3,
    frozenset(['La', 'Al']): -38, frozenset(['Ce', 'Al']): -38, frozenset(['La', 'Ce']): 0,
    frozenset(['Cu', 'Sn']): -2,  frozenset(['Cu', 'Al']): -1,
    frozenset(['Mg', 'Ni']): -4,  frozenset(['Mg', 'Cu']): -3,  frozenset(['Mg', 'Al']): 2,
    frozenset(['Ag', 'Cu']): 2,   frozenset(['Ag', 'Al']): -4,
    frozenset(['Pd', 'Al']): -56, frozenset(['Pd', 'Cu']): -14,
}

def get_binary_H(el1, el2):
    return BINARY_ENTHALPIES.get(frozenset([el1, el2]), 0.0)

# =============================================================================
# 3. LOGIKA A VÝPOČTY
# =============================================================================
def parse_formula(notation: str) -> dict:
    """Zpracuje vstupní řetězec na slovník složení."""
    notation = notation.strip().replace(" ", "")
    composition = {}
    
    try:
        # 1. Závorková notace: (TiVCr)95Ni5
        base_match = re.match(r'\(([A-Za-z]+)\)(\d+(?:\.\d+)?)', notation)
        if base_match:
            base_str = base_match.group(1)
            base_pct = float(base_match.group(2))
            base_elements = re.findall(r'[A-Z][a-z]?', base_str)
            if not base_elements: return None
            
            # Rozdělení procenta báze
            for el in base_elements:
                composition[el] = base_pct / len(base_elements)
            
            # Zbytek
            rest = notation[base_match.end():]
            additional = re.findall(r'([A-Z][a-z]?)(\d+(?:\.\d+)?)', rest)
            for el, val in additional:
                composition[el] = float(val)
                
        # 2. Standardní notace: Ti30V30...
        else:
            parts = re.findall(r'([A-Z][a-z]?)(\d+(?:\.\d+)?)?', notation)
            for el, val in parts:
                if not el: continue
                amount = float(val) if val else 1.0
                composition[el] = amount

        # Validace a normalizace
        for el in composition:
            if el not in ELEMENTS_DB:
                st.error(f"Chyba: Prvek '{el}' není v databázi.")
                return None
        
        total = sum(composition.values())
        return {k: v/total for k, v in composition.items()}

    except:
        return None

def calculate_parameters(comp):
    """Vypočítá všechny termodynamické parametry."""
    elements = list(comp.keys())
    R = 8.31446
    
    # Průměrné hodnoty
    r_bar = sum(comp[el] * ELEMENTS_DB[el].r for el in elements)
    Tm_avg = sum(comp[el] * ELEMENTS_DB[el].Tm for el in elements)
    VEC_avg = sum(comp[el] * ELEMENTS_DB[el].VEC for el in elements)
    H_inf = sum(comp[el] * ELEMENTS_DB[el].H_inf for el in elements)
    H_f = sum(comp[el] * ELEMENTS_DB[el].H_f for el in elements)
    
    # Delta (rozdíl poloměrů)
    delta_sq = sum(comp[el] * (1 - ELEMENTS_DB[el].r / r_bar)**2 for el in elements)
    delta = 100 * math.sqrt(delta_sq)
    
    # Entropie (S_mix)
    S_mix = -R * sum(c * math.log(c) for c in comp.values() if c > 0)
    
    # Entalpie (H_mix) - Miedema
    H_mix = 0.0
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            el1, el2 = elements[i], elements[j]
            H_ij = get_binary_H(el1, el2)
            H_mix += 4 * H_ij * comp[el1] * comp[el2]
    
    # Omega
    Omega = (Tm_avg * S_mix) / (abs(H_mix) * 1000) if abs(H_mix) > 0.001 else 9999.0
    
    return {
        "S_mix": S_mix, "H_mix": H_mix, "delta": delta, "Omega": Omega,
        "VEC": VEC_avg, "Tm": Tm_avg, "H_inf": H_inf, "H_f": H_f
    }

def get_prediction(res):
    """Slovní interpretace výsledků."""
    # Tuhý roztok (Solid Solution)
    is_ss = (res['Omega'] >= 1.1) and (res['delta'] <= 6.6)
    
    if is_ss:
        if res['VEC'] < 6.87: struct = "BCC (Tělesně středěná)"
        elif res['VEC'] >= 8.0: struct = "FCC (Plošně středěná)"
        else: struct = "Směs BCC + FCC"
        return f"✅ Tuhý roztok ({struct})", "success"
    else:
        return "⚠️ Pravděpodobně intermetalické fáze nebo segregace", "warning"

# =============================================================================
# 4. POKROČILÉ VIZUALIZACE (ASHBY, HEATMAP, RADAR)
# =============================================================================
def create_ashby_plot(res):
    """Vytvoří Ashbyho diagram stability (Omega vs Delta)."""
    # Data pro graf (Zóna stability + Aktuální bod)
    source = pd.DataFrame([
        {"label": "Aktuální slitina", "Omega": res['Omega'], "Delta": res['delta'], "Type": "Alloy"},
    ])
    
    # Základní graf bodu
    points = alt.Chart(source).mark_circle(size=200, color='red').encode(
        x=alt.X('Omega', scale=alt.Scale(domain=[0, max(2.0, res['Omega']+0.5)])),
        y=alt.Y('Delta', scale=alt.Scale(domain=[0, max(10, res['delta']+2)]), title='Delta (%)'),
        tooltip=['label', 'Omega', 'Delta']
    )
    
    # Zóna stability (Obdélník)
    rect = alt.Chart(pd.DataFrame([{'x': 1.1, 'x2': 100, 'y': 0, 'y2': 6.6}])).mark_rect(
        color='green', opacity=0.1
    ).encode(
        x='x', x2='x2', y='y', y2='y2'
    )
    
    text = points.mark_text(align='left', dx=15).encode(text='label')
    
    return (rect + points + text).properties(
        title="Ashbyho diagram stability (Solid Solution Zone)",
        height=300
    )

def create_heatmap(comp):
    """Vytvoří matici binárních entalpií."""
    elements = list(comp.keys())
    data = []
    for el1 in elements:
        for el2 in elements:
            if el1 != el2:
                h = get_binary_H(el1, el2)
                data.append({"Prvek 1": el1, "Prvek 2": el2, "H_mix": h})
            else:
                data.append({"Prvek 1": el1, "Prvek 2": el2, "H_mix": 0})
                
    df = pd.DataFrame(data)
    
    chart = alt.Chart(df).mark_rect().encode(
        x='Prvek 1:N', y='Prvek 2:N',
        color=alt.Color('H_mix:Q', scale=alt.Scale(scheme='redblue', domainMid=0), title="H_mix (kJ/mol)"),
        tooltip=['Prvek 1', 'Prvek 2', 'H_mix']
    ).properties(title="Matice interakcí (Modrá = Přitahování)", width=350, height=350)
    
    return chart

def create_property_chart(res):
    """Sloupcový graf vlastností."""
    data = pd.DataFrame([
        {"Parametr": "Stabilita (Ω)", "Hodnota": min(res['Omega'], 5.0), "Norm": 1.1},
        {"Parametr": "Velikost (δ %)", "Hodnota": res['delta'], "Norm": 6.6},
        {"Parametr": "H-absorpce (-ΔH∞)", "Hodnota": -res['H_inf'] if res['H_inf'] < 0 else 0, "Norm": 20},
    ])
    
    chart = alt.Chart(data).mark_bar().encode(
        x='Hodnota:Q',
        y='Parametr:N',
        color=alt.condition(
            alt.datum.Hodnota > alt.datum.Norm,
            alt.value("green"),
            alt.value("steelblue")
        )
    ).properties(title="Klíčové metriky")
    return chart

# =============================================================================
# 5. HLAVNÍ APLIKACE
# =============================================================================
def main():
    col1, col2 = st.columns([1, 10])
    with col1: st.write("# 🧪")
    with col2: st.title("HEA Kalkulačka Expert")
    
    st.write("Expertní nástroj pro návrh slitin a predikci vodíkových vlastností.")
    st.caption("Založeno na modelech: Miedema, Yang-Zhang (Ω, δ), Guo-Liu (VEC).")
    st.divider()

    # --- VSTUP ---
    c_in, c_help = st.columns([2, 1])
    with c_in:
        formula = st.text_input("Zadejte vzorec slitiny:", value="(TiVCr)95Ni5", help="Např. (TiVCr)95Ni5 nebo Ti30V30Cr30Ni10")
        calc_btn = st.button("🚀 SPOČÍTAT PARAMETRY", type="primary")
    
    with c_help:
        st.info("Příklady formátu:\n- `(TiVCr)95Ni5`\n- `Ti32 V32 Cr32 Ni4`\n- `Ti1 V1 Cr1` (ekvimolární)")

    if calc_btn and formula:
        comp = parse_formula(formula)
        
        if comp:
            res = calculate_parameters(comp)
            pred_text, pred_type = get_prediction(res)
            
            # --- VÝSLEDKY ---
            st.divider()
            st.subheader(f"Výsledky pro: {formula}")
            
            if pred_type == "success": st.success(pred_text)
            else: st.warning(pred_text)
            
            # Karty s metrikami
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Entropie (ΔSmix)", f"{res['S_mix']:.3f}", "J/K·mol")
            c2.metric("Entalpie (ΔHmix)", f"{res['H_mix']:.3f}", "kJ/mol")
            c3.metric("Delta (δ)", f"{res['delta']:.2f}", "%", help="Musí být < 6.6%")
            c4.metric("Omega (Ω)", f"{res['Omega']:.2f}", help="Musí být > 1.1")
            
            c5, c6, c7, c8 = st.columns(4)
            c5.metric("VEC", f"{res['VEC']:.2f}")
            c6.metric("ΔH∞ (H-absorpce)", f"{res['H_inf']:.2f}", "kJ/mol", delta_color="inverse")
            c7.metric("ΔHf (Hydrid)", f"{res['H_f']:.2f}", "kJ/mol")
            c8.metric("T_m (Tání)", f"{res['Tm']:.0f}", "K")

            # --- GRAFICKÁ ANALÝZA (TABS) ---
            st.write("### 📊 Grafická analýza")
            tab1, tab2, tab3 = st.tabs(["Fázová stabilita (Ashby)", "Interakce prvků", "Složení & Data"])
            
            with tab1:
                st.altair_chart(create_ashby_plot(res), use_container_width=True)
                st.info("💡 Zelená zóna označuje oblast stabilních tuhých roztoků (Yang & Zhang criteria).")
            
            with tab2:
                col_h, col_p = st.columns([1, 1])
                with col_h:
                    st.altair_chart(create_heatmap(comp), use_container_width=True)
                with col_p:
                    st.altair_chart(create_property_chart(res), use_container_width=True)
            
            with tab3:
                 # Graf složení
                df_chart = pd.DataFrame({"Prvek": list(comp.keys()), "Podíl": [v*100 for v in comp.values()]})
                chart = alt.Chart(df_chart).mark_arc(innerRadius=60).encode(
                    theta="Podíl", color="Prvek", tooltip=["Prvek", "Podíl"]
                ).properties(height=250)
                
                c_pie, c_table = st.columns([1, 2])
                with c_pie: st.altair_chart(chart)
                with c_table:
                    # Tabulka s daty
                    element_data_list = []
                    for el, frac in comp.items():
                        d = ELEMENTS_DB[el]
                        element_data_list.append({
                            "Prvek": el, "Atom. %": f"{frac*100:.1f}",
                            "r (Å)": d.r, "VEC": d.VEC, "Tm (K)": d.Tm,
                            "ΔH∞": d.H_inf
                        })
                    st.dataframe(pd.DataFrame(element_data_list), hide_index=True)

            # --- EXPORT ---
            st.divider()
            export_data = {"Vzorec": formula}
            for k, v in res.items(): export_data[k] = round(v, 4)
            for el, frac in comp.items(): export_data[f"El_{el}_%"] = round(frac*100, 2)
            
            df_export = pd.DataFrame([export_data])
            csv = df_export.to_csv(index=False).encode('utf-8-sig')
            
            st.download_button(
                label="📥 Stáhnout kompletní report (CSV)",
                data=csv,
                file_name="hea_report.csv",
                mime="text/csv",
                type="primary"
            )
            
            # --- CITACE ---
            with st.expander("📚 Zdroje a metodika"):
                st.markdown("""
                **Použité modely a citace:**
                * **Stabilita fáze (Ω, δ):** Yang, X. & Zhang, Y. *Prediction of high-entropy stabilized solid-solution alloys*. Mater. Chem. Phys. (2012).
                * **Valenční koncentrace (VEC):** Guo, S. & Liu, C.T. *Phase stability in high entropy alloys*. Prog. Nat. Sci. (2011).
                * **Entalpie (Miedema):** Boer, F.R. et al. *Cohesion in Metals*. North-Holland (1988).
                * **Vodík (Griessen):** Griessen, R. & Driessen, A. *Heat of formation and band structure of binary and ternary metal hydrides*. Phys. Rev. B (1984).
                """)

        else:
            st.error("Nepodařilo se přečíst vzorec. Zkontrolujte značky prvků a formát.")

if __name__ == "__main__":
    main()