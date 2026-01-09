import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import re
import io
from dataclasses import dataclass
from docx import Document
from docx.shared import Inches, Pt, RGBColor

# =============================================================================
# 1. KONFIGURACE APLIKACE
# =============================================================================
st.set_page_config(
    page_title="HEA Ultimate Calculator v2",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# 2. DATABÁZE PRVKŮ (Čistá data bez citačních značek)
# =============================================================================
@dataclass
class ElementData:
    symbol: str
    name: str
    r: float      # Poloměr (Å)
    vec: int      # Valenční elektrony
    tm: float     # Teplota tání (K)
    mass: float   # Atomová hmotnost (g/mol)
    phi: float    # Miedema Work function (V)
    nws: float    # Miedema Electron density (d.u.)
    h_inf: float  # Entalpie rozpouštění H (kJ/mol H)
    h_f: float    # Entalpie tvorby hydridu (kJ/mol H2)

ELEMENTS_DB = {
    'Al': ElementData('Al', 'Hliník', 1.43, 3, 933, 26.98, 4.20, 1.39, 60, -6),
    'Mg': ElementData('Mg', 'Hořčík', 1.60, 2, 923, 24.31, 3.45, 1.17, 21, -75),
    'Si': ElementData('Si', 'Křemík', 1.17, 4, 1687, 28.09, 4.70, 1.50, 180, 0),
    'Ti': ElementData('Ti', 'Titan', 1.47, 4, 1941, 47.87, 3.65, 1.47, -52, -137),
    'V':  ElementData('V', 'Vanad', 1.35, 5, 2183, 50.94, 4.25, 1.64, -30, -47),
    'Cr': ElementData('Cr', 'Chrom', 1.29, 6, 2180, 52.00, 4.65, 1.73, 28, 6),
    'Mn': ElementData('Mn', 'Mangan', 1.37, 7, 1519, 54.94, 4.45, 1.61, 1, 21),
    'Fe': ElementData('Fe', 'Železo', 1.26, 8, 1811, 55.85, 4.93, 1.77, 25, 20),
    'Co': ElementData('Co', 'Kobalt', 1.25, 9, 1768, 58.93, 5.10, 1.75, 21, 31),
    'Ni': ElementData('Ni', 'Nikl', 1.25, 10, 1728, 58.69, 5.20, 1.75, 12, 15),
    'Cu': ElementData('Cu', 'Měď', 1.28, 11, 1358, 63.55, 4.55, 1.47, 46, 45),
    'Zr': ElementData('Zr', 'Zirkonium', 1.60, 4, 2128, 91.22, 3.45, 1.41, -58, -164),
    'Nb': ElementData('Nb', 'Niob', 1.47, 5, 2750, 92.91, 4.05, 1.62, -35, -40),
    'Mo': ElementData('Mo', 'Molybden', 1.40, 6, 2896, 95.95, 4.65, 1.77, 25, 92),
    'Pd': ElementData('Pd', 'Palladium', 1.37, 10, 1828, 106.4, 5.45, 1.67, -10, -20),
    'Hf': ElementData('Hf', 'Hafnium', 1.59, 4, 2506, 178.5, 3.55, 1.43, -38, -130),
    'Ta': ElementData('Ta', 'Tantal', 1.47, 5, 3290, 180.9, 4.05, 1.63, -36, -78),
    'W':  ElementData('W', 'Wolfram', 1.41, 6, 3695, 183.8, 4.80, 1.81, 96, 74),
    'La': ElementData('La', 'Lanthan', 1.88, 3, 1193, 138.9, 3.05, 1.18, -67, -206),
    'Ce': ElementData('Ce', 'Cer', 1.82, 4, 1068, 140.1, 3.15, 1.19, -74, -200),
    'Y':  ElementData('Y', 'Yttrium', 1.80, 3, 1799, 88.91, 3.20, 1.21, -79, -228)
}

# =============================================================================
# 3. MIEDEMŮV MODEL (Jádro výpočtů)
# =============================================================================
def calculate_miedema_enthalpy(el1_sym, el2_sym):
    """Vypočítá binární entalpii míšení (kJ/mol)."""
    e1 = ELEMENTS_DB.get(el1_sym)
    e2 = ELEMENTS_DB.get(el2_sym)
    
    if not e1 or not e2 or el1_sym == el2_sym:
        return 0.0

    # Konstanty pro přechodné kovy
    P = 14.1
    Q = 9.4 
    
    d_phi = e1.phi - e2.phi
    d_nws_direct = e1.nws - e2.nws

    term_phi = -P * (d_phi**2)
    term_nws = Q * (d_nws_direct**2)
    
    # Scaling factor
    enthalpy = (term_phi + term_nws) * 5.0 
    return enthalpy

# =============================================================================
# 4. LOGIKA A PARSOVÁNÍ
# =============================================================================
def parse_composition(formula):
    """Parsuje textový vstup (např. (TiZr)80Ni20)."""
    formula = formula.strip().replace(" ", "")
    composition = {}
    try:
        # Závorky
        bracket_match = re.match(r'\(([A-Za-z]+)\)(\d+(?:\.\d+)?)', formula)
        remaining_formula = formula
        
        if bracket_match:
            base_str = bracket_match.group(1)
            base_pct = float(bracket_match.group(2))
            base_elems = re.findall(r'[A-Z][a-z]?', base_str)
            if base_elems:
                share = base_percent = base_pct / len(base_elems)
                for el in base_elems: composition[el] = share
            remaining_formula = formula[bracket_match.end():]

        # Zbytek
        matches = re.findall(r'([A-Z][a-z]?)(\d+(?:\.\d+)?)?', remaining_formula)
        for el, qty in matches:
            if not el: continue
            amount = float(qty) if qty else 1.0 
            composition[el] = amount

        # Normalizace
        total = sum(composition.values())
        if total == 0: return None
        return {k: v/total for k, v in composition.items()}
    except:
        return None

def calculate_hea_properties(comp):
    elements = list(comp.keys())
    R_GAS = 8.314
    
    # Průměry
    r_bar = sum(comp[el] * ELEMENTS_DB[el].r for el in elements)
    tm_avg = sum(comp[el] * ELEMENTS_DB[el].tm for el in elements)
    vec_avg = sum(comp[el] * ELEMENTS_DB[el].vec for el in elements)
    
    # Termodynamika
    s_mix = -R_GAS * sum(c * np.log(c) for c in comp.values() if c > 0)
    
    h_mix = 0.0
    binary_contributions = [] # Pro graf příspěvků
    
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            el_i, el_j = elements[i], elements[j]
            h_ij = calculate_miedema_enthalpy(el_i, el_j)
            contribution = 4 * h_ij * comp[el_i] * comp[el_j]
            h_mix += contribution
            binary_contributions.append({
                "Pair": f"{el_i}-{el_j}",
                "H_ij": h_ij,
                "Contribution": contribution
            })
            
    delta = 100 * np.sqrt(sum(comp[el] * (1 - ELEMENTS_DB[el].r / r_bar)**2 for el in elements))
    omega = (tm_avg * s_mix) / (abs(h_mix) * 1000) if abs(h_mix) > 1e-4 else 100.0
    
    # Vodík
    h_inf_mix = sum(comp[el] * ELEMENTS_DB[el].h_inf for el in elements)
    h_f_mix = sum(comp[el] * ELEMENTS_DB[el].h_f for el in elements)
    
    return {
        "s_mix": s_mix, "h_mix": h_mix, "delta": delta, "omega": omega,
        "vec": vec_avg, "tm": tm_avg, "h_inf": h_inf_mix, "h_f": h_f_mix,
        "contributions": binary_contributions
    }

# =============================================================================
# 5. EXPERTNÍ LOGIKA (NOVÉ)
# =============================================================================
def get_expert_insights(comp, res):
    insights = []
    
    # 1. Katalýza
    has_catalyst = any(el in comp for el in ['Ni', 'Pd', 'Co'])
    if not has_catalyst:
        insights.append("⚠️ **Pomalá kinetika:** Chybí katalytické prvky (Ni, Pd). Absorpce vodíku může být pomalá i při dobré termodynamice.")
    else:
        insights.append("⚡ **Kinetika:** Přítomnost Ni/Pd/Co by měla urychlit disociaci vodíku.")

    # 2. VEC Pravidla
    if 6.87 <= res['vec'] <= 8.0:
        insights.append("🏗️ **Fázový přechod:** Hodnota VEC je v přechodové oblasti. Očekávejte směs BCC + FCC fází.")
    
    # 3. Lavesovy fáze
    if res['delta'] > 6.6 and res['h_mix'] < -10:
        insights.append("🧪 **Lavesovy fáze:** Vysoká neshoda atomů a silná vazba indikují možný vznik C14/C15 fází (dobré pro kapacitu, horší pro cyklování).")

    # 4. Stabilita hydridu
    if res['h_f'] < -50:
        insights.append("🔥 **Příliš stabilní:** Entalpie hydridu je velmi nízká. Pro desorpci bude potřeba vysoká teplota (>300°C).")
    
    return insights

# =============================================================================
# 6. UI A GRAFY
# =============================================================================
def main():
    st.title("🧬 HEA Expert: Hydrogen & Thermodynamics")
    
    # Vstup
    col1, col2 = st.columns([3, 1])
    with col1:
        formula = st.text_input("Zadejte složení (např. TiVCr, (TiVCr)95Ni5):", "TiVCrNb")
    with col2:
        st.write("")
        st.write("")
        btn = st.button("🚀 Analyzovat", type="primary", use_container_width=True)

    if btn and formula:
        comp = parse_composition(formula)
        if not comp:
            st.error("Chyba ve vzorci nebo neznámý prvek.")
            return

        res = calculate_hea_properties(comp)
        
        # --- ZOBRAZENÍ METRIK (SEMAFOR) ---
        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Entropie (ΔS)", f"{res['s_mix']:.2f} J/mol·K", delta="High Entropy" if res['s_mix']>12 else "Low")
        c2.metric("Entalpie (ΔH)", f"{res['h_mix']:.2f} kJ/mol", delta_color="inverse", delta="Stable" if -20 < res['h_mix'] < 5 else "Unstable")
        c3.metric("Neshoda (δ)", f"{res['delta']:.2f} %", delta_color="inverse", delta="Solid Solution" if res['delta'] < 6.6 else "Multiphase")
        c4.metric("Omega (Ω)", f"{res['omega']:.2f}", delta="Stable" if res['omega'] > 1.1 else "Unstable")

        # --- EXPERTNÍ INSIGHTS (NOVÉ) ---
        insights = get_expert_insights(comp, res)
        with st.container():
            st.info("### 🧠 Expertní analýza\n" + "\n\n".join(insights))

        # --- GRAFY (ALTAIR - MODERNÍ) ---
        tab1, tab2, tab3 = st.tabs(["📊 Diagram Stability", "🧪 Vodíková Afinita", "📉 Detail Entalpie"])
        
        with tab1: # Ashby Map
            # Reference data
            ref_data = pd.DataFrame([
                {'Omega': 12.5, 'Delta': 4.1, 'Label': 'Cantor (FCC)'},
                {'Omega': 5.2, 'Delta': 5.8, 'Label': 'Senkov (BCC)'},
                {'Omega': res['omega'], 'Delta': res['delta'], 'Label': 'Tvoje Slitina'}
            ])
            
            chart = alt.Chart(ref_data).mark_circle(size=200).encode(
                x=alt.X('Omega', scale=alt.Scale(type='log', domain=[0.1, 100])),
                y='Delta',
                color=alt.Color('Label', scale=alt.Scale(domain=['Tvoje Slitina', 'Cantor (FCC)', 'Senkov (BCC)'], range=['red', 'gray', 'gray'])),
                tooltip=['Label', 'Omega', 'Delta']
            ).properties(height=400)
            
            # Zelená zóna (Solid Solution)
            rect = alt.Chart(pd.DataFrame({'x': [1.1], 'x2': [100], 'y': [0], 'y2': [6.6]})).mark_rect(
                color='green', opacity=0.1
            ).encode(x='x', x2='x2', y='y', y2='y2')
            
            st.altair_chart(rect + chart, use_container_width=True)

        with tab2: # Vodík
            col_h1, col_h2 = st.columns(2)
            with col_h1:
                st.metric("ΔH (Rozpouštění)", f"{res['h_inf']:.1f} kJ/mol H")
            with col_h2:
                st.metric("ΔH (Hydrid)", f"{res['h_f']:.1f} kJ/mol H2")
                
            # Gauge chart (jednoduchý bar)
            h_df = pd.DataFrame({'Val': [res['h_f']], 'Type': ['Hydrid']})
            h_chart = alt.Chart(h_df).mark_bar().encode(
                x=alt.X('Val', scale=alt.Scale(domain=[-100, 50])),
                color=alt.condition(alt.datum.Val < -40, alt.value('red'), # Příliš silné
                      alt.condition(alt.datum.Val < -10, alt.value('green'), # Ideál
                      alt.value('orange'))) # Bariéra
            ).properties(height=100, title="Stabilita Hydridu (Zelená = Ideální skladování)")
            st.altair_chart(h_chart, use_container_width=True)

        with tab3: # Rozpad entalpie (NOVÉ)
            contrib_df = pd.DataFrame(res['contributions'])
            if not contrib_df.empty:
                bar_chart = alt.Chart(contrib_df).mark_bar().encode(
                    y=alt.Y('Pair', sort='-x'),
                    x='Contribution',
                    color=alt.condition(alt.datum.Contribution < 0, alt.value('blue'), alt.value('red')),
                    tooltip=['Pair', 'H_ij', 'Contribution']
                ).properties(title="Které prvky se 'mají rády'? (Modrá = Přitažlivost)")
                st.altair_chart(bar_chart, use_container_width=True)

        # --- EXPORT ---
        st.divider()
        doc = Document()
        doc.add_heading(f"Report: {formula}", 0)
        doc.add_paragraph(f"VEC: {res['vec']:.2f}")
        for i in insights: doc.add_paragraph(i)
        
        f = io.BytesIO()
        doc.save(f)
        f.seek(0)
        
        st.download_button("📄 Stáhnout DOCX Report", f, file_name=f"{formula}_report.docx")

if __name__ == "__main__":
    main()
