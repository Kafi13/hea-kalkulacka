import streamlit as st
import math
import pandas as pd
import altair as alt
import numpy as np
import re
from dataclasses import dataclass
from enum import Enum
from fpdf import FPDF
import matplotlib.pyplot as plt
import io

# =============================================================================
# 1. KONFIGURACE APLIKACE
# =============================================================================
st.set_page_config(
    page_title="HEA Kalkulačka Expert Pro",
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
    .reportview-container .main .block-container{
        padding-top: 2rem;
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
    density: float # Hustota (g/cm3)
    price: float   # Cena (CZK/kg - orientační odhad)

# Databáze prvků (Zdroje: Griessen, Driessen, Takeuchi + ceny/hustoty)
# Ceny jsou hrubé odhady pro akademické účely
ELEMENTS_DB = {
    'Sc': ElementData('Sc', 1.64, 3, 1814, -90, -100, 2.99, 350000),
    'Y':  ElementData('Y', 1.80, 3, 1799, -79, -110, 4.47, 8500),
    'La': ElementData('La', 1.87, 3, 1193, -67, -150, 6.16, 1500),
    'Ce': ElementData('Ce', 1.82, 3, 1068, -74, -140, 6.77, 1200),
    'Ti': ElementData('Ti', 1.47, 4, 1941, -52, -68, 4.51, 3500),
    'Zr': ElementData('Zr', 1.60, 4, 2128, -58, -82, 6.51, 8000),
    'Hf': ElementData('Hf', 1.59, 4, 2506, -38, -70, 13.31, 120000),
    'V':  ElementData('V', 1.31, 5, 2183, -30, -40, 5.96, 7500),
    'Nb': ElementData('Nb', 1.43, 5, 2750, -35, -50, 8.57, 18000),
    'Ta': ElementData('Ta', 1.43, 5, 3290, -36, -45, 16.65, 85000),
    'Cr': ElementData('Cr', 1.25, 6, 2180, 28, -10, 7.15, 2500),
    'Mo': ElementData('Mo', 1.39, 6, 2896, 25, 5, 10.22, 12000),
    'W':  ElementData('W', 1.39, 6, 3695, 96, 10, 19.25, 11000),
    'Mn': ElementData('Mn', 1.27, 7, 1519, 1, -8, 7.43, 600),
    'Fe': ElementData('Fe', 1.26, 8, 1811, 25, 15, 7.87, 25),
    'Co': ElementData('Co', 1.25, 9, 1768, 21, 18, 8.86, 9500),
    'Ni': ElementData('Ni', 1.24, 10, 1728, 12, 5, 8.91, 5500),
    'Pd': ElementData('Pd', 1.37, 10, 1828, -10, -20, 12.02, 1200000),
    'Cu': ElementData('Cu', 1.28, 11, 1358, 46, 25, 8.96, 250),
    'Ag': ElementData('Ag', 1.44, 11, 1234, 63, 30, 10.50, 22000),
    'Zn': ElementData('Zn', 1.34, 12, 692, 15, 5, 7.13, 80),
    'Al': ElementData('Al', 1.43, 3, 933, 60, -6, 2.70, 60),
    'Mg': ElementData('Mg', 1.60, 2, 923, 21, -75, 1.74, 120),
    'Si': ElementData('Si', 1.32, 4, 1687, 180, 20, 2.33, 80),
    'Ca': ElementData('Ca', 1.97, 2, 1115, -94, -180, 1.54, 1500),
    'Sn': ElementData('Sn', 1.62, 4, 505, 125, 40, 7.29, 700),
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

        for el in composition:
            if el not in ELEMENTS_DB:
                st.error(f"Chyba: Prvek '{el}' není v databázi.")
                return None
        
        total = sum(composition.values())
        return {k: v/total for k, v in composition.items()}

    except:
        return None

def calculate_parameters(comp):
    """Vypočítá všechny termodynamické parametry, hustotu a cenu."""
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
    
    # Entropie a Entalpie
    S_mix = -R * sum(c * math.log(c) for c in comp.values() if c > 0)
    H_mix = 0.0
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            el1, el2 = elements[i], elements[j]
            H_ij = get_binary_H(el1, el2)
            H_mix += 4 * H_ij * comp[el1] * comp[el2]
    
    Omega = (Tm_avg * S_mix) / (abs(H_mix) * 1000) if abs(H_mix) > 0.001 else 9999.0
    
    # Ekonomika a Hustota
    # Odhad hmotnostních zlomků pro přesnější hustotu/cenu
    total_mass = sum(comp[el] * 1 for el in elements) # Zjednodušeno (atomová hmotnost by byla lepší, ale stačí pro odhad) 
    # Pro lepší přesnost potřebujeme atomové hmotnosti. Přidáme zástupné konstanty nebo použijeme hustotu jako proxy.
    # Použijeme jednoduchý Rule of Mixtures na hustotu (nejběžnější v HEA papers pro odhad) a cenu.
    density_mix = sum(comp[el] * ELEMENTS_DB[el].density for el in elements)
    cost_mix = sum(comp[el] * ELEMENTS_DB[el].price for el in elements)

    return {
        "S_mix": S_mix, "H_mix": H_mix, "delta": delta, "Omega": Omega,
        "VEC": VEC_avg, "Tm": Tm_avg, "H_inf": H_inf, "H_f": H_f,
        "Density": density_mix, "Cost": cost_mix
    }

def get_prediction(res):
    is_ss = (res['Omega'] >= 1.1) and (res['delta'] <= 6.6)
    if is_ss:
        if res['VEC'] < 6.87: struct = "BCC"
        elif res['VEC'] >= 8.0: struct = "FCC"
        else: struct = "BCC + FCC"
        return f"✅ Tuhý roztok ({struct})", "success"
    else:
        return "⚠️ Intermetalické fáze", "warning"

# =============================================================================
# 4. EXPORT A PDF
# =============================================================================

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Laboratorní protokol - HEA Analýza', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Strana {self.page_no()}', 0, 0, 'C')

def create_pdf(res, formula, comp):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # 1. Základní info
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, f"Slitina: {formula}", 0, 1)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 10, f"Datum: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", 0, 1)
    pdf.ln(5)

    # 2. Složení
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Chemické složení:", 0, 1)
    pdf.set_font("Arial", size=11)
    comp_str = ", ".join([f"{el}: {val*100:.1f}%" for el, val in comp.items()])
    pdf.multi_cell(0, 10, comp_str)
    pdf.ln(5)

    # 3. Výsledky
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Vypočtené parametry:", 0, 1)
    pdf.set_font("Arial", size=11)
    
    metrics = [
        f"Entropie (S_mix): {res['S_mix']:.3f} J/K/mol",
        f"Entalpie (H_mix): {res['H_mix']:.3f} kJ/mol",
        f"Delta (size mismatch): {res['delta']:.2f} %",
        f"Omega (stabilita): {res['Omega']:.2f}",
        f"Valenční elektrony (VEC): {res['VEC']:.2f}",
        f"Teplota tání (Tm): {res['Tm']:.0f} K",
        f"Hustota (odhad): {res['Density']:.2f} g/cm3",
        f"Cena materiálu: {res['Cost']:.0f} CZK/kg"
    ]
    
    for m in metrics:
        pdf.cell(0, 8, m, 0, 1)

    pdf.ln(5)
    pred_text, _ = get_prediction(res)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"Závěr predikce: {pred_text}", 0, 1)
    
    # Poznámka k PDF
    pdf.set_font("Arial", 'I', 9)
    pdf.ln(10)
    pdf.multi_cell(0, 5, "Pozn: Jedná se o teoretický výpočet modely Miedema, Yang-Zhang a Guo.")

    return pdf.output(dest='S').encode('latin-1', 'replace')

# =============================================================================
# 5. VIZUALIZACE (ASHBY, RADAR)
# =============================================================================
def create_ashby_plot(res_list, labels):
    data = []
    for i, res in enumerate(res_list):
        data.append({"label": labels[i], "Omega": res['Omega'], "Delta": res['delta']})
    
    source = pd.DataFrame(data)
    
    points = alt.Chart(source).mark_circle(size=200).encode(
        x=alt.X('Omega', scale=alt.Scale(domain=[0, max(2.0, max(d['Omega'] for d in data)+0.5)])),
        y=alt.Y('Delta', scale=alt.Scale(domain=[0, max(10, max(d['Omega'] for d in data)+2)]), title='Delta (%)'),
        color='label',
        tooltip=['label', 'Omega', 'Delta']
    )
    
    rect = alt.Chart(pd.DataFrame([{'x': 1.1, 'x2': 100, 'y': 0, 'y2': 6.6}])).mark_rect(
        color='green', opacity=0.1
    ).encode(x='x', x2='x2', y='y', y2='y2')
    
    return (rect + points).properties(title="Ashbyho diagram stability", height=350)

def create_radar_comparison(res1, label1, res2=None, label2=None):
    # Normalizace pro radar chart (aby se vešly do grafu)
    # Omega/5, Delta/10, VEC/12
    
    def normalize(r):
        return [
            {"key": "Omega (x5)", "value": min(r['Omega'], 5.0), "max": 5.0},
            {"key": "Delta (x10)", "value": min(r['delta'], 10.0), "max": 10.0},
            {"key": "VEC (x12)", "value": min(r['VEC'], 12.0), "max": 12.0},
            {"key": "Tm (x4000)", "value": min(r['Tm'], 4000)/400, "max": 10.0}, # Scaled down
            {"key": "Cena (rel)", "value": min(math.log10(r['Cost']+1), 6), "max": 6.0} # Log scale
        ]

    data = []
    for item in normalize(res1):
        item['category'] = label1
        data.append(item)
    
    if res2:
        for item in normalize(res2):
            item['category'] = label2
            data.append(item)
            
    df = pd.DataFrame(data)
    
    chart = alt.Chart(df).mark_line(point=True).encode(
        x='key',
        y='value',
        color='category'
    ).properties(title="Porovnání vlastností (Normalizováno)")
    return chart

# =============================================================================
# 6. HLAVNÍ APLIKACE
# =============================================================================
def main():
    st.sidebar.title("Nastavení")
    mode = st.sidebar.radio("Režim:", ["Jedna slitina", "Porovnání (A/B)"])
    
    st.title("HEA Kalkulačka Expert Pro 🔬")
    st.caption("Verze 2.1 | Včetně ekonomické analýzy a PDF reportů")
    
    col1, col2 = st.columns(2)
    
    formula1 = col1.text_input("Vzorec slitiny A:", "(TiVCr)95Ni5")
    formula2 = ""
    if mode == "Porovnání (A/B)":
        formula2 = col2.text_input("Vzorec slitiny B (pro srovnání):", "Ti20V20Cr20Ni20Al20")

    if st.button("🚀 PROVÉST ANALÝZU", type="primary"):
        comp1 = parse_formula(formula1)
        if not comp1: return
        res1 = calculate_parameters(comp1)
        
        res2 = None
        if mode == "Porovnání (A/B)" and formula2:
            comp2 = parse_formula(formula2)
            if comp2:
                res2 = calculate_parameters(comp2)

        # --- Výsledky Slitina A ---
        st.divider()
        st.subheader(f"Analýza: {formula1}")
        pred_text, pred_type = get_prediction(res1)
        if pred_type == "success": st.success(pred_text)
        else: st.warning(pred_text)
        
        # Metriky A
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Omega (Ω)", f"{res1['Omega']:.2f}")
        c2.metric("Delta (δ)", f"{res1['delta']:.2f}", "%")
        c3.metric("Hustota", f"{res1['Density']:.2f}", "g/cm³")
        c4.metric("Cena", f"{res1['Cost']:.0f}", "CZK/kg")

        # --- Výsledky Slitina B (pokud existuje) ---
        if res2:
            st.divider()
            st.subheader(f"Srovnání s: {formula2}")
            pred_text2, pred_type2 = get_prediction(res2)
            if pred_type2 == "success": st.success(pred_text2)
            else: st.warning(pred_text2)
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Omega (Ω)", f"{res2['Omega']:.2f}", f"{res2['Omega']-res1['Omega']:.2f}")
            c2.metric("Delta (δ)", f"{res2['delta']:.2f}", f"{res2['delta']-res1['delta']:.2f} %", delta_color="inverse")
            c3.metric("Hustota", f"{res2['Density']:.2f}", f"{res2['Density']-res1['Density']:.2f} g/cm³")
            c4.metric("Cena", f"{res2['Cost']:.0f}", f"{res2['Cost']-res1['Cost']:.0f} CZK", delta_color="inverse")
            
            # Srovnávací grafy
            st.write("### ⚔️ Grafické srovnání")
            g1, g2 = st.columns(2)
            with g1:
                st.altair_chart(create_ashby_plot([res1, res2], [formula1, formula2]), use_container_width=True)
            with g2:
                st.altair_chart(create_radar_comparison(res1, formula1, res2, formula2), use_container_width=True)

        else:
            # Grafy pouze pro A
            res_list = [res1]
            labels = [formula1]
            st.altair_chart(create_ashby_plot(res_list, labels), use_container_width=True)
            
        # --- PDF Report ---
        st.divider()
        st.write("### 📑 Export")
        pdf_bytes = create_pdf(res1, formula1, comp1)
        st.download_button(
            label="📄 Stáhnout Laboratorní Protokol (PDF)",
            data=pdf_bytes,
            file_name=f"HEA_Report_{formula1}.pdf",
            mime="application/pdf"
        )

if __name__ == "__main__":
    main()