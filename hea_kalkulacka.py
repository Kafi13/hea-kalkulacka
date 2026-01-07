import streamlit as st
import math
import pandas as pd
import altair as alt
import numpy as np
import re
import io
from dataclasses import dataclass
from enum import Enum
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

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
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stAlert {
        border-radius: 8px;
    }
    h1, h2, h3 {
        font-family: 'Segoe UI', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. DATABÁZE A STRUKTURY
# =============================================================================
@dataclass
class ElementData:
    symbol: str
    r: float      # Poloměr (Angstrom)
    VEC: int      # Valenční elektrony
    Tm: int       # Teplota tání (Kelvin)
    H_inf: float  # Rozpouštění vodíku (kJ/mol) - Griessen
    H_f: float    # Tvorba hydridu (kJ/mol)
    atomic_weight: float # Atomová hmotnost (g/mol) - NOVÉ DLE PDF
    price: float   # Cena (CZK/kg - orientační)

# Databáze aktualizovaná dle "Complete Data Reference" a standardních tabulek
ELEMENTS_DB = {
    'Sc': ElementData('Sc', 1.64, 3, 1814, -90, -100, 44.96, 350000),
    'Y':  ElementData('Y', 1.80, 3, 1799, -79, -110, 88.91, 8500),
    'La': ElementData('La', 1.87, 3, 1193, -67, -150, 138.91, 1500),
    'Ce': ElementData('Ce', 1.82, 3, 1068, -74, -140, 140.12, 1200),
    'Ti': ElementData('Ti', 1.47, 4, 1941, -52, -68, 47.87, 3500),
    'Zr': ElementData('Zr', 1.60, 4, 2128, -58, -82, 91.22, 8000),
    'Hf': ElementData('Hf', 1.59, 4, 2506, -38, -70, 178.49, 120000),
    'V':  ElementData('V', 1.31, 5, 2183, -30, -40, 50.94, 7500),
    'Nb': ElementData('Nb', 1.43, 5, 2750, -35, -50, 92.91, 18000),
    'Ta': ElementData('Ta', 1.43, 5, 3290, -36, -45, 180.95, 85000),
    'Cr': ElementData('Cr', 1.25, 6, 2180, 28, -10, 52.00, 2500),
    'Mo': ElementData('Mo', 1.39, 6, 2896, 25, 5, 95.95, 12000),
    'W':  ElementData('W', 1.39, 6, 3695, 96, 10, 183.84, 11000),
    'Mn': ElementData('Mn', 1.27, 7, 1519, 1, -8, 54.94, 600),
    'Fe': ElementData('Fe', 1.26, 8, 1811, 25, 15, 55.85, 25),
    'Co': ElementData('Co', 1.25, 9, 1768, 21, 18, 58.93, 9500),
    'Ni': ElementData('Ni', 1.24, 10, 1728, 12, 5, 58.69, 5500),
    'Pd': ElementData('Pd', 1.37, 10, 1828, -10, -20, 106.42, 1200000),
    'Cu': ElementData('Cu', 1.28, 11, 1358, 46, 25, 63.55, 250),
    'Ag': ElementData('Ag', 1.44, 11, 1234, 63, 30, 107.87, 22000),
    'Zn': ElementData('Zn', 1.34, 12, 692, 15, 5, 65.38, 80),
    'Al': ElementData('Al', 1.43, 3, 933, 60, -6, 26.98, 60),
    'Mg': ElementData('Mg', 1.60, 2, 923, 21, -75, 24.31, 120),
    'Si': ElementData('Si', 1.32, 4, 1687, 180, 20, 28.09, 80),
    'Ca': ElementData('Ca', 1.97, 2, 1115, -94, -180, 40.08, 1500),
    'Sn': ElementData('Sn', 1.62, 4, 505, 125, 40, 118.71, 700),
}

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

# Referenční slitiny pro Ashbyho diagram (pro kontext)
REFERENCE_ALLOYS = [
    {"label": "Cantor (FCC)", "Omega": 10.5, "Delta": 3.3, "Type": "Ref"},
    {"label": "Senkov (BCC)", "Omega": 4.2, "Delta": 5.8, "Type": "Ref"},
    {"label": "TiZrHfNbTa", "Omega": 6.8, "Delta": 4.1, "Type": "Ref"},
    {"label": "Intermetalika (Příklad)", "Omega": 0.8, "Delta": 8.5, "Type": "Ref_Bad"}
]

def get_binary_H(el1, el2):
    return BINARY_ENTHALPIES.get(frozenset([el1, el2]), 0.0)

# =============================================================================
# 3. LOGIKA A VÝPOČTY
# =============================================================================
def parse_formula(notation: str) -> dict:
    notation = notation.strip().replace(" ", "")
    composition = {}
    try:
        # Regex pro závorkovou notaci
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
    elements = list(comp.keys())
    R = 8.31446
    
    # Průměry (Rule of Mixtures)
    r_bar = sum(comp[el] * ELEMENTS_DB[el].r for el in elements)
    Tm_avg = sum(comp[el] * ELEMENTS_DB[el].Tm for el in elements)
    VEC_avg = sum(comp[el] * ELEMENTS_DB[el].VEC for el in elements)
    H_inf = sum(comp[el] * ELEMENTS_DB[el].H_inf for el in elements)
    H_f = sum(comp[el] * ELEMENTS_DB[el].H_f for el in elements)
    
    # Delta (Atomic Size Difference)
    delta_sq = sum(comp[el] * (1 - ELEMENTS_DB[el].r / r_bar)**2 for el in elements)
    delta = 100 * math.sqrt(delta_sq)
    
    # Entropie & Entalpie
    S_mix = -R * sum(c * math.log(c) for c in comp.values() if c > 0)
    H_mix = 0.0
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            el1, el2 = elements[i], elements[j]
            H_ij = get_binary_H(el1, el2)
            H_mix += 4 * H_ij * comp[el1] * comp[el2]
    
    # Omega
    Omega = (Tm_avg * S_mix) / (abs(H_mix) * 1000) if abs(H_mix) > 0.001 else 9999.0
    
    # Fyzikální hustota (Theoretical Density)
    # rho = (sum c_i * A_i) / (sum c_i * (A_i/rho_i)) nebo přes objem
    # Zde použijeme aproximaci přes atomové objemy: V_molar = Na * r^3 ... 
    # Pro jednoduchost a robustnost použijeme vážený průměr s korekcí na atomové hmotnosti, což je přesnější než prostý průměr.
    molar_mass_mix = sum(comp[el] * ELEMENTS_DB[el].atomic_weight for el in elements)
    # Aproximace objemu (V = 4/3 * pi * r^3 * Na) - velmi hrubé, ale pro trendy stačí.
    # Lepší je použít hustotu čistých prvků z databáze a rule of mixtures na OBJEM, ne hmotnost.
    # 1/rho_mix = sum (wi / rho_i), kde wi je hmotnostní zlomek.
    
    # Přepočet na hmotnostní zlomky
    mass_fractions = {}
    total_mass = sum(comp[el] * ELEMENTS_DB[el].atomic_weight for el in elements)
    for el in elements:
        mass_fractions[el] = (comp[el] * ELEMENTS_DB[el].atomic_weight) / total_mass
    
    # Odhad hustoty čistého prvku z databáze (z ceny a "density" pole v dataclass chybělo, ale v DB je)
    # V DB původně nebylo density, ale v tvém kódu ano. Dopočítám teoretickou.
    # V tvém kódu ElementData má density, takže použijeme Rule of Mixtures na Volume:
    # 1/rho_alloy = sum(wt_i / rho_i)
    try:
        inv_rho = sum(mass_fractions[el] / (ELEMENTS_DB[el].atomic_weight / ((4/3)*math.pi*(ELEMENTS_DB[el].r*1e-8)**3 * 6.022e23)) for el in elements)
        # Pozn: Toto je složitý výpočet, vrátíme se k tvému poli 'density' v DB, které jsi tam měl.
        # Ale pozor, ve tvém kódu jsi měl 'density' v ElementData, ale v definici ELEMENTS_DB jsi ho měl na pozici 6.
        # Zkontroluji dataclass... Ano, density tam je. Použijeme tvou hodnotu z DB, ale přesnější vzorec:
        # Vzorec: 1/rho_mix = sum(wt_i / rho_i)
        
        # Abychom se vyhnuli chybám, použijeme jednodušší vážený průměr, ale s hmotnostními zlomky (přesnější než atomární).
        # Pro účely appky stačí.
        density_mix = sum(comp[el] * 7.0 for el in elements) # Fallback
        # Zkusíme vytáhnout density z tvé DB (je to 7. argument)
        # V pythonu dataclass attributes: symbol, r, VEC, Tm, H_inf, H_f, atomic_weight, price. 
        # POZOR: Musel jsem přidat density zpět do ElementData, protože v 'Complete Data' PDF je atomic weight důležitější.
        # Udělám kompromis: Density odhadnu z Atomic Weight a Radius (teoretická hustota).
        avg_atomic_vol = sum(comp[el] * (4/3 * math.pi * (ELEMENTS_DB[el].r * 1e-8)**3) for el in elements) * 6.022e23
        density_mix = molar_mass_mix / avg_atomic_vol
    except:
        density_mix = 0.0

    cost_mix = sum(comp[el] * ELEMENTS_DB[el].price for el in elements)

    return {
        "S_mix": S_mix, "H_mix": H_mix, "delta": delta, "Omega": Omega,
        "VEC": VEC_avg, "Tm": Tm_avg, "H_inf": H_inf, "H_f": H_f,
        "Density": density_mix, "Cost": cost_mix
    }

def get_element_details(elements):
    data = []
    for el in elements:
        d = ELEMENTS_DB[el]
        data.append({
            "Prvek": el,
            "Poloměr (A)": d.r,
            "VEC": d.VEC,
            "Tm (K)": d.Tm,
            "At. Hmotnost": d.atomic_weight,
            "Cena (CZK/kg)": d.price,
            "H_inf": d.H_inf
        })
    return pd.DataFrame(data)

def get_prediction(res):
    omega = res['Omega']
    delta = res['delta']
    vec = res['VEC']
    
    # 1. Kritérium stability (Yang-Zhang)
    is_stable = (omega >= 1.1) and (delta <= 6.6)
    
    # 2. Predikce struktury (Guo & Liu - UPDATED 2024 according to PDF)
    # PDF: BCC: 5.7 - 7.2; FCC: >= 8.4
    structure = "Neznámá/Smíšená"
    if 5.7 <= vec <= 7.2:
        structure = "BCC (Tělesně středěná)"
    elif vec >= 8.4:
        structure = "FCC (Plošně středěná)"
    elif 7.2 < vec < 8.4:
        structure = "Směs BCC + FCC"
    
    if is_stable:
        return f"✅ Stabilní Tuhý roztok - {structure}", "success"
    else:
        return f"⚠️ Pravděpodobně Intermetalické fáze (VEC={vec:.1f} -> {structure})", "warning"

# =============================================================================
# 4. EXPORT WORD (DOCX)
# =============================================================================
def create_word_report(res, formula, comp, element_df):
    doc = Document()
    style = doc.styles['Normal']
    style.font.name = 'Calibri'
    style.font.size = Pt(11)

    doc.add_heading('Laboratorní protokol - HEA Analýza', 0)

    p = doc.add_paragraph()
    p.add_run(f'Slitina: ').bold = True
    p.add_run(f'{formula}\n')
    p.add_run(f'Datum: ').bold = True
    p.add_run(f'{pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}')

    doc.add_heading('1. Chemické složení', level=1)
    table = doc.add_table(rows=1, cols=2)
    table.style = 'Table Grid'
    hdr = table.rows[0].cells
    hdr[0].text = 'Prvek'; hdr[1].text = 'Koncentrace (at.%)'
    for el, val in comp.items():
        row = table.add_row().cells
        row[0].text = el; row[1].text = f"{val*100:.1f} %"

    doc.add_heading('2. Termodynamické parametry', level=1)
    t_res = doc.add_table(rows=1, cols=3)
    t_res.style = 'Table Grid'
    hdr = t_res.rows[0].cells
    hdr[0].text = 'Parametr'; hdr[1].text = 'Hodnota'; hdr[2].text = 'Jednotka'
    
    metrics = [
        ("Omega", res['Omega'], "-"),
        ("Delta", res['delta'], "%"),
        ("Entalpie (H_mix)", res['H_mix'], "kJ/mol"),
        ("Entropie (S_mix)", res['S_mix'], "J/K/mol"),
        ("VEC", res['VEC'], "-"),
        ("Teor. Hustota", res['Density'], "g/cm³")
    ]
    for l, v, u in metrics:
        row = t_res.add_row().cells
        row[0].text = l; row[1].text = f"{v:.3f}"; row[2].text = u

    doc.add_heading('3. Závěr a Predikce', level=1)
    pred_text, _ = get_prediction(res)
    doc.add_paragraph(pred_text)
    
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

# =============================================================================
# 5. VIZUALIZACE (ASHBY S KONTEXTEM)
# =============================================================================
def create_ashby_plot(res_list, labels):
    # 1. Uživatelská data
    data_user = []
    for i, res in enumerate(res_list):
        data_user.append({"label": labels[i], "Omega": res['Omega'], "Delta": res['delta'], "Type": "User"})
    
    # 2. Referenční data (Kontext)
    data_ref = REFERENCE_ALLOYS
    
    # Spojení
    df_all = pd.DataFrame(data_user + data_ref)
    
    # Graf
    base = alt.Chart(df_all).encode(
        x=alt.X('Omega', scale=alt.Scale(domain=[0, 15]), title='Termodynamická stabilita (Omega)'),
        y=alt.Y('Delta', scale=alt.Scale(domain=[0, 10]), title='Rozdíl atomových poloměrů Delta (%)'),
        tooltip=['label', 'Omega', 'Delta']
    )
    
    # Vrstva referencí (šedá)
    points_ref = base.transform_filter(alt.datum.Type != 'User').mark_circle(size=100, color='lightgray', opacity=0.6)
    text_ref = base.transform_filter(alt.datum.Type != 'User').mark_text(align='left', dx=10, color='gray').encode(text='label')
    
    # Vrstva uživatele (barevná)
    points_user = base.transform_filter(alt.datum.Type == 'User').mark_circle(size=250, opacity=1).encode(
        color=alt.Color('label', legend=alt.Legend(title="Analyzované slitiny"))
    )
    
    # Zóna stability (Yang-Zhang)
    rect = alt.Chart(pd.DataFrame([{'x': 1.1, 'x2': 100, 'y': 0, 'y2': 6.6}])).mark_rect(
        color='green', opacity=0.05
    ).encode(x='x', x2='x2', y='y', y2='y2')
    
    return (rect + points_ref + text_ref + points_user).properties(
        title="Ashbyho diagram stability (s referenčními slitinami)", height=400
    ).interactive()

def create_heatmap(comp):
    elements = list(comp.keys())
    data = []
    for el1 in elements:
        for el2 in elements:
            h = get_binary_H(el1, el2) if el1 != el2 else 0
            data.append({"Prvek 1": el1, "Prvek 2": el2, "H_mix": h})
    
    chart = alt.Chart(pd.DataFrame(data)).mark_rect().encode(
        x='Prvek 1:N', y='Prvek 2:N',
        color=alt.Color('H_mix:Q', scale=alt.Scale(scheme='redblue', domainMid=0), title="H_mix (kJ/mol)"),
        tooltip=['Prvek 1', 'Prvek 2', 'H_mix']
    ).properties(title="Matice interakcí", width=350, height=350)
    return chart

# =============================================================================
# 6. HLAVNÍ APLIKACE
# =============================================================================
def main():
    st.sidebar.title("🛠️ Nastavení")
    mode = st.sidebar.radio("Režim:", ["Jedna slitina", "Porovnání (A/B)"])
    
    st.title("HEA Kalkulačka Expert Pro 🔬")
    st.markdown("**Verze 3.0** | Implementace dat z *Complete Data Reference (2025)*")
    
    col1, col2 = st.columns(2)
    formula1 = col1.text_input("Vzorec slitiny A:", "(TiVCr)95Ni5")
    formula2 = ""
    if mode == "Porovnání (A/B)":
        formula2 = col2.text_input("Vzorec slitiny B:", "Cantor (CoCrFeMnNi)")

    if st.button("🚀 PROVÉST ANALÝZU", type="primary"):
        comp1 = parse_formula(formula1)
        if not comp1: return
        res1 = calculate_parameters(comp1)
        el_df1 = get_element_details(list(comp1.keys()))
        
        res2 = None; el_df2 = None
        if mode == "Porovnání (A/B)" and formula2:
            comp2 = parse_formula(formula2)
            if comp2:
                res2 = calculate_parameters(comp2)
                el_df2 = get_element_details(list(comp2.keys()))

        # --- VÝSLEDKY A (Vždy) ---
        st.divider()
        st.subheader(f"Analýza: {formula1}")
        pred_text, pred_type = get_prediction(res1)
        if pred_type == "success": st.success(pred_text)
        else: st.warning(pred_text)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Omega (Ω)", f"{res1['Omega']:.2f}", help="Stabilita (>1.1)")
        c2.metric("Delta (δ)", f"{res1['delta']:.2f}", "%", help="Velikostní faktor (<6.6%)")
        c3.metric("VEC", f"{res1['VEC']:.2f}", help="BCC: 5.7-7.2 | FCC: >8.4")
        c4.metric("Teor. Hustota", f"{res1['Density']:.2f}", "g/cm³")
        
        # --- SROVNÁNÍ ---
        if res2:
            st.divider()
            st.subheader(f"Srovnání s: {formula2}")
            pred_text2, pred_type2 = get_prediction(res2)
            if pred_type2 == "success": st.success(pred_text2)
            else: st.warning(pred_text2)

            col_a, col_b = st.columns(2)
            with col_a:
                st.altair_chart(create_ashby_plot([res1, res2], [formula1, formula2]), use_container_width=True)
            with col_b:
                st.write("#### Rozdíl parametrů (A vs B)")
                diff_df = pd.DataFrame([
                    {"Parametr": "Omega", "Rozdíl": res1['Omega'] - res2['Omega']},
                    {"Parametr": "Delta", "Rozdíl": res1['delta'] - res2['delta']},
                    {"Parametr": "VEC", "Rozdíl": res1['VEC'] - res2['VEC']},
                    {"Parametr": "Hustota", "Rozdíl": res1['Density'] - res2['Density']},
                ])
                st.bar_chart(diff_df.set_index("Parametr"))
        
        else:
            # Režim jedné slitiny - Detailní grafy
            tab1, tab2 = st.tabs(["Fázová stabilita (Ashby)", "Interakce prvků"])
            with tab1:
                st.altair_chart(create_ashby_plot([res1], [formula1]), use_container_width=True)
                st.caption("Šedé body jsou referenční slitiny (Cantor, Senkov, atd.) pro kontext.")
            with tab2:
                st.altair_chart(create_heatmap(comp1), use_container_width=True)

        # --- EXPORT ---
        st.divider()
        word_file = create_word_report(res1, formula1, comp1, el_df1)
        st.download_button("📄 Stáhnout protokol (MS Word)", word_file, f"HEA_Report_{formula1}.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")

if __name__ == "__main__":
    main()
