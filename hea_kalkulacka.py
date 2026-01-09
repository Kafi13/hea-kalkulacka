import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re
import io
from dataclasses import dataclass
from docx import Document
from docx.shared import Inches, Pt, RGBColor

# =============================================================================
# 1. KONFIGURACE APLIKACE
# =============================================================================
st.set_page_config(
    page_title="HEA Ultimate Calculator",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS pro vědecký vzhled
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #dce0e6;
        padding: 10px;
        border-radius: 5px;
    }
    h1 { color: #1f77b4; }
    h3 { color: #2c3e50; border-bottom: 2px solid #1f77b4; padding-bottom: 5px; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. DATABÁZE PRVKŮ 
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
    v23: float    # Miedema Molar Volume term (cm^2)
    h_inf: float  # Entalpie rozpouštění H (kJ/mol H)
    h_f: float    # Entalpie tvorby hydridu (kJ/mol H2)

# [cite_start]Data zkompletována z PDF "Complete Data Reference" [cite: 2313]
ELEMENTS_DB = {
    # Lehké kovy
    'Al': ElementData('Al', 'Hliník', 1.43, 3, 933, 26.98, 4.20, 1.39, 4.64, 60, -6),
    'Mg': ElementData('Mg', 'Hořčík', 1.60, 2, 923, 24.31, 3.45, 1.17, 5.81, 21, -75),
    'Si': ElementData('Si', 'Křemík', 1.17, 4, 1687, 28.09, 4.70, 1.50, 4.20, 180, 0), 
    
    # 4. Perioda (Transition Metals)
    'Ti': ElementData('Ti', 'Titan', 1.47, 4, 1941, 47.87, 3.65, 1.47, 4.12, -52, -137),
    'V':  ElementData('V', 'Vanad', 1.35, 5, 2183, 50.94, 4.25, 1.64, 4.12, -30, -47),
    'Cr': ElementData('Cr', 'Chrom', 1.29, 6, 2180, 52.00, 4.65, 1.73, 3.74, 28, 6),
    'Mn': ElementData('Mn', 'Mangan', 1.37, 7, 1519, 54.94, 4.45, 1.61, 3.78, 1, 21),
    'Fe': ElementData('Fe', 'Železo', 1.26, 8, 1811, 55.85, 4.93, 1.77, 3.69, 25, 20),
    'Co': ElementData('Co', 'Kobalt', 1.25, 9, 1768, 58.93, 5.10, 1.75, 3.55, 21, 31),
    'Ni': ElementData('Ni', 'Nikl', 1.25, 10, 1728, 58.69, 5.20, 1.75, 3.52, 12, 15),
    'Cu': ElementData('Cu', 'Měď', 1.28, 11, 1358, 63.55, 4.55, 1.47, 3.70, 46, 45),
    
    # 5. Perioda
    'Zr': ElementData('Zr', 'Zirkonium', 1.60, 4, 2128, 91.22, 3.45, 1.41, 5.81, -58, -164),
    'Nb': ElementData('Nb', 'Niob', 1.47, 5, 2750, 92.91, 4.05, 1.62, 4.89, -35, -40),
    'Mo': ElementData('Mo', 'Molybden', 1.40, 6, 2896, 95.95, 4.65, 1.77, 4.45, 25, 92),
    'Pd': ElementData('Pd', 'Palladium', 1.37, 10, 1828, 106.4, 5.45, 1.67, 3.90, -10, -20),
    
    # 6. Perioda
    'Hf': ElementData('Hf', 'Hafnium', 1.59, 4, 2506, 178.5, 3.55, 1.43, 5.65, -38, -130),
    'Ta': ElementData('Ta', 'Tantal', 1.47, 5, 3290, 180.9, 4.05, 1.63, 4.89, -36, -78),
    'W':  ElementData('W', 'Wolfram', 1.41, 6, 3695, 183.8, 4.80, 1.81, 4.50, 96, 74),
    
    # Vzácné zeminy
    'La': ElementData('La', 'Lanthan', 1.88, 3, 1193, 138.9, 3.05, 1.18, 6.60, -67, -206),
    'Ce': ElementData('Ce', 'Cer', 1.82, 4, 1068, 140.1, 3.15, 1.19, 6.30, -74, -200),
}

# =============================================================================
# 3. MIEDEMŮV MODEL
# =============================================================================
def calculate_miedema_enthalpy(el1_sym, el2_sym):
    """
    Vypočítá binární entalpii míšení pomocí fyzikálního modelu.
    """
    e1 = ELEMENTS_DB.get(el1_sym)
    e2 = ELEMENTS_DB.get(el2_sym)
    
    if not e1 or not e2 or el1_sym == el2_sym:
        return 0.0

    # Konstanty pro slitiny přechodných kovů (standardní Miedema)
    P = 14.1
    Q = 9.4 
    
    # Rozdíly parametrů
    d_phi = e1.phi - e2.phi
    d_nws_direct = e1.nws - e2.nws

    # Výpočet chemické části (přitažlivá - exotermická)
    term_phi = -P * (d_phi**2)
    
    # Výpočet elastické části (odpudivá - endotermická)
    term_nws = Q * (d_nws_direct**2)
    
    # Empirický scaling faktor pro kJ/mol
    enthalpy = (term_phi + term_nws) * 5.0 
    
    return enthalpy

# =============================================================================
# 4. LOGIKA PARSOVÁNÍ
# =============================================================================
def parse_composition(formula):
    """
    Parsuje vzorce typu (TiZr)80Ni20 nebo Ti20V20Cr20.
    """
    formula = formula.strip().replace(" ", "")
    composition = {}
    
    try:
        # 1. Zpracování závorek: (Base)Percent
        bracket_match = re.match(r'\(([A-Za-z]+)\)(\d+(?:\.\d+)?)', formula)
        remaining_formula = formula
        
        if bracket_match:
            base_elements_str = bracket_match.group(1)
            base_percent = float(bracket_match.group(2))
            
            # Najdi prvky v závorce
            base_elements = re.findall(r'[A-Z][a-z]?', base_elements_str)
            if base_elements:
                share = base_percent / len(base_elements)
                for el in base_elements:
                    composition[el] = share
            
            remaining_formula = formula[bracket_match.end():]

        # 2. Zpracování zbytku
        matches = re.findall(r'([A-Z][a-z]?)(\d+(?:\.\d+)?)?', remaining_formula)
        for el, qty in matches:
            if not el: continue
            amount = float(qty) if qty else 1.0 
            composition[el] = amount

        # 3. Validace a Normalizace
        total = sum(composition.values())
        if total == 0: return None
        
        normalized = {k: v/total for k, v in composition.items()}
        
        # Check if elements exist in DB
        for el in normalized:
            if el not in ELEMENTS_DB:
                st.error(f"Prvek '{el}' není v databázi.")
                return None
                
        return normalized

    except Exception as e:
        st.error(f"Chyba při čtení vzorce: {e}")
        return None

# =============================================================================
# 5. VÝPOČTY
# =============================================================================
def calculate_hea_properties(comp):
    elements = list(comp.keys())
    R_GAS = 8.314
    
    # 1. Průměry (Rule of Mixtures)
    r_bar = sum(comp[el] * ELEMENTS_DB[el].r for el in elements)
    tm_avg = sum(comp[el] * ELEMENTS_DB[el].tm for el in elements)
    vec_avg = sum(comp[el] * ELEMENTS_DB[el].vec for el in elements)
    mass_avg = sum(comp[el] * ELEMENTS_DB[el].mass for el in elements)
    
    # 2. Termodynamika
    # Entropie (S_mix)
    s_mix = -R_GAS * sum(c * np.log(c) for c in comp.values() if c > 0)
    
    # Entalpie (H_mix) - Miedema Loop
    h_mix = 0.0
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            el_i, el_j = elements[i], elements[j]
            h_ij = calculate_miedema_enthalpy(el_i, el_j)
            h_mix += 4 * h_ij * comp[el_i] * comp[el_j] 
            
    # Atomová neshoda (Delta)
    delta_sq = sum(comp[el] * (1 - ELEMENTS_DB[el].r / r_bar)**2 for el in elements)
    delta = 100 * np.sqrt(delta_sq)
    
    # Omega Parameter
    omega = (tm_avg * s_mix) / (abs(h_mix) * 1000) if abs(h_mix) > 1e-4 else 100.0
    
    # 3. Vodík (Griessen-Driessen)
    h_inf_mix = sum(comp[el] * ELEMENTS_DB[el].h_inf for el in elements)
    h_f_mix = sum(comp[el] * ELEMENTS_DB[el].h_f for el in elements)
    
    return {
        "s_mix": s_mix, "h_mix": h_mix, "delta": delta, "omega": omega,
        "vec": vec_avg, "tm": tm_avg, "mass": mass_avg,
        "h_inf": h_inf_mix, "h_f": h_f_mix
    }

# =============================================================================
# 6. KLASIFIKACE 
# =============================================================================
def get_phase_prediction(props):
    """Predikce fáze podle VEC a termodynamiky"""
    vec = props['vec']
    delta = props['delta']
    omega = props['omega']
    
    status = []
    
    # Stabilita tuhého roztoku
    if omega >= 1.1 and delta <= 6.6:
        status.append("✅ Stabilní tuhý roztok")
    elif delta > 6.6:
        status.append("⚠️ Riziko Lavesových fází / Amorfní (δ > 6.6%)")
    else:
        status.append("❌ Vícefázový systém (Ω < 1.1)")
        
    # Typ mřížky (VEC)
    if vec < 6.87:
        status.append("🟦 Typ: BCC (Vhodné pro H2)")
    elif vec >= 8.0:
        status.append("🟧 Typ: FCC")
    else:
        status.append("🟪 Typ: BCC + FCC")
        
    return " | ".join(status)

def get_hydrogen_prediction(h_f):
    """Klasifikace podle entalpie hydridu"""
    if h_f < -40:
        return "🔥 Silný hydrid (Past/Trap) - Vysoká desorpční teplota", "error"
    elif -40 <= h_f <= -10:
        return "🔋 Optimální pro skladování (Reverzibilní RT)", "success"
    elif -10 < h_f <= 0:
        return "⚖️ Slabá vazba (Tuhý roztok)", "warning"
    else:
        return "🛡️ Vodíková bariéra (Endotermická)", "info"

# =============================================================================
# 7. VIZUALIZACE 
# =============================================================================
def plot_ashby(props, label):
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Zóny stability
    rect = patches.Rectangle((1.1, 0), 100, 6.6, linewidth=1, edgecolor='none', facecolor='#d4edda', alpha=0.5, label='Solid Solution Zone')
    ax.add_patch(rect)
    
    # Referenční body
    ax.scatter([10.8], [3.2], color='gray', alpha=0.5, label='Cantor Alloy')
    ax.scatter([12.5], [4.1], color='gray', alpha=0.5)
    
    # User point
    ax.scatter([props['omega']], [props['delta']], color='red', s=150, marker='*', label=label, zorder=10)
    
    ax.set_xlabel(r'Omega Parameter ($\Omega$)')
    ax.set_ylabel(r'Atomic Size Difference ($\delta$ %)')
    ax.set_xlim(0, 20) 
    ax.set_ylim(0, 15)
    ax.axhline(6.6, color='black', linestyle='--', linewidth=0.8)
    ax.axvline(1.1, color='black', linestyle='--', linewidth=0.8)
    ax.legend(loc='upper right')
    ax.set_title("Yang-Zhang Stability Map")
    ax.grid(True, alpha=0.3)
    
    return fig

# =============================================================================
# 8. EXPORT
# =============================================================================
def create_word_report(comp, props, formula):
    doc = Document()
    doc.add_heading('HEA Thermodynamic Report', 0)
    
    doc.add_paragraph(f'Analyzovaná slitina: {formula}')
    doc.add_paragraph(f'Datum: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}')
    
    doc.add_heading('1. Složení', level=1)
    s = ""
    for el, val in comp.items():
        s += f"{el}: {val*100:.1f} at.%, "
    doc.add_paragraph(s)
    
    doc.add_heading('2. Termodynamické parametry', level=1)
    table = doc.add_table(rows=1, cols=2)
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'Parametr'
    hdr_cells[1].text = 'Hodnota'
    
    data_rows = [
        ('Entropie míšení (S_mix)', f"{props['s_mix']:.4f} J/mol.K"),
        ('Entalpie míšení (H_mix)', f"{props['h_mix']:.4f} kJ/mol"),
        ('Atomová neshoda (Delta)', f"{props['delta']:.2f} %"),
        ('Omega (Ω)', f"{props['omega']:.2f}"),
        ('VEC', f"{props['vec']:.2f}"),
        ('H_f (Hydrid)', f"{props['h_f']:.2f} kJ/mol"),
    ]
    
    for k, v in data_rows:
        row_cells = table.add_row().cells
        row_cells[0].text = k
        row_cells[1].text = v

    doc.add_heading('3. Predikce', level=1)
    doc.add_paragraph(f"Fáze: {get_phase_prediction(props)}")
    h_text, _ = get_hydrogen_prediction(props['h_f'])
    doc.add_paragraph(f"Vodík: {h_text}")
    
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

# =============================================================================
# 9. HLAVNÍ UI (STREAMLIT)
# =============================================================================
def main():
    st.sidebar.image("https://img.icons8.com/color/96/000000/atom-editor.png", width=64)
    st.sidebar.title("HEA Ultimate")
    st.sidebar.info("Režim: Advanced Academic")
    
    st.title("HEA Expert System: Thermodynamics & Hydrogen")
    st.markdown("Komplexní nástroj pro návrh slitin s využitím modelů **Miedema**, **Griessen-Driessen** a **Yang-Zhang**.")
    
    # VSTUP
    col_in1, col_in2 = st.columns([3, 1])
    with col_in1:
        formula_input = st.text_input("Zadejte vzorec slitiny (např. TiVCr, (TiVCr)95Ni5):", value="TiVCrNb")
    with col_in2:
        st.write("") # Spacer
        st.write("")
        calculate_btn = st.button("🚀 Vypočítat", type="primary", use_container_width=True)
        
    if calculate_btn and formula_input:
        comp = parse_composition(formula_input)
        
        if comp:
            # VÝPOČET
            res = calculate_hea_properties(comp)
            
            # 1. ZOBRAZENÍ HLAVNÍCH METRIK
            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Entropie (ΔS)", f"{res['s_mix']:.2f} R", help="Ideálně > 1.5 R")
            c2.metric("Entalpie (ΔH)", f"{res['h_mix']:.2f} kJ", help="Ideálně -15 až +5 kJ/mol")
            c3.metric("Neshoda (δ)", f"{res['delta']:.2f} %", help="Ideálně < 6.6 %")
            c4.metric("Omega (Ω)", f"{res['omega']:.2f}", help="Ideálně > 1.1")
            
            # 2. PREDIKCE A VODÍK
            st.subheader("🤖 Expertní Predikce")
            
            phase_text = get_phase_prediction(res)
            st.info(f"**Struktura:** {phase_text}")
            
            h_text, h_status = get_hydrogen_prediction(res['h_f'])
            if h_status == 'success': st.success(f"**Vodík:** {h_text} (ΔHf = {res['h_f']:.1f} kJ/mol)")
            elif h_status == 'error': st.error(f"**Vodík:** {h_text} (ΔHf = {res['h_f']:.1f} kJ/mol)")
            elif h_status == 'warning': st.warning(f"**Vodík:** {h_text} (ΔHf = {res['h_f']:.1f} kJ/mol)")
            else: st.info(f"**Vodík:** {h_text} (ΔHf = {res['h_f']:.1f} kJ/mol)")
            
            # 3. GRAFY A DATA
            tab1, tab2 = st.tabs(["📊 Fázový Diagram", "📝 Export Protokolu"])
            
            with tab1:
                col_g1, col_g2 = st.columns([2, 1])
                with col_g1:
                    fig = plot_ashby(res, formula_input)
                    st.pyplot(fig)
                with col_g2:
                    st.markdown("### Detailní Složení")
                    df_comp = pd.DataFrame(list(comp.items()), columns=['Prvek', 'Podíl'])
                    df_comp['Podíl'] = df_comp['Podíl'] * 100
                    st.dataframe(df_comp.style.format({"Podíl": "{:.2f} %"}), hide_index=True)
                    
            with tab2:
                st.write("Vygenerovat formální report pro výzkumnou zprávu.")
                docx_file = create_word_report(comp, res, formula_input)
                st.download_button(
                    label="📄 Stáhnout DOCX Report",
                    data=docx_file,
                    file_name=f"HEA_Report_{formula_input}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

if __name__ == "__main__":
    main()
