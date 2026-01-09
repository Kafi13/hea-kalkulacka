<!DOCTYPE html>
<html lang="cs">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HEA Termodynamický & Vodíkový Kalkulátor</title>
    <script src="https://unpkg.com/react@18/umd/react.production.min.js" crossorigin></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js" crossorigin></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <script src="https://unpkg.com/recharts@2.12.7/umd/Recharts.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'JetBrains Mono', monospace;
            background: linear-gradient(135deg, #0a0f1a 0%, #1a1f2e 50%, #0d1421 100%);
            min-height: 100vh;
            color: #e0e0e0;
        }
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #1a1f2e;
        }
        ::-webkit-scrollbar-thumb {
            background: #3a4a6e;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #4a5a7e;
        }
    </style>
</head>
<body>
    <div id="root"></div>
    <script type="text/babel">
        const { useState, useMemo } = React;
        const { 
            ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, 
            ResponsiveContainer, ReferenceLine, BarChart, Bar, Cell,
            PieChart, Pie, Legend
        } = Recharts;

        // ============== DATABÁZE PRVKŮ ==============
        const ELEMENTS = {
            // Lehké kovy
            Al: { name: "Hliník", symbol: "Al", group: "Lehké kovy", r: 143, VEC: 3, Tm: 933, phi: 4.20, nWS: 1.39, V23: 4.64, deltaH_H: -4.0, deltaH_inf: 6.0, color: "#90CAF9", note: "Nízká H afinita" },
            Mg: { name: "Hořčík", symbol: "Mg", group: "Lehké kovy", r: 160, VEC: 2, Tm: 923, phi: 3.45, nWS: 1.17, V23: 5.89, deltaH_H: -37.0, deltaH_inf: -19.0, color: "#81D4FA", note: "Stabilní hydrid MgH2" },
            Si: { name: "Křemík", symbol: "Si", group: "Polokovy", r: 118, VEC: 4, Tm: 1687, phi: 4.70, nWS: 1.50, V23: 4.20, deltaH_H: 0.0, deltaH_inf: 46.0, color: "#B39DDB", note: "Polovodič" },
            
            // 3d přechodné kovy
            Sc: { name: "Skandium", symbol: "Sc", group: "3d přechodné", r: 162, VEC: 3, Tm: 1814, phi: 3.25, nWS: 1.27, V23: 6.09, deltaH_H: -100.0, deltaH_inf: -78.0, color: "#CE93D8", note: "Velmi stabilní hydrid" },
            Ti: { name: "Titan", symbol: "Ti", group: "3d přechodné", r: 147, VEC: 4, Tm: 1941, phi: 3.80, nWS: 1.52, V23: 4.12, deltaH_H: -67.0, deltaH_inf: -52.0, color: "#7986CB", note: "Velmi stabilní hydrid TiH2" },
            V: { name: "Vanad", symbol: "V", group: "3d přechodné", r: 134, VEC: 5, Tm: 2183, phi: 4.25, nWS: 1.64, V23: 3.71, deltaH_H: -34.0, deltaH_inf: -26.0, color: "#64B5F6", note: "Dobrá reverzibilita" },
            Cr: { name: "Chrom", symbol: "Cr", group: "3d přechodné", r: 128, VEC: 6, Tm: 2180, phi: 4.65, nWS: 1.73, V23: 3.35, deltaH_H: 6.0, deltaH_inf: 25.0, color: "#4FC3F7", note: "Odolný vůči H" },
            Mn: { name: "Mangan", symbol: "Mn", group: "3d přechodné", r: 127, VEC: 7, Tm: 1519, phi: 4.45, nWS: 1.61, V23: 3.37, deltaH_H: -8.0, deltaH_inf: 8.0, color: "#4DD0E1", note: "Nestabilní hydrid" },
            Fe: { name: "Železo", symbol: "Fe", group: "3d přechodné", r: 126, VEC: 8, Tm: 1811, phi: 4.93, nWS: 1.77, V23: 3.29, deltaH_H: 14.0, deltaH_inf: 29.0, color: "#4DB6AC", note: "Velmi nízká H afinita" },
            Co: { name: "Kobalt", symbol: "Co", group: "3d přechodné", r: 125, VEC: 9, Tm: 1768, phi: 5.10, nWS: 1.79, V23: 3.18, deltaH_H: 11.0, deltaH_inf: 25.0, color: "#81C784", note: "Nízká H rozpustnost" },
            Ni: { name: "Nikl", symbol: "Ni", group: "3d přechodné", r: 124, VEC: 10, Tm: 1728, phi: 5.20, nWS: 1.80, V23: 3.09, deltaH_H: -2.0, deltaH_inf: 12.0, color: "#AED581", note: "Katalyzátor H2 disociace" },
            Cu: { name: "Měď", symbol: "Cu", group: "3d přechodné", r: 128, VEC: 11, Tm: 1358, phi: 4.55, nWS: 1.47, V23: 3.37, deltaH_H: 21.0, deltaH_inf: 46.0, color: "#FFB74D", note: "H bariéra" },
            Zn: { name: "Zinek", symbol: "Zn", group: "3d přechodné", r: 134, VEC: 12, Tm: 693, phi: 4.10, nWS: 1.32, V23: 4.02, deltaH_H: 13.0, deltaH_inf: 35.0, color: "#FF8A65", note: "Nízká Tm" },
            
            // 4d přechodné kovy
            Y: { name: "Yttrium", symbol: "Y", group: "4d přechodné", r: 180, VEC: 3, Tm: 1799, phi: 3.20, nWS: 1.21, V23: 8.26, deltaH_H: -114.0, deltaH_inf: -91.0, color: "#F48FB1", note: "Velmi stabilní YH2, YH3" },
            Zr: { name: "Zirkonium", symbol: "Zr", group: "4d přechodné", r: 160, VEC: 4, Tm: 2128, phi: 3.45, nWS: 1.41, V23: 5.89, deltaH_H: -80.0, deltaH_inf: -65.0, color: "#CE93D8", note: "Stabilní ZrH2" },
            Nb: { name: "Niob", symbol: "Nb", group: "4d přechodné", r: 146, VEC: 5, Tm: 2750, phi: 4.05, nWS: 1.62, V23: 4.93, deltaH_H: -40.0, deltaH_inf: -32.0, color: "#B39DDB", note: "Dobrý pro H skladování" },
            Mo: { name: "Molybden", symbol: "Mo", group: "4d přechodné", r: 139, VEC: 6, Tm: 2896, phi: 4.65, nWS: 1.77, V23: 4.24, deltaH_H: 20.0, deltaH_inf: 37.0, color: "#9FA8DA", note: "H bariéra, vysoká Tm" },
            Pd: { name: "Palladium", symbol: "Pd", group: "4d přechodné", r: 137, VEC: 10, Tm: 1828, phi: 5.45, nWS: 1.67, V23: 4.03, deltaH_H: -10.0, deltaH_inf: 1.0, color: "#80DEEA", note: "Výborná H absorpce" },
            Ag: { name: "Stříbro", symbol: "Ag", group: "4d přechodné", r: 144, VEC: 11, Tm: 1235, phi: 4.45, nWS: 1.39, V23: 4.61, deltaH_H: 31.0, deltaH_inf: 62.0, color: "#B0BEC5", note: "H bariéra" },
            
            // 5d přechodné kovy
            Hf: { name: "Hafnium", symbol: "Hf", group: "5d přechodné", r: 159, VEC: 4, Tm: 2506, phi: 3.55, nWS: 1.46, V23: 5.76, deltaH_H: -65.0, deltaH_inf: -52.0, color: "#FFCC80", note: "Podobné Zr" },
            Ta: { name: "Tantal", symbol: "Ta", group: "5d přechodné", r: 146, VEC: 5, Tm: 3290, phi: 4.15, nWS: 1.63, V23: 4.93, deltaH_H: -38.0, deltaH_inf: -30.0, color: "#FFAB91", note: "Vysoká Tm, dobrá H absorpce" },
            W: { name: "Wolfram", symbol: "W", group: "5d přechodné", r: 139, VEC: 6, Tm: 3695, phi: 4.80, nWS: 1.81, V23: 4.24, deltaH_H: 40.0, deltaH_inf: 56.0, color: "#BCAAA4", note: "Nejvyšší Tm, H bariéra" },
            Re: { name: "Rhenium", symbol: "Re", group: "5d přechodné", r: 137, VEC: 7, Tm: 3459, phi: 5.20, nWS: 1.86, V23: 4.03, deltaH_H: 25.0, deltaH_inf: 40.0, color: "#B0BEC5", note: "Vysoká Tm" },
            Pt: { name: "Platina", symbol: "Pt", group: "5d přechodné", r: 139, VEC: 10, Tm: 2041, phi: 5.65, nWS: 1.78, V23: 4.24, deltaH_H: -5.0, deltaH_inf: 10.0, color: "#E0E0E0", note: "Katalyzátor" },
            Au: { name: "Zlato", symbol: "Au", group: "5d přechodné", r: 144, VEC: 11, Tm: 1337, phi: 5.15, nWS: 1.57, V23: 4.61, deltaH_H: 25.0, deltaH_inf: 52.0, color: "#FFD54F", note: "Inertní" },
            
            // Lanthanoidy
            La: { name: "Lanthan", symbol: "La", group: "Lanthanoidy", r: 187, VEC: 3, Tm: 1193, phi: 3.05, nWS: 1.18, V23: 9.04, deltaH_H: -100.0, deltaH_inf: -80.0, color: "#FFE082", note: "LaH2, LaH3" },
            Ce: { name: "Cer", symbol: "Ce", group: "Lanthanoidy", r: 182, VEC: 3, Tm: 1068, phi: 3.18, nWS: 1.19, V23: 8.54, deltaH_H: -95.0, deltaH_inf: -75.0, color: "#FFF59D", note: "CeH2, CeH3" }
        };

        // ============== BINÁRNÍ ENTALPIE MÍŠENÍ ==============
        const BINARY_ENTHALPIES = {
            // Ti páry
            "Ti-V": -2, "Ti-Cr": -7, "Ti-Mn": -8, "Ti-Fe": -17, "Ti-Co": -28,
            "Ti-Ni": -35, "Ti-Cu": -9, "Ti-Zn": -15, "Ti-Al": -30, "Ti-Zr": 0,
            "Ti-Nb": 2, "Ti-Mo": -4, "Ti-Hf": 0, "Ti-Ta": 1, "Ti-W": -6,
            "Ti-La": 13, "Ti-Ce": 10, "Ti-Y": 8, "Ti-Sc": 4, "Ti-Pd": -63,
            "Ti-Mg": 16, "Ti-Si": -66,
            // V páry
            "V-Cr": -2, "V-Mn": -2, "V-Fe": -7, "V-Co": -14, "V-Ni": -18,
            "V-Cu": 5, "V-Zn": -4, "V-Al": -16, "V-Zr": -4, "V-Nb": -1,
            "V-Mo": 0, "V-Hf": -2, "V-Ta": -1, "V-W": -1, "V-La": 22,
            "V-Pd": -44, "V-Si": -45,
            // Cr páry
            "Cr-Mn": 2, "Cr-Fe": -1, "Cr-Co": -4, "Cr-Ni": -7, "Cr-Cu": 12,
            "Cr-Zn": 5, "Cr-Al": -10, "Cr-Zr": -12, "Cr-Nb": -7, "Cr-Mo": 0,
            "Cr-Hf": -9, "Cr-Ta": -7, "Cr-W": 1, "Cr-La": 27, "Cr-Pd": -27,
            "Cr-Si": -37,
            // Mn páry
            "Mn-Fe": 0, "Mn-Co": -5, "Mn-Ni": -8, "Mn-Cu": 4, "Mn-Zn": -4,
            "Mn-Al": -19, "Mn-Zr": -20, "Mn-Nb": -13, "Mn-Mo": -5,
            "Mn-Pd": -23, "Mn-Si": -45,
            // Fe páry
            "Fe-Co": -1, "Fe-Ni": -2, "Fe-Cu": 13, "Fe-Zn": -1, "Fe-Al": -11,
            "Fe-Zr": -25, "Fe-Nb": -16, "Fe-Mo": -2, "Fe-Hf": -21, "Fe-Ta": -15,
            "Fe-W": 0, "Fe-La": 18, "Fe-Pd": -4, "Fe-Si": -35,
            // Co páry
            "Co-Ni": 0, "Co-Cu": 6, "Co-Zn": -7, "Co-Al": -19, "Co-Zr": -41,
            "Co-Nb": -25, "Co-Mo": -5, "Co-Hf": -35, "Co-Ta": -24, "Co-W": -1,
            "Co-La": 7, "Co-Pd": 0, "Co-Si": -38,
            // Ni páry
            "Ni-Cu": 4, "Ni-Zn": -9, "Ni-Al": -22, "Ni-Zr": -49, "Ni-Nb": -30,
            "Ni-Mo": -7, "Ni-Hf": -42, "Ni-Ta": -29, "Ni-W": -3, "Ni-La": -4,
            "Ni-Pd": 0, "Ni-Si": -40, "Ni-Mg": -4,
            // Cu páry
            "Cu-Zn": -6, "Cu-Al": -1, "Cu-Zr": -23, "Cu-Nb": -3, "Cu-Mo": 19,
            "Cu-Hf": -21, "Cu-Ta": -2, "Cu-W": 22, "Cu-La": -15, "Cu-Pd": -14,
            "Cu-Mg": -3, "Cu-Si": -10,
            // Zn páry
            "Zn-Al": 1, "Zn-Zr": -37, "Zn-Nb": -19, "Zn-La": -29, "Zn-Mg": 4,
            // Al páry
            "Al-Zr": -44, "Al-Nb": -18, "Al-Mo": -5, "Al-Hf": -39, "Al-Ta": -19,
            "Al-W": -2, "Al-La": -38, "Al-Ce": -38, "Al-Y": -38, "Al-Sc": -38,
            "Al-Pd": -55, "Al-Mg": 2, "Al-Si": -4,
            // Zr páry
            "Zr-Nb": 4, "Zr-Mo": -6, "Zr-Hf": 0, "Zr-Ta": 3, "Zr-W": -9,
            "Zr-La": 10, "Zr-Pd": -91, "Zr-Si": -84,
            // Nb páry
            "Nb-Mo": -6, "Nb-Hf": 4, "Nb-Ta": 0, "Nb-W": -8, "Nb-La": 16,
            "Nb-Pd": -64, "Nb-Si": -56,
            // Mo páry
            "Mo-Hf": -4, "Mo-Ta": -5, "Mo-W": 0, "Mo-La": 32, "Mo-Pd": -18,
            // Hf páry
            "Hf-Ta": 3, "Hf-W": -6, "Hf-La": 11, "Hf-Pd": -84,
            // Ta páry
            "Ta-W": -7, "Ta-La": 17, "Ta-Pd": -60,
            // W páry
            "W-La": 36, "W-Pd": -12,
            // La páry
            "La-Ce": 0, "La-Pd": -76, "La-Mg": 6,
            // Ce páry
            "Ce-Pd": -75,
            // Pd páry
            "Pd-Si": -55, "Pd-Mg": -45,
            // Y páry
            "Y-Zr": 9, "Y-Sc": 0,
            // Sc páry
            "Sc-Zr": 5,
            // Mg páry
            "Mg-Si": -3
        };

        // Funkce pro získání binární entalpie
        function getBinaryEnthalpy(el1, el2) {
            const key1 = `${el1}-${el2}`;
            const key2 = `${el2}-${el1}`;
            if (BINARY_ENTHALPIES[key1] !== undefined) return BINARY_ENTHALPIES[key1];
            if (BINARY_ENTHALPIES[key2] !== undefined) return BINARY_ENTHALPIES[key2];
            // Miedema odhad
            const a = ELEMENTS[el1];
            const b = ELEMENTS[el2];
            if (a && b && a.phi && b.phi && a.nWS && b.nWS && a.V23 && b.V23) {
                const P = 14.2, Q = 9.4;
                const dPhi = a.phi - b.phi;
                const dN = a.nWS - b.nWS;
                const Vavg = 2 * a.V23 * b.V23 / (a.V23 + b.V23);
                return Math.round(Vavg * (-P * dPhi * dPhi + Q * dN * dN));
            }
            return 0;
        }

        // ============== HLAVNÍ KOMPONENTA ==============
        function HEACalculator() {
            const [selectedElements, setSelectedElements] = useState(['Ti', 'V', 'Cr', 'Ni', 'Nb']);
            const [compositions, setCompositions] = useState({ Ti: 20, V: 20, Cr: 20, Ni: 20, Nb: 20 });

            // Normalizace kompozice
            const normalizedComp = useMemo(() => {
                const total = Object.values(compositions).reduce((a, b) => a + b, 0);
                if (total === 0) return compositions;
                const norm = {};
                for (const el of selectedElements) {
                    norm[el] = (compositions[el] || 0) / total;
                }
                return norm;
            }, [compositions, selectedElements]);

            // Výpočet termodynamických parametrů
            const results = useMemo(() => {
                const n = selectedElements.length;
                if (n < 2) return null;

                const R = 8.314;
                let Smix = 0, Hmix = 0, rAvg = 0, TmAvg = 0, VEC = 0;
                let deltaH_inf = 0, deltaH_f = 0;

                // Průměrné vlastnosti a vodíkové entalpie
                for (const el of selectedElements) {
                    const c = normalizedComp[el];
                    const data = ELEMENTS[el];
                    rAvg += c * data.r;
                    TmAvg += c * data.Tm;
                    VEC += c * data.VEC;
                    deltaH_inf += c * data.deltaH_inf;
                    deltaH_f += c * data.deltaH_H;
                    if (c > 0) {
                        Smix -= c * Math.log(c);
                    }
                }
                Smix *= R;

                // Entalpie míšení (Miedema)
                for (let i = 0; i < n; i++) {
                    for (let j = i + 1; j < n; j++) {
                        const el1 = selectedElements[i];
                        const el2 = selectedElements[j];
                        const c1 = normalizedComp[el1];
                        const c2 = normalizedComp[el2];
                        const Hij = getBinaryEnthalpy(el1, el2);
                        Hmix += 4 * Hij * c1 * c2;
                    }
                }

                // Atomový mismatch δ
                let delta = 0;
                for (const el of selectedElements) {
                    const c = normalizedComp[el];
                    const ri = ELEMENTS[el].r;
                    delta += c * Math.pow(1 - ri / rAvg, 2);
                }
                delta = 100 * Math.sqrt(delta);

                // Omega parametr
                const Omega = Math.abs(Hmix) > 0.01 ? (TmAvg * Smix / 1000) / Math.abs(Hmix) : Infinity;

                return {
                    Smix: Smix / R, // v jednotkách R
                    Hmix,
                    delta,
                    Omega,
                    VEC,
                    rAvg,
                    TmAvg,
                    deltaH_inf,
                    deltaH_f
                };
            }, [selectedElements, normalizedComp]);

            // Predikce struktury
            const structurePrediction = useMemo(() => {
                if (!results) return null;
                const { delta, Omega, VEC, Hmix } = results;

                let structure = "BCC";
                let confidence = "vysoká";
                let icon = "🔷";

                if (delta > 6.6) {
                    structure = "Amorfní/Intermetalická";
                    confidence = "vysoká";
                    icon = "⚠️";
                } else if (Omega < 1.1) {
                    structure = "Intermetalické sloučeniny";
                    confidence = "střední";
                    icon = "⚠️";
                } else if (VEC < 6.87) {
                    structure = "BCC";
                    confidence = "vysoká";
                    icon = "🔷";
                } else if (VEC > 8.0) {
                    structure = "FCC";
                    confidence = "vysoká";
                    icon = "🔶";
                } else {
                    structure = "BCC + FCC (duální)";
                    confidence = "střední";
                    icon = "🔷🔶";
                }

                return { structure, confidence, icon };
            }, [results]);

            // Klasifikace vodíkové afinity
            const hydrogenClass = useMemo(() => {
                if (!results) return null;
                const { deltaH_f, deltaH_inf } = results;

                let category, color, icon, description;

                if (deltaH_f < -60) {
                    category = "Vodíková past";
                    color = "#E53935";
                    icon = "🔥";
                    description = "Příliš stabilní hydrid. Desorpce vyžaduje >300°C.";
                } else if (deltaH_f < -20) {
                    category = "Skladovací materiál";
                    color = "#43A047";
                    icon = "✓";
                    description = "Ideální pro reverzibilní skladování H2.";
                } else if (deltaH_f < 0) {
                    category = "Mírná absorpce";
                    color = "#FB8C00";
                    icon = "~";
                    description = "Slabá tvorba hydridů.";
                } else {
                    category = "Vodíková bariéra";
                    color = "#1E88E5";
                    icon = "🛡️";
                    description = "Odolný vůči vodíkové křehkosti.";
                }

                return { category, color, icon, description };
            }, [results]);

            // Toggle prvku
            const toggleElement = (el) => {
                if (selectedElements.includes(el)) {
                    if (selectedElements.length > 2) {
                        setSelectedElements(prev => prev.filter(e => e !== el));
                        setCompositions(prev => {
                            const newComp = { ...prev };
                            delete newComp[el];
                            return newComp;
                        });
                    }
                } else {
                    if (selectedElements.length < 8) {
                        setSelectedElements(prev => [...prev, el]);
                        setCompositions(prev => ({ ...prev, [el]: 10 }));
                    }
                }
            };

            // Ekvlatomární
            const setEquiatomic = () => {
                const n = selectedElements.length;
                const val = 100 / n;
                const newComp = {};
                selectedElements.forEach(el => newComp[el] = val);
                setCompositions(newComp);
            };

            // Data pro grafy
            const pieData = selectedElements.map(el => ({
                name: el,
                value: normalizedComp[el] * 100,
                fill: ELEMENTS[el].color
            }));

            const barData = selectedElements.map(el => ({
                name: el,
                value: ELEMENTS[el].deltaH_H,
                contribution: normalizedComp[el] * ELEMENTS[el].deltaH_H,
                fill: ELEMENTS[el].color
            }));

            const scatterData = results ? [{ x: results.Hmix, y: results.delta, z: 10 }] : [];

            // Periodická tabulka (zjednodušená)
            const periodicTable = [
                ['', '', '', 'Al', 'Si', '', '', '', '', '', '', ''],
                ['Mg', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', ''],
                ['', 'Y', 'Zr', 'Nb', 'Mo', '', '', '', 'Pd', 'Ag', '', ''],
                ['', 'La', 'Hf', 'Ta', 'W', 'Re', '', '', 'Pt', 'Au', '', ''],
                ['', 'Ce', '', '', '', '', '', '', '', '', '', '']
            ];

            return (
                <div style={{ padding: '20px', maxWidth: '1600px', margin: '0 auto' }}>
                    {/* Header */}
                    <div style={{ textAlign: 'center', marginBottom: '30px' }}>
                        <h1 style={{ fontSize: '28px', fontWeight: '700', background: 'linear-gradient(90deg, #64B5F6, #81C784)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', marginBottom: '8px' }}>
                            HEA Termodynamický & Vodíkový Kalkulátor
                        </h1>
                        <p style={{ color: '#888', fontSize: '14px' }}>
                            Výpočet parametrů vysokoentropických slitin a predikce vodíkové afinity
                        </p>
                    </div>

                    {/* Main Grid */}
                    <div style={{ display: 'grid', gridTemplateColumns: '350px 1fr 380px', gap: '20px' }}>
                        
                        {/* LEFT PANEL - Výběr prvků */}
                        <div style={{ background: 'rgba(255,255,255,0.03)', borderRadius: '16px', padding: '20px', border: '1px solid rgba(255,255,255,0.1)' }}>
                            <h2 style={{ fontSize: '16px', marginBottom: '15px', color: '#90CAF9' }}>Výběr prvků</h2>
                            
                            {/* Mini periodická tabulka */}
                            <div style={{ marginBottom: '20px' }}>
                                {periodicTable.map((row, ri) => (
                                    <div key={ri} style={{ display: 'flex', gap: '3px', marginBottom: '3px' }}>
                                        {row.map((el, ci) => (
                                            <div
                                                key={ci}
                                                onClick={() => el && ELEMENTS[el] && toggleElement(el)}
                                                style={{
                                                    width: '28px',
                                                    height: '28px',
                                                    display: 'flex',
                                                    alignItems: 'center',
                                                    justifyContent: 'center',
                                                    fontSize: '10px',
                                                    fontWeight: '600',
                                                    borderRadius: '4px',
                                                    cursor: el && ELEMENTS[el] ? 'pointer' : 'default',
                                                    background: el && selectedElements.includes(el) ? ELEMENTS[el].color : (el && ELEMENTS[el] ? 'rgba(255,255,255,0.1)' : 'transparent'),
                                                    color: el && selectedElements.includes(el) ? '#000' : '#888',
                                                    border: el && ELEMENTS[el] ? '1px solid rgba(255,255,255,0.2)' : 'none',
                                                    transition: 'all 0.2s'
                                                }}
                                            >
                                                {el}
                                            </div>
                                        ))}
                                    </div>
                                ))}
                            </div>

                            {/* Složení */}
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                                <h3 style={{ fontSize: '14px', color: '#81C784' }}>Složení</h3>
                                <button
                                    onClick={setEquiatomic}
                                    style={{
                                        background: 'linear-gradient(135deg, #4FC3F7, #64B5F6)',
                                        border: 'none',
                                        padding: '4px 12px',
                                        borderRadius: '6px',
                                        color: '#000',
                                        fontSize: '11px',
                                        fontWeight: '600',
                                        cursor: 'pointer'
                                    }}
                                >
                                    Ekvlatomární
                                </button>
                            </div>

                            {/* Slidery */}
                            {selectedElements.map(el => (
                                <div key={el} style={{ marginBottom: '8px' }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '2px' }}>
                                        <span style={{ color: ELEMENTS[el].color }}>{el}</span>
                                        <span>{(normalizedComp[el] * 100).toFixed(1)}%</span>
                                    </div>
                                    <input
                                        type="range"
                                        min="0"
                                        max="100"
                                        value={compositions[el] || 0}
                                        onChange={(e) => setCompositions(prev => ({ ...prev, [el]: parseFloat(e.target.value) }))}
                                        style={{ width: '100%', accentColor: ELEMENTS[el].color }}
                                    />
                                </div>
                            ))}

                            {/* Pie chart */}
                            <div style={{ height: '180px', marginTop: '15px' }}>
                                <ResponsiveContainer>
                                    <PieChart>
                                        <Pie
                                            data={pieData}
                                            dataKey="value"
                                            nameKey="name"
                                            cx="50%"
                                            cy="50%"
                                            innerRadius={40}
                                            outerRadius={70}
                                            paddingAngle={2}
                                        >
                                            {pieData.map((entry, index) => (
                                                <Cell key={index} fill={entry.fill} />
                                            ))}
                                        </Pie>
                                        <Tooltip formatter={(v) => `${v.toFixed(1)}%`} />
                                    </PieChart>
                                </ResponsiveContainer>
                            </div>
                        </div>

                        {/* CENTER PANEL - Výsledky */}
                        <div>
                            {results && (
                                <>
                                    {/* Parametry grid */}
                                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px', marginBottom: '20px' }}>
                                        {/* ΔSmix */}
                                        <div style={{ background: 'linear-gradient(135deg, rgba(76,175,80,0.2), rgba(76,175,80,0.05))', borderRadius: '12px', padding: '15px', border: '1px solid rgba(76,175,80,0.3)' }}>
                                            <div style={{ fontSize: '11px', color: '#81C784', marginBottom: '4px' }}>ΔS<sub>mix</sub></div>
                                            <div style={{ fontSize: '24px', fontWeight: '700', color: '#A5D6A7' }}>{results.Smix.toFixed(2)}</div>
                                            <div style={{ fontSize: '10px', color: '#666' }}>× R (J/mol·K)</div>
                                        </div>

                                        {/* ΔHmix */}
                                        <div style={{ background: `linear-gradient(135deg, ${results.Hmix < 0 ? 'rgba(33,150,243,0.2)' : 'rgba(244,67,54,0.2)'}, transparent)`, borderRadius: '12px', padding: '15px', border: `1px solid ${results.Hmix < 0 ? 'rgba(33,150,243,0.3)' : 'rgba(244,67,54,0.3)'}` }}>
                                            <div style={{ fontSize: '11px', color: results.Hmix < 0 ? '#64B5F6' : '#EF5350', marginBottom: '4px' }}>ΔH<sub>mix</sub></div>
                                            <div style={{ fontSize: '24px', fontWeight: '700', color: results.Hmix < 0 ? '#90CAF9' : '#EF9A9A' }}>{results.Hmix.toFixed(1)}</div>
                                            <div style={{ fontSize: '10px', color: '#666' }}>kJ/mol {results.Hmix >= -11.6 && results.Hmix <= 3.2 ? '✓' : '⚠'}</div>
                                        </div>

                                        {/* δ */}
                                        <div style={{ background: `linear-gradient(135deg, ${results.delta < 6.6 ? 'rgba(156,39,176,0.2)' : 'rgba(244,67,54,0.2)'}, transparent)`, borderRadius: '12px', padding: '15px', border: `1px solid ${results.delta < 6.6 ? 'rgba(156,39,176,0.3)' : 'rgba(244,67,54,0.3)'}` }}>
                                            <div style={{ fontSize: '11px', color: results.delta < 6.6 ? '#BA68C8' : '#EF5350', marginBottom: '4px' }}>δ (mismatch)</div>
                                            <div style={{ fontSize: '24px', fontWeight: '700', color: results.delta < 6.6 ? '#CE93D8' : '#EF9A9A' }}>{results.delta.toFixed(2)}</div>
                                            <div style={{ fontSize: '10px', color: '#666' }}>% {results.delta < 6.6 ? '✓ < 6.6%' : '⚠ > 6.6%'}</div>
                                        </div>

                                        {/* Ω */}
                                        <div style={{ background: `linear-gradient(135deg, ${results.Omega > 1.1 ? 'rgba(0,150,136,0.2)' : 'rgba(255,152,0,0.2)'}, transparent)`, borderRadius: '12px', padding: '15px', border: `1px solid ${results.Omega > 1.1 ? 'rgba(0,150,136,0.3)' : 'rgba(255,152,0,0.3)'}` }}>
                                            <div style={{ fontSize: '11px', color: results.Omega > 1.1 ? '#4DB6AC' : '#FFB74D', marginBottom: '4px' }}>Ω</div>
                                            <div style={{ fontSize: '24px', fontWeight: '700', color: results.Omega > 1.1 ? '#80CBC4' : '#FFCC80' }}>{results.Omega === Infinity ? '∞' : results.Omega.toFixed(2)}</div>
                                            <div style={{ fontSize: '10px', color: '#666' }}>{results.Omega > 1.1 ? '✓ > 1.1' : '⚠ < 1.1'}</div>
                                        </div>

                                        {/* VEC */}
                                        <div style={{ background: 'linear-gradient(135deg, rgba(255,235,59,0.15), transparent)', borderRadius: '12px', padding: '15px', border: '1px solid rgba(255,235,59,0.3)' }}>
                                            <div style={{ fontSize: '11px', color: '#FFF176', marginBottom: '4px' }}>VEC</div>
                                            <div style={{ fontSize: '24px', fontWeight: '700', color: '#FFEE58' }}>{results.VEC.toFixed(2)}</div>
                                            <div style={{ fontSize: '10px', color: '#666' }}>
                                                {results.VEC < 6.87 ? 'BCC' : results.VEC > 8 ? 'FCC' : 'BCC+FCC'}
                                            </div>
                                        </div>

                                        {/* Průměry */}
                                        <div style={{ background: 'linear-gradient(135deg, rgba(158,158,158,0.15), transparent)', borderRadius: '12px', padding: '15px', border: '1px solid rgba(158,158,158,0.3)' }}>
                                            <div style={{ fontSize: '11px', color: '#BDBDBD', marginBottom: '4px' }}>Průměry</div>
                                            <div style={{ fontSize: '14px', color: '#E0E0E0' }}>r̄ = {results.rAvg.toFixed(1)} pm</div>
                                            <div style={{ fontSize: '14px', color: '#E0E0E0' }}>T<sub>m</sub> = {results.TmAvg.toFixed(0)} K</div>
                                        </div>
                                    </div>

                                    {/* Predikce struktury */}
                                    {structurePrediction && (
                                        <div style={{ background: 'rgba(255,255,255,0.03)', borderRadius: '12px', padding: '15px', border: '1px solid rgba(255,255,255,0.1)', marginBottom: '20px' }}>
                                            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                                                <span style={{ fontSize: '28px' }}>{structurePrediction.icon}</span>
                                                <div>
                                                    <div style={{ fontSize: '18px', fontWeight: '600' }}>{structurePrediction.structure}</div>
                                                    <div style={{ fontSize: '11px', color: '#888' }}>Spolehlivost: {structurePrediction.confidence}</div>
                                                </div>
                                            </div>
                                        </div>
                                    )}

                                    {/* Diagram stability */}
                                    <div style={{ background: 'rgba(255,255,255,0.03)', borderRadius: '12px', padding: '15px', border: '1px solid rgba(255,255,255,0.1)' }}>
                                        <h3 style={{ fontSize: '14px', marginBottom: '10px', color: '#90CAF9' }}>Diagram stability (δ vs ΔH<sub>mix</sub>)</h3>
                                        <div style={{ height: '250px' }}>
                                            <ResponsiveContainer>
                                                <ScatterChart margin={{ top: 20, right: 30, bottom: 20, left: 10 }}>
                                                    <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                                                    <XAxis 
                                                        type="number" 
                                                        dataKey="x" 
                                                        domain={[-25, 10]} 
                                                        name="ΔHmix"
                                                        tick={{ fill: '#888', fontSize: 10 }}
                                                        label={{ value: 'ΔHmix (kJ/mol)', position: 'bottom', fill: '#888', fontSize: 11 }}
                                                    />
                                                    <YAxis 
                                                        type="number" 
                                                        dataKey="y" 
                                                        domain={[0, 10]} 
                                                        name="δ"
                                                        tick={{ fill: '#888', fontSize: 10 }}
                                                        label={{ value: 'δ (%)', angle: -90, position: 'insideLeft', fill: '#888', fontSize: 11 }}
                                                    />
                                                    <ReferenceLine y={6.6} stroke="#E53935" strokeDasharray="5 5" label={{ value: 'δ=6.6%', fill: '#E53935', fontSize: 10 }} />
                                                    <ReferenceLine x={-11.6} stroke="#64B5F6" strokeDasharray="5 5" />
                                                    <ReferenceLine x={3.2} stroke="#64B5F6" strokeDasharray="5 5" />
                                                    <Scatter 
                                                        data={scatterData} 
                                                        fill="#4FC3F7"
                                                        shape="circle"
                                                    >
                                                        {scatterData.map((entry, index) => (
                                                            <Cell key={index} fill="#4FC3F7" />
                                                        ))}
                                                    </Scatter>
                                                    <Tooltip 
                                                        formatter={(value, name) => [value.toFixed(2), name === 'x' ? 'ΔHmix' : 'δ']}
                                                        contentStyle={{ background: '#1a1f2e', border: '1px solid #333' }}
                                                    />
                                                </ScatterChart>
                                            </ResponsiveContainer>
                                        </div>
                                    </div>
                                </>
                            )}
                        </div>

                        {/* RIGHT PANEL - Vodíková afinita */}
                        <div style={{ background: 'rgba(255,255,255,0.03)', borderRadius: '16px', padding: '20px', border: '1px solid rgba(255,255,255,0.1)' }}>
                            <h2 style={{ fontSize: '16px', marginBottom: '15px', color: '#81C784' }}>Vodíková afinita</h2>

                            {results && hydrogenClass && (
                                <>
                                    {/* Klasifikace */}
                                    <div style={{ background: `linear-gradient(135deg, ${hydrogenClass.color}33, transparent)`, borderRadius: '12px', padding: '15px', border: `1px solid ${hydrogenClass.color}66`, marginBottom: '15px' }}>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
                                            <span style={{ fontSize: '24px' }}>{hydrogenClass.icon}</span>
                                            <span style={{ fontSize: '18px', fontWeight: '600', color: hydrogenClass.color }}>{hydrogenClass.category}</span>
                                        </div>
                                        <p style={{ fontSize: '12px', color: '#aaa' }}>{hydrogenClass.description}</p>
                                    </div>

                                    {/* Hodnoty */}
                                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginBottom: '15px' }}>
                                        <div style={{ background: 'rgba(100,181,246,0.1)', borderRadius: '8px', padding: '12px', textAlign: 'center' }}>
                                            <div style={{ fontSize: '10px', color: '#64B5F6' }}>ΔH<sup>∞</sup> (roztok)</div>
                                            <div style={{ fontSize: '20px', fontWeight: '600', color: '#90CAF9' }}>{results.deltaH_inf.toFixed(1)}</div>
                                            <div style={{ fontSize: '9px', color: '#666' }}>kJ/mol H</div>
                                        </div>
                                        <div style={{ background: 'rgba(129,199,132,0.1)', borderRadius: '8px', padding: '12px', textAlign: 'center' }}>
                                            <div style={{ fontSize: '10px', color: '#81C784' }}>ΔH<sub>f</sub> (hydrid)</div>
                                            <div style={{ fontSize: '20px', fontWeight: '600', color: '#A5D6A7' }}>{results.deltaH_f.toFixed(1)}</div>
                                            <div style={{ fontSize: '9px', color: '#666' }}>kJ/mol H</div>
                                        </div>
                                    </div>

                                    {/* Bar chart příspěvků */}
                                    <h3 style={{ fontSize: '12px', marginBottom: '8px', color: '#aaa' }}>Příspěvky prvků k ΔH<sub>f</sub></h3>
                                    <div style={{ height: '200px' }}>
                                        <ResponsiveContainer>
                                            <BarChart data={barData} layout="vertical" margin={{ left: 30, right: 10 }}>
                                                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                                                <XAxis type="number" tick={{ fill: '#888', fontSize: 10 }} />
                                                <YAxis type="category" dataKey="name" tick={{ fill: '#888', fontSize: 10 }} />
                                                <Tooltip 
                                                    formatter={(v, name) => [v.toFixed(1) + ' kJ/mol', name === 'contribution' ? 'Příspěvek' : 'ΔH_H']}
                                                    contentStyle={{ background: '#1a1f2e', border: '1px solid #333', fontSize: 11 }}
                                                />
                                                <Bar dataKey="contribution" name="Příspěvek">
                                                    {barData.map((entry, index) => (
                                                        <Cell key={index} fill={entry.fill} />
                                                    ))}
                                                </Bar>
                                            </BarChart>
                                        </ResponsiveContainer>
                                    </div>

                                    {/* Interpretace */}
                                    <div style={{ marginTop: '15px', padding: '12px', background: 'rgba(255,255,255,0.02)', borderRadius: '8px', fontSize: '11px', color: '#888' }}>
                                        <strong style={{ color: '#aaa' }}>Interpretace:</strong><br />
                                        {results.deltaH_f < -40 && "Slitina silně váže vodík. Vhodná pro trvalé skladování, ale desorpce vyžaduje vysoké teploty."}
                                        {results.deltaH_f >= -40 && results.deltaH_f < -20 && "Optimální rozsah pro reverzibilní skladování vodíku při provozních teplotách."}
                                        {results.deltaH_f >= -20 && results.deltaH_f < 0 && "Slabá interakce s vodíkem. Možné použití jako membrána nebo povlak."}
                                        {results.deltaH_f >= 0 && "Materiál je odolný vůči absorpci vodíku. Vhodný pro aplikace vyžadující odolnost vůči vodíkové křehkosti."}
                                    </div>
                                </>
                            )}
                        </div>
                    </div>

                    {/* Footer */}
                    <div style={{ marginTop: '30px', textAlign: 'center', fontSize: '11px', color: '#555', borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '15px' }}>
                        <p>Modely: Hume-Rothery pravidla • Miedema model (binární entalpie) • Griessen-Driessen (vodíková afinita)</p>
                        <p style={{ marginTop: '5px' }}>© 2025 HEA Calculator | RTI ZČU</p>
                    </div>
                </div>
            );
        }

        // Render
        const root = ReactDOM.createRoot(document.getElementById('root'));
        root.render(<HEACalculator />);
    </script>
</body>
</html>
