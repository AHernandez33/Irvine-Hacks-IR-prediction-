"""
fetch_ir_spectra.py
--------------------
Fetches IR spectra from NIST and saves to CSV in the format:
    'Name of molecule', 'SMILES', point 1, point 2, ..., point 250

Requirements:
    pip install nistchempy pandas numpy scipy requests

Usage:
    python fetch_ir_spectra.py
"""

import re
import os
import time
import numpy as np
import pandas as pd
import requests
from scipy.interpolate import interp1d
import nistchempy as nist

# ── Configuration ──────────────────────────────────────────────────────────────

N_POINTS       = 250
WAVENUMBER_MIN = 400
WAVENUMBER_MAX = 4000
OUTPUT_FILE    = "ir_spectra.csv"
DELAY          = 1.5

# ── 200 Compounds ──────────────────────────────────────────────────────────────
# Common organic molecules with well-known NIST IR spectra

COMPOUNDS = [
    # Alcohols
    ("methanol",                "CO",                       "C67561"),
    ("ethanol",                 "CCO",                      "C64175"),
    ("propanol",                "CCCO",                     "C71238"),
    ("isopropanol",             "CC(O)C",                   "C67630"),
    ("butanol",                 "CCCCO",                    "C71363"),
    ("isobutanol",              "CC(C)CO",                  "C78831"),
    ("tert-butanol",            "CC(C)(C)O",                "C75650"),
    ("pentanol",                "CCCCCO",                   "C71410"),
    ("hexanol",                 "CCCCCCO",                  "C111273"),
    ("cyclohexanol",            "OC1CCCCC1",                "C108930"),
    ("benzyl alcohol",          "OCc1ccccc1",               "C100516"),
    ("ethylene glycol",         "OCCO",                     "C107211"),
    ("glycerol",                "OCC(O)CO",                 "C56815"),
    ("phenol",                  "Oc1ccccc1",                "C108952"),
    ("cresol",                  "Cc1ccccc1O",               "C95487"),
    # Aldehydes
    ("formaldehyde",            "C=O",                      "C50000"),
    ("acetaldehyde",            "CC=O",                     "C75070"),
    ("propanal",                "CCC=O",                    "C123386"),
    ("butanal",                 "CCCC=O",                   "C123728"),
    ("benzaldehyde",            "O=Cc1ccccc1",              "C100527"),
    ("cinnamaldehyde",          "O=C/C=C/c1ccccc1",        "C104552"),
    # Ketones
    ("acetone",                 "CC(C)=O",                  "C67641"),
    ("butanone",                "CCC(C)=O",                 "C78933"),
    ("pentan-2-one",            "CCCC(C)=O",                "C107879"),
    ("pentan-3-one",            "CCC(=O)CC",                "C96220"),
    ("cyclohexanone",           "O=C1CCCCC1",               "C108941"),
    ("acetophenone",            "CC(=O)c1ccccc1",           "C98862"),
    ("benzophenone",            "O=C(c1ccccc1)c1ccccc1",   "C119619"),
    ("methyl vinyl ketone",     "CC(=O)C=C",                "C78944"),
    # Carboxylic acids
    ("formic acid",             "OC=O",                     "C64186"),
    ("acetic acid",             "CC(=O)O",                  "C64197"),
    ("propionic acid",          "CCC(=O)O",                 "C79094"),
    ("butyric acid",            "CCCC(=O)O",                "C107926"),
    ("valeric acid",            "CCCCC(=O)O",               "C109524"),
    ("hexanoic acid",           "CCCCCC(=O)O",              "C142621"),
    ("benzoic acid",            "OC(=O)c1ccccc1",           "C65850"),
    ("oxalic acid",             "OC(=O)C(=O)O",             "C144627"),
    ("malonic acid",            "OC(=O)CC(=O)O",            "C141829"),
    ("succinic acid",           "OC(=O)CCC(=O)O",           "C110156"),
    ("lactic acid",             "CC(O)C(=O)O",              "C50215"),
    # Esters
    ("methyl acetate",          "COC(C)=O",                 "C79209"),
    ("ethyl acetate",           "CCOC(C)=O",                "C141786"),
    ("propyl acetate",          "CCCOC(C)=O",               "C109604"),
    ("butyl acetate",           "CCCCOC(C)=O",              "C123864"),
    ("methyl formate",          "COC=O",                    "C107313"),
    ("ethyl formate",           "CCOC=O",                   "C109944"),
    ("methyl propanoate",       "CCOC(=O)C",                "C554122"),
    ("dimethyl carbonate",      "COC(=O)OC",                "C616386"),
    ("ethyl benzoate",          "CCOC(=O)c1ccccc1",         "C93890"),
    # Amines
    ("methylamine",             "CN",                       "C74895"),
    ("dimethylamine",           "CNC",                      "C124403"),
    ("trimethylamine",          "CN(C)C",                   "C75503"),
    ("ethylamine",              "CCN",                      "C75047"),
    ("diethylamine",            "CCNCC",                    "C109897"),
    ("triethylamine",           "CCN(CC)CC",                "C121448"),
    ("propylamine",             "CCCN",                     "C107108"),
    ("butylamine",              "CCCCN",                    "C109739"),
    ("aniline",                 "Nc1ccccc1",                "C62533"),
    ("N-methylaniline",         "CNc1ccccc1",               "C100618"),
    ("diphenylamine",           "c1ccc(Nc2ccccc2)cc1",     "C122394"),
    ("cyclohexylamine",         "NC1CCCCC1",                "C108918"),
    # Amides
    ("formamide",               "NC=O",                     "C75127"),
    ("acetamide",               "CC(N)=O",                  "C60355"),
    ("N-methylacetamide",       "CNC(C)=O",                 "C79163"),
    ("N,N-dimethylformamide",   "CN(C)C=O",                 "C68122"),
    ("N,N-dimethylacetamide",   "CN(C)C(C)=O",              "C127191"),
    ("benzamide",               "NC(=O)c1ccccc1",           "C55210"),
    # Nitriles
    ("acetonitrile",            "CC#N",                     "C75058"),
    ("propionitrile",           "CCC#N",                    "C107120"),
    ("butyronitrile",           "CCCC#N",                   "C109740"),
    ("benzonitrile",            "N#Cc1ccccc1",              "C100470"),
    ("acrylonitrile",           "C=CC#N",                   "C107131"),
    # Alkanes
    ("pentane",                 "CCCCC",                    "C109660"),
    ("hexane",                  "CCCCCC",                   "C110543"),
    ("heptane",                 "CCCCCCC",                  "C142825"),
    ("octane",                  "CCCCCCCC",                 "C111659"),
    ("nonane",                  "CCCCCCCCC",                "C111842"),
    ("decane",                  "CCCCCCCCCC",               "C124185"),
    ("isopentane",              "CCC(C)C",                  "C78784"),
    ("neopentane",              "CC(C)(C)C",                "C463821"),
    ("cyclopentane",            "C1CCCC1",                  "C287923"),
    ("cyclohexane",             "C1CCCCC1",                 "C110827"),
    ("methylcyclohexane",       "CC1CCCCC1",                "C108872"),
    # Alkenes
    ("ethylene",                "C=C",                      "C74851"),
    ("propylene",               "CC=C",                     "C115071"),
    ("1-butene",                "CCC=C",                    "C106989"),
    ("isobutylene",             "CC(C)=C",                  "C115117"),
    ("1-pentene",               "CCCC=C",                   "C109676"),
    ("styrene",                 "C=Cc1ccccc1",              "C100425"),
    ("1,3-butadiene",           "C=CC=C",                   "C106990"),
    ("cyclohexene",             "C1=CCCCC1",                "C110838"),
    # Alkynes
    ("acetylene",               "C#C",                      "C74862"),
    ("propyne",                 "CC#C",                     "C74997"),
    ("1-butyne",                "CCC#C",                    "C107006"),
    ("1-pentyne",               "CCCC#C",                   "C627191"),
    # Aromatics
    ("benzene",                 "c1ccccc1",                  "C71432"),
    ("toluene",                 "Cc1ccccc1",                 "C108883"),
    ("ethylbenzene",            "CCc1ccccc1",               "C100414"),
    ("o-xylene",                "Cc1ccccc1C",               "C95476"),
    ("m-xylene",                "Cc1cccc(C)c1",             "C108383"),
    ("p-xylene",                "Cc1ccc(C)cc1",             "C106423"),
    ("naphthalene",             "c1ccc2ccccc2c1",           "C91203"),
    ("biphenyl",                "c1ccc(-c2ccccc2)cc1",      "C92524"),
    ("indene",                  "C1=Cc2ccccc21",            "C95136"),
    ("anthracene",              "c1ccc2cc3ccccc3cc2c1",     "C120127"),
    # Halogenated
    ("chloromethane",           "CCl",                      "C74873"),
    ("dichloromethane",         "ClCCl",                    "C75092"),
    ("chloroform",              "ClC(Cl)Cl",                "C67663"),
    ("carbon tetrachloride",    "ClC(Cl)(Cl)Cl",            "C56235"),
    ("chloroethane",            "CCCl",                     "C75003"),
    ("1,2-dichloroethane",      "ClCCCl",                   "C107062"),
    ("chlorobenzene",           "Clc1ccccc1",               "C108907"),
    ("1,2-dichlorobenzene",     "Clc1ccccc1Cl",             "C95501"),
    ("bromethane",              "CCBr",                     "C74964"),
    ("bromoform",               "BrC(Br)Br",                "C75252"),
    ("bromobenzene",            "Brc1ccccc1",               "C108861"),
    ("fluorobenzene",           "Fc1ccccc1",                "C462066"),
    ("iodoethane",              "CCI",                      "C75030"),
    # Ethers
    ("diethyl ether",           "CCOCC",                    "C60297"),
    ("diisopropyl ether",       "CC(C)OC(C)C",              "C108203"),
    ("methyl tert-butyl ether", "CC(C)(C)OC",               "C1634044"),
    ("tetrahydrofuran",         "C1CCOC1",                  "C109999"),
    ("1,4-dioxane",             "C1COCCO1",                 "C123911"),
    ("anisole",                 "COc1ccccc1",               "C100663"),
    ("diphenyl ether",          "O(c1ccccc1)c1ccccc1",     "C101848"),
    ("1,2-dimethoxyethane",     "COCCOC",                   "C110714"),
    # Sulfur compounds
    ("dimethyl sulfoxide",      "CS(C)=O",                  "C67685"),
    ("dimethyl sulfide",        "CSC",                      "C75183"),
    ("diethyl sulfide",         "CCSCC",                    "C352939"),
    ("carbon disulfide",        "S=C=S",                    "C75150"),
    ("thiophene",               "c1ccsc1",                  "C110021"),
    ("thiophenol",              "Sc1ccccc1",                "C108985"),
    ("dimethyl sulfone",        "CS(C)(=O)=O",              "C67715"),
    # Nitrogen heterocycles
    ("pyridine",                "c1ccncc1",                 "C110861"),
    ("pyrimidine",              "c1cnccn1",                 "C289"),
    ("pyrrole",                 "c1cc[nH]c1",               "C109977"),
    ("indole",                  "c1ccc2[nH]ccc2c1",         "C120726"),
    ("imidazole",               "c1cn[nH]c1",               "C288329"),
    ("quinoline",               "c1ccc2ncccc2c1",           "C91222"),
    ("piperidine",              "C1CCNCC1",                 "C110894"),
    ("morpholine",              "C1COCCN1",                 "C110918"),
    ("piperazine",              "C1CNCCN1",                 "C110851"),
    # Oxygen heterocycles
    ("furan",                   "c1ccoc1",                  "C110009"),
    ("benzofuran",              "c1ccc2occc2c1",            "C271896"),
    ("2-methylfuran",           "Cc1ccco1",                 "C534222"),
    ("tetrahydropyran",         "C1CCOCC1",                 "C142587"),
    # Nitro compounds
    ("nitromethane",            "C[N+](=O)[O-]",            "C75525"),
    ("nitroethane",             "CC[N+](=O)[O-]",           "C79243"),
    ("nitrobenzene",            "O=[N+]([O-])c1ccccc1",    "C98953"),
    ("2-nitrotoluene",          "Cc1ccccc1[N+](=O)[O-]",   "C88722"),
    ("4-nitrotoluene",          "Cc1ccc([N+](=O)[O-])cc1", "C99990"),
    # Anhydrides and acyl halides
    ("acetic anhydride",        "CC(=O)OC(C)=O",            "C108247"),
    ("acetyl chloride",         "CC(Cl)=O",                 "C75365"),
    ("benzoyl chloride",        "O=C(Cl)c1ccccc1",          "C98884"),
    # Misc important molecules
    ("water",                   "O",                        "C7732185"),
    ("acetonitrile",            "CC#N",                     "C75058"),
    ("dimethylformamide",       "CN(C)C=O",                 "C68122"),
    ("carbon disulfide",        "S=C=S",                    "C75150"),
    ("acrolein",                "C=CC=O",                   "C107028"),
    ("crotonaldehyde",          "CC=CC=O",                  "C123739"),
    ("methyl acrylate",         "C=CC(=O)OC",               "C96333"),
    ("ethyl acrylate",          "C=CC(=O)OCC",              "C140885"),
    ("methyl methacrylate",     "C=C(C)C(=O)OC",            "C80626"),
    ("caprolactam",             "O=C1CCCCCN1",              "C105602"),
    ("butyrolactone",           "O=C1CCCO1",                "C96484"),
    ("propylene oxide",         "CC1CO1",                   "C75569"),
    ("epichlorohydrin",         "ClCC1CO1",                 "C106898"),
    ("dimethyl phthalate",      "COC(=O)c1ccccc1C(=O)OC",  "C131113"),
    ("salicylaldehyde",         "O=Cc1ccccc1O",             "C90028"),
    ("vanillin",                "O=Cc1ccc(O)c(OC)c1",      "C121335"),
    ("coumarin",                "O=c1ccc2ccccc2o1",         "C91640"),
    ("caffeine",                "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "C58082"),
    ("glucose",                 "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", "C50997"),
    ("fructose",                "OC[C@@H]1OC(O)(CO)[C@@H](O)[C@@H]1O",    "C57483"),
    ("urea",                    "NC(N)=O",                  "C57136"),
    ("thiourea",                "NC(N)=S",                  "C62566"),
    ("succinimide",             "O=C1CCC(=O)N1",            "C123561"),
    ("maleic anhydride",        "O=C1OC(=O)C=C1",           "C108316"),
    ("phthalic anhydride",      "O=C1OC(=O)c2ccccc21",     "C85449"),
    ("adipic acid",             "OC(=O)CCCCC(=O)O",         "C124049"),
    ("caprylic acid",           "CCCCCCCC(=O)O",            "C124072"),
    ("lauric acid",             "CCCCCCCCCCCC(=O)O",        "C143077"),
    ("stearic acid",            "CCCCCCCCCCCCCCCCCC(=O)O",  "C57114"),
    ("oleic acid",              "CCCCCCCC/C=C\\CCCCCCCC(=O)O", "C112801"),
    ("cholesterol",             "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C", "C57885"),
]

# ── JCAMP-DX parser ────────────────────────────────────────────────────────────

def parse_jdx(jdx_text):
    lines = jdx_text.splitlines()

    def get_header(key):
        for line in lines:
            if line.upper().startswith(f'##{key.upper()}='):
                return line.split('=', 1)[1].strip()
        return None

    try:
        xfactor = float(get_header('XFACTOR') or 1.0)
        yfactor = float(get_header('YFACTOR') or 1.0)
        firstx  = float(get_header('FIRSTX')  or 0.0)
        lastx   = float(get_header('LASTX')   or 0.0)
        npoints = int(get_header('NPOINTS')   or 0)
    except (TypeError, ValueError):
        return None, None

    xydata_type = (get_header('XYDATA') or '').upper()
    y_vals = []

    if 'X++' in xydata_type:
        in_data = False
        for line in lines:
            if '##XYDATA=' in line.upper():
                in_data = True
                continue
            if in_data:
                if line.startswith('##'):
                    break
                line = line.strip()
                if not line:
                    continue
                tokens = re.split(r'[\s,]+', line)
                for tok in tokens[1:]:
                    try:
                        y_vals.append(float(tok) * yfactor)
                    except ValueError:
                        pass
        if not y_vals or npoints == 0:
            return None, None
        x_vals = np.linspace(firstx * xfactor, lastx * xfactor, len(y_vals))
        y_vals = np.array(y_vals)
    else:
        x_vals, y_vals = [], []
        in_data = False
        for line in lines:
            if re.match(r'##(PEAK TABLE|XYPOINTS)=', line, re.IGNORECASE):
                in_data = True
                continue
            if in_data:
                if line.startswith('##'):
                    break
                parts = re.split(r'[\s,;]+', line.strip())
                if len(parts) >= 2:
                    try:
                        x_vals.append(float(parts[0]) * xfactor)
                        y_vals.append(float(parts[1]) * yfactor)
                    except ValueError:
                        pass
        if len(x_vals) < 2:
            return None, None
        x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)

    if x_vals[0] > x_vals[-1]:
        x_vals = x_vals[::-1]
        y_vals = y_vals[::-1]

    return x_vals, y_vals


def interpolate(x, y):
    mask = (x >= WAVENUMBER_MIN) & (x <= WAVENUMBER_MAX)
    xc, yc = x[mask], y[mask]
    if len(xc) < 2:
        return None
    f = interp1d(xc, yc, kind='linear', bounds_error=False,
                 fill_value=(yc[0], yc[-1]))
    return f(np.linspace(WAVENUMBER_MIN, WAVENUMBER_MAX, N_POINTS)).astype(np.float32)


# ── Method 1: nistchempy ───────────────────────────────────────────────────────

def try_nistchempy(nist_id):
    try:
        compound = nist.get_compound(nist_id)
        if compound is None:
            return None
        compound.get_ir_spectra()
        for spec in compound.ir_specs:
            if hasattr(spec, 'jdx_text') and spec.jdx_text:
                x, y = parse_jdx(spec.jdx_text)
                if x is not None:
                    pts = interpolate(x, y)
                    if pts is not None:
                        return pts
    except Exception:
        pass
    return None


# ── Method 2: direct NIST JDX URL ─────────────────────────────────────────────

def try_direct_jdx(nist_id):
    headers = {'User-Agent': 'Mozilla/5.0 (IR research data collection)'}
    for spec_type in range(1, 6):
        url = (f"https://webbook.nist.gov/cgi/cbook.cgi"
               f"?JCAMP={nist_id}&Index={spec_type}&Type=IR")
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            time.sleep(0.5)
            if resp.status_code != 200 or '##TITLE=' not in resp.text:
                continue
            x, y = parse_jdx(resp.text)
            if x is None:
                continue
            pts = interpolate(x, y)
            if pts is not None:
                print(f"    → direct JDX (spectrum type {spec_type})")
                return pts
        except Exception:
            continue
    return None


# ── Fetch one compound ─────────────────────────────────────────────────────────

def fetch_compound(name, smiles, nist_id):
    pts = try_nistchempy(nist_id)
    if pts is not None:
        print(f"  OK via nistchempy")
        return pts
    time.sleep(DELAY)
    pts = try_direct_jdx(nist_id)
    return pts


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Deduplicate compound list by name (in case of duplicates above)
    seen = set()
    unique_compounds = []
    for entry in COMPOUNDS:
        if entry[0] not in seen:
            seen.add(entry[0])
            unique_compounds.append(entry)

    rows = []
    failed = []

    for i, (name, smiles, nist_id) in enumerate(unique_compounds):
        print(f"[{i+1}/{len(unique_compounds)}] {name} ({nist_id})")
        pts = fetch_compound(name, smiles, nist_id)
        time.sleep(DELAY)

        if pts is not None:
            row = {'Name of molecule': name, 'SMILES': smiles}
            for j, val in enumerate(pts, start=1):
                row[f'point {j}'] = round(float(val), 6)
            rows.append(row)
            print()
        else:
            failed.append(name)
            print(f"  FAILED\n")

    if not rows:
        print("No data collected. Check your internet connection.")
        return

    point_cols = [f'point {i}' for i in range(1, N_POINTS + 1)]
    df = pd.DataFrame(rows, columns=['Name of molecule', 'SMILES'] + point_cols)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n{'='*50}")
    print(f"Saved {len(df)}/{len(unique_compounds)} molecules → '{OUTPUT_FILE}'")
    if failed:
        print(f"Failed ({len(failed)}): {', '.join(failed)}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()