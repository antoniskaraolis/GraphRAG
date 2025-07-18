# graphrag/utils.py
import re
import datetime
import random

LATEX_ESCAPES = {
    r"\\'a": "á", r"\\'e": "é", r"\\'i": "í", r"\\'o": "ó", r"\\'u": "ú",
    r"\\'A": "Á", r"\\'E": "É", r"\\'I": "Í", r"\\'O": "Ó", r"\\'U": "Ú",
    r'\\"a': "ä", r'\\"e': "ë", r'\\"i': "ï", r'\\"o': "ö", r'\\"u': "ü",
    r'\\"A': "Ä", r'\\"E': "Ë", r'\\"I': "Ï", r'\\"O': "Ö", r'\\"U': "Ü",
    r'\\`a': "à", r'\\`e': "è", r'\\`i': "ì", r'\\`o': "ò", r'\\`u': "ù",
}

CATEGORY_MAP = {
    "Computer Science": [
        "cs.AI", "cs.CL", "cs.CC", "cs.CE", "cs.CV", "cs.CY", "cs.CR",
        "cs.DS", "cs.DB", "cs.DL", "cs.DM", "cs.DC", "cs.ET", "cs.FL",
        "cs.GL", "cs.GR", "cs.GT", "cs.HC", "cs.IR", "cs.IT", "cs.LG",
        "cs.LO", "cs.MS", "cs.MA", "cs.MM", "cs.NI", "cs.NE", "cs.NA",
        "cs.OS", "cs.OH", "cs.PF", "cs.PL", "cs.RO", "cs.SI", "cs.SE",
        "cs.SD", "cs.SC", "cs.SY"
    ],
    "Economics": [
        "econ.EM", "econ.GN", "econ.TH"
    ],
    "Electrical Engineering and Systems Science": [
        "eess.AS", "eess.IV", "eess.SP", "eess.SY"
    ],
    "Mathematics": [
        "math.AG", "math.AT", "math.AP", "math.CT", "math.CA", "math.CO",
        "math.AC", "math.CV", "math.DG", "math.DS", "math.FA", "math.GM",
        "math.GN", "math.GT", "math.GR", "math.HO", "math.IT", "math.KT",
        "math.LO", "math.MP", "math.MG", "math.NT", "math.NA", "math.OA",
        "math.OC", "math.PR", "math.QA", "math.RT", "math.RA", "math.SP",
        "math.ST", "math.SG"
    ],
    "Physics": [
        "physics.acc-ph", "physics.app-ph", "physics.ao-ph", "physics.atom-ph",
        "physics.atm-clus", "physics.bio-ph", "physics.chem-ph", "physics.class-ph",
        "physics.comp-ph", "physics.data-an", "physics.flu-dyn", "physics.gen-ph",
        "physics.geo-ph", "physics.hist-ph", "physics.ins-det", "physics.med-ph",
        "physics.optics", "physics.ed-ph", "physics.soc-ph", "physics.plasm-ph",
        "physics.pop-ph", "physics.space-ph", "quant-ph", "hep-th", "hep-ph",
        "hep-ex", "hep-lat", "astro-ph", "astro-ph.CO", "astro-ph.HE", "astro-ph.GA",
        "astro-ph.IM", "astro-ph.SR", "astro-ph.EP", "cond-mat", "cond-mat.mtrl-sci",
        "cond-mat.mes-hall", "cond-mat.other", "cond-mat.quant-gas", "cond-mat.soft",
        "cond-mat.stat-mech", "cond-mat.str-el", "cond-mat.supr-con", "nucl-ex",
        "nucl-th"
    ],
    "Quantitative Biology": [
        "q-bio.BM", "q-bio.CB", "q-bio.GN", "q-bio.MN", "q-bio.NC", "q-bio.OT",
        "q-bio.PE", "q-bio.QM", "q-bio.SC", "q-bio.TO"
    ],
    "Quantitative Finance": [
        "q-fin.CP", "q-fin.EC", "q-fin.GN", "q-fin.MF", "q-fin.PM", "q-fin.PR",
        "q-fin.RM", "q-fin.ST", "q-fin.TR"
    ],
    "Statistics": [
        "stat.AP", "stat.CO", "stat.ML", "stat.ME", "stat.OT", "stat.TH"
    ]
}

def clean_text(text):
    if not text:
        return ""
    for esc, char in LATEX_ESCAPES.items():
        text = text.replace(esc, char)
    text = re.sub(r'\s+', ' ', text.replace('\n', ' ')).strip()
    return text

def parse_author(author_parts):
    if isinstance(author_parts, list) and len(author_parts) > 0:
        if isinstance(author_parts[0], list):
            if len(author_parts[0]) >= 2:
                return clean_text(f"{author_parts[0][1]} {author_parts[0][0]}")
            return clean_text(author_parts[0][0]) if author_parts[0] else "Unknown"
        else:
            if len(author_parts) >= 2:
                return clean_text(f"{author_parts[1]} {author_parts[0]}")
            return clean_text(author_parts[0]) if author_parts else "Unknown"
    return "Unknown"

def extract_year(version_dates):
    if not version_dates:
        return None
    try:
        date_str = version_dates[0].split(' GMT')[0].strip()
        return datetime.datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S').year
    except Exception:
        return None

def map_to_domain(categories_str):
    if not categories_str:
        return None
    categories = categories_str.split()
    for domain, keywords in CATEGORY_MAP.items():
        for kw in keywords:
            if any(cat == kw or cat.startswith(kw + '.') for cat in categories):
                return domain
    return None
