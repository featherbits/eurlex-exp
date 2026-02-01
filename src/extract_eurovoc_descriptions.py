# src/extract_eurovoc_descriptions.py

import json
from pathlib import Path
import rdflib

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

EUROVOC_SKOS_URL = "https://op.europa.eu/o/opportal-service/euvoc-download-handler?cellarURI=http%3A%2F%2Fpublications.europa.eu%2Fresource%2Fdistribution%2Feurovoc%2F20260128-0%2Frdf%2Fskos_ap_act%2Feurovoc-skos-ap-act.rdf&fileName=eurovoc-skos-ap-act.rdf"


def download_skos(path: Path):
    import requests

    print("Downloading EUROVOC SKOS file...")
    r = requests.get(EUROVOC_SKOS_URL)
    r.raise_for_status()

    path.write_bytes(r.content)
    print(f"Saved SKOS file to {path}")


def extract_descriptions():
    skos_path = DATA_DIR / "eurovoc-skos-ap-act.rdf"

    if not skos_path.exists():
        download_skos(skos_path)

    print("Parsing SKOS file (this may take a moment)...")

    g = rdflib.Graph()
    g.parse(str(skos_path))

    SKOS = rdflib.Namespace("http://www.w3.org/2004/02/skos/core#")

    eurovoc = {}

    for s, p, o in g.triples((None, SKOS.prefLabel, None)):
        if o.language == "en":
            # Extract EUROVOC code from URI
            uri = str(s)
            if uri.endswith("/"):
                continue

            code = uri.split("/")[-1]

            eurovoc[code] = str(o)

    # Save mapping
    out_path = DATA_DIR / "eurovoc_descriptions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(eurovoc, f, ensure_ascii=False, indent=2)

    print(f"Saved English EUROVOC descriptors to {out_path}")


if __name__ == "__main__":
    extract_descriptions()
