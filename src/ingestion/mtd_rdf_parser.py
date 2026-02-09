from rdflib import Graph, Namespace, URIRef
from typing import Optional, Dict


def parse_mtd_rdf(rdf_path: str) -> Optional[Dict]:
    """
    Parse MTD RDF metadata file and extract CELEX, document type, dates.
    Adapted to CDM-based RDF where CELEX is in cdm:resource_legal_id_celex.
    """

    g = Graph()
    try:
        g.parse(rdf_path)
    except Exception:
        return None

    CDM = Namespace("http://publications.europa.eu/ontology/cdm#")

    meta: Dict = {
        "celex": None,
        "document_type": None,
        "date_document": None,
        "date_effect": None,
        "eurovoc": [],  # stays empty for your current dumps
    }

    # CELEX
    for _, _, o in g.triples((None, CDM.resource_legal_id_celex, None)):
        meta["celex"] = str(o)

    # Document type (R, L, D, M, etc.)
    for _, _, o in g.triples((None, CDM.resource_legal_type, None)):
        meta["document_type"] = str(o)

    # Date of document
    for _, _, o in g.triples((None, CDM.work_date_document, None)):
        meta["date_document"] = str(o)

    # Date of entry into force (hyphen in predicate → explicit URI)
    date_effect_pred = URIRef(
        "http://publications.europa.eu/ontology/cdm#resource_legal_date_entry-into-force"
    )
    for _, _, o in g.triples((None, date_effect_pred, None)):
        meta["date_effect"] = str(o)

    return meta if meta["celex"] else None
