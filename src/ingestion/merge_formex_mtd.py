import json
import os
from typing import Optional, Dict, Any

from .formex_parser import parse_formex_directory
from .mtd_rdf_parser import parse_mtd_rdf


def merge_formex_mtd(
    formex_dir: str,
    mtd_rdf_path: Optional[str],
    output_dir: str,
    language: str,
    uuid: Optional[str] = None,
) -> Optional[str]:
    """
    Merge one FMX UUID directory with its MTD RDF into a unified JSON document.
    Returns CELEX if successful, otherwise None.
    """

    formex_doc = parse_formex_directory(formex_dir, language=language)
    if formex_doc is None:
        print("No formex doc")
        return None

    mtd_meta: Optional[Dict[str, Any]] = (
        parse_mtd_rdf(mtd_rdf_path) if mtd_rdf_path else None
    )
    if not mtd_meta or not mtd_meta.get("celex"):
        print("Metadata does not contain celex")
        return None

    celex = mtd_meta["celex"]

    # Fill top-level metadata from MTD
    formex_doc["celex"] = celex
    formex_doc["document_type"] = mtd_meta.get("document_type")
    formex_doc["date_document"] = mtd_meta.get("date_document")
    formex_doc["date_effect"] = mtd_meta.get("date_effect")
    formex_doc["eurovoc"] = mtd_meta.get("eurovoc", [])

    # Attach MTD + UUID into metadata
    formex_doc["metadata"]["mtd"] = mtd_meta
    if uuid:
        formex_doc["metadata"]["uuid"] = uuid

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{celex}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(formex_doc, f, ensure_ascii=False, indent=2)

    return celex
