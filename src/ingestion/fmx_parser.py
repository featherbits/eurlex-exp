# src/ingestion/fmx_parser.py

import os
import xml.etree.ElementTree as ET
from typing import Optional
from .document_model import LegalDocument


def parse_fmx_directory(fmx_dir: str, language: str = "en") -> Optional[LegalDocument]:
    """
    Parse all FMX XML files inside a fmx4/ directory and merge them into a single LegalDocument.
    """

    xml_files = [
        os.path.join(fmx_dir, f)
        for f in os.listdir(fmx_dir)
        if f.endswith(".xml") and not f.endswith(".doc.xml")
    ]

    if not xml_files:
        return None

    doc = LegalDocument(
        celex=None,
        language=language,
        title=None,
        document_type=None,
        date_document=None,
        date_effect=None,
        recitals=[],
        articles=[],
        annexes=[],
        eurovoc=[],
        metadata={"source_files": []}
    )

    # FMX namespaces
    ns = {
        "fmx": "http://publications.europa.eu/resource/schema/fmx4",
        "xhtml": "http://www.w3.org/1999/xhtml"
    }

    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception:
            continue

        doc.metadata["source_files"].append(os.path.basename(xml_path))

        # CELEX (only once)
        if doc.celex is None:
            celex = root.findtext(".//fmx:CELEX", namespaces=ns)
            doc.celex = celex

        # Title (only once)
        if doc.title is None:
            title = root.findtext(".//fmx:title", namespaces=ns)
            doc.title = title

        # Recitals
        for rec in root.findall(".//fmx:recital", namespaces=ns):
            paragraphs = [
                p.text.strip()
                for p in rec.findall(".//xhtml:p", namespaces=ns)
                if p.text
            ]
            if paragraphs:
                doc.recitals.append("\n".join(paragraphs))

        # Articles
        for art in root.findall(".//fmx:article", namespaces=ns):
            number = art.findtext(".//fmx:num", namespaces=ns)
            title = art.findtext(".//fmx:title", namespaces=ns)

            paragraphs = [
                p.text.strip()
                for p in art.findall(".//xhtml:p", namespaces=ns)
                if p.text
            ]

            doc.articles.append({
                "number": number,
                "title": title,
                "paragraphs": paragraphs
            })

        # Annexes
        for ann in root.findall(".//fmx:annex", namespaces=ns):
            number = ann.findtext(".//fmx:num", namespaces=ns)
            title = ann.findtext(".//fmx:title", namespaces=ns)

            paragraphs = [
                p.text.strip()
                for p in ann.findall(".//xhtml:p", namespaces=ns)
                if p.text
            ]

            doc.annexes.append({
                "number": number,
                "title": title,
                "content": "\n".join(paragraphs)
            })

    return doc if doc.celex else None
