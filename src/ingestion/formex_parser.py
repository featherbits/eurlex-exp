import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
from pathlib import Path


# ============================================================
# TEXT EXTRACTION HELPERS
# ============================================================

def _extract_text(node: ET.Element) -> str:
    """
    Recursively extract visible text from a FORMEX node, ignoring formatting semantics
    but keeping all textual content. Normalizes whitespace.
    """
    parts: List[str] = []

    def walk(n: ET.Element):
        if n.text:
            parts.append(n.text)
        for child in n:
            walk(child)
            if child.tail:
                parts.append(child.tail)

    walk(node)
    text = " ".join(" ".join(parts).split())
    return text.strip()


def _extract_title(root: ET.Element) -> Optional[str]:
    title_el = root.find(".//TITLE")
    if title_el is None:
        return None
    ti = title_el.find(".//TI")
    if ti is None:
        return _extract_text(title_el) or None
    return _extract_text(ti) or None


def _extract_recitals(root: ET.Element) -> List[Dict[str, Any]]:
    recitals: List[Dict[str, Any]] = []
    for consid in root.findall(".//PREAMBLE/GR.CONSID/CONSID"):
        np = consid.find(".//NP")
        if np is None:
            continue
        no_p = np.find(".//NO.P")
        number = _extract_text(no_p) if no_p is not None else None
        txt = np.find(".//TXT")
        text = _extract_text(txt) if txt is not None else _extract_text(np)
        text = text or ""
        if text.strip():
            recitals.append(
                {
                    "number": number,
                    "text": text,
                }
            )
    return recitals


def _extract_list(list_el: ET.Element) -> Dict[str, Any]:
    style = list_el.attrib.get("TYPE")
    items: List[Dict[str, Any]] = []

    for item in list_el.findall("./ITEM"):
        np = item.find(".//NP")
        if np is None:
            text = _extract_text(item)
            if text:
                items.append({"number": None, "text": text})
            continue

        no_p = np.find(".//NO.P")
        number = _extract_text(no_p) if no_p is not None else None

        txt = np.find(".//TXT")
        text = _extract_text(txt) if txt is not None else _extract_text(np)
        text = text or ""
        if text.strip():
            items.append({"number": number, "text": text})

    return {
        "type": "list",
        "style": style,
        "items": items,
    }


def _extract_blocks_from_container(container: ET.Element) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []

    for child in container:
        tag = child.tag

        if tag.endswith("ALINEA"):
            text = _extract_text(child)
            if text:
                blocks.append({"type": "paragraph", "text": text})
            for list_el in child.findall(".//LIST"):
                blocks.append(_extract_list(list_el))

        elif tag.endswith("LIST"):
            blocks.append(_extract_list(child))

        elif tag.endswith("P"):
            text = _extract_text(child)
            if text:
                blocks.append({"type": "paragraph", "text": text})

    return blocks


def _extract_articles(root: ET.Element) -> List[Dict[str, Any]]:
    articles: List[Dict[str, Any]] = []

    for art in root.findall(".//ARTICLE"):
        identifier = art.attrib.get("IDENTIFIER")
        ti_art = art.find("./TI.ART")
        title = _extract_text(ti_art) if ti_art is not None else None

        blocks = _extract_blocks_from_container(art)

        articles.append(
            {
                "number": identifier,
                "title": title,
                "blocks": blocks,
            }
        )

    return articles


def _extract_annexes(root: ET.Element) -> List[Dict[str, Any]]:
    annexes: List[Dict[str, Any]] = []

    for annex in root.findall(".//ANNEX"):
        title_el = annex.find("./TITLE")
        title = _extract_text(title_el) if title_el is not None else None

        contents = annex.find("./CONTENTS")
        if contents is None:
            blocks = _extract_blocks_from_container(annex)
        else:
            blocks = _extract_blocks_from_container(contents)

        annexes.append(
            {
                "title": title,
                "blocks": blocks,
            }
        )

    return annexes


# ============================================================
# SINGLE-FILE PARSER
# ============================================================

def parse_formex_file(xml_path: str, language: str) -> Optional[Dict[str, Any]]:
    try:
        tree = ET.parse(xml_path)
    except Exception:
        return None

    root = tree.getroot()

    doc: Dict[str, Any] = {
        "celex": None,
        "language": language,
        "title": _extract_title(root),
        "document_type": None,
        "date_document": None,
        "date_effect": None,
        "eurovoc": [],
        "recitals": _extract_recitals(root),
        "articles": _extract_articles(root),
        "annexes": _extract_annexes(root),
        "metadata": {},
    }

    return doc


# ============================================================
# MULTI-FRAGMENT MERGING LOGIC (Scenario A)
# ============================================================

IGNORE_SUFFIXES = (
    ".doc.xml",
    ".doc.fmx.xml",
    ".toc.xml",
    ".toc.fmx.xml",
    ".tif",
    ".log",
    ".ds_store",
)

IGNORE_EXACT = {
    "[cellar] content metadata datastream",
}


def _is_ignored_file(filename: str) -> bool:
    f = filename.lower()
    if f in IGNORE_EXACT:
        return True
    return any(f.endswith(sfx) for sfx in IGNORE_SUFFIXES)


def _find_all_valid_xml(fmx4_dir: Path) -> List[Path]:
    xmls = []
    for root, dirs, files in os.walk(fmx4_dir):
        for f in files:
            f_lower = f.lower()
            if not f_lower.endswith(".xml"):
                continue
            if _is_ignored_file(f_lower):
                continue
            xmls.append(Path(root) / f)
    return xmls

def debug_dump_merged_xml(root: ET.Element, out_path: Path):
    xml_string = ET.tostring(root, encoding="unicode")
    out_path.write_text(xml_string, encoding="utf-8")

def parse_formex_directory(formex_dir: str, language: str) -> Optional[Dict[str, Any]]:
    """
    Full ACT merging version using only xml.etree.ElementTree.
    Preserves original XML structure exactly (Scenario A).
    Keeps only the first <TITLE> encountered (A3).
    """
    fmx4_dir = Path(formex_dir) / "fmx4"
    if not fmx4_dir.is_dir():
        print(f"No fmx4 directory in {formex_dir}")
        return None

    # 1. Collect all valid XML files
    xml_paths = _find_all_valid_xml(fmx4_dir)
    if not xml_paths:
        print(f"No valid XML files in {formex_dir}")
        return None

    # 2. Parse all XML fragments
    fragments = []
    for p in sorted(xml_paths):
        try:
            tree = ET.parse(str(p))
            fragments.append(tree)
        except Exception:
            continue

    if not fragments:
        print(f"Failed to parse any XML in {formex_dir}")
        return None

    # 3. Classify fragments
    acts = []
    annexes = []
    others = []

    for tree in fragments:
        root = tree.getroot()
        tag = root.tag.upper()

        if tag == "ACT":
            acts.append(tree)
        elif tag == "ANNEX":
            annexes.append(tree)
        else:
            others.append(tree)

    # If no ACT exists, fallback to first OTHER
    if not acts:
        acts = [others.pop(0)] if others else [fragments[0]]

    # 4. Merge ACT fragments
    base = acts[0]
    base_root = base.getroot()

    base_preamble = base_root.find("PREAMBLE")
    base_terms = base_root.find("ENACTING.TERMS")

    for frag in acts[1:]:
        root = frag.getroot()

        # PREAMBLE
        pre = root.find("PREAMBLE")
        if pre is not None and base_preamble is not None:
            for child in list(pre):
                base_preamble.append(child)

        # ENACTING.TERMS
        terms = root.find("ENACTING.TERMS")
        if terms is not None and base_terms is not None:
            for child in list(terms):
                base_terms.append(child)

        # ANNEX blocks
        for annex in root.findall("ANNEX"):
            base_root.append(annex)

    # 5. Attach ANNEX fragments
    for ann in annexes:
        base_root.append(ann.getroot())

    # 6. Attach OTHER fragments into ENACTING.TERMS
    if base_terms is not None:
        for other in others:
            for child in list(other.getroot()):
                base_terms.append(child)

    # 7. Convert merged tree to string and feed into existing parser
    xml_string = ET.tostring(base_root, encoding="utf-8")
    try:
        merged_root = ET.fromstring(xml_string)
    except Exception:
        print(f"Failed to parse merged XML in {formex_dir}")
        return None

    # 8. Reuse your existing extraction logic
    doc: Dict[str, Any] = {
        "celex": None,
        "language": language,
        "title": _extract_title(merged_root),
        "document_type": None,
        "date_document": None,
        "date_effect": None,
        "eurovoc": [],
        "recitals": _extract_recitals(merged_root),
        "articles": _extract_articles(merged_root),
        "annexes": _extract_annexes(merged_root),
        "metadata": {},
    }

    # DEBUG: dump merged XML for inspection
    # debug_dump_merged_xml(merged_root, Path(formex_dir) / "merged_debug.xml")

    return doc
