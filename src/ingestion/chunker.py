import re
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def sentence_split(text: str) -> List[str]:
    """
    Simple sentence splitter: split on . ? ! followed by space + capital letter.
    """
    pattern = r"(?<=[.!?])\s+(?=[A-Z])"
    parts = re.split(pattern, text)
    return [p.strip() for p in parts if p.strip()]


def count_tokens(text: str) -> int:
    return len(text.split())


def base_metadata(doc: Dict[str, Any]) -> Dict[str, Any]:
    mtd = doc.get("metadata", {}).get("mtd", {})
    return {
        "celex": doc.get("celex"),
        "language": doc.get("language"),
        "document_type": mtd.get("document_type"),
        "date_document": mtd.get("date_document"),
        "date_effect": mtd.get("date_effect"),
        "uuid": doc.get("metadata", {}).get("uuid"),
    }


# ---------------------------------------------------------------------------
# Chunk ID builder
# ---------------------------------------------------------------------------


def build_chunk_id(
    celex: str,
    chunk_type: str,
    article_number: Optional[str] = None,
    block_index: Optional[int] = None,
    recital_number: Optional[str] = None,
    annex_index: Optional[int] = None,
    annex_block_index: Optional[int] = None,
    part: int = 0,
) -> str:
    parts = [celex, chunk_type]

    if chunk_type == "recital":
        parts.append(str(recital_number))

    if chunk_type == "article":
        parts.append(str(article_number))
        if block_index is not None:
            parts.append(str(block_index))

    if chunk_type == "annex":
        parts.append(str(annex_index))
        if annex_block_index is not None:
            parts.append(str(annex_block_index))

    parts.append(str(part))
    return ":".join(parts)


# ---------------------------------------------------------------------------
# Recitals
# ---------------------------------------------------------------------------


def chunk_recital(
    doc: Dict[str, Any],
    recital: Dict[str, Any],
    max_tokens: int = 800,
    min_tokens: int = 150,
) -> List[Dict[str, Any]]:
    celex = doc["celex"]
    number = recital["number"]
    header = f"Recital {number}"
    body = normalize_text(recital["text"])
    full = f"{header} {body}"

    if count_tokens(full) <= max_tokens:
        return [
            {
                "id": build_chunk_id(celex, "recital", recital_number=number, part=0),
                "celex": celex,
                "chunk_type": "recital",
                "recital_number": number,
                "position": 0,
                "text": full,
                "metadata": base_metadata(doc),
            }
        ]

    sentences = sentence_split(body)
    chunks: List[Dict[str, Any]] = []
    buf: List[str] = []
    buf_tokens = 0
    part = 0
    pos = 0

    for s in sentences:
        t = count_tokens(s)
        if buf_tokens + t > max_tokens and buf_tokens >= min_tokens:
            text = " ".join(buf)
            prefix = header if part == 0 else f"{header} (continued)"
            chunks.append(
                {
                    "id": build_chunk_id(
                        celex, "recital", recital_number=number, part=part
                    ),
                    "celex": celex,
                    "chunk_type": "recital",
                    "recital_number": number,
                    "position": pos,
                    "text": f"{prefix} {text}",
                    "metadata": base_metadata(doc),
                }
            )
            part += 1
            pos += 1
            buf = []
            buf_tokens = 0

        buf.append(s)
        buf_tokens += t

    if buf:
        text = " ".join(buf)
        prefix = header if part == 0 else f"{header} (continued)"
        chunks.append(
            {
                "id": build_chunk_id(
                    celex, "recital", recital_number=number, part=part
                ),
                "celex": celex,
                "chunk_type": "recital",
                "recital_number": number,
                "position": pos,
                "text": f"{prefix} {text}",
                "metadata": base_metadata(doc),
            }
        )

    return chunks


# ---------------------------------------------------------------------------
# Article blocks
# ---------------------------------------------------------------------------


def chunk_article_block(
    doc: Dict[str, Any],
    article_number: str,
    block: Dict[str, Any],
    block_index: int,
    max_tokens: int = 800,
    min_tokens: int = 150,
) -> List[Dict[str, Any]]:
    celex = doc["celex"]
    header = f"Article {article_number}"

    if block["type"] == "paragraph":
        body = normalize_text(block["text"])
        full = f"{header} {body}"

        if count_tokens(full) <= max_tokens:
            return [
                {
                    "id": build_chunk_id(
                        celex, "article", article_number, block_index, part=0
                    ),
                    "celex": celex,
                    "chunk_type": "article",
                    "article_number": article_number,
                    "block_index": block_index,
                    "position": 0,
                    "text": full,
                    "metadata": base_metadata(doc),
                }
            ]

        sentences = sentence_split(body)
        chunks: List[Dict[str, Any]] = []
        buf: List[str] = []
        buf_tokens = 0
        part = 0
        pos = 0

        for s in sentences:
            t = count_tokens(s)
            if buf_tokens + t > max_tokens and buf_tokens >= min_tokens:
                text = " ".join(buf)
                prefix = header if part == 0 else f"{header} (continued)"
                chunks.append(
                    {
                        "id": build_chunk_id(
                            celex, "article", article_number, block_index, part
                        ),
                        "celex": celex,
                        "chunk_type": "article",
                        "article_number": article_number,
                        "block_index": block_index,
                        "position": pos,
                        "text": f"{prefix} {text}",
                        "metadata": base_metadata(doc),
                    }
                )
                part += 1
                pos += 1
                buf = []
                buf_tokens = 0

            buf.append(s)
            buf_tokens += t

        if buf:
            text = " ".join(buf)
            prefix = header if part == 0 else f"{header} (continued)"
            chunks.append(
                {
                    "id": build_chunk_id(
                        celex, "article", article_number, block_index, part
                    ),
                    "celex": celex,
                    "chunk_type": "article",
                    "article_number": article_number,
                    "block_index": block_index,
                    "position": pos,
                    "text": f"{prefix} {text}",
                    "metadata": base_metadata(doc),
                }
            )

        return chunks

    if block["type"] == "list":
        chunks: List[Dict[str, Any]] = []
        pos = 0
        for item in block.get("items", []):
            item_text = normalize_text(item["text"])
            full = f"{header} {item['number']} {item_text}"
            chunks.append(
                {
                    "id": build_chunk_id(
                        celex, "article", article_number, block_index, part=pos
                    ),
                    "celex": celex,
                    "chunk_type": "article",
                    "article_number": article_number,
                    "block_index": block_index,
                    "position": pos,
                    "text": full,
                    "metadata": base_metadata(doc),
                }
            )
            pos += 1
        return chunks

    return []


def chunk_article(
    doc: Dict[str, Any],
    article: Dict[str, Any],
    max_tokens: int = 800,
    min_tokens: int = 150,
) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    article_number = article["number"]
    blocks = article.get("blocks", [])

    pos = 0
    for i, block in enumerate(blocks):
        block_chunks = chunk_article_block(
            doc, article_number, block, i, max_tokens=max_tokens, min_tokens=min_tokens
        )
        for ch in block_chunks:
            ch["position_global"] = pos
            pos += 1
        chunks.extend(block_chunks)

    return chunks


# ---------------------------------------------------------------------------
# Annex blocks (same pattern as articles, but with annex title)
# ---------------------------------------------------------------------------


def chunk_annex_block(
    doc: Dict[str, Any],
    annex_index: int,
    annex_title: str,
    block: Dict[str, Any],
    block_index: int,
    max_tokens: int = 800,
    min_tokens: int = 150,
) -> List[Dict[str, Any]]:
    celex = doc["celex"]
    header = f"Annex {annex_index + 1}"
    if annex_title:
        header = f"{header} — {annex_title}"

    if block["type"] == "paragraph":
        body = normalize_text(block["text"])
        full = f"{header} {body}"

        if count_tokens(full) <= max_tokens:
            return [
                {
                    "id": build_chunk_id(
                        celex,
                        "annex",
                        annex_index=annex_index,
                        annex_block_index=block_index,
                        part=0,
                    ),
                    "celex": celex,
                    "chunk_type": "annex",
                    "annex_index": annex_index,
                    "annex_block_index": block_index,
                    "position": 0,
                    "text": full,
                    "metadata": base_metadata(doc),
                }
            ]

        sentences = sentence_split(body)
        chunks: List[Dict[str, Any]] = []
        buf: List[str] = []
        buf_tokens = 0
        part = 0
        pos = 0

        for s in sentences:
            t = count_tokens(s)
            if buf_tokens + t > max_tokens and buf_tokens >= min_tokens:
                text = " ".join(buf)
                prefix = header if part == 0 else f"{header} (continued)"
                chunks.append(
                    {
                        "id": build_chunk_id(
                            celex,
                            "annex",
                            annex_index=annex_index,
                            annex_block_index=block_index,
                            part=part,
                        ),
                        "celex": celex,
                        "chunk_type": "annex",
                        "annex_index": annex_index,
                        "annex_block_index": block_index,
                        "position": pos,
                        "text": f"{prefix} {text}",
                        "metadata": base_metadata(doc),
                    }
                )
                part += 1
                pos += 1
                buf = []
                buf_tokens = 0

            buf.append(s)
            buf_tokens += t

        if buf:
            text = " ".join(buf)
            prefix = header if part == 0 else f"{header} (continued)"
            chunks.append(
                {
                    "id": build_chunk_id(
                        celex,
                        "annex",
                        annex_index=annex_index,
                        annex_block_index=block_index,
                        part=part,
                    ),
                    "celex": celex,
                    "chunk_type": "annex",
                    "annex_index": annex_index,
                    "annex_block_index": block_index,
                    "position": pos,
                    "text": f"{prefix} {text}",
                    "metadata": base_metadata(doc),
                }
            )

        return chunks

    if block["type"] == "list":
        chunks: List[Dict[str, Any]] = []
        pos = 0
        for item in block.get("items", []):
            item_text = normalize_text(item["text"])
            full = f"{header} {item['number']} {item_text}"
            chunks.append(
                {
                    "id": build_chunk_id(
                        celex,
                        "annex",
                        annex_index=annex_index,
                        annex_block_index=block_index,
                        part=pos,
                    ),
                    "celex": celex,
                    "chunk_type": "annex",
                    "annex_index": annex_index,
                    "annex_block_index": block_index,
                    "position": pos,
                    "text": full,
                    "metadata": base_metadata(doc),
                }
            )
            pos += 1
        return chunks

    return []


def chunk_annex(
    doc: Dict[str, Any],
    annex: Dict[str, Any],
    annex_index: int,
    max_tokens: int = 800,
    min_tokens: int = 150,
) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    title = annex.get("title", "")
    blocks = annex.get("blocks", [])

    pos = 0
    for i, block in enumerate(blocks):
        block_chunks = chunk_annex_block(
            doc,
            annex_index=annex_index,
            annex_title=title,
            block=block,
            block_index=i,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
        )
        for ch in block_chunks:
            ch["position_global"] = pos
            pos += 1
        chunks.extend(block_chunks)

    return chunks


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def chunk_document(
    doc: Dict[str, Any],
    max_tokens: int = 800,
    min_tokens: int = 150,
) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    global_pos = 0

    # Recitals
    for recital in doc.get("recitals", []):
        rc = chunk_recital(doc, recital, max_tokens, min_tokens)
        for c in rc:
            c["position_global"] = global_pos
            global_pos += 1
        chunks.extend(rc)

    # Articles
    for article in doc.get("articles", []):
        ac = chunk_article(doc, article, max_tokens, min_tokens)
        for c in ac:
            c["position_global"] = global_pos
            global_pos += 1
        chunks.extend(ac)

    # Annexes
    for idx, annex in enumerate(doc.get("annexes", [])):
        ax = chunk_annex(
            doc, annex, annex_index=idx, max_tokens=max_tokens, min_tokens=min_tokens
        )
        for c in ax:
            c["position_global"] = global_pos
            global_pos += 1
        chunks.extend(ax)

    return chunks
