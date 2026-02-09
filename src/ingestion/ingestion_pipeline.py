import os
from typing import Optional

from .merge_formex_mtd import merge_formex_mtd


def _resolve_single_subdir(root: str) -> Optional[str]:
    """
    Given e.g. data/cellar_raw/fmx/en, which contains a single dump directory
    like LEG_EN_FMX_20260201_01_00, return its full path.
    """
    if not os.path.isdir(root):
        return None
    entries = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if not entries:
        return None
    # If there are multiple, you can later add selection logic; for now take the first.
    return os.path.join(root, sorted(entries)[0])


def ingest_language(
    fmx_root: str,
    mtd_root: str,
    language: str,
    output_root: str,
) -> None:
    """
    Full ingestion pipeline for one language:
      - resolve FMX and MTD dump directories
      - walk UUIDs
      - merge FMX + MTD
      - save CELEX.json
    Expected layout:
      FMX: data/cellar_raw/fmx/en/LEG_EN_FMX_.../<UUID>/...xml
      MTD: data/cellar_raw/mtd/en/LEG_MTD_.../<UUID>/tree_non_inferred.rdf
    """

    # Resolve language roots
    fmx_lang_root = os.path.join(fmx_root, language)  # e.g. data/cellar_raw/fmx/en
    mtd_lang_root = os.path.join(mtd_root, language)  # e.g. data/cellar_raw/mtd/en

    fmx_dump_root = _resolve_single_subdir(fmx_lang_root)
    mtd_dump_root = _resolve_single_subdir(mtd_lang_root)

    if not fmx_dump_root or not mtd_dump_root:
        print(f"[{language}] FMX or MTD dump root not found.")
        return

    print(f"[{language}] FMX dump root: {fmx_dump_root}")
    print(f"[{language}] MTD dump root: {mtd_dump_root}")

    out_dir = os.path.join(output_root, language)
    os.makedirs(out_dir, exist_ok=True)

    success = 0
    skipped = 0

    for uuid in sorted(os.listdir(fmx_dump_root)):
        formex_dir = os.path.join(fmx_dump_root, uuid)
        if not os.path.isdir(formex_dir):
            continue

        mtd_rdf_path = os.path.join(mtd_dump_root, uuid, "tree_non_inferred.rdf")
        if not os.path.isfile(mtd_rdf_path):
            print(f"[{language}] UUID {uuid} skipped (no RDF)")
            skipped += 1
            continue

        try:
            celex = merge_formex_mtd(
                formex_dir=formex_dir,
                mtd_rdf_path=mtd_rdf_path,
                output_dir=out_dir,
                language=language,
                uuid=uuid,
            )
        except Exception as e:
            print(f"[{language}] UUID {uuid} failed: {e}")
            skipped += 1
            continue

        if celex:
            print(f"[{language}] UUID {uuid} → CELEX {celex}")
            success += 1
        else:
            print(f"[{language}] UUID {uuid} skipped (no CELEX or parse failure)")
            skipped += 1

    print(f"[{language}] Done. {success} CELEX docs ingested, {skipped} skipped.")
