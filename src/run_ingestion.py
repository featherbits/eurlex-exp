import os

from ingestion.ingestion_pipeline import ingest_language

# build metadata json documents
def main():
    # Root directories for FMX and MTD dumps
    # Example structure:
    #   ./data/cellar_raw/fmx/en/LEG_EN_FMX_20260201_01_00/<UUID>/...
    #   ./data/cellar_raw/mtd/en/LEG_MTD_20260201_01_00/<UUID>/tree_non_inferred.rdf
    fmx_root = os.path.join(".", "data", "cellar_raw", "fmx")
    mtd_root = os.path.join(".", "data", "cellar_raw", "mtd")

    # Where unified JSON will be written:
    #   ./data/cellar_json/<lang>/<CELEX>.json
    output_root = os.path.join(".", "data", "cellar_json")

    # Extend this list as you add more languages
    languages = ["en"]

    for lang in languages:
        print(f"=== Ingesting language: {lang} ===")
        ingest_language(
            fmx_root=fmx_root,
            mtd_root=mtd_root,
            language=lang,
            output_root=output_root,
        )


if __name__ == "__main__":
    main()
