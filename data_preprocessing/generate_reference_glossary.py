import json
import re


def normalize_gloss(gloss):
    """
    Remove trailing numeric variant.
    
    BANK1 -> BANK
    BANK2 -> BANK
    UNIVERSITY -> UNIVERSITY
    """

    return re.sub(r"\d+$", "", gloss)



def generate_canonical_glossary(
        mapping_file,
        output_file
):

    with open(mapping_file, "r", encoding="utf-8") as f:
        mapping = json.load(f)


    best_variants = {}


    for key, item in mapping.items():

        base = normalize_gloss(key).upper()

        score = item.get(
            "score",
            0
        )


        if (
            base not in best_variants
            or score > best_variants[base]["score"]
        ):

            best_variants[base] = {
                "variant": key,
                "score": score
            }


    glossary = {
        "source": "ASL_SIGN_MAPPING_DATABASE",
        "description":
            "Canonical ASL gloss vocabulary generated from highest scoring sign assets.",
        "glosses": sorted(
            list(best_variants.keys())
        )
    }


    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            glossary,
            f,
            indent=2,
            ensure_ascii=False
        )


if __name__ == "__main__":

    generate_canonical_glossary(
        "./data_preprocessing/best_asl_videos.json",
        "canonical_glossary.json"
    )