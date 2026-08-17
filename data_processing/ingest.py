"""
Module d'ingestion du corpus FORCE-N réel.

Le corpus fourni par l'encadrant se présente sous forme de fichiers .txt
individuels (un par page scrapée), chacun avec un en-tête structuré :

    URL: https://force-n.sn/...
    SCRAPED_AT_UTC: 2026-05-08T18:48:48.063648+00:00
    TITLE: Titre de la page | FORCE-N
    <contenu de la page>

Ce script parcourt un dossier contenant ces fichiers, extrait les
métadonnées et le contenu de chacun, déduit une catégorie à partir de
l'URL, et produit un fichier JSON unique au format attendu par
data_processing/cleaning.py (la suite du pipeline).
"""

import json
import re
from pathlib import Path
from urllib.parse import urlparse

RAW_TXT_FOLDER = "data/raw_corpus_txt"   # dossier où placer les .txt reçus
OUTPUT_PATH = "data/sample_corpus.json"  # même nom que le corpus factice :
                                          # le reste du pipeline n'a rien à changer


def infer_category(url: str) -> str:
    """
    Déduit une catégorie à partir du chemin de l'URL, en l'absence d'une
    colonne "catégorie" explicite dans index.csv fourni par l'encadrant.
    Les règles ci-dessous sont basées sur l'arborescence réelle du site
    observée dans le corpus (index.csv).
    """
    path = urlparse(url).path.lower()

    if "/articles/" in path:
        return "actualite"
    if "actualites" in path:
        return "actualite"
    if "alumni" in path or "temoignage" in path:
        return "temoignage"
    if "opportunite" in path:
        return "opportunite"
    if "partenaire" in path:
        return "partenaire"
    if "faq" in path:
        return "faq"
    if "formation" in path or "catalogue" in path or "composantes" in path \
            or "sigui" in path or "parcours-initiatique" in path:
        return "formation"
    if "promotion-des-sciences" in path:
        return "formation"
    if "webinaire" in path:
        return "webinaire"
    if path in ("/", "/node/1") or path.startswith("/node/1"):
        return "accueil"
    if "a-propos" in path or "qui-sommes-nous" in path or "missions" in path \
            or "les-services" in path:
        return "a_propos"

    return "autre"


def parse_scraped_file(file_path: Path) -> dict | None:
    """
    Parse un fichier .txt scrapé et retourne un dictionnaire
    {url, title, content, category, last_modified}, ou None si le
    fichier ne respecte pas le format attendu (pour ne pas faire
    planter tout le lot sur un seul fichier corrompu).
    """
    text = file_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    url, scraped_at, title = None, None, None
    body_start_index = 0

    for i, line in enumerate(lines):
        if line.startswith("URL:"):
            url = line[len("URL:"):].strip()
        elif line.startswith("SCRAPED_AT_UTC:"):
            scraped_at = line[len("SCRAPED_AT_UTC:"):].strip()
        elif line.startswith("TITLE:"):
            title = line[len("TITLE:"):].strip()
            body_start_index = i + 1
            break  # les 3 lignes d'en-tête sont toujours dans cet ordre

    if not url or not title:
        print(f"  Ignoré (en-tête incomplet) : {file_path.name}")
        return None

    # Le titre contient souvent " | FORCE-N" à la fin, on le retire
    # pour un titre plus propre
    clean_title = re.sub(r"\s*\|\s*FORCE-N\s*$", "", title)

    content = "\n".join(lines[body_start_index:]).strip()

    return {
        "title": clean_title,
        "content": content,
        "category": infer_category(url),
        "url": url,
        "last_modified": scraped_at or "",
    }


def ingest_corpus(txt_folder: str = RAW_TXT_FOLDER, output_path: str = OUTPUT_PATH) -> list[dict]:
    """
    Parcourt tous les fichiers .txt d'un dossier, les convertit, et
    sauvegarde le résultat au format JSON attendu par le reste du pipeline.
    """
    folder = Path(txt_folder)
    if not folder.exists():
        raise FileNotFoundError(
            f"Le dossier {txt_folder} n'existe pas. Crée-le et places-y "
            f"tous les fichiers .txt du corpus (et index.csv si tu veux, "
            f"il est ignoré par ce script)."
        )

    txt_files = sorted(folder.glob("*.txt"))
    print(f"{len(txt_files)} fichier(s) .txt trouvé(s) dans {txt_folder}")

    pages = []
    for file_path in txt_files:
        page = parse_scraped_file(file_path)
        if page:
            pages.append(page)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)

    print(f"{len(pages)} page(s) converties et sauvegardées dans : {output_path}")

    # Petit récapitulatif par catégorie, utile pour vérifier que la
    # déduction de catégorie a bien fonctionné
    from collections import Counter
    category_counts = Counter(page["category"] for page in pages)
    print("\nRépartition par catégorie :")
    for category, count in category_counts.most_common():
        print(f"  {category} : {count}")

    return pages


if __name__ == "__main__":
    ingest_corpus()