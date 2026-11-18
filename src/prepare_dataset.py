import os
import json
import random
import urllib.request
import zipfile
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Résolution QQVGA exigée par l'ISP et le NNPA de ColibryNPU
TARGET_WIDTH = 160
TARGET_HEIGHT = 120

# Filtrage des boîtes trop petites en 160x120 (min 8px large, 14px haut)
MIN_BOX_WIDTH_PX = 8.0
MIN_BOX_HEIGHT_PX = 14.0


def setup_dirs():
    """Crée l'arborescence de dossiers pour les données brutes et prétraitées."""
    for split in ["train", "val"]:
        (PROCESSED_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (PROCESSED_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def download_coco_val2017():
    """Télécharge et extrait COCO val2017 si les fichiers ne sont pas déjà présents."""
    img_url = "http://images.cocodataset.org/zips/val2017.zip"
    ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

    img_zip = RAW_DIR / "val2017.zip"
    ann_zip = RAW_DIR / "annotations.zip"

    if not (RAW_DIR / "val2017").exists():
        print("Téléchargement des images COCO val2017...")
        urllib.request.urlretrieve(img_url, img_zip)
        with zipfile.ZipFile(img_zip, "r") as zf:
            zf.extractall(RAW_DIR)

    if not (RAW_DIR / "annotations").exists():
        print("Téléchargement des annotations COCO...")
        urllib.request.urlretrieve(ann_url, ann_zip)
        with zipfile.ZipFile(ann_zip, "r") as zf:
            zf.extractall(RAW_DIR)


def process_dataset():
    """Extrait les images 'person', filtre le bruit et sauvegarde en QQVGA Gris."""
    ann_path = RAW_DIR / "annotations" / "instances_val2017.json"
    img_dir = RAW_DIR / "val2017"

    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images_dict = {img["id"]: img for img in coco["images"]}
    person_cat_id = 1
    valid_person_annotations = {}
    all_person_img_ids = set()

    for ann in coco["annotations"]:
        if ann["category_id"] == person_cat_id:
            img_id = ann["image_id"]
            all_person_img_ids.add(img_id)
            if ann.get("iscrowd", 0) == 1:
                continue

            img_info = images_dict[img_id]
            orig_w, orig_h = img_info["width"], img_info["height"]
            x, y, w, h = ann["bbox"]

            # Calcul de la taille de la boîte une fois redimensionnée en 160x120
            resized_w = (w / orig_w) * TARGET_WIDTH
            resized_h = (h / orig_h) * TARGET_HEIGHT

            if resized_w >= MIN_BOX_WIDTH_PX and resized_h >= MIN_BOX_HEIGHT_PX:
                valid_person_annotations.setdefault(img_id, []).append(ann["bbox"])

    person_img_ids = set(valid_person_annotations.keys())
    # Images garanties sans aucune personne (vrais négatifs)
    all_img_ids = set(images_dict.keys())
    non_person_img_ids = list(all_img_ids - all_person_img_ids)

    random.seed(42)
    min_count = min(len(person_img_ids), len(non_person_img_ids))
    sampled_person_ids = list(random.sample(list(person_img_ids), min_count))
    sampled_non_person_ids = list(random.sample(non_person_img_ids, min_count))

    final_img_ids = sampled_person_ids + sampled_non_person_ids
    random.shuffle(final_img_ids)

    split_idx = int(len(final_img_ids) * 0.8)
    train_ids = set(final_img_ids[:split_idx])

    print(f"Dataset équilibré : {len(sampled_person_ids)} pos / {len(sampled_non_person_ids)} neg.")

    # Nettoyage des anciens fichiers
    for split in ["train", "val"]:
        for p in (PROCESSED_DIR / "images" / split).glob("*.png"):
            p.unlink()
        for p in (PROCESSED_DIR / "labels" / split).glob("*.txt"):
            p.unlink()

    for img_id in final_img_ids:
        img_info = images_dict[img_id]
        src_path = img_dir / img_info["file_name"]
        if not src_path.exists():
            continue

        img = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        orig_h, orig_w = img.shape

        img_resized = cv2.resize(
            img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA
        )

        split = "train" if img_id in train_ids else "val"
        stem = Path(img_info["file_name"]).stem
        out_img_path = PROCESSED_DIR / "images" / split / f"{stem}.png"
        cv2.imwrite(str(out_img_path), img_resized)

        if img_id in valid_person_annotations:
            out_lbl_path = PROCESSED_DIR / "labels" / split / f"{stem}.txt"
            with open(out_lbl_path, "w", encoding="utf-8") as lf:
                for x, y, w, h in valid_person_annotations[img_id]:
                    xc = (x + w / 2.0) / orig_w
                    yc = (y + h / 2.0) / orig_h
                    nw = w / orig_w
                    nh = h / orig_h
                    lf.write(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")

    print(f"Prétraitement QQVGA terminé dans : {PROCESSED_DIR}")


if __name__ == "__main__":
    setup_dirs()
    download_coco_val2017()
    process_dataset()