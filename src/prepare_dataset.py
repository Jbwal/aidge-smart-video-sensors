import os
import zipfile
import urllib.request
import cv2
import json
from tqdm import tqdm
import random

DATA_DIR = "./data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

#nouvelles dimensions pour la puce
TARGET_WIDTH = 160
TARGET_HEIGHT = 120

def setup_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(os.path.join(PROCESSED_DIR, "images/train"), exist_ok=True)
    os.makedirs(os.path.join(PROCESSED_DIR, "images/val"), exist_ok=True)
    os.makedirs(os.path.join(PROCESSED_DIR, "labels/train"), exist_ok=True)
    os.makedirs(os.path.join(PROCESSED_DIR, "labels/val"), exist_ok=True)

def download_coco_val2017():
    """Télécharge le dataset de validation COCO 2017 (5000 images, idéal pour débuter rapidement)"""
    img_url = "http://images.cocodataset.org/zips/val2017.zip"
    ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    
    img_zip = os.path.join(RAW_DIR, "val2017.zip")
    ann_zip = os.path.join(RAW_DIR, "annotations.zip")

    if not os.path.exists(os.path.join(RAW_DIR, "val2017")):
        #telechargement images coco val2017
        urllib.request.urlretrieve(img_url, img_zip)
        with zipfile.ZipFile(img_zip, 'r') as zip_ref:
            zip_ref.extractall(RAW_DIR)

    if not os.path.exists(os.path.join(RAW_DIR, "annotations")):
        #telechargement des annotations coco
        urllib.request.urlretrieve(ann_url, ann_zip)
        with zipfile.ZipFile(ann_zip, 'r') as zip_ref:
            zip_ref.extractall(RAW_DIR)

def process_dataset():
    ann_path = os.path.join(RAW_DIR, "annotations/instances_val2017.json")
    img_dir = os.path.join(RAW_DIR, "val2017")

    with open(ann_path, 'r') as f:
        coco = json.load(f)

    #on prend uniquement la categorie "person"
    person_cat_id = 1
    person_img_ids = set()
    img_annotations = {}

    for ann in coco['annotations']:
        if ann['category_id'] == person_cat_id and ann.get('iscrowd', 0) == 0:
            img_id = ann['image_id']
            person_img_ids.add(img_id)
            if img_id not in img_annotations:
                img_annotations[img_id] = []
            img_annotations[img_id].append(ann['bbox'])  #[x,y,largeur,hauteur]
    
    other_img_ids = set(img['id'] for img in coco['images'])
    non_person_img_ids = list(other_img_ids - person_img_ids)

    random.seed(42)

    min_count = min(len(person_img_ids), len(non_person_img_ids))
    sampled_person_ids = set(random.sample(list(person_img_ids), min_count))
    sampled_non_person_ids = set(random.sample(list(non_person_img_ids), min_count))

    final_img_ids = list(sampled_person_ids)+list(sampled_non_person_ids)
    random.shuffle(final_img_ids)

    images_dict = {img['id']: img for img in coco['images']}

    #on garde 80% des images pour l'entrainement et 20% pour la validation
    split_idx = int(len(final_img_ids) * 0.8)
    train_ids = set(final_img_ids[:split_idx])
    val_ids = set(final_img_ids[split_idx:])

    print(f"{len(sampled_person_ids)} images avec personnes et {len(sampled_non_person_ids)} images sans personnes")

    for img_id in final_img_ids:
        img_info = images_dict[img_id]
        file_name = img_info['file_name']
        src_path = os.path.join(img_dir, file_name)

        if not os.path.exists(src_path):
            continue

        #conversion en niveau de gris
        img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        orig_h, orig_w = img.shape

        #redimensionement avec les bonnes dimensions
        img_resized = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)

        #determiner si train ou val
        split = "train" if img_id in train_ids else "val"
        out_img_path = os.path.join(PROCESSED_DIR, f"images/{split}/{os.path.splitext(file_name)[0]}.png")
        #sauvegarder l'image
        cv2.imwrite(out_img_path, img_resized)

        if img_id in sampled_person_ids:
            out_label_path = os.path.join(PROCESSED_DIR,f"labels/{split}/{os.path.splitext(file_name)[0]}.txt")
            #normalisation des bounding box pour le format YOLO
            with open(out_label_path, 'w') as lf:
                for bbox in img_annotations[img_id]:
                    x, y, w, h = bbox
                    #conversion en coordonnees normalisées (0.0 à 1.0)
                    x_center = (x + w / 2.0) / orig_w
                    y_center = (y + h / 2.0) / orig_h
                    norm_w = w / orig_w
                    norm_h = h / orig_h

                    lf.write(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
    print(f"images disponibles dans {PROCESSED_DIR}")

if __name__ == "__main__":
    setup_dirs()
    download_coco_val2017()
    process_dataset()