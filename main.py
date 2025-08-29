import argparse
from pathlib import Path
import sys
import cv2
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo

# Extensions supportées
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

# --- CONFIG DETECTRON2 ---
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "model_final.pth"  # Poids à récupérer sur Hugging Face
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cpu"  # change en "cuda" si GPU dispo

predictor = DefaultPredictor(cfg)
# --------------------------

def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS

def process_image_file(src: Path, dst: Path) -> None:
    # Lecture image
    img = cv2.imread(str(src))
    if img is None:
        raise ValueError("Impossible de lire l'image")

    # Prédiction
    outputs = predictor(img)

    # Visualisation avec bounding boxes / masks
    v = Visualizer(img[:, :, ::-1], scale=1.0)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Sauvegarde
    cv2.imwrite(str(dst), out.get_image()[:, :, ::-1])

def run(input_dir: Path, output_dir: Path) -> int:
    if not input_dir.exists():
        print(f"[ERREUR] Dossier d'entrée introuvable: {input_dir}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    images = [p for p in input_dir.iterdir() if is_image(p)]
    if not images:
        print(f"[INFO] Aucune image trouvée dans: {input_dir}")
        return 0

    print(f"[INFO] {len(images)} image(s) détectée(s) dans '{input_dir}'.")
    for src in images:
        dst = output_dir / src.name
        try:
            process_image_file(src, dst)
            print(f"[OK] {src.name} -> {dst}")
        except Exception as e:
            print(f"[SKIP] {src.name}: {e}")

    print(f"[FINI] Sortie écrite dans: {output_dir}")
    return 0

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch Detectron2 car damage detection"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("pictures"),
        help="Dossier d'entrée (par défaut: pictures)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("output"),
        help="Dossier de sortie (par défaut: output)"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    sys.exit(run(args.input, args.output))
