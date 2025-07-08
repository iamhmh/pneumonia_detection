import os, glob, cv2

SPLITS = ["train", "val", "test"]
CLASSES = ["NORMAL", "PNEUMONIA"]
TARGET_SIZE = (224, 224)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed")

def preprocess_and_save():
    for split in SPLITS:
        for cls in CLASSES:
            in_dir  = os.path.join(INPUT_DIR, split, cls)
            out_dir = os.path.join(OUTPUT_DIR, split, cls)
            os.makedirs(out_dir, exist_ok=True)

            pattern = os.path.join(in_dir, "*.*")
            for img_path in glob.glob(pattern):
                # Charger
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print("⚠️  échec lecture:", img_path)
                    continue

                # Redimensionner
                img_resized = cv2.resize(img, TARGET_SIZE)

                # Normaliser sur 0–255 et convertir en uint8
                img_norm = cv2.normalize(img_resized, None, 0, 255,
                                          cv2.NORM_MINMAX).astype("uint8")

                # Sauvegarder en PNG
                fname = os.path.splitext(os.path.basename(img_path))[0] + ".png"
                cv2.imwrite(os.path.join(out_dir, fname), img_norm)

            print(f"[OK] {split}/{cls} → {len(os.listdir(out_dir))} images traitées")

if __name__ == "__main__":
    preprocess_and_save()
