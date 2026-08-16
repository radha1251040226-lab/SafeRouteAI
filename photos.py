import os
import random
import shutil
import zipfile
import urllib.request
from pathlib import Path

from PIL import Image

# ============================================================
# SETTINGS
# ============================================================

NUM_IMAGES = 100
IMAGE_SIZE = (500, 500)

BASE_DIR = Path.cwd()
DOWNLOAD_DIR = BASE_DIR / "pothole_download"
OUTPUT_DIR = BASE_DIR / "pothole_100"

# Roboflow raw pothole dataset
DATASET_URL = "https://public.roboflow.com/object-detection/pothole/1"

ZIP_FILE = DOWNLOAD_DIR / "pothole_dataset.zip"

random.seed(42)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

DOWNLOAD_DIR.mkdir(exist_ok=True)

if OUTPUT_DIR.exists():
    print("Removing old pothole_100 folder...")
    shutil.rmtree(OUTPUT_DIR)

(OUTPUT_DIR / "images").mkdir(parents=True)
(OUTPUT_DIR / "labels").mkdir(parents=True)


# ============================================================
# DOWNLOAD DATASET
# ============================================================

print("\n======================================")
print("DOWNLOADING POTHOLE DATASET")
print("======================================")

print("Dataset page:")
print(DATASET_URL)

print("\nIMPORTANT:")
print("The Roboflow public dataset page may require you to")
print("download the YOLO export manually if direct downloading")
print("is not permitted by the server.")

print("\nIf the automatic download fails:")
print("1. Open:")
print("   https://public.roboflow.com/object-detection/pothole/1")
print("2. Select YOLO format.")
print("3. Download the dataset.")
print("4. Extract it into:")
print(f"   {DOWNLOAD_DIR.resolve()}")


# ============================================================
# TRY AUTOMATIC DOWNLOAD
# ============================================================

try:

    print("\nTrying automatic download...")

    urllib.request.urlretrieve(
        DATASET_URL,
        ZIP_FILE
    )

    print("Download completed.")

except Exception as e:

    print("\nAutomatic download was not available.")
    print("Reason:", e)

    print("\nPlease manually download the YOLO dataset")
    print("from the Roboflow page and extract it into:")

    print(DOWNLOAD_DIR.resolve())

    input("\nPress ENTER after you have extracted the dataset...")


# ============================================================
# EXTRACT ZIP IF NECESSARY
# ============================================================

if ZIP_FILE.exists():

    try:

        print("\nExtracting dataset...")

        with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
            zip_ref.extractall(DOWNLOAD_DIR)

        print("Extraction completed.")

    except zipfile.BadZipFile:

        print("\nDownloaded file was not a ZIP dataset.")
        print("Please manually download the YOLO dataset.")


# ============================================================
# SEARCH FOR IMAGES AND LABELS
# ============================================================

print("\n======================================")
print("SEARCHING DATASET")
print("======================================")

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

all_images = []

for path in DOWNLOAD_DIR.rglob("*"):

    if path.is_file() and path.suffix.lower() in image_extensions:

        all_images.append(path)


print("Images found:", len(all_images))


# ============================================================
# FIND IMAGES HAVING YOLO LABELS
# ============================================================

valid_pairs = []

for image_path in all_images:

    label_path = image_path.with_suffix(".txt")

    if label_path.exists():

        try:

            with open(label_path, "r", encoding="utf-8") as f:
                lines = [
                    line.strip()
                    for line in f.readlines()
                    if line.strip()
                ]

            if len(lines) == 0:
                continue

            # Verify YOLO annotation format
            valid = False

            for line in lines:

                parts = line.split()

                if len(parts) >= 5:

                    try:

                        class_id = int(float(parts[0]))

                        x = float(parts[1])
                        y = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])

                        if (
                            0 <= x <= 1
                            and 0 <= y <= 1
                            and 0 <= w <= 1
                            and 0 <= h <= 1
                        ):
                            valid = True

                    except ValueError:
                        pass

            if valid:

                valid_pairs.append(
                    (image_path, label_path)
                )

        except Exception:
            pass


print("Images with valid YOLO labels:", len(valid_pairs))


# ============================================================
# CHECK ENOUGH IMAGES
# ============================================================

if len(valid_pairs) < NUM_IMAGES:

    raise RuntimeError(
        f"Only {len(valid_pairs)} valid annotated images found. "
        f"Need {NUM_IMAGES}."
    )


# ============================================================
# SELECT 100 DIFFERENT IMAGES
# ============================================================

print("\n======================================")
print("SELECTING 100 DIFFERENT IMAGES")
print("======================================")

selected = random.sample(
    valid_pairs,
    NUM_IMAGES
)

print("Selected:", len(selected))


# ============================================================
# FUNCTION TO RESIZE IMAGE + UPDATE YOLO LABELS
# ============================================================

def resize_and_update_labels(
    image_path,
    label_path,
    output_image,
    output_label
):

    image = Image.open(image_path).convert("RGB")

    original_width, original_height = image.size

    new_width, new_height = IMAGE_SIZE

    # Resize image
    image_resized = image.resize(
        IMAGE_SIZE,
        Image.Resampling.LANCZOS
    )

    image_resized.save(
        output_image,
        quality=95
    )

    # --------------------------------------------------------
    # YOLO coordinates are normalized.
    #
    # Because we resize the entire image uniformly,
    # normalized coordinates remain the same.
    # --------------------------------------------------------

    with open(label_path, "r", encoding="utf-8") as f:

        lines = [
            line.strip()
            for line in f.readlines()
            if line.strip()
        ]

    new_lines = []

    for line in lines:

        parts = line.split()

        if len(parts) < 5:
            continue

        class_id = parts[0]

        x = float(parts[1])
        y = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])

        # Clamp values
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))

        new_lines.append(
            f"{class_id} "
            f"{x:.6f} "
            f"{y:.6f} "
            f"{w:.6f} "
            f"{h:.6f}"
        )

    with open(
        output_label,
        "w",
        encoding="utf-8"
    ) as f:

        f.write("\n".join(new_lines))


# ============================================================
# CREATE 100 IMAGES
# ============================================================

print("\n======================================")
print("CREATING POTHOLE_100")
print("======================================")

for index, (image_path, label_path) in enumerate(
    selected,
    start=1
):

    if index == 1:

        filename = "road.jpg"
        label_filename = "road.txt"

    else:

        filename = f"road{index}.jpg"
        label_filename = f"road{index}.txt"

    output_image = (
        OUTPUT_DIR /
        "images" /
        filename
    )

    output_label = (
        OUTPUT_DIR /
        "labels" /
        label_filename
    )

    try:

        resize_and_update_labels(
            image_path,
            label_path,
            output_image,
            output_label
        )

        print(
            f"[{index:03d}/100] "
            f"{filename}"
        )

    except Exception as e:

        print(
            f"ERROR processing {image_path.name}: "
            f"{e}"
        )


# ============================================================
# CREATE YOLOV8 DATA.YAML
# ============================================================

yaml_content = f"""path: {OUTPUT_DIR.resolve().as_posix()}
train: images
val: images

nc: 1

names:
  0: pothole
"""

with open(
    OUTPUT_DIR / "data.yaml",
    "w",
    encoding="utf-8"
) as f:

    f.write(yaml_content)


# ============================================================
# CREATE README
# ============================================================

readme = """POTHOLE 100 DATASET
====================

Number of images: 100

Image size:
500 x 500

Class:
0 = pothole

Format:
YOLO

Structure:

pothole_100/
    images/
        road.jpg
        road2.jpg
        ...
        road100.jpg

    labels/
        road.txt
        road2.txt
        ...
        road100.txt

    data.yaml

The labels use normalized YOLO bounding-box coordinates.
"""

with open(
    OUTPUT_DIR / "README.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(readme)


# ============================================================
# VERIFY DATASET
# ============================================================

print("\n======================================")
print("VERIFYING DATASET")
print("======================================")

images_created = list(
    (OUTPUT_DIR / "images").glob("*.jpg")
)

labels_created = list(
    (OUTPUT_DIR / "labels").glob("*.txt")
)

print(
    "Images created:",
    len(images_created)
)

print(
    "Labels created:",
    len(labels_created)
)


# Check matching image-label pairs

missing_labels = []

for image in images_created:

    label = (
        OUTPUT_DIR /
        "labels" /
        f"{image.stem}.txt"
    )

    if not label.exists():

        missing_labels.append(
            image.name
        )


if missing_labels:

    print("\nWARNING:")
    print("Images without labels:")

    for name in missing_labels:
        print(name)

else:

    print(
        "Every image has a corresponding YOLO label."
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n======================================")
print("DONE!")
print("======================================")

print(
    "\nDataset location:"
)

print(
    OUTPUT_DIR.resolve()
)

print(
    "\nYou now have:"
)

print(
    "100 different pothole images"
)

print(
    "100 YOLO annotation files"
)

print(
    "500 x 500 images"
)

print(
    "YOLOv8 data.yaml"
)

print(
    "\nFolder:"
)

print(
    OUTPUT_DIR.resolve()
)
