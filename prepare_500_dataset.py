import os
import shutil
import random
from pathlib import Path

# ============================================================
# SafeRouteAI
# Create a 500-image Pothole + Animal YOLO dataset
# ============================================================

# ------------------------------------------------------------
# CHANGE THESE TWO PATHS
# ------------------------------------------------------------

# Your existing pothole dataset
POTHOLE_IMAGES = Path("pothole_dataset/images")
POTHOLE_LABELS = Path("pothole_dataset/labels")

# Your animal dataset
ANIMAL_IMAGES = Path("animal_dataset/images")
ANIMAL_LABELS = Path("animal_dataset/labels")


# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

OUTPUT = Path("combined_dataset")

TRAIN_IMAGES = OUTPUT / "images" / "train"
VAL_IMAGES = OUTPUT / "images" / "val"

TRAIN_LABELS = OUTPUT / "labels" / "train"
VAL_LABELS = OUTPUT / "labels" / "val"


# ------------------------------------------------------------
# CREATE DIRECTORIES
# ------------------------------------------------------------

for folder in [
    TRAIN_IMAGES,
    VAL_IMAGES,
    TRAIN_LABELS,
    VAL_LABELS
]:
    folder.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# FIND IMAGES
# ------------------------------------------------------------

def get_images(folder):

    extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.JPG",
        "*.JPEG",
        "*.PNG"
    ]

    images = []

    for ext in extensions:
        images.extend(folder.glob(ext))

    return images


pothole_images = get_images(POTHOLE_IMAGES)
animal_images = get_images(ANIMAL_IMAGES)


print("Pothole images found:", len(pothole_images))
print("Animal images found:", len(animal_images))


# ------------------------------------------------------------
# CHECK
# ------------------------------------------------------------

if len(pothole_images) == 0:
    print("\nERROR: No pothole images found.")
    print("Check:", POTHOLE_IMAGES)
    exit()

if len(animal_images) == 0:
    print("\nERROR: No animal images found.")
    print("Check:", ANIMAL_IMAGES)
    exit()


# ------------------------------------------------------------
# SELECT 250 EACH
# ------------------------------------------------------------

random.seed(42)

random.shuffle(pothole_images)
random.shuffle(animal_images)

pothole_selected = pothole_images[:250]
animal_selected = animal_images[:250]


print("\nSelected:")
print("Pothole:", len(pothole_selected))
print("Animal :", len(animal_selected))


# ------------------------------------------------------------
# COMBINE
# ------------------------------------------------------------

all_data = []

for image in pothole_selected:
    all_data.append(
        (image, POTHOLE_LABELS, 0, "pothole")
    )

for image in animal_selected:
    all_data.append(
        (image, ANIMAL_LABELS, 1, "animal")
    )


random.shuffle(all_data)


# ------------------------------------------------------------
# COPY AND CONVERT LABELS
# ------------------------------------------------------------

for index, (image, label_folder, new_class, category) in enumerate(all_data):

    # 80% train
    if index < 400:

        image_output = TRAIN_IMAGES
        label_output = TRAIN_LABELS

    else:

        image_output = VAL_IMAGES
        label_output = VAL_LABELS


    # New unique filename
    new_name = f"road_{index + 1:03d}"

    destination_image = image_output / f"{new_name}.jpg"
    destination_label = label_output / f"{new_name}.txt"


    # Copy image
    shutil.copy2(
        image,
        destination_image
    )


    # Original label
    original_label = label_folder / f"{image.stem}.txt"


    # Create new YOLO label
    with open(
        destination_label,
        "w",
        encoding="utf-8"
    ) as output_file:

        if original_label.exists():

            with open(
                original_label,
                "r",
                encoding="utf-8"
            ) as input_file:

                for line in input_file:

                    parts = line.strip().split()

                    if len(parts) != 5:
                        continue

                    # Change class:
                    # pothole -> 0
                    # animal  -> 1

                    parts[0] = str(new_class)

                    output_file.write(
                        " ".join(parts) + "\n"
                    )


print("\n===================================")
print("DATASET CREATED")
print("===================================")

print("Total images : 500")
print("Training     : 400")
print("Validation   : 100")

print("\nLocation:")
print(OUTPUT)

print("\nClasses:")
print("0 = pothole")
print("1 = animal")

print("===================================")