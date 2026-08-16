from ultralytics import YOLO
import os
import csv

# ==============================
# SafeRouteAI - Driver Assistant
# ==============================

MODEL_PATH = r"runs\detect\pothole_detection-4\weights\best.pt"
IMAGE_FOLDER = r"dataset\images\val"

OUTPUT_FOLDER = r"runs\detect\saferoute_results\images"
REPORT_FOLDER = r"reports"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# Load trained YOLO model
model = YOLO(MODEL_PATH)

# Get images
images = [
    os.path.join(IMAGE_FOLDER, f)
    for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

images.sort()

# ==============================
# Position calculation
# ==============================

def get_position(x_center, image_width):
    ratio = x_center / image_width

    if ratio < 0.33:
        return "LEFT"
    elif ratio < 0.66:
        return "CENTER"
    else:
        return "RIGHT"


# ==============================
# Driver guidance
# ==============================

def get_guidance(position, area_ratio, confidence):

    # Very large pothole
    if area_ratio >= 20:
        if position == "LEFT":
            return "HIGH RISK - SLOW DOWN - AVOID LEFT"
        elif position == "RIGHT":
            return "HIGH RISK - SLOW DOWN - AVOID RIGHT"
        else:
            return "HIGH RISK - SLOW DOWN - AVOID CENTER"

    # Medium pothole
    if area_ratio >= 5:
        if position == "LEFT":
            return "CAUTION - REDUCE SPEED - MOVE RIGHT"
        elif position == "RIGHT":
            return "CAUTION - REDUCE SPEED - MOVE LEFT"
        else:
            return "CAUTION - REDUCE SPEED"

    # Small pothole
    if position == "LEFT":
        return "SMALL POTHOLE - KEEP RIGHT"
    elif position == "RIGHT":
        return "SMALL POTHOLE - KEEP LEFT"
    else:
        return "SMALL POTHOLE - SLOW DOWN"


# ==============================
# CSV report
# ==============================

csv_path = os.path.join(REPORT_FOLDER, "driver_guidance_report.csv")

rows = []

total_potholes = 0
high_risk = 0
medium_risk = 0
low_risk = 0

print("\n" + "=" * 70)
print("                 SafeRouteAI")
print("        AI ROAD SAFETY DRIVER ASSISTANT")
print("=" * 70)

# ==============================
# Detection
# ==============================

for image_path in images:

    results = model.predict(
        source=image_path,
        conf=0.25,
        save=False,
        verbose=False
    )

    result = results[0]

    image_name = os.path.basename(image_path)

    image_height, image_width = result.orig_shape

    pothole_count = 0

    print(f"\nImage: {image_name}")

    # Start with road clear
    image_guidance = "ROAD CLEAR - DRIVE SAFELY"

    if result.boxes is not None:

        for i, box in enumerate(result.boxes):

            confidence = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            box_width = x2 - x1
            box_height = y2 - y1

            area = box_width * box_height

            image_area = image_width * image_height

            area_ratio = (area / image_area) * 100

            x_center = (x1 + x2) / 2

            position = get_position(
                x_center,
                image_width
            )

            # Severity
            if area_ratio >= 20:
                severity = "HIGH"
                high_risk += 1

            elif area_ratio >= 5:
                severity = "MEDIUM"
                medium_risk += 1

            else:
                severity = "LOW"
                low_risk += 1

            guidance = get_guidance(
                position,
                area_ratio,
                confidence
            )

            pothole_count += 1
            total_potholes += 1

            image_guidance = guidance

            print(f"\n  Pothole {pothole_count}")
            print(f"  Confidence : {confidence * 100:.2f}%")
            print(f"  Position   : {position}")
            print(f"  Area       : {area_ratio:.2f}%")
            print(f"  Severity   : {severity}")
            print(f"  Guidance   : {guidance}")

            rows.append([
                image_name,
                pothole_count,
                round(confidence * 100, 2),
                position,
                round(area_ratio, 2),
                severity,
                guidance
            ])

    print(f"\n  Potholes detected: {pothole_count}")

    if pothole_count == 0:
        print("  ROAD STATUS: CLEAR")
        print("  Driver Guidance: Continue safely.")
    else:
        print(f"  DRIVER GUIDANCE: {image_guidance}")

    # Save annotated image
    annotated = result.plot()

    output_path = os.path.join(
        OUTPUT_FOLDER,
        image_name
    )

    import cv2

    # Add guidance text
    cv2.putText(
        annotated,
        image_guidance,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.imwrite(output_path, annotated)


# ==============================
# Save CSV
# ==============================

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Image",
        "Pothole",
        "Confidence",
        "Position",
        "Area_Ratio",
        "Severity",
        "Driver_Guidance"
    ])

    writer.writerows(rows)


# ==============================
# Final analysis
# ==============================

print("\n" + "=" * 70)
print("                 FINAL ROAD ANALYSIS")
print("=" * 70)

print(f"Images analyzed : {len(images)}")
print(f"Total potholes  : {total_potholes}")

print(f"High risk       : {high_risk}")
print(f"Medium risk     : {medium_risk}")
print(f"Low risk        : {low_risk}")

if total_potholes == 0:

    risk_score = 0
    risk_level = "LOW"

    recommendation = (
        "ROAD CLEAR - Continue driving safely."
    )

elif high_risk >= 3:

    risk_score = 100
    risk_level = "HIGH"

    recommendation = (
        "HIGH RISK - SLOW DOWN AND AVOID DAMAGED AREAS."
    )

elif high_risk >= 1:

    risk_score = 70
    risk_level = "MEDIUM-HIGH"

    recommendation = (
        "CAUTION - REDUCE SPEED AND AVOID POTHOLES."
    )

else:

    risk_score = 40
    risk_level = "MEDIUM"

    recommendation = (
        "MINOR ROAD DAMAGE - DRIVE CAREFULLY."
    )


print(f"\nRisk Score      : {risk_score}/100")
print(f"Risk Level      : {risk_level}")

print("\nDRIVER RECOMMENDATION:")
print(recommendation)

print("\n" + "=" * 70)

print("CSV report:")
print(csv_path)

print("\nAnnotated images:")
print(OUTPUT_FOLDER)

print("=" * 70)
print("SafeRouteAI completed successfully!")