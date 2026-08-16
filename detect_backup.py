from ultralytics import YOLO
import os
import csv
from datetime import datetime

# Load trained pothole detection model
model = YOLO(r"runs\detect\pothole_detection-4\weights\best.pt")

# Input images
source = r"dataset\images\val"

# Create reports folder if it doesn't exist
os.makedirs("reports", exist_ok=True)

# CSV report path
csv_path = r"reports\pothole_report.csv"

# Run detection
results = model.predict(
    source=source,
    conf=0.25,
    save=True
)

# Prepare CSV data
report_data = []

print("\n" + "=" * 50)
print("             SafeRouteAI")
print("        Pothole Detection Report")
print("=" * 50)

total_potholes = 0

for result in results:

    image_name = os.path.basename(result.path)

    print(f"\nImage: {image_name}")

    if result.boxes is None or len(result.boxes) == 0:
        print("No potholes detected.")
        continue

    pothole_count = len(result.boxes)
    total_potholes += pothole_count

    print(f"Potholes detected: {pothole_count}")

    for i, box in enumerate(result.boxes):

        confidence = float(box.conf[0])

        # Bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        width = x2 - x1
        height = y2 - y1

        area = width * height

        # Estimated severity
        if area < 15000:
            severity = "Low"
        elif area < 40000:
            severity = "Medium"
        else:
            severity = "High"

        confidence_percent = confidence * 100

        print(f"\n  Pothole {i + 1}")
        print(f"  Confidence: {confidence_percent:.2f}%")
        print(f"  Severity: {severity}")

        # Store information for CSV
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Location will be added when using GPS-enabled input
latitude = ""
longitude = ""

report_data.append([
    image_name,
    i + 1,
    f"{confidence_percent:.2f}%",
    severity,
    latitude,
    longitude,
    timestamp,
    round(x1, 2),
    round(y1, 2),
    round(x2, 2),
    round(y2, 2)
])

# Write CSV report
with open(csv_path, "w", newline="", encoding="utf-8") as file:

    writer = csv.writer(file)

    writer.writerow([
    "Image",
    "Pothole_Number",
    "Confidence",
    "Severity",
    "Latitude",
    "Longitude",
    "Timestamp",
    "X1",
    "Y1",
    "X2",
    "Y2"
])

    writer.writerows(report_data)

print("\n" + "=" * 50)
print(f"Total potholes detected: {total_potholes}")
print(f"Report saved to: {csv_path}")
print("=" * 50)

print("\nDetection completed successfully!")