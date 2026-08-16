from ultralytics import YOLO
import cv2
import os

# ==============================
# LOAD MODELS
# ==============================

# Your trained pothole model
pothole_model = YOLO(
    r"runs\detect\pothole_detection-4\weights\best.pt"
)

# General object detection model
object_model = YOLO("yolo11n.pt")

# ==============================
# TEST IMAGE
# ==============================

image_path = "test.jpg"

if not os.path.exists(image_path):
    print("ERROR: test.jpg not found!")
    print("Put test.jpg inside the SafeRouteAI folder.")
    exit()

frame = cv2.imread(image_path)

if frame is None:
    print("ERROR: Could not open test.jpg")
    exit()

height, width = frame.shape[:2]
frame_area = height * width

warning = "ROAD CLEAR"
danger_score = 0

# ==============================
# POTHOLE DETECTION
# ==============================

results = pothole_model(
    frame,
    conf=0.35,
    verbose=False
)

for result in results:

    for box in result.boxes:

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        confidence = float(box.conf[0])

        box_area = (x2 - x1) * (y2 - y1)

        area_ratio = box_area / frame_area

        if area_ratio > 0.15:

            warning = "DANGER! POTHOLE - SLOW DOWN"
            danger_score = 100

        elif area_ratio > 0.07:

            warning = "POTHOLE AHEAD - REDUCE SPEED"
            danger_score = 70

        else:

            warning = "POTHOLE AHEAD - BE CAREFUL"
            danger_score = 40

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            3
        )

        cv2.putText(
            frame,
            f"POTHOLE {confidence:.2f}",
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

# ==============================
# GENERAL OBJECT DETECTION
# ==============================

results = object_model(
    frame,
    conf=0.35,
    verbose=False
)

animals = {
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe"
}

vehicles = {
    "car",
    "motorcycle",
    "bus",
    "truck",
    "bicycle"
}

for result in results:

    for box in result.boxes:

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        confidence = float(box.conf[0])

        class_id = int(box.cls[0])

        name = object_model.names[class_id]

        # Ignore irrelevant objects
        if (
            name not in animals
            and name not in vehicles
            and name != "person"
        ):
            continue

        box_area = (x2 - x1) * (y2 - y1)

        area_ratio = box_area / frame_area

        # ==============================
        # ANIMAL
        # ==============================

        if name in animals:

            if area_ratio > 0.15:

                current_warning = "STOP! ANIMAL VERY CLOSE"
                current_score = 100

            elif area_ratio > 0.07:

                current_warning = "ANIMAL AHEAD - SLOW DOWN"
                current_score = 70

            else:

                current_warning = "ANIMAL AHEAD - BE CAREFUL"
                current_score = 40

        # ==============================
        # PERSON
        # ==============================

        elif name == "person":

            if area_ratio > 0.15:

                current_warning = "STOP! PERSON AHEAD"
                current_score = 100

            else:

                current_warning = "PERSON AHEAD - SLOW DOWN"
                current_score = 70

        # ==============================
        # VEHICLE
        # ==============================

        else:

            if area_ratio > 0.15:

                current_warning = "DANGER! VEHICLE VERY CLOSE"
                current_score = 100

            elif area_ratio > 0.07:

                current_warning = "VEHICLE AHEAD - SLOW DOWN"
                current_score = 70

            else:

                current_warning = "VEHICLE AHEAD - MAINTAIN DISTANCE"
                current_score = 40

        # Update warning only if more dangerous
        if current_score > danger_score:

            warning = current_warning
            danger_score = current_score

        # Draw object
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 165, 255),
            3
        )

        cv2.putText(
            frame,
            f"{name} {confidence:.2f}",
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 255),
            2
        )

# ==============================
# WARNING DISPLAY
# ==============================

cv2.rectangle(
    frame,
    (0, 0),
    (width, 70),
    (0, 0, 0),
    -1
)

cv2.putText(
    frame,
    warning,
    (20, 45),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.85,
    (0, 255, 255),
    3
)

# ==============================
# SHOW RESULT
# ==============================

cv2.imshow(
    "SafeRouteAI Detection",
    frame
)

print()
print("================================")
print("SafeRouteAI RESULT")
print("================================")
print("Instruction:", warning)
print("================================")
print()
print("Press any key on the image window to close.")

cv2.waitKey(0)

cv2.destroyAllWindows()