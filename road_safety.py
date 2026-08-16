from ultralytics import YOLO
import cv2
import pyttsx3
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    print("Try changing camera index from 0 to 1.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# ============================================================
# SafeRouteAI - Multi Hazard Road Safety Detection
# ============================================================

# Your trained pothole model
pothole_model = YOLO(
    r"runs\detect\pothole_detection-4\weights\best.pt"
)

# Pretrained YOLO model for general objects
object_model = YOLO("yolo11n.pt")
# ============================================================
# VOICE ALERT SYSTEM
# ============================================================

engine = pyttsx3.init()

engine.setProperty("rate", 155)
engine.setProperty("volume", 1.0)

last_warning = ""
last_warning_time = 0

VOICE_DELAY = 3


def speak_warning(message):
    global last_warning
    global last_warning_time

    current_time = time.time()

    if (
        message != last_warning
        or current_time - last_warning_time >= VOICE_DELAY
    ):
        print("VOICE:", message)

        engine.say(message)
        engine.runAndWait()

        last_warning = message
        last_warning_time = current_time

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

print("======================================")
print("       SafeRouteAI Started")
print("======================================")
print("Detecting:")
print("- Potholes")
print("- Animals")
print("- Cars")
print("- Motorcycles")
print("- Buses")
print("- Trucks")
print("- People")
print("- Bicycles")
print("Press Q to quit.")
print("======================================")


# ------------------------------------------------------------
# Object categories
# ------------------------------------------------------------

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

danger_objects = {
    "person"
}


# ------------------------------------------------------------
# Function to determine danger level
# ------------------------------------------------------------

def get_danger_level(area_ratio, object_name):

    # VERY CLOSE
    if area_ratio > 0.15:

        if object_name in animals:
            return (
                "STOP! ANIMAL VERY CLOSE",
                (0, 0, 255)
            )

        elif object_name == "person":
            return (
                "STOP! PERSON AHEAD",
                (0, 0, 255)
            )

        elif object_name in vehicles:
            return (
                "DANGER! VEHICLE VERY CLOSE",
                (0, 0, 255)
            )

        else:
            return (
                "STOP! OBSTACLE AHEAD",
                (0, 0, 255)
            )

    # CLOSE
    elif area_ratio > 0.07:

        if object_name in animals:
            return (
                "ANIMAL AHEAD - SLOW DOWN",
                (0, 140, 255)
            )

        elif object_name == "person":
            return (
                "PERSON AHEAD - SLOW DOWN",
                (0, 140, 255)
            )

        elif object_name in vehicles:
            return (
                "VEHICLE AHEAD - SLOW DOWN",
                (0, 140, 255)
            )

        else:
            return (
                "OBSTACLE AHEAD - SLOW DOWN",
                (0, 140, 255)
            )

    # FAR / MEDIUM DISTANCE
    else:

        if object_name in animals:
            return (
                "ANIMAL AHEAD - BE CAREFUL",
                (0, 255, 255)
            )

        elif object_name == "person":
            return (
                "PERSON AHEAD - BE CAREFUL",
                (0, 255, 255)
            )

        elif object_name in vehicles:
            return (
                "VEHICLE AHEAD - MAINTAIN DISTANCE",
                (0, 255, 255)
            )

        else:
            return (
                "OBSTACLE AHEAD - BE CAREFUL",
                (0, 255, 255)
            )


# ============================================================
# Main loop
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera.")
        break

    frame_height, frame_width = frame.shape[:2]

    frame_area = frame_width * frame_height

    # Default status
    warning = "ROAD CLEAR"
    warning_color = (0, 255, 0)

    danger_score = 0

    # ========================================================
    # 1. POTHOLE DETECTION
    # ========================================================

    pothole_results = pothole_model(
        frame,
        conf=0.35,
        verbose=False
    )

    for result in pothole_results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            confidence = float(box.conf[0])

            width = x2 - x1
            height = y2 - y1

            area = width * height

            area_ratio = area / frame_area

            # -----------------------------------------------
            # Pothole danger calculation
            # -----------------------------------------------

            if area_ratio > 0.15:

                current_warning = (
                    "DANGER! LARGE POTHOLE - SLOW DOWN"
                )

                current_color = (0, 0, 255)

                current_score = 100

            elif area_ratio > 0.07:

                current_warning = (
                    "POTHOLE AHEAD - REDUCE SPEED"
                )

                current_color = (0, 140, 255)

                current_score = 70

            else:

                current_warning = (
                    "POTHOLE AHEAD - BE CAREFUL"
                )

                current_color = (0, 255, 255)

                current_score = 40

            # Update warning if more dangerous
            if current_score > danger_score:

                warning = current_warning

                warning_color = current_color

                danger_score = current_score

            # Draw pothole box
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
                0.65,
                (0, 0, 255),
                2
            )

    # ========================================================
    # 2. GENERAL OBJECT DETECTION
    # ========================================================

    object_results = object_model(
        frame,
        conf=0.35,
        verbose=False
    )

    for result in object_results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            confidence = float(box.conf[0])

            class_id = int(box.cls[0])

            object_name = object_model.names[class_id]

            # Ignore irrelevant objects
            if (
                object_name not in animals
                and object_name not in vehicles
                and object_name not in danger_objects
            ):
                continue

            width = x2 - x1
            height = y2 - y1

            area = width * height

            area_ratio = area / frame_area

            # ------------------------------------------------
            # Determine warning
            # ------------------------------------------------

            current_warning, current_color = get_danger_level(
                area_ratio,
                object_name
            )

            # Danger score
            if area_ratio > 0.15:
                current_score = 100

            elif area_ratio > 0.07:
                current_score = 70

            else:
                current_score = 40

            # ------------------------------------------------
            # Update main warning
            # ------------------------------------------------

            if current_score > danger_score:

                warning = current_warning

                warning_color = current_color

                danger_score = current_score

            # ------------------------------------------------
            # Draw bounding box
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                current_color,
                3
            )

            label = f"{object_name} {confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                current_color,
                2
            )

    # ========================================================
    # 3. DRIVER INSTRUCTION
    # ========================================================

    cv2.rectangle(
        frame,
        (0, 0),
        (frame_width, 75),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        warning,
        (20, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        warning_color,
        3
    )

    # ========================================================
    # 4. PROJECT NAME
    # ========================================================

    cv2.putText(
        frame,
        "SafeRouteAI - Intelligent Road Safety",
        (15, frame_height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # ========================================================
    # 5. SHOW CAMERA
    # ========================================================

    cv2.imshow(
        "SafeRouteAI - Road Hazard Detection",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# Close everything
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("SafeRouteAI stopped.")