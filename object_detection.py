from ultralytics import YOLO
import cv2
import os

def get_driver_instruction(object_name, position, area_ratio=0, confidence=0):

    # =========================
    # POTHOLE INSTRUCTIONS
    # =========================

    if object_name == "pothole":

        # Dangerous / large pothole
        if area_ratio >= 20:

            if position == "LEFT":
                return "DANGER! LARGE POTHOLE ON LEFT - SLOW DOWN AND MOVE RIGHT"

            elif position == "RIGHT":
                return "DANGER! LARGE POTHOLE ON RIGHT - SLOW DOWN AND MOVE LEFT"

            else:
                return "DANGER! LARGE POTHOLE AHEAD - SLOW DOWN AND AVOID IT"

        # Medium pothole
        elif area_ratio >= 5:

            if position == "LEFT":
                return "CAUTION! POTHOLE ON LEFT - REDUCE SPEED AND KEEP RIGHT"

            elif position == "RIGHT":
                return "CAUTION! POTHOLE ON RIGHT - REDUCE SPEED AND KEEP LEFT"

            else:
                return "CAUTION! POTHOLE AHEAD - REDUCE SPEED"

        # Small pothole
        else:

            if position == "LEFT":
                return "SMALL POTHOLE LEFT - KEEP RIGHT"

            elif position == "RIGHT":
                return "SMALL POTHOLE RIGHT - KEEP LEFT"

            else:
                return "SMALL POTHOLE AHEAD - SLOW DOWN"


    # =========================
    # VEHICLE INSTRUCTIONS
    # =========================

    if object_name in ["car", "truck", "bus", "motorcycle", "bicycle"]:

        if position == "CENTER":
            return f"WARNING! {object_name.upper()} AHEAD - SLOW DOWN"

        elif position == "LEFT":
            return f"CAUTION! {object_name.upper()} ON LEFT"

        else:
            return f"CAUTION! {object_name.upper()} ON RIGHT"


    # =========================
    # PERSON INSTRUCTIONS
    # =========================

    if object_name == "person":

        if position == "CENTER":
            return "DANGER! PERSON AHEAD - SLOW DOWN AND BE READY TO STOP"

        elif position == "LEFT":
            return "CAUTION! PERSON ON LEFT - SLOW DOWN"

        else:
            return "CAUTION! PERSON ON RIGHT - SLOW DOWN"


    # =========================
    # ANIMAL INSTRUCTIONS
    # =========================

    if object_name in [
        "dog",
        "cat",
        "horse",
        "cow",
        "sheep",
        "bird"
    ]:

        if position == "CENTER":
            return "DANGER! ANIMAL AHEAD - SLOW DOWN AND BE READY TO STOP"

        elif position == "LEFT":
            return "CAUTION! ANIMAL ON LEFT - REDUCE SPEED"

        else:
            return "CAUTION! ANIMAL ON RIGHT - REDUCE SPEED"


    # =========================
    # OTHER OBJECT
    # =========================

    return f"CAUTION! {object_name.upper()} DETECTED - DRIVE CAREFULLY"
instruction = get_driver_instruction(
    "pothole",
    position,
    area_ratio,
    confidence
)

print("Driver Instruction:", instruction)
instruction = get_driver_instruction(
    name,
    position,
    0,
    confidence
)

print("Driver Instruction:", instruction)
# Pretrained COCO model
model = YOLO("yolov8n.pt")

# Test image
image_path = r"input\test\test_road.jpg"

if not os.path.exists(image_path):
    print("Image not found:", image_path)
    exit()

results = model.predict(
    source=image_path,
    conf=0.30,
    save=False
)

result = results[0]

print("\n" + "=" * 60)
print("             SafeRouteAI")
print("       ROAD OBJECT DETECTION")
print("=" * 60)

objects_detected = 0

for box in result.boxes:

    class_id = int(box.cls[0])
    confidence = float(box.conf[0])

    name = model.names[class_id]

    x1, y1, x2, y2 = box.xyxy[0].tolist()

    center_x = (x1 + x2) / 2
    image_width = result.orig_shape[1]

    # Position relative to camera
    if center_x < image_width * 0.33:
        position = "LEFT"
    elif center_x < image_width * 0.66:
        position = "CENTER"
    else:
        position = "RIGHT"

    objects_detected += 1

    print("\nObject:", name)
    print("Confidence:", f"{confidence * 100:.2f}%")
    print("Position:", position)

    # Driver warning
    if name in ["car", "truck", "bus", "motorcycle", "bicycle", "person"]:

        if position == "LEFT":
            warning = f"CAUTION - {name.upper()} ON LEFT"

        elif position == "RIGHT":
            warning = f"CAUTION - {name.upper()} ON RIGHT"

        else:
            warning = f"WARNING - {name.upper()} AHEAD"

    elif name in ["dog", "cat", "horse", "cow", "sheep", "bird"]:

        if position == "CENTER":
            warning = f"WARNING - ANIMAL AHEAD - SLOW DOWN"

        else:
            warning = f"CAUTION - ANIMAL ON {position}"

    else:
        warning = f"CAUTION - {name.upper()} DETECTED"

    print("Driver Guidance:", warning)


# Save annotated image
output_folder = r"runs\detect\road_objects"

os.makedirs(output_folder, exist_ok=True)

annotated = result.plot()

output_path = os.path.join(
    output_folder,
    "road_object_detection.jpg"
)

cv2.imwrite(output_path, annotated)

print("\n" + "=" * 60)

if objects_detected == 0:
    print("ROAD STATUS: CLEAR")
    print("Driver Guidance: Continue carefully.")
else:
    print("Objects detected:", objects_detected)
    print("Driver should slow down and remain alert.")

print("\nAnnotated image:")
print(output_path)

print("=" * 60)