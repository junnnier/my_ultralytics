from ultralytics import YOLO

# Load a model
model = YOLO("./runs/face_keypoint/20251210/weights/best.pt")  # load a custom model
model.model.model[-1].to_onnx = False

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map  # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps  # a list containing mAP50-95 for each category