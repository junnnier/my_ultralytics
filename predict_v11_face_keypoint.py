from ultralytics import YOLO

model = YOLO("runs/face_keypoint/20251210/weights/best.pt")
model.model.model[-1].to_onnx = False
model.predict("test.jpg",
              save=True,
              imgsz=[384,288],
              device=2,
              conf=0.4)