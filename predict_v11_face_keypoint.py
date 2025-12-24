from ultralytics import YOLO

model = YOLO("runs/light_face_keypoint/20251219/weights/best.pt")
model.model.model[-1].to_onnx = False
model.predict("test.jpg",
              save=True,
              imgsz=[288,384],
              device=2,
              conf=0.4,
              project="runs/light_face_keypoint")