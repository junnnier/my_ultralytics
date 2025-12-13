from ultralytics import YOLO

model = YOLO("runs/face_keypoint/20251210/weights/best.pt")

model.predict("test.jpg",
              ave=True,
              imgsz=640,
              device=2,
              conf=0.4)