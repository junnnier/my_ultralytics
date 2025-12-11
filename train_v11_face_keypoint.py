from ultralytics import YOLO

# Load a model
model = YOLO("ultralytics/cfg/models/11/yolo11-face-keypoint.yaml")  # load a pretrained model (recommended for training)
model.load("weight/yolo11n-pose.pt")

# Train the model
results = model.train(data="ultralytics/cfg/datasets/face-keypoint.yaml",
                      epochs=300,
                      imgsz=640,
                      batch=16,
                      workers=6,
                      close_mosaic=0,
                      patience=50,
                      optimizer='SGD',
                      device="3",
                      project="runs/face_keypoint",
                      name="20251210")