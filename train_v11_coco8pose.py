from ultralytics import YOLO

# Load a model
model = YOLO("ultralytics/cfg/models/11/yolo11-coco8-pose.yaml")  # load a pretrained model (recommended for training)
model.load("weight/yolo11n-pose.pt")

# Train the model
results = model.train(data="coco8-pose.yaml",
                      epochs=100,
                      imgsz=640,
                      device="3",
                      name="20251210")