from ultralytics import YOLO

# Load a model
model = YOLO("ultralytics/cfg/models/11/yolo11-seg.yaml")  # build a new model from YAML
model.load("pretrain_weights/yolo11n-seg.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="coco8-seg.yaml",
                      epochs=100,
                      imgsz=640,
                      batch=32,
                      device="3",
                      project="runs/seg",
                      name="20260106")