from ultralytics import YOLO

# Load a model
model = YOLO("ultralytics/cfg/models/11/yolo11-face-keypoint-light.yaml")  # load a pretrained model (recommended for training)
model.load("pretrain_face.pt")

# Train the model
results = model.train(data="ultralytics/cfg/datasets/face-keypoint.yaml",
                      epochs=300,
                      imgsz=512,  # image [h,w]
                      batch=32,
                      workers=8,
                      mosaic=0.5,
                      patience=50,
                      optimizer='SGD',
                      device="3",
                      project="runs/light_face_keypoint",
                      name="20251224")