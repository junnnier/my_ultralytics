from ultralytics import YOLO

# Load a model
model = YOLO("ultralytics/cfg/models/11/yolo11-face-keypoint.yaml")  # load a pretrained model (recommended for training)
model.load("weight/yolo11n-pose.pt")

# Train the model
results = model.train(data="ultralytics/cfg/datasets/face-keypoint.yaml",
                      epochs=300,
                      imgsz=[288,384],  # image [h,w]
                      batch=16,
                      workers=4,
                      close_mosaic=0,
                      patience=50,
                      optimizer='SGD',
                      device="2,3",
                      project="runs/face_keypoint",
                      name="20251218_test")