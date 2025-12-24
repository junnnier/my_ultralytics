from ultralytics import YOLO

model = YOLO("./runs/light_face_keypoint/20251224/weights/best.pt")  # load a custom-trained model
model.model.model[-1].to_onnx = True  # 导出onnx模型时去除后处理中decode相关函数
# Export the model
model.export(format="onnx",
             imgsz=(288,384),
             simplify=True,
             device="2",
             opset=11)