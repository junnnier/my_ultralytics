from ultralytics import YOLO

model = YOLO("./runs/seg/20260106/weights/best.pt")  # load a custom-trained model
model.model.model[-1].to_onnx = True  # 导出onnx模型时去除后处理中decode相关函数
# Export the model
model.export(format="onnx",
             imgsz=(640,640),  # (h, w)
             simplify=True,
             device="3",
             opset=11)