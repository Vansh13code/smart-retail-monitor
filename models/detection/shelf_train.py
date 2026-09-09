from ultralytics import YOLO

model = YOLO("yolo26n.pt")

model.train(
    data="../../Shelves.yolo26/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name="shelf_detector",
    device = 0
)