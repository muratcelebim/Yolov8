from ultralytics import YOLO
import time

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
start_train = time.time()
model.train(data="config.yaml", epochs=500)  # train the model
print(f"Train Time: {time.time() - start_train}")
print("*")