from yolov5.yolov5 import YoLov5
import numpy as np
import os


class Model:

    def __init__(self, engine_file_path, class_num, classnames):
        if not os.path.exists(engine_file_path):
            raise ValueError(f"engine file path: {engine_file_path} not exists!")
        if class_num == 0 or class_num != len(classnames):
            raise ValueError(f"class number or classnames error!")
        self.yolo = YoLov5(engine_file_path, classnames=classnames)

    def __call__(self, image): 
        data = []
        boxes, scores, labels = self.yolo.infer(image)
        for box, score, label in zip(boxes, scores, labels):
            data.append({'label': label, "box": [str(int(p)) for p in box], "confidence": "{:.2f}".format(score)})  
        return data

    def switch(self, model_name, ctx):
        return True

    def check_health(self):
        return True

    def __del__(self):
        self.yolo.destroy()