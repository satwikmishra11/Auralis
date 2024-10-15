import cv2
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import pyttsx3
import time
from threading import Thread
from queue import Queue
import asyncio

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class AdvancedObjectDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = fasterrcnn_resnet50_fpn(pretrained=True).to(self.device)
        self.model.eval()

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 90)  # Set frame rate to 30 FPS

        self.engine = pyttsx3.init()
        self.audio_queue = Queue()
        self.last_detection_time = 0
        self.detection_cooldown = 3  # seconds
        self.object_history = {}
        self.confidence_threshold = 0.7

    async def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = F.to_tensor(rgb_frame).to(self.device)

        with torch.no_grad():
            prediction = self.model([input_tensor])[0]

        return frame, prediction

    def filter_predictions(self, prediction):
        masks = prediction['scores'] > self.confidence_threshold
        boxes = prediction['boxes'][masks].cpu().numpy().astype(int)
        labels = prediction['labels'][masks].cpu().numpy()
        scores = prediction['scores'][masks].cpu().numpy()
        return boxes, labels, scores

    def update_object_history(self, labels, scores):
        current_time = time.time()

        detected_objects = []
        for label, score in zip(labels, scores):
            object_name = COCO_INSTANCE_CATEGORY_NAMES[label]
            detected_objects.append(object_name)

            if object_name not in self.object_history or \
               current_time - self.object_history[object_name] > self.detection_cooldown:
                self.object_history[object_name] = current_time
                if self.audio_queue.empty():
                    self.audio_queue.put(f"I see {', '.join(detected_objects)}")

    def draw_boxes(self, frame, boxes, labels, scores):
        detected_objects = set()

        for box, label, score in zip(boxes, labels, scores):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            object_name = COCO_INSTANCE_CATEGORY_NAMES[label]
            detected_objects.add(object_name)

            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame, f"{object_name}: {score:.2f}", (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
            y_offset = 50
        for idx, obj in enumerate(detected_objects):
            cv2.rectangle(frame, (10, y_offset - 30), (300, y_offset + 10), (0, 0, 0), -1)
            cv2.putText(frame, f"Detected: {obj}", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            y_offset += 40  

        return frame

    def speak_thread(self):
        while True:
            text = self.audio_queue.get()
            if text is None:
                break
            self.engine.say(text)
            self.engine.runAndWait()

    async def run(self):
        speak_thread = Thread(target=self.speak_thread)
        speak_thread.start()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame, prediction = await self.process_frame(frame)
            if prediction:
                boxes, labels, scores = self.filter_predictions(prediction)
                self.update_object_history(labels, scores)
                frame = self.draw_boxes(frame, boxes, labels, scores)

            cv2.imshow('Advanced Object Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.audio_queue.put(None)  
        speak_thread.join()
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = AdvancedObjectDetector()
    asyncio.run(detector.run())