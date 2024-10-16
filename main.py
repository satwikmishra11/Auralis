import cv2
import numpy as np
import torch
from ultralytics import YOLO
import pyttsx3
import time
from threading import Thread
from queue import Queue
import asyncio

class AdvancedObjectDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO('yolov8n.pt').to(self.device)  # Using YOLOv8n (nano) for speed and accuracy
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 90)  # Set frame rate to 90 FPS

        self.engine = pyttsx3.init()
        self.audio_queue = Queue()
        self.last_detection_time = 0
        self.detection_cooldown = 3  # seconds
        self.object_history = {}
        self.confidence_threshold = 0.7

    async def process_frame(self, frame):
        results = self.model(frame)  # Use YOLOv8 model to detect objects
        predictions = results[0].boxes.data.cpu().numpy()  # Extract detection information

        return frame, predictions

    def filter_predictions(self, predictions):
        boxes = []
        labels = []
        scores = []

        for pred in predictions:
            score = pred[4]  # Confidence score
            if score > self.confidence_threshold:
                x1, y1, x2, y2 = map(int, pred[:4])
                label = int(pred[5])  # Class index
                boxes.append([x1, y1, x2, y2])
                labels.append(label)
                scores.append(score)

        return np.array(boxes), np.array(labels), np.array(scores)

    def update_object_history(self, labels, scores):
        current_time = time.time()

        detected_objects = []
        for label, score in zip(labels, scores):
            object_name = self.model.names[label]
            detected_objects.append(object_name)

            if object_name not in self.object_history or \
               current_time - self.object_history[object_name] > self.detection_cooldown:
                self.object_history[object_name] = current_time
                # Combine all detected objects into one announcement if the queue is empty
                if self.audio_queue.empty():
                    combined_names = ', '.join(detected_objects)
                    self.audio_queue.put(f"I see {combined_names}")

    def draw_boxes(self, frame, boxes, labels, scores):
        detected_objects = set()

        for box, label, score in zip(boxes, labels, scores):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            object_name = self.model.names[label]
            detected_objects.add(object_name)

            # Draw bounding box and label with score
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame, f"{object_name}: {score:.2f}", (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display all detected objects as a list in a corner
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

            frame, predictions = await self.process_frame(frame)
            if predictions is not None:
                boxes, labels, scores = self.filter_predictions(predictions)
                self.update_object_history(labels, scores)
                frame = self.draw_boxes(frame, boxes, labels, scores)

            # Display the frame with detected objects and bounding boxes
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
