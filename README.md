<h1>Auralis: Real-Time Object Detection</h1>h1>
Empowering Vision with AI-Powered Real-Time Object Detection
Auralis is a web-based application designed to detect objects in real-time using a webcam. Powered by Google's Teachable Machine and TensorFlow.js, Auralis recognizes four objects: person, cell phone, bottle, and laptops. This project aims to assist users by identifying objects in the camera's view and providing immediate visual feedback.

#Project Features
Real-Time Object Detection: Detects and classifies objects using a Teachable Machine model.
Webcam Integration: Seamlessly accesses the webcam for live video feed.
Visual Feedback with Animations: Displays dynamic overlays and animations on detected objects.
Teachable Machine Integration: Uses Google's Teachable Machine model for object detection.
Demo
Teachable Machine Model

Table of Contents
Installation
Usage
Project Structure
How It Works
Contributing
License
Installation
Prerequisites
A modern web browser (Chrome, Firefox, Edge)
Internet access to load Teachable Machine model
Steps to Set Up
Clone the repository:
bash
Copy
git clone https://github.com/yourusername/auralis-object-detection.git
Navigate to the project folder:
```bash
cd auralis-object-detection
```
Run the project locally: Open the index.html file in your browser or use a live server extension.
Usage
Open the project in a browser.
Allow the application to access your webcam when prompted.
Watch as objects such as person, cell phone, bottle, and laptops are detected in real-time.
Visual feedback will appear with animated overlays highlighting the detected object.
Project Structure
graphql
```bash
├── index.html              # Main HTML file that renders the webcam and UI
├── script.js               # JavaScript for handling webcam, object detection, and animations
├── styles.css              # CSS file for styling the UI and animations
└── README.md               # Project documentation (this file)
```
How It Works
Webcam Access: The app uses navigator.mediaDevices.getUserMedia() to access the webcam feed.
Teachable Machine Model: The project is integrated with a Teachable Machine model that detects person, cell phone, bottle, and laptops.
Real-Time Detection: The webcam feed is passed through TensorFlow.js to the model, which provides real-time predictions about the objects in the video.
Visual Feedback: Once an object is detected, the app displays the result with a flash animation and an overlay showing the detected object class.
Contributing
We welcome contributions to make Auralis even better. To contribute:

Fork this repository.
Create a new branch:
```bash
git checkout -b feature/your-feature-name
```
Commit your changes:
```bash
git commit -m "Add new feature"
```
Push the branch:
```bash
git push origin feature/your-feature-name
```
Open a pull request.
License
This project is licensed under the MIT License. See the LICENSE file for more details.

Contact
For any inquiries or support, feel free to reach out at:
Email: satwikmishra46@gmail.com/
GitHub: satwikmishra11
