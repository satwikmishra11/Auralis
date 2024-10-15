document.addEventListener('DOMContentLoaded', async function () {
    const videoElement = document.getElementById('webcam');
    const canvasElement = document.getElementById('overlay');
    const context = canvasElement.getContext('2d');
    const detectionResults = document.getElementById('detection-results');

    // Teachable Machine model URL
    const URL = "https://teachablemachine.withgoogle.com/models/dAph--AcR/";

    let model, webcam;

    // Load the Teachable Machine model
    async function loadModel() {
        model = await tmImage.load(URL + 'model.json', URL + 'metadata.json');
        console.log("Model loaded successfully.");
    }

    // Request access to the webcam
    async function setupWebcam() {
        return new Promise((resolve, reject) => {
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(stream => {
                    videoElement.srcObject = stream;
                    videoElement.addEventListener('loadeddata', () => resolve(), false);
                })
                .catch(err => reject(err));
        });
    }

    // Start the detection loop
    async function detectObjects() {
        const prediction = await model.predict(webcam.canvas);
        let highestPrediction = { className: '', probability: 0 };

        // Find the class with the highest prediction
        prediction.forEach(pred => {
            if (pred.probability > highestPrediction.probability) {
                highestPrediction = pred;
            }
        });

        // Display the detection result
        detectionResults.textContent = `Detected: ${highestPrediction.className}`;

        // Flash overlay and animate results
        context.clearRect(0, 0, canvasElement.width, canvasElement.height);
        context.fillStyle = 'rgba(0, 150, 136, 0.3)';
        context.fillRect(0, 0, canvasElement.width, canvasElement.height);

        // Redraw the frame after the flash
        setTimeout(() => {
            context.clearRect(0, 0, canvasElement.width, canvasElement.height);
        }, 500);
    }

    // Initialize and run the webcam and detection
    async function run() {
        // Load the Teachable Machine model and setup webcam
        await loadModel();
        await setupWebcam();

        webcam = new tmImage.Webcam(640, 480, true);
        await webcam.setup();  // Set up the webcam
        await webcam.play();   // Start the webcam stream
        window.requestAnimationFrame(loop);  // Start the prediction loop

        // Display the webcam feed on the page
        videoElement.srcObject = webcam.canvas;
    }

    // Prediction loop
    async function loop() {
        webcam.update();  // Update the webcam feed
        await detectObjects();  // Run object detection
        window.requestAnimationFrame(loop);  // Loop again
    }

    // Start the application
    run().catch(err => console.error("Error starting the app: ", err));
});
