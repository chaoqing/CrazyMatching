# Product Requirement Document (PRD): Crazy Matching (AR Version)
- Version: 1.1
- Date: July 21, 2025
- Project: An augmented reality version of the "Crazy Matching" game.

## 1.0 Background
Crazy Matching is a web-based, real-time visual perception game. It uses a device's camera to create an augmented reality experience. The application is inspired by physical card games like "Spot It!" where players race to find the single matching symbol between two cards. This project will build a functional prototype that recognizes specific, custom-designed game symbols in a live video feed.

## 2.0 Core Functionality
The primary function of the app is to host a two-player game. In the game, the system will present two sets of symbols (either on-screen or on physical cards). A player will point their device's camera at the cards, and the application will:

- Access the live camera feed.

- Perform real-time analysis of the video stream.

- Identify all the known custom symbols it sees on both cards.

- Determine which symbol is common to both cards.

- Visually highlight the matching symbol on the screen to declare a winner for the round.

## 3.0 Technology Stack & Architecture

**Frontend**: Standard web technologies (HTML, CSS, JavaScript). No complex framework is required for the prototype. The application must be mobile-first and responsive.

**Core Algorithm Engine**: TensorFlow.js. This will be used to load and run our custom machine learning model directly in the browser.

**Supporting Libraries**: We may use OpenCV.js for preliminary image processing tasks (like perspective correction on the cards) if needed, but the primary recognition task will be handled by TensorFlow.js.

## 4.0 AI Model Workflow: Custom Object Detection
We will follow Option A and train our own custom object detection model. This ensures the application can recognize our unique, stylized game symbols accurately.

- Step 1: Data Collection & Preparation

Create a clean, high-resolution image of each unique game symbol (e.g., cow.png, zebra.png, star.png).

For each symbol, capture at least 20-30 photographs from various angles, in different lighting conditions, and at different sizes to create a robust training dataset.

- Step 2: Model Training

We will use a high-level tool like Google's Teachable Machine for the initial training.

Upload the prepared datasets for each symbol and label them accordingly.

Use the tool's interface to train the object detection model until a satisfactory accuracy is achieved.

- Step 3: Model Deployment & Integration

Export the trained model in the TensorFlow.js format.

Integrate the exported model into our web application. The application will load this custom model instead of a generic one like COCO-SSD.

The core application logic will feed video frames to this model for real-time inference, as inspired by the V2EX article.
