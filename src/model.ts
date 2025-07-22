/**
 * @license
 * Copyright 2025 Nicolas Wang. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';

// Define a type for your custom detected objects, similar to cocoSsd.DetectedObject
// Adjust properties based on your model's output and what you need for your game logic.
export interface CustomDetectedObject {
    bbox: [number, number, number, number]; // [x, y, width, height]
    class: string; // The detected class name (e.g., 'cow', 'zebra')
    score: number; // Confidence score
}

export class Model {
    // Use tf.GraphModel for custom models
    private model: tf.GraphModel | null = null;
    // Define your class names in the order they were trained
    private classNames: string[] = ['cow', 'zebra', 'star']; // TODO: Update with your actual class names

    async load() {
        await tf.ready(); // Ensure TensorFlow.js backend is ready
        // Load your custom TensorFlow.js model from the public directory
        // The path should be relative to your web server's root.
        const modelPath = './assets/crazy_matching/model.json'; // Path to your converted custom model
        this.model = await tf.loadGraphModel(modelPath);
        console.log('Custom model loaded from:', modelPath);
    }

    async detect(input: HTMLVideoElement): Promise<CustomDetectedObject[]> {
        if (!this.model) {
            console.log('Model not loaded.');
            return [];
        }

        const imgTensor = tf.browser.fromPixels(input);
        // Depending on your model's input requirements, you might need to resize,
        // normalize, or expand dimensions of the image tensor.
        // Example: Resize to model's expected input size (e.g., 300x300 for SSD)
        const resized = tf.image.resizeBilinear(imgTensor, [300, 300]); // TODO: Adjust size to your model's input
        const expanded = resized.expandDims(0); // Add batch dimension
        const normalized = expanded.div(255.0); // Normalize to [0, 1] if your model expects it

        // Perform inference
        // The output of your custom model will be raw tensors.
        // You need to know the names of your model's output tensors.
        // Common names for object detection are 'detection_boxes', 'detection_scores', 'detection_classes', 'num_detections'.
        const predictions = this.model.execute(normalized) as tf.Tensor[]; // Cast to Tensor[]

        // TODO: Process the raw tensor outputs to extract bounding boxes, scores, and classes.
        // The exact processing depends on your model's output structure.
        // This is a generic example and will likely need adjustment.

        const bboxPrediction = predictions[0].squeeze(); // Assuming the model outputs a single tensor of shape [8]
        const bboxCoords = bboxPrediction.arraySync() as number[];

        const detectedObjects: CustomDetectedObject[] = [];
        const width = input.videoWidth;
        const height = input.videoHeight;

        // Process the first bounding box
        const [x1_norm, y1_norm, w1_norm, h1_norm] = bboxCoords.slice(0, 4);
        detectedObjects.push({
            bbox: [x1_norm * width, y1_norm * height, w1_norm * width, h1_norm * height],
            class: this.classNames[0], // Assuming first box is for class 0
            score: 1.0 // No score predicted by this model, assume 1.0 for now
        });

        // Process the second bounding box
        const [x2_norm, y2_norm, w2_norm, h2_norm] = bboxCoords.slice(4, 8);
        detectedObjects.push({
            bbox: [x2_norm * width, y2_norm * height, w2_norm * width, h2_norm * height],
            class: this.classNames[1], // Assuming second box is for class 1
            score: 1.0 // No score predicted by this model, assume 1.0 for now
        });

        // Dispose of tensors to free up memory
        tf.dispose([imgTensor, resized, expanded, normalized, ...predictions, bboxPrediction]);

        return detectedObjects;
    }
}
