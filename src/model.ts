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
uu distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';
import * as ort from 'onnxruntime-web';
import { cv2_detect } from './cv2.js';
declare const cv: any;

// Define a type for your custom detected objects, similar to cocoSsd.DetectedObject
// Adjust properties based on your model's output and what you need for your game logic.
// 新接口，兼容 raw 和 success 字段
export interface ModelDetectResult {
    raw: number[];
    success: boolean;
    allBboxes: { cx: number; cy: number; w: number; h: number; r: number; }[];
}

// Helper function to convert a tensor or array of tensors to a serializable array
const DUMP_TENSOR_DATA_AND_SHAPE = true; // Constant to control dumping of tensor data and shape

function tensorToFlatArray(tensor: tf.Tensor | tf.Tensor[] | null): any {
    if (!tensor) {
        return null;
    }

    if (Array.isArray(tensor)) {
        return tensor.map(t => ({ data: Array.from(t.dataSync()), shape: t.shape }));
    }
    return { data: Array.from(tensor.dataSync()), shape: tensor.shape };
}

// Helper function to send debug data to the backend
async function sendDebugData(data: any) {
    if (DUMP_TENSOR_DATA_AND_SHAPE && import.meta.env.DEV) {
        try {
            await fetch('/log-debug-data', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data),
            });
            // console.log('Debug data sent successfully.');
        } catch (error) {
            console.error('Failed to send debug data:', error);
        }
    }
}

function getDimensions(input: HTMLVideoElement | HTMLImageElement) {
    if (input instanceof HTMLVideoElement) {
        return { width: input.videoWidth, height: input.videoHeight };
    } else {
        return { width: input.width, height: input.height };
    }
}

export class Model {
    // Use tf.GraphModel for custom models
    private model: tf.GraphModel | null = null;
    // Define your class names in the order they were trained

    async load() {
        await tf.ready(); // Ensure TensorFlow.js backend is ready
        // Load your custom TensorFlow.js model from the public directory
        // The path should be relative to your web server's root.
        const modelPath = './models/crazy_matching/model.json'; // Path to your converted custom model
        this.model = await tf.loadGraphModel(modelPath);
        console.log('Custom model loaded from:', modelPath);
    }

    async detect(input: HTMLVideoElement | HTMLImageElement): Promise<ModelDetectResult[]> {
        if (!this.model) {
            console.log('Model not loaded.');
            return [];
        }

        let xRatio = 0.;
        let yRatio = 0.;

        const img = tf.browser.fromPixels(input);
        const dims = getDimensions(input);
        console.log(`Input dimensions: ${img.shape}, ${dims.width}x${dims.height}`);
        const [h, w] = img.shape.slice(0, 2); // get source width and height
        const maxSize = Math.max(w, h); // get max size
        const imgTensor = tf.pad(img, [
            [0, maxSize - h], // padding y [bottom only]
            [0, maxSize - w], // padding x [right only]
            [0, 0],
        ]);

        xRatio = 640 / maxSize; // update xRatio
        yRatio = 640 / maxSize; // update yRatio
        console.log("ratio: ", xRatio, yRatio)

        // Depending on your model's input requirements, you might need to resize,
        // normalize, or expand dimensions of the image tensor.
        // Example: Resize to model's expected input size (e.g., 300x300 for SSD)
        const resized = tf.image.resizeBilinear(imgTensor, [640, 640]); // 修正为模型要求的输入尺寸
        const expanded = resized.expandDims(0); // Add batch dimension
        const normalized = expanded.div(255.0); // Normalize to [0, 1] if your model expects it

        // Perform inference
        // The output of your custom model will be raw tensors.
        // You need to know the names of your model's output tensors.
        // Common names for object detection are 'detection_boxes', 'detection_scores', 'detection_classes', 'num_detections'.
        const model_results = this.model.execute(normalized);
        const numClass = 15;
        const transRes = tf.transpose(model_results as tf.Tensor, [0, 2, 1]); // transpose result [b, det, n] => [b, n, det]
        const boxes = tf.tidy(() => {
            const w = transRes.slice([0, 0, 2], [-1, -1, 1]); // get width
            const h = transRes.slice([0, 0, 3], [-1, -1, 1]); // get height
            const x1 = tf.sub(transRes.slice([0, 0, 0], [-1, -1, 1]), tf.div(w, 2)); // x1
            const y1 = tf.sub(transRes.slice([0, 0, 1], [-1, -1, 1]), tf.div(h, 2)); // y1
            return tf
                .concat(
                    [
                        y1,
                        x1,
                        tf.add(y1, h), //y2
                        tf.add(x1, w), //x2
                    ],
                    2
                )
                .squeeze();
        }); // process boxes [y1, x1, y2, x2]

        const [scores, classes] = tf.tidy(() => {
            // class scores
            const rawScores = transRes.slice([0, 0, 4], [-1, -1, numClass]).squeeze([0]); // #6 only squeeze axis 0 to handle only 1 class models
            return [rawScores.max(1), rawScores.argMax(1)];
        }); // get max scores and classes index

        const nms = await tf.image.nonMaxSuppressionAsync(boxes as tf.Tensor2D, scores, 500, 0.45, 0.2); // NMS to filter boxes


        const boxes_data = boxes.gather(nms, 0).dataSync(); // indexing boxes by nms index
        const scores_data = scores.gather(nms, 0).dataSync(); // indexing scores by nms index
        const classes_data = classes.gather(nms, 0).dataSync(); // indexing classes by nms index

        const width = dims.width;
        const height = dims.height;

        const finalDetections: { box: number[], label: number, score: number }[] = [];
        const numDetections = nms.size; // Number of detections after NMS

        for (let i = 0; i < numDetections; i++) {
            const score = scores_data[i];
            const label = classes_data[i];

            // boxes_data is [y1, x1, y2, x2] relative to maxSize
            const y1_scaled = boxes_data[i * 4];
            const x1_scaled = boxes_data[i * 4 + 1];
            const y2_scaled = boxes_data[i * 4 + 2];
            const x2_scaled = boxes_data[i * 4 + 3];

            // Scale back to original video dimensions
            const x1 = x1_scaled / xRatio;
            const y1 = y1_scaled / yRatio;
            const x2 = x2_scaled / xRatio;
            const y2 = y2_scaled / yRatio;

            // Ensure coordinates are within bounds
            const finalX1 = Math.max(0, x1);
            const finalY1 = Math.max(0, y1);
            const finalX2 = Math.min(width, x2);
            const finalY2 = Math.min(height, y2);

            // Filter by confidence threshold
            if (score > 0.2) {
                finalDetections.push({
                    box: [finalX1, finalY1, finalX2, finalY2], // xmin, ymin, xmax, ymax
                    label: label,
                    score: score
                });
            }
        }
        console.log("mid:", finalDetections)

        let success = false;
        let rawOutput: number[] = [];
        let maxCompositeScore = -1;
        let bestPair: { det1: typeof finalDetections[0], det2: typeof finalDetections[0] } | null = null;

        // Iterate through all unique pairs of detections
        for (let i = 0; i < finalDetections.length; i++) {
            for (let j = i + 1; j < finalDetections.length; j++) {
                const det1 = finalDetections[i];
                const det2 = finalDetections[j];

                // Only consider pairs of the same class
                if (det1.label === det2.label) {
                    // Calculate composite score: balances total score and score similarity
                    const compositeScore = (det1.score + det2.score) * (1 - Math.abs(det1.score - det2.score));

                    if (compositeScore > maxCompositeScore) {
                        maxCompositeScore = compositeScore;
                        bestPair = { det1, det2 };
                    }
                }
            }
        }
        console.log("final:", bestPair)

        if (bestPair) {
            // Convert bbox to [cx, cy, w, h]
            const boxToCxCyWh = (box: number[]) => {
                const xmin = box[0];
                const ymin = box[1];
                const xmax = box[2];
                const ymax = box[3];
                const w = xmax - xmin;
                const h = ymax - ymin;
                const cx = xmin + w / 2;
                const cy = ymin + h / 2;
                return [cx, cy, w, h];
            };

            const [cx1, cy1, w1, h1] = boxToCxCyWh(bestPair.det1.box);
            const [cx2, cy2, w2, h2] = boxToCxCyWh(bestPair.det2.box);

            // r1 and r2 are always 0 for this model
            const dims = getDimensions(input);
            rawOutput = [cx1 / dims.width, cy1 / dims.height, w1 / dims.width, h1 / dims.height, 0, cx2 / dims.width, cy2 / dims.height, w2 / dims.width, h2 / dims.height, 0];
            success = true;
        }

        // Debugging: Send data to backend in development mode
        if (DUMP_TENSOR_DATA_AND_SHAPE && import.meta.env.DEV) {
            sendDebugData({
                timestamp: new Date().toISOString(),
                model: 'Model',
                img: tensorToFlatArray(img),
                model_results: tensorToFlatArray(model_results),
                nms: tensorToFlatArray(nms),
                finalDetections: finalDetections,
                raw: rawOutput,
            });
        }

        // Dispose of all tensors
        tf.dispose([img, imgTensor, resized, expanded, normalized, model_results, transRes, boxes, scores, classes, nms]);

        // Convert all finalDetections to the desired format for allBboxes
        const allBboxes = finalDetections.map(det => {
            const [xmin, ymin, xmax, ymax] = det.box;
            const w = xmax - xmin;
            const h = ymax - ymin;
            const cx = xmin + w / 2;
            const cy = ymin + h / 2;
            const dims = getDimensions(input);
            return { cx: cx / dims.width, cy: cy / dims.height, w: w / dims.width, h: h / dims.height, r: 0 };
        });

        // Return raw, success, and allBboxes fields
        return [{ raw: rawOutput, success, allBboxes }];
    }
}

// Helper function for Intersection over Union (IoU)
function iou(box1: number[], box2: number[]): number {
    const x1 = Math.max(box1[0], box2[0]);
    const y1 = Math.max(box1[1], box2[1]);
    const x2 = Math.min(box1[2], box2[2]);
    const y2 = Math.min(box1[3], box2[3]);

    const intersectionWidth = Math.max(0, x2 - x1);
    const intersectionHeight = Math.max(0, y2 - y1);
    const intersectionArea = intersectionWidth * intersectionHeight;

    const box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    const box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1]);

    const unionArea = box1Area + box2Area - intersectionArea;

    return unionArea === 0 ? 0 : intersectionArea / unionArea;
}

// Non-Maximum Suppression (NMS) implementation
function nms(boxes: number[][], scores: number[], iouThreshold: number): number[] {
    const sortedIndices = scores.map((score, index) => ({ score, index }))
        .sort((a, b) => b.score - a.score)
        .map(item => item.index);

    const selectedIndices: number[] = [];
    const suppressed = new Array(boxes.length).fill(false);

    for (const currentIdx of sortedIndices) {
        if (suppressed[currentIdx]) {
            continue;
        }

        selectedIndices.push(currentIdx);
        const currentBox = boxes[currentIdx];

        for (let i = 0; i < boxes.length; i++) {
            if (i === currentIdx || suppressed[i]) {
                continue;
            }

            const otherBox = boxes[i];
            if (iou(currentBox, otherBox) > iouThreshold) {
                suppressed[i] = true;
            }
        }
    }
    return selectedIndices;
}

export class SSDModel {
    private session: ort.InferenceSession | null = null;
    private inputShape: [number, number, number, number] = [1, 3, 320, 320]; // Default input shape for SSDLite


    async load() {
        ort.env.wasm.numThreads = 1; // Use single thread for WASM for better compatibility
        ort.env.wasm.simd = true; // Enable SIMD for performance if available
        //ort.env.wasm.proxy = true; // Use web worker for inference
        //ort.env.wasm.wasmPaths = './'; // Set the relative path for WASM files to the assets folder

        const modelPath = './models/crazy_matching.onnx';
        try {
            this.session = await ort.InferenceSession.create(modelPath, {
                executionProviders: ['webgl', 'wasm'],
                graphOptimizationLevel: 'all'
            });
            console.log('ONNX model loaded from:', modelPath);

            // Get input shape from inputMetadata
            if (this.session.inputNames.length > 0) {
                const inputMeta = this.session.inputMetadata[0];
                if (inputMeta && inputMeta.isTensor && (inputMeta as any).shape && (inputMeta as any).shape.length === 4) {
                    // Handle dynamic input shapes: replace string dimensions (like 'batch_size') with 1
                    this.inputShape = (inputMeta as any).shape.map((dim: string | number) =>
                        typeof dim === 'string' ? 1 : dim
                    ) as [number, number, number, number];
                }
            }

        } catch (e) {
            console.error('Failed to load ONNX model:', e);
        }
    }

    async detect(input: HTMLVideoElement | HTMLImageElement): Promise<ModelDetectResult[]> {
        if (!this.session) {
            console.log('ONNX Model not loaded.');
            return [];
        }

        const dims = getDimensions(input);
        const width = dims.width;
        const height = dims.height;
        const resized_height = this.inputShape[2];
        const resized_width = this.inputShape[3];
        console.log(`Input video dimensions: ${width}x${height}`);

        // Preprocess image for ONNX model
        const imgTensor = tf.browser.fromPixels(input);
        const resized = tf.image.resizeBilinear(imgTensor, [resized_height, resized_width]);
        const normalized = resized.div(255.0);
        const transposed = normalized.transpose([2, 0, 1]); // HWC to CHW
        const expanded = transposed.expandDims(0); // Add batch dimension
        const inputData = new Float32Array(expanded.dataSync());

        tf.dispose([imgTensor, resized, normalized, transposed, expanded]);

        const inputName = this.session.inputNames[0];
        const feeds: { [key: string]: ort.Tensor } = {}; // Use a mutable object for feeds
        feeds[inputName] = new ort.Tensor('float32', inputData, this.inputShape);

        try {
            const results = await this.session.run(feeds);
            const boxes = results.boxes.data as Float32Array; // [num_detections, 4] (xmin, ymin, xmax, ymax)
            const scores = results.scores.data as Float32Array; // [num_detections]
            const labels = results.labels.data as Int32Array; // [num_detections]

            const detections: { box: number[], label: number, score: number }[] = [];
            for (let i = 0; i < labels.length; i++) {
                const score = scores[i];
                if (score > 0.2) { // Confidence threshold
                    // Convert box coordinates to scale free [0,1]
                    const box = [
                        boxes[i * 4] / resized_width, // xmin
                        boxes[i * 4 + 1] / resized_height, // ymin
                        boxes[i * 4 + 2] / resized_width, // xmax
                        boxes[i * 4 + 3] / resized_height, // ymax
                    ];
                    detections.push({ box, label: labels[i], score });
                }
            }

            // Apply NMS
            const nmsBoxes = detections.map(d => d.box);
            const nmsScores = detections.map(d => d.score);
            const selectedIndices = nms(nmsBoxes, nmsScores, 0.45); // IoU threshold for NMS

            const finalDetections = selectedIndices.map(idx => detections[idx]);

            // Group detections by class
            const detectionsByClass: { [key: number]: typeof finalDetections } = {};
            for (const det of finalDetections) {
                if (det.label !== 0) { // Exclude background class
                    if (!detectionsByClass[det.label]) {
                        detectionsByClass[det.label] = [];
                    }
                    detectionsByClass[det.label].push(det);
                }
            }
            console.log('Detections after NMS:', detectionsByClass);

            let success = false;
            let rawOutput: number[] = [];
            let maxCompositeScore = -1;
            let bestPair: { det1: typeof finalDetections[0], det2: typeof finalDetections[0] } | null = null;

            // Iterate through all unique pairs of detections
            for (let i = 0; i < finalDetections.length; i++) {
                for (let j = i + 1; j < finalDetections.length; j++) {
                    const det1 = finalDetections[i];
                    const det2 = finalDetections[j];

                    // Only consider pairs of the same class
                    if (det1.label === det2.label) {
                        // Calculate composite score: balances total score and score similarity
                        // Higher total score is better, smaller difference is better
                        const compositeScore = (det1.score + det2.score) * (1 - Math.abs(det1.score - det2.score));

                        if (compositeScore > maxCompositeScore) {
                            maxCompositeScore = compositeScore;
                            bestPair = { det1, det2 };
                        }
                    }
                }
            }

            if (bestPair) {
                // Convert bbox to [cx, cy, w, h]
                const boxToCxCyWh = (box: number[]) => {
                    const xmin = box[0];
                    const ymin = box[1];
                    const xmax = box[2];
                    const ymax = box[3];
                    const w = xmax - xmin;
                    const h = ymax - ymin;
                    const cx = xmin + w / 2;
                    const cy = ymin + h / 2;
                    return [cx, cy, w, h];
                };

                const [cx1, cy1, w1, h1] = boxToCxCyWh(bestPair.det1.box);
                const [cx2, cy2, w2, h2] = boxToCxCyWh(bestPair.det2.box);

                // r1 and r2 are always 0 for this model
                rawOutput = [cx1, cy1, w1, h1, 0, cx2, cy2, w2, h2, 0];
                success = true;
            }

            // Debugging: Send data to backend in development mode
            if (DUMP_TENSOR_DATA_AND_SHAPE && import.meta.env.DEV) {
                sendDebugData({
                    timestamp: new Date().toISOString(),
                    model: 'SSDModel',
                    img: tensorToFlatArray(imgTensor),
                    model_results: {
                        boxes: Array.from(boxes),
                        scores: Array.from(scores),
                        labels: Array.from(labels),
                    },
                    nms: selectedIndices, // selectedIndices is already an array
                    finalDetections: finalDetections,
                    raw: rawOutput,
                });
            }

            // Convert all finalDetections to the desired format for allBboxes
            const allBboxes = finalDetections.map(det => {
                const [xmin, ymin, xmax, ymax] = det.box;
                const w = xmax - xmin;
                const h = ymax - ymin;
                const cx = xmin + w / 2;
                const cy = ymin + h / 2;
                return { cx, cy, w, h, r: 0 };
            });

            return [{ raw: rawOutput, success, allBboxes }];

        } catch (e) {
            console.error('Failed to run ONNX inference:', e);
            return [];
        }
    }
}

export class CVModel {
    private minArea = 500; // Default minimum area, can be adjusted
    private isLoaded = false;

    async load() {
        // Wait for OpenCV.js to be loaded
        if (typeof cv === 'undefined') {
            await new Promise<void>((resolve) => {
                // Check if OpenCV is already loaded
                if (typeof cv !== 'undefined') {
                    this.isLoaded = true;
                    resolve();
                    return;
                }

                // Otherwise wait for it to load
                window.addEventListener('opencv-loaded', () => {
                    this.isLoaded = true;
                    resolve();
                }, { once: true });
            });
        } else {
            this.isLoaded = true;
        }
    }

    async detect(input: HTMLVideoElement | HTMLImageElement): Promise<ModelDetectResult[]> {
        if (!this.isLoaded) {
            console.error('OpenCV.js not loaded. Call load() first.');
            return [];
        }

        if (input instanceof HTMLVideoElement) {
            const imgTensor = tf.browser.fromPixels(input);
            const dims = getDimensions(input);

            let rgbaTensor = imgTensor;
            if (imgTensor.shape[2] === 3) {
                // If the tensor has 3 channels (RGB), add an opaque alpha channel
                const ones = tf.ones([dims.height, dims.width, 1]).mul(255);
                rgbaTensor = tf.concat([imgTensor, ones], 2) as tf.Tensor3D;
                tf.dispose(ones);
            }

            const pixels = new Uint8ClampedArray(rgbaTensor.dataSync());
            let imageData = new ImageData(pixels, dims.width, dims.height);
            tf.dispose([imgTensor, rgbaTensor]);
            return cv2_detect(imageData, this.minArea);
        }else{
            return cv2_detect(input, this.minArea);
        }

    }

}

