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

// Define a type for your custom detected objects, similar to cocoSsd.DetectedObject
// Adjust properties based on your model's output and what you need for your game logic.
// 新接口，兼容 raw 和 success 字段
export interface ModelDetectResult {
    raw: number[];
    success: boolean;
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

    async detect(input: HTMLVideoElement): Promise<ModelDetectResult[]> {
        if (!this.model) {
            console.log('Model not loaded.');
            return [];
        }

        const imgTensor = tf.browser.fromPixels(input);
        // Depending on your model's input requirements, you might need to resize,
        // normalize, or expand dimensions of the image tensor.
        // Example: Resize to model's expected input size (e.g., 300x300 for SSD)
        const resized = tf.image.resizeBilinear(imgTensor, [224, 224]); // 修正为模型要求的输入尺寸
        const expanded = resized.expandDims(0); // Add batch dimension
        const normalized = expanded.div(255.0); // Normalize to [0, 1] if your model expects it

        // Perform inference
        // The output of your custom model will be raw tensors.
        // You need to know the names of your model's output tensors.
        // Common names for object detection are 'detection_boxes', 'detection_scores', 'detection_classes', 'num_detections'.
        const predictions = this.model.execute(normalized);
        let bboxPrediction: tf.Tensor;
        if (Array.isArray(predictions)) {
            if (!predictions[0]) {
                console.error('模型推理结果为空或格式不正确:', predictions);
                tf.dispose([imgTensor, resized, expanded, normalized]);
                return [];
            }
            bboxPrediction = predictions[0].squeeze();
        } else if (predictions instanceof tf.Tensor) {
            bboxPrediction = predictions.squeeze();
        } else {
            console.error('模型推理结果类型未知:', predictions);
            tf.dispose([imgTensor, resized, expanded, normalized]);
            return [];
        }
        const bboxCoords = bboxPrediction.arraySync() as number[];
        const width = input.videoWidth;
        const height = input.videoHeight;
        // 新模型输出格式：[cx1, cy1, w1, h1, r1, cx2, cy2, w2, h2, r2]
        let success = false;
        const raw = bboxCoords;
        if (bboxCoords.length === 10) {
            const [cx1, cy1, w1, h1, r1, cx2, cy2, w2, h2, r2] = bboxCoords;
            // 检查参数范围
            const valid1 = cx1 >= 0 && cx1 <= width && cy1 >= 0 && cy1 <= height && w1 > 0 && w1 <= width && h1 > 0 && h1 <= height;
            const valid2 = cx2 >= 0 && cx2 <= width && cy2 >= 0 && cy2 <= height && w2 > 0 && w2 <= width && h2 > 0 && h2 <= height;
            const rot1ok = r1 >= -Math.PI && r1 <= Math.PI;
            const rot2ok = r2 >= -Math.PI && r2 <= Math.PI;
            const box1 = getRectCorners(cx1, cy1, w1, h1, r1);
            const box2 = getRectCorners(cx2, cy2, w2, h2, r2);
            const notOverlap = !isRectOverlap(box1, box2);
            success = valid1 && valid2 && rot1ok && rot2ok && notOverlap;
        }
        if (Array.isArray(predictions)) {
            tf.dispose([imgTensor, resized, expanded, normalized, ...predictions, bboxPrediction]);
        } else {
            tf.dispose([imgTensor, resized, expanded, normalized, predictions as tf.Tensor, bboxPrediction]);
        }
        // 返回 raw 和 success 字段，供前端判断和绘制
        return [{ raw, success }];
        // 获取旋转矩形四个顶点
        function getRectCorners(cx: number, cy: number, w: number, h: number, angle: number): Array<{ x: number, y: number }> {
            const hw = w / 2, hh = h / 2;
            const corners = [
                { x: -hw, y: -hh },
                { x: hw, y: -hh },
                { x: hw, y: hh },
                { x: -hw, y: hh }
            ];
            return corners.map(pt => {
                const x = pt.x * Math.cos(angle) - pt.y * Math.sin(angle) + cx;
                const y = pt.x * Math.sin(angle) + pt.y * Math.cos(angle) + cy;
                return { x, y };
            });
        }

        // 判断两个旋转矩形是否重叠（分离轴定理，近似实现）
        function isRectOverlap(a: Array<{ x: number, y: number }>, b: Array<{ x: number, y: number }>): boolean {
            // 检查所有轴
            const axes = getAxes(a).concat(getAxes(b));
            for (const axis of axes) {
                const [minA, maxA] = projectPolygon(a, axis);
                const [minB, maxB] = projectPolygon(b, axis);
                if (maxA < minB || maxB < minA) {
                    return false; // 存在分离轴
                }
            }
            return true; // 所有轴都重叠
        }

        function getAxes(corners: Array<{ x: number, y: number }>): Array<{ x: number, y: number }> {
            const axes = [];
            for (let i = 0; i < corners.length; i++) {
                const p1 = corners[i];
                const p2 = corners[(i + 1) % corners.length];
                const edge = { x: p2.x - p1.x, y: p2.y - p1.y };
                // 垂直向量
                axes.push({ x: -edge.y, y: edge.x });
            }
            return axes;
        }

        function projectPolygon(corners: Array<{ x: number, y: number }>, axis: { x: number, y: number }): [number, number] {
            const norm = Math.sqrt(axis.x * axis.x + axis.y * axis.y);
            const ax = axis.x / norm, ay = axis.y / norm;
            let min = Infinity, max = -Infinity;
            for (const pt of corners) {
                const proj = pt.x * ax + pt.y * ay;
                min = Math.min(min, proj);
                max = Math.max(max, proj);
            }
            return [min, max];
        }
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
        ort.env.wasm.proxy = true; // Use web worker for inference
        ort.env.wasm.wasmPaths = './'; // Set the relative path for WASM files to the assets folder

        const modelPath = './models/crazy_matching.onnx';
        try {
            this.session = await ort.InferenceSession.create(modelPath, {
                executionProviders: ['wasm'],
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

    async detect(input: HTMLVideoElement): Promise<ModelDetectResult[]> {
        if (!this.session) {
            console.log('ONNX Model not loaded.');
            return [];
        }

        const width = input.videoWidth;
        const height = input.videoHeight;
        console.log(`Input video dimensions: ${width}x${height}`);

        // Preprocess image for ONNX model
        const imgTensor = tf.browser.fromPixels(input);
        const resized = tf.image.resizeBilinear(imgTensor, [this.inputShape[2], this.inputShape[3]]);
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
            // Assuming output names are 'boxes', 'labels', 'scores'
            const boxes = results.boxes.data as Float32Array; // [num_detections, 4] (xmin, ymin, xmax, ymax)
            const labels = results.labels.data as Int32Array; // [num_detections]
            const scores = Array.from(results.scores.data as any).map((s: any) => Number(s)); // [num_detections]

            const detections: { box: number[], label: number, score: number }[] = [];
            for (let i = 0; i < labels.length; i++) {
                const score = scores[i];
                if (score > 0.5) { // Confidence threshold
                    const box = [
                        boxes[i * 4] * width, // xmin
                        boxes[i * 4 + 1] * height, // ymin
                        boxes[i * 4 + 2] * width, // xmax
                        boxes[i * 4 + 3] * height // ymax
                    ];
                    detections.push({ box, label: labels[i], score });
                }
            }

            // Apply NMS
            const nmsBoxes = detections.map(d => d.box);
            const nmsScores = detections.map(d => d.score);
            console.log('Detections before NMS:', nmsBoxes, nmsScores);
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

            // Find if there are two animals of the same class
            for (const classId in detectionsByClass) {
                const classDetections = detectionsByClass[classId];
                if (classDetections.length >= 2) {
                    // Take the first two detections of the same class
                    const det1 = classDetections[0];
                    const det2 = classDetections[1];

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

                    const [cx1, cy1, w1, h1] = boxToCxCyWh(det1.box);
                    const [cx2, cy2, w2, h2] = boxToCxCyWh(det2.box);

                    // r1 and r2 are always 0 for this model
                    rawOutput = [cx1, cy1, w1, h1, 0, cx2, cy2, w2, h2, 0];
                    success = true;
                    break; // Found a match, no need to check other classes
                }
            }

            return [{ raw: rawOutput, success }];

        } catch (e) {
            console.error('Failed to run ONNX inference:', e);
            return [];
        }
    }
}
