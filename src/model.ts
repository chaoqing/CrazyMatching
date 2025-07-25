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
        function getRectCorners(cx: number, cy: number, w: number, h: number, angle: number): Array<{x: number, y: number}> {
            const hw = w / 2, hh = h / 2;
            const corners = [
                {x: -hw, y: -hh},
                {x: hw, y: -hh},
                {x: hw, y: hh},
                {x: -hw, y: hh}
            ];
            return corners.map(pt => {
                const x = pt.x * Math.cos(angle) - pt.y * Math.sin(angle) + cx;
                const y = pt.x * Math.sin(angle) + pt.y * Math.cos(angle) + cy;
                return {x, y};
            });
        }

        // 判断两个旋转矩形是否重叠（分离轴定理，近似实现）
        function isRectOverlap(a: Array<{x: number, y: number}>, b: Array<{x: number, y: number}>): boolean {
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

        function getAxes(corners: Array<{x: number, y: number}>): Array<{x: number, y: number}> {
            const axes = [];
            for (let i = 0; i < corners.length; i++) {
                const p1 = corners[i];
                const p2 = corners[(i + 1) % corners.length];
                const edge = {x: p2.x - p1.x, y: p2.y - p1.y};
                // 垂直向量
                axes.push({x: -edge.y, y: edge.x});
            }
            return axes;
        }

        function projectPolygon(corners: Array<{x: number, y: number}>, axis: {x: number, y: number}): [number, number] {
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
