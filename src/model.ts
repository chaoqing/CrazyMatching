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
import * as cocoSsd from '@tensorflow-models/coco-ssd';

export class Model {
    private model: cocoSsd.ObjectDetection | null = null;

    async load() {
        await tf.ready(); // Ensure TensorFlow.js backend is ready
        this.model = await cocoSsd.load();
        console.log('COCO-SSD model loaded from CDN.');
    }

    async detect(input: HTMLVideoElement): Promise<cocoSsd.DetectedObject[]> {
        if (!this.model) {
            console.log('Model not loaded.');
            return [];
        }
        return this.model.detect(input);
    }
}
