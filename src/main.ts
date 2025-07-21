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

import './style.css';
import { Model } from './model';

class Main {
    private video: HTMLVideoElement;
    private canvas: HTMLCanvasElement;
    private model: Model;
    private matchResult: HTMLDivElement;

    constructor() {
        this.video = document.getElementById('video') as HTMLVideoElement;
        this.canvas = document.getElementById('canvas') as HTMLCanvasElement;
        this.matchResult = document.getElementById('match-result') as HTMLDivElement;
        this.model = new Model();
    }

    async setupCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'environment' },
                audio: false,
            });
            this.video.srcObject = stream;
            return new Promise((resolve) => {
                this.video.onloadedmetadata = () => {
                    resolve(this.video);
                };
            });
        } catch (error) {
            console.error('Error in getUserMedia:', error);
            throw error; // Re-throw the error to be caught by the run() method
        }
    }

    async detect() {
        const predictions = await this.model.detect(this.video);
        const ctx = this.canvas.getContext('2d');
        if (!ctx) {
            return;
        }
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        const detectedObjects = new Set();
        for (const prediction of predictions) {
            const [x, y, width, height] = prediction.bbox;
            ctx.strokeStyle = '#00FFFF';
            ctx.lineWidth = 4;
            ctx.strokeRect(x, y, width, height);
            ctx.fillStyle = '#00FFFF';
            const text = `${prediction.class} (${Math.round(prediction.score * 100)}%)`;
            ctx.fillText(text, x, y > 10 ? y - 5 : 10);
            detectedObjects.add(prediction.class);
        }

        // For now, we just check if we have detected more than one object of the same class
        const duplicates = this.findDuplicates(predictions);
        if (duplicates.length > 0) {
            this.matchResult.innerText = `Match found: ${duplicates.join(', ')}`;
        } else {
            this.matchResult.innerText = '';
        }


        requestAnimationFrame(() => this.detect());
    }

    findDuplicates(predictions: any[]): string[] {
        const counts: Record<string, number> = {};
        for (const prediction of predictions) {
            counts[prediction.class] = (counts[prediction.class] || 0) + 1;
        }
        return Object.keys(counts).filter(key => counts[key] > 1);
    }

    async run() {
        try {
            await this.model.load();
            console.log('Model loaded successfully.');
        } catch (error) {
            console.error('Error loading model:', error);
            this.matchResult.innerText = 'Error loading model. Please check console for details.';
            return;
        }

        try {
            await this.setupCamera();
            this.video.play();
            this.canvas.width = this.video.videoWidth;
            this.canvas.height = this.video.videoHeight;
            console.log('Camera setup and video started.');
        } catch (error) {
            console.error('Error setting up camera:', error);
            this.matchResult.innerText = 'Error accessing camera. Please ensure camera permissions are granted.';
            return;
        }
        this.detect();
    }
}

const main = new Main();
main.run();

