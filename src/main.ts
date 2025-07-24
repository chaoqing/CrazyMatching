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
    private useAltAlgo: boolean = false;

    constructor() {
        this.video = document.getElementById('video') as HTMLVideoElement;
        this.canvas = document.getElementById('canvas') as HTMLCanvasElement;
        this.matchResult = document.getElementById('match-result') as HTMLDivElement;
        this.model = new Model();
        const toggleBtn = document.getElementById('toggle-algo') as HTMLButtonElement;
        if (toggleBtn) {
            toggleBtn.addEventListener('click', () => {
                this.useAltAlgo = !this.useAltAlgo;
                toggleBtn.textContent = this.useAltAlgo ? '切换到默认算法' : '切换检测算法';
            });
        }
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
        let predictions;
        if (this.useAltAlgo) {
            predictions = await this.altDetect();
        } else {
            predictions = await this.model.detect(this.video);
        }
        const ctx = this.canvas.getContext('2d');
        if (!ctx) {
            return;
        }
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // 直接使用模型 detect 返回的 success 字段
        if (Array.isArray(predictions) && predictions.length === 1 && typeof predictions[0].success === 'boolean') {
            if (predictions[0].success) {
                this.matchResult.innerText = `Match found`;
                console.info('Match found:', predictions[0].raw);
            } else {
                this.matchResult.innerText = '';
                console.info('Match not found:', predictions[0].raw);
            }
        }

        // 新模型输出格式：[cx1, cy1, w1, h1, r1, dx, dy, w2, h2, r2]
        if (Array.isArray(predictions) && predictions.length === 1 && Array.isArray(predictions[0].raw)) {
            const arr = predictions[0].raw;
            if (arr.length === 10) {
                const [cx1, cy1, w1, h1, r1, dx, dy, w2, h2, r2] = arr;
                // 第一个框
                drawRotatedRect(ctx, cx1, cy1, w1, h1, r1, '#00FFFF');
                ctx.fillText('1', cx1, cy1);
                // 第二个框
                const cx2 = cx1 + dx;
                const cy2 = cy1 + dy;
                drawRotatedRect(ctx, cx2, cy2, w2, h2, r2, '#FF00FF');
                ctx.fillText('2', cx2, cy2);
            }
        }
// 绘制旋转矩形
function drawRotatedRect(ctx: CanvasRenderingContext2D, cx: number, cy: number, w: number, h: number, angle: number, color: string) {
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(angle);
    ctx.strokeStyle = color;
    ctx.lineWidth = 4;
    ctx.strokeRect(-w/2, -h/2, w, h);
    ctx.restore();
}



        requestAnimationFrame(() => this.detect());
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

    // 示例：备用检测算法（可自定义实现）
    async altDetect(): Promise<any[]> {
        // 这里可以实现另一种检测逻辑，当前仅返回空数组
        // 你可以在此处集成其他模型或算法
        return [];
    }
}

const main = new Main();
main.run();

