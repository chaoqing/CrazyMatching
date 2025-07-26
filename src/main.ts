if (import.meta.env.DEV) {
  import('eruda').then((eruda) => {
    eruda.default.init();
  });
}

import './style.css';
import { Model } from './model';

class Main {
    private video: HTMLVideoElement;
    private canvas: HTMLCanvasElement;
    private model: Model;
    private isDetecting: boolean = false;
    private videoDevices: MediaDeviceInfo[] = [];
    private currentStream: MediaStream | null = null;

    constructor() {
        this.video = document.getElementById('video') as HTMLVideoElement;
        this.canvas = document.getElementById('canvas') as HTMLCanvasElement;
        this.model = new Model();
        const toggleBtn = document.getElementById('toggle-algo') as HTMLButtonElement;
        if (toggleBtn) {
            toggleBtn.addEventListener('click', () => {
                this.isDetecting = !this.isDetecting;
                toggleBtn.textContent = this.isDetecting ? 'Stop' : 'Start';
                if (this.isDetecting) {
                    this.detect();
                } else {
                    const ctx = this.canvas.getContext('2d');
                    if (ctx) {
                        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
                    }
                }
            });
        }
    }

    async run() {
        await this.model.load();
        await this.setupCamera();
    }

    async setupCamera() {
        // 1. Enumerate devices
        const devices = await navigator.mediaDevices.enumerateDevices();
        this.videoDevices = devices.filter(device => device.kind === 'videoinput');
        this.videoDevices.sort((a, b) => a.label.localeCompare(b.label));
        console.log('Available video devices:', this.videoDevices.map(d => ({ label: d.label, deviceId: d.deviceId })));

        // 2. Determine the best initial camera
        const wideCamera = this.videoDevices.find(d => d.label.toLowerCase().includes('wide'));
        const initialDeviceId = wideCamera ? wideCamera.deviceId : (this.videoDevices.length > 0 ? this.videoDevices[0].deviceId : undefined);
        console.log('Initial camera selected:', initialDeviceId);

        // 3. Start the stream with the chosen camera
        if (initialDeviceId) {
            await this.startStream(initialDeviceId);
        } else {
            // Fallback if no specific device is found
            await this.startStream();
        }

        // 4. Setup UI after stream is running
        this.setupCameraSelectorUI();
    }

    setupCameraSelectorUI() {
        const cameraSelector = document.getElementById('right-controls');
        if (!cameraSelector || this.videoDevices.length <= 1) {
            return;
        }

        cameraSelector.innerHTML = '';
        this.videoDevices.forEach(device => {
            const button = document.createElement('button');
            button.classList.add('camera-select-button');
            button.dataset.deviceId = device.deviceId;
            button.title = device.label;
            button.addEventListener('click', () => this.startStream(device.deviceId));
            cameraSelector.appendChild(button);
        });
        cameraSelector.style.display = 'flex';
        
        const currentDeviceId = this.currentStream?.getVideoTracks()[0].getSettings().deviceId;
        if(currentDeviceId) {
            this.updateActiveButton(currentDeviceId);
        }
    }

    async startStream(deviceId?: string) {
        if (this.currentStream) {
            this.currentStream.getTracks().forEach(track => track.stop());
        }

        const videoConstraints: MediaTrackConstraints = deviceId 
            ? { deviceId: { exact: deviceId } } 
            : { facingMode: 'environment' };

        try {
            this.currentStream = await navigator.mediaDevices.getUserMedia({ video: videoConstraints, audio: false });
            this.video.srcObject = this.currentStream;

            await new Promise(resolve => {
                this.video.onloadedmetadata = () => resolve(true);
            });

            await this.video.play();

            this.canvas.width = this.video.videoWidth;
            this.canvas.height = this.video.videoHeight;
            
            const currentDeviceId = this.currentStream.getVideoTracks()[0].getSettings().deviceId;
            if (currentDeviceId) {
                this.updateActiveButton(currentDeviceId);
            }
            this.setupZoomSlider();

        } catch (error) {
            console.error(`Error starting stream for device ${deviceId}:`, error);
        }
    }

    updateActiveButton(deviceId: string) {
        document.querySelectorAll('.camera-select-button').forEach(button => {
            const btn = button as HTMLButtonElement;
            btn.classList.toggle('active', btn.dataset.deviceId === deviceId);
        });
    }

    setupZoomSlider() {
        if (!this.currentStream) return;

        const [track] = this.currentStream.getVideoTracks();
        const capabilities = track.getCapabilities();
        const zoomSlider = document.getElementById('zoom') as HTMLInputElement;

        if (capabilities.zoom && zoomSlider) {
            zoomSlider.min = capabilities.zoom.min.toString();
            zoomSlider.max = capabilities.zoom.max.toString();
            zoomSlider.step = capabilities.zoom.step.toString();
            zoomSlider.value = track.getSettings().zoom?.toString() ?? capabilities.zoom.min.toString();
            zoomSlider.disabled = false;

            zoomSlider.oninput = (event) => {
                track.applyConstraints({ advanced: [{ zoom: (event.target as HTMLInputElement).valueAsNumber }] });
            };
        } else {
            zoomSlider.disabled = true;
        }
    }

    async detect() {
        if (!this.isDetecting) return;

        const predictions = await this.model.detect(this.video);
        const ctx = this.canvas.getContext('2d');
        if (!ctx) return;

        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        if (Array.isArray(predictions) && predictions.length > 0 && Array.isArray(predictions[0].raw)) {
            const arr = predictions[0].raw;
            if (arr.length === 10) {
                const [cx1, cy1, w1, h1, r1, cx2, cy2, w2, h2, r2] = arr;
                const scaleX = this.canvas.clientWidth;
                const scaleY = this.canvas.clientHeight;
                drawRotatedRect(ctx, cx1 * scaleX, cy1 * scaleY, w1 * scaleX, h1 * scaleY, r1, '#00FFFF');
                ctx.fillText('1', cx1 * scaleX, cy1 * scaleY);
                drawRotatedRect(ctx, cx2 * scaleX, cy2 * scaleY, w2 * scaleX, h2 * scaleY, r2, '#FF00FF');
                ctx.fillText('2', cx2 * scaleX, cy2 * scaleY);
            }
        }
        
        requestAnimationFrame(() => this.detect());
    }
}

const main = new Main();
main.run();

function drawRotatedRect(ctx: CanvasRenderingContext2D, cx: number, cy: number, w: number, h: number, angle: number, color: string) {
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(angle);
    ctx.strokeStyle = color;
    ctx.lineWidth = 4;
    ctx.strokeRect(-w/2, -h/2, w, h);
    ctx.restore();
}