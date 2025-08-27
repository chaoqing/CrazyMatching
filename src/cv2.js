export const DUMP_TENSOR_DATA_AND_SHAPE = true;

export async function cv2_detect(input, minArea) {
    let src;
    let currentInputWidth;
    let currentInputHeight;

    if (input instanceof HTMLImageElement) {
        src = cv.imread(input);
        currentInputWidth = input.width;
        currentInputHeight = input.height;
    } else if (input instanceof ImageData) {
        src = new cv.Mat(input.height, input.width, cv.CV_8UC4);
        src.data.set(input.data);
        cv.cvtColor(src, src, cv.COLOR_RGBA2BGR); // Convert from RGBA to BGR
        currentInputWidth = input.width;
        currentInputHeight = input.height;
    } else {
        throw new Error("Please input a valid HTMLImageElement or ImageData.");
    }
    console.log(`Input image size: ${src.cols}x${src.rows}`);
        
        const gray = new cv.Mat();
        const objectMask = new cv.Mat();
        const labels = new cv.Mat();
        const stats = new cv.Mat();
        const centroids = new cv.Mat();
        
        // Get canvas for debugging visualization
        const canvas = document.getElementById('canvas');

        try {
            cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
            cv.threshold(gray, objectMask, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
            cv.bitwise_not(objectMask, objectMask); // Invert the mask

            const numLabels = cv.connectedComponentsWithStats(objectMask, labels, stats, centroids, 4, cv.CV_32S);
            console.log(`Found ${numLabels - 1} connected components (excluding background).`);

            const allBboxes = [];
            const componentAreas = [];

            for (let i = 1; i < numLabels; ++i) { // Skip background label 0
                const area = stats.data32S[i * cv.CC_STAT_MAX + cv.CC_STAT_AREA];
                if (area > 0) {
                    componentAreas.push(area);
                }
            }

            let lowerBound = -1;
            let upperBound = Infinity;

            console.log(`Component areas: ${componentAreas.join(', ')}`);
            if (componentAreas.length > 2) {
                const meanArea = componentAreas.reduce((sum, a) => sum + a, 0) / componentAreas.length;
                const stdDevArea = Math.sqrt(componentAreas.map(a => (a - meanArea) ** 2).reduce((sum, sq) => sum + sq, 0) / componentAreas.length);
                lowerBound = meanArea - 2 * stdDevArea;
                upperBound = meanArea + 2 * stdDevArea;
            }

            for (let i = 1; i < numLabels; ++i) {
                const x = stats.data32S[i * cv.CC_STAT_MAX + cv.CC_STAT_LEFT];
                const y = stats.data32S[i * cv.CC_STAT_MAX + cv.CC_STAT_TOP];
                const w = stats.data32S[i * cv.CC_STAT_MAX + cv.CC_STAT_WIDTH];
                const h = stats.data32S[i * cv.CC_STAT_MAX + cv.CC_STAT_HEIGHT];
                const area = stats.data32S[i * cv.CC_STAT_MAX + cv.CC_STAT_AREA];

                if (area < minArea) {
                    continue;
                }

                const isOutlier = !(lowerBound < area && area < upperBound);

                if (!isOutlier) {
                    const inputWidth = input instanceof HTMLVideoElement ? input.videoWidth : input.width;
                    const inputHeight = input instanceof HTMLVideoElement ? input.videoHeight : input.height;
                    const cx = (x + w / 2) / currentInputWidth;
                    const cy = (y + h / 2) / currentInputHeight;
                    const normalizedW = w / currentInputWidth;
                    const normalizedH = h / currentInputHeight;
                    allBboxes.push({ cx, cy, w: normalizedW, h: normalizedH, r: 0 });
                    
                    // Draw component boundaries for debugging
                    if (DUMP_TENSOR_DATA_AND_SHAPE && import.meta.env.DEV) {
                        const ctx = canvas.getContext('2d');
                        if (ctx) {
                            ctx.strokeStyle = isOutlier ? 'red' : 'green';
                            ctx.lineWidth = 2;
                            ctx.strokeRect(x, y, w, h);
                            
                            // Add area text
                            ctx.fillStyle = isOutlier ? 'red' : 'green';
                            ctx.font = '12px Arial';
                            ctx.fillText(`Area: ${area}`, x, y - 5);
                        }
                    }
                }
            }

            // Find the two largest bounding boxes for the 'raw' output, similar to card detection
            // Sort by area (descending) and take the top 2
            const sortedBboxes = [...allBboxes].sort((a, b) => (b.w * b.h) - (a.w * a.h));

            let rawOutput = [];
            let success = false;

            if (sortedBboxes.length >= 2) {
                const det1 = sortedBboxes[0];
                const det2 = sortedBboxes[1];
                rawOutput = [det1.cx, det1.cy, det1.w, det1.h, det1.r, det2.cx, det2.cy, det2.w, det2.h, det2.r];
                success = true;
            }

            // Save debug visualization if enabled
            if (DUMP_TENSOR_DATA_AND_SHAPE) {
                const ctx = canvas.getContext('2d');
                if (ctx) {
                    cv.imshow(canvas, objectMask);
                }
            }

            return [{ raw: rawOutput, success, allBboxes }];

        } catch (error) {
            console.error('OpenCV.js detection failed:', error);
            return [];
        } finally {
            src.delete();
            gray.delete();
            objectMask.delete();
            labels.delete();
            stats.delete();
            centroids.delete();
        }
    }