declare module 'opencv.js' {
    export interface Mat {
        data32S: Int32Array;
        delete(): void;
    }

    export interface MatConstructor {
        new(): Mat;
    }

    export type Size = { width: number; height: number };
    export type Point = { x: number; y: number };
    export type Scalar = number[];
    
    const cv: {
        Mat: MatConstructor;
        onRuntimeInitialized?: () => void;
        
        // Constants
        CV_8U: number;
        CV_8UC4: number;
        CV_32F: number;
        CV_32S: number;
        COLOR_RGBA2BGR: number;
        COLOR_BGR2GRAY: number;
        COLOR_RGBA2GRAY: number;
        THRESH_BINARY: number;
        THRESH_OTSU: number;
        
        // Connected Components constants
        CC_STAT_LEFT: number;
        CC_STAT_TOP: number;
        CC_STAT_WIDTH: number;
        CC_STAT_HEIGHT: number;
        CC_STAT_AREA: number;
        CC_STAT_MAX: number;
        
        // Functions
        imread(imageSource: HTMLImageElement | HTMLCanvasElement | HTMLVideoElement): Mat;
        imshow(canvasOutput: HTMLCanvasElement, mat: Mat): void;
        matFromArray(rows: number, cols: number, type: number, array: number[]): Mat;
        matFromImageData(imageData: ImageData): Mat;
        cvtColor(src: Mat, dst: Mat, code: number): void;
        resize(src: Mat, dst: Mat, dsize: Size, fx?: number, fy?: number, interpolation?: number): void;
        threshold(src: Mat, dst: Mat, thresh: number, maxval: number, type: number): number;
        bitwise_not(src: Mat, dst: Mat): void;
        connectedComponentsWithStats(
            image: Mat,
            labels: Mat,
            stats: Mat,
            centroids: Mat,
            connectivity: number,
            ltype: number
        ): number;
    };

    export default cv;
    
    // Common functions and classes
    export function imread(imageSource: HTMLImageElement | HTMLCanvasElement | HTMLVideoElement): Mat;
    export function imshow(canvasOutput: HTMLCanvasElement, mat: Mat): void;
    export function matFromArray(rows: number, cols: number, type: number, array: number[]): Mat;
    export function matFromImageData(imageData: ImageData): Mat;
    
    // Constants
    export const CV_8U: number;
    export const CV_8UC4: number;
    export const CV_32F: number;
    export const CV_32S: number;
    export const COLOR_RGBA2BGR: number;
    export const COLOR_BGR2GRAY: number;
    export const COLOR_RGBA2GRAY: number;
    export const THRESH_BINARY: number;
    export const THRESH_OTSU: number;
    
    // Connected Components constants
    export const CC_STAT_LEFT: number;
    export const CC_STAT_TOP: number;
    export const CC_STAT_WIDTH: number;
    export const CC_STAT_HEIGHT: number;
    export const CC_STAT_AREA: number;
    export const CC_STAT_MAX: number;
    
    // Functions
    export function cvtColor(src: Mat, dst: Mat, code: number): void;
    export function resize(src: Mat, dst: Mat, dsize: Size, fx?: number, fy?: number, interpolation?: number): void;
    export function threshold(src: Mat, dst: Mat, thresh: number, maxval: number, type: number): number;
    export function bitwise_not(src: Mat, dst: Mat): void;
    export function connectedComponentsWithStats(
        image: Mat,
        labels: Mat,
        stats: Mat,
        centroids: Mat,
        connectivity: number,
        ltype: number
    ): number;
}
