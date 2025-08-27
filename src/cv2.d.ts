declare function cv2_detect(input: HTMLImageElement | ImageData, minArea: number): Promise<{
    raw: number[];
    success: boolean;
    allBboxes: { cx: number; cy: number; w: number; h: number; r: number; }[];
}[]>;

export { cv2_detect };
