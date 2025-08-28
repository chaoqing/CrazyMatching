export const DUMP_TENSOR_DATA_AND_SHAPE = true;

function computeSimilarity(image1, image2) {
    return 1.0
    // Compute similarity between two RGBA images using Hu Moments.
    // This method is invariant to translation, rotation, and scale.

    // Convert to grayscale
    let image1_gray = new cv.Mat();
    let image2_gray = new cv.Mat();

    if (image1.channels() === 4) { // RGBA
        cv.cvtColor(image1, image1_gray, cv.COLOR_RGBA2GRAY);
        cv.cvtColor(image2, image2_gray, cv.COLOR_RGBA2GRAY);
    } else { // Assume RGB/BGR
        cv.cvtColor(image1, image1_gray, cv.COLOR_RGB2GRAY);
        cv.cvtColor(image2, image2_gray, cv.COLOR_RGB2GRAY);
    }

    // Calculate Hu Moments for both images
    let moments1 = cv.moments(image1_gray, false);
    let moments2 = cv.moments(image2_gray, false);

    // Handle empty images or failed moment calculation
    if (moments1.m00 === 0 || moments2.m00 === 0) {
        image1_gray.delete();
        image2_gray.delete();
        return 0.0;
    }

    // Calculate Hu Moments
    let huMoments1 = cv.HuMoments(moments1);
    let huMoments2 = cv.HuMoments(moments2);

    // Convert to log scale to handle small values better
    for (let i = 0; i < 7; i++) {
        if (huMoments1.data64F[i] !== 0) {
            huMoments1.data64F[i] = -Math.sign(huMoments1.data64F[i]) * Math.log10(Math.abs(huMoments1.data64F[i]));
        }
        if (huMoments2.data64F[i] !== 0) {
            huMoments2.data64F[i] = -Math.sign(huMoments2.data64F[i]) * Math.log10(Math.abs(huMoments2.data64F[i]));
        }
    }

    // Calculate similarity using normalized L2 distance between Hu moments
    let distance = 0;
    for (let i = 0; i < 7; i++) {
        distance += Math.pow(huMoments1.data64F[i] - huMoments2.data64F[i], 2);
    }
    distance = Math.sqrt(distance);

    // Convert distance to similarity score (0 to 1)
    let similarity = Math.exp(-distance / 2.0);

    image1_gray.delete();
    image2_gray.delete();
    huMoments1.delete();
    huMoments2.delete();

    return similarity;
}

async function cv2_crazy_matching(inputImage) {
    // Get canvas for debugging visualization
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    let image = inputImage.clone(); // Work on a clone to avoid modifying the original input

    let gray = new cv.Mat();
    let bilateral = new cv.Mat();
    let edges = new cv.Mat();
    let dilated = new cv.Mat();
    let closed = new cv.Mat();
    let contours = new cv.MatVector();
    let hierarchy = new cv.Mat();

    let card1_bbox = null;
    let card2_bbox = null;
    let patches1 = [];
    let patches2 = [];
    let best_pair = null;
    let max_similarity = 0.0;
    let kernel = cv.Mat.ones(3, 3, cv.CV_8U);
    let small_kernel = cv.Mat.ones(3, 3, cv.CV_8U);
    let large_kernel = cv.Mat.ones(10, 10, cv.CV_8U);
    let card_contours = [];

    try {
        // Preprocessing
        cv.cvtColor(image, gray, cv.COLOR_BGR2GRAY);
        cv.bilateralFilter(gray, bilateral, 9, 75, 75);
        cv.Canny(bilateral, edges, 50, 150);
        cv.dilate(edges, dilated, kernel, new cv.Point(-1, -1), 2);
        cv.morphologyEx(dilated, closed, cv.MORPH_CLOSE, kernel, new cv.Point(-1, -1), 2);

        if (DUMP_TENSOR_DATA_AND_SHAPE && import.meta.env.DEV) {
            if (ctx) {
                cv.imshow(canvas, closed);
            }
        }

        // Find contours for cards
        cv.findContours(closed, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
        let min_card_area = image.rows * image.cols * 0.05;

        for (let i = 0; i < contours.size(); ++i) {
            let cnt = contours.get(i);
            let area = cv.contourArea(cnt);
            if (area < min_card_area) {
                cnt.delete();
                continue;
            }
            let peri = cv.arcLength(cnt, true);
            let approx = new cv.Mat();
            cv.approxPolyDP(cnt, approx, 0.02 * peri, true);
            if (approx.rows === 4) {
                card_contours.push(cnt);
            } else {
                cnt.delete();
            }
            approx.delete();
        }

        if (card_contours.length < 2) {
            return null;
        }

        // Find pair with most similar areas, which should be the two cards
        let areas = card_contours.map(cnt => cv.contourArea(cnt));
        let min_diff_ratio = Infinity;
        let best_pair_idx = [0, 1];

        for (let i = 0; i < card_contours.length; ++i) {
            for (let j = i + 1; j < card_contours.length; ++j) {
                let diff_ratio = Math.abs(areas[i] - areas[j]) / Math.max(areas[i], areas[j]);
                if (diff_ratio < min_diff_ratio) {
                    min_diff_ratio = diff_ratio;
                    best_pair_idx = [i, j];
                }
            }
        }

        if (min_diff_ratio > 0.2) {
            return null;
        }

        let final_card_contours = [card_contours[best_pair_idx[0]], card_contours[best_pair_idx[1]]];
        let bboxes = final_card_contours.map(c => cv.boundingRect(c));
        card1_bbox = bboxes[0];
        card2_bbox = bboxes[1];

        if (DUMP_TENSOR_DATA_AND_SHAPE) {
            if (ctx) {
                ctx.strokeStyle = 'green';
                ctx.lineWidth = 2;
                ctx.strokeRect(card1_bbox.x, card1_bbox.y, card1_bbox.width, card1_bbox.height);
                ctx.strokeRect(card2_bbox.x, card2_bbox.y, card2_bbox.width, card2_bbox.height);

                ctx.fillStyle = 'red';
                ctx.font = '12px Arial';
                ctx.fillText(`Card1`, card1_bbox.x, card1_bbox.y - 5);
                ctx.fillText(`Card2`, card2_bbox.x, card2_bbox.y - 5);
            }
        }

        // Process each card
        let all_patches = [[], []];
        for (let card_idx = 0; card_idx < final_card_contours.length; ++card_idx) {
            let card_cnt = final_card_contours[card_idx];
            let bbox = bboxes[card_idx];
            let x = bbox.x;
            let y = bbox.y;
            let w = bbox.width;
            let h = bbox.height;

            // Extract card region
            let card_mask = new cv.Mat.zeros(gray.rows, gray.cols, cv.CV_8U);
            let card_cnt_vec = new cv.MatVector();
            card_cnt_vec.push_back(card_cnt);
            cv.drawContours(card_mask, card_cnt_vec, 0, new cv.Scalar(255), -1);
            card_cnt_vec.delete();
            let card_region = new cv.Mat();
            cv.bitwise_and(image, image, card_region, card_mask);
            let card_roi = card_region.roi(bbox);
            let gray_roi = new cv.Mat();
            cv.cvtColor(card_roi, gray_roi, cv.COLOR_BGR2GRAY);

            // Adaptive thresholding
            let white_mask = new cv.Mat();
            cv.adaptiveThreshold(gray_roi, white_mask, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 10);
            let animal_mask = new cv.Mat();
            cv.bitwise_not(white_mask, animal_mask);

            // Morphological close
            cv.morphologyEx(animal_mask, animal_mask, cv.MORPH_CLOSE, small_kernel, new cv.Point(-1, -1), 2);

            // Connected components
            let labels = new cv.Mat();
            let stats = new cv.Mat();
            let centroids = new cv.Mat();
            let num_labels = cv.connectedComponentsWithStats(animal_mask, labels, stats, centroids, 8, cv.CV_32S);

            let filled_components = [];
            let card_area = w * h;
            let min_component_area = 0.001 * card_area;
            let max_component_area = 0.5 * card_area;

            for (let i = 1; i < num_labels; ++i) {
                let component_mask = new cv.Mat();
                cv.compare(labels, new cv.Mat(1, 1, labels.type(), new cv.Scalar(i)), component_mask, cv.CMP_EQ);
                component_mask.convertTo(component_mask, cv.CV_8U, 255);

                let contours_comp = new cv.MatVector();
                let hierarchy_comp = new cv.Mat();
                cv.findContours(component_mask, contours_comp, hierarchy_comp, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

                let hull_mask = new cv.Mat.zeros(component_mask.rows, component_mask.cols, cv.CV_8U);
                for (let j = 0; j < contours_comp.size(); ++j) {
                    let cont_comp = contours_comp.get(j);
                    let hull = new cv.Mat();
                    cv.convexHull(cont_comp, hull, false, true);
                    let hull_vec = new cv.MatVector();
                    hull_vec.push_back(hull);
                    cv.drawContours(hull_mask, hull_vec, 0, new cv.Scalar(255), -1);
                    hull_vec.delete();
                    hull.delete();
                    cont_comp.delete();
                }
                contours_comp.delete();
                hierarchy_comp.delete();

                let filled_hull_mask = new cv.Mat();
                cv.bitwise_and(hull_mask, animal_mask, filled_hull_mask);

                let component_mask_closed = new cv.Mat();
                cv.morphologyEx(filled_hull_mask, component_mask_closed, cv.MORPH_CLOSE, large_kernel, new cv.Point(-1, -1), 2);

                let final_component_mask = new cv.Mat();
                cv.bitwise_and(hull_mask, component_mask_closed, final_component_mask);

                let contours_final = new cv.MatVector();
                let hierarchy_final = new cv.Mat();
                cv.findContours(final_component_mask, contours_final, hierarchy_final, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

                let filled_mask = new cv.Mat.zeros(final_component_mask.rows, final_component_mask.cols, cv.CV_8U);
                cv.drawContours(filled_mask, contours_final, -1, new cv.Scalar(255), cv.FILLED);
                contours_final.delete();
                hierarchy_final.delete();

                let solid_area = cv.countNonZero(hull_mask);
                if (min_component_area < solid_area && solid_area < max_component_area) {
                    filled_components.push({ mask: filled_mask, area: solid_area });
                } else {
                    filled_mask.delete();
                }

                component_mask.delete();
                hull_mask.delete();
                filled_hull_mask.delete();
                component_mask_closed.delete();
                final_component_mask.delete();
            }

            // Sort and rebuild animal_mask from all valid components
            filled_components.sort((a, b) => b.area - a.area);
            animal_mask.setTo(new cv.Scalar(0)); // Clear animal_mask
            for (let comp of filled_components) {
                cv.bitwise_or(animal_mask, comp.mask, animal_mask);
                comp.mask.delete(); // Delete after use
            }

            // Final connected components for patches
            let labels_final = new cv.Mat();
            let stats_final = new cv.Mat();
            let centroids_final = new cv.Mat();
            let num_labels_final = cv.connectedComponentsWithStats(animal_mask, labels_final, stats_final, centroids_final, 8, cv.CV_32S);
            console.log(`Card ${card_idx + 1}: Found ${num_labels_final - 1} patches`);

            for (let i = 1; i < num_labels_final; ++i) {
                let x_comp = stats_final.data32S[i * cv.CC_STAT_MAX + cv.CC_STAT_LEFT];
                let y_comp = stats_final.data32S[i * cv.CC_STAT_MAX + cv.CC_STAT_TOP];
                let w_comp = stats_final.data32S[i * cv.CC_STAT_MAX + cv.CC_STAT_WIDTH];
                let h_comp = stats_final.data32S[i * cv.CC_STAT_MAX + cv.CC_STAT_HEIGHT];

                let component_mask_full = new cv.Mat();
                cv.compare(labels_final, new cv.Mat(1, 1, labels.type(), new cv.Scalar(i)), component_mask_full, cv.CMP_EQ);
                component_mask_full.convertTo(component_mask_full, cv.CV_8U, 255);

                let component_roi_rect = new cv.Rect(x_comp, y_comp, w_comp, h_comp);
                let component_roi = card_roi.roi(component_roi_rect);
                let component_mask = component_mask_full.roi(component_roi_rect);

                let channels = new cv.MatVector();
                cv.split(component_roi, channels);
                let b = channels.get(0);
                let g = channels.get(1);
                let r = channels.get(2);
                channels.delete();

                let rgba_channels = new cv.MatVector();
                rgba_channels.push_back(b);
                rgba_channels.push_back(g);
                rgba_channels.push_back(r);
                rgba_channels.push_back(component_mask);

                let rgba = new cv.Mat();
                cv.merge(rgba_channels, rgba);
                rgba_channels.delete();

                let abs_bbox = { x: x + x_comp, y: y + y_comp, width: w_comp, height: h_comp };
                all_patches[card_idx].push({ bbox: abs_bbox, img: rgba, mask: component_mask.clone() }); // Clone mask for storage

                if (DUMP_TENSOR_DATA_AND_SHAPE && import.meta.env.DEV) {
                    if (ctx) {
                        ctx.strokeStyle = 'orange';
                        ctx.lineWidth = 2;
                        ctx.strokeRect(abs_bbox.x, abs_bbox.y, abs_bbox.width, abs_bbox.height);

                        ctx.fillStyle = 'red';
                        ctx.font = '12px Arial';
                        ctx.fillText(`Patch ${card_idx}.${i}`, abs_bbox.x, abs_bbox.y);
                    }
                }

                b.delete(); g.delete(); r.delete();
                component_mask_full.delete();
                component_roi.delete();
                component_mask.delete();
            }
            labels_final.delete();
            stats_final.delete();
            centroids_final.delete();

            card_mask.delete();
            card_region.delete();
            card_roi.delete();
            gray_roi.delete();
            white_mask.delete();
            animal_mask.delete();
            labels.delete();
            stats.delete();
            centroids.delete();
        }
        for (let cnt of final_card_contours) {
            cnt.delete();
        }
        final_card_contours = [];

        patches1 = all_patches[0];
        patches2 = all_patches[1];

        if (patches1.length === 0 || patches2.length === 0) {
            return { card1_bbox, card2_bbox, patches1, patches2, best_pair: null, similarity: 0.0 };
        }

        // Find best matching pair
        max_similarity = -Infinity;
        best_pair = [0, 0];
        for (let i = 0; i < patches1.length; ++i) {
            for (let j = 0; j < patches2.length; ++j) {
                let p1 = patches1[i];
                let p2 = patches2[j];
                let similarity = computeSimilarity(p1.img, p2.img);
                if (similarity > max_similarity) {
                    max_similarity = similarity;
                    best_pair = [i, j];
                }
            }
        }

        return { card1_bbox, card2_bbox, patches1, patches2, best_pair, similarity: max_similarity };

    } catch (error) {
        console.error('cv2_crazy_matching failed:', error);
        return null;
    } finally {
        // Clean up all Mats
        image.delete();
        gray.delete();
        bilateral.delete();
        edges.delete();
        dilated.delete();
        closed.delete();
        contours.delete();
        hierarchy.delete();
        kernel.delete();
        small_kernel.delete();
        large_kernel.delete();

        // Delete Mats within patches if they were created
        for (let p of patches1) {
            p.img.delete();
            p.mask.delete();
        }
        patches1 = [];
        for (let p of patches2) {
            p.img.delete();
            p.mask.delete();
        }
        patches2 = [];
        // // Delete card contours
        // for (let cnt of card_contours) {
        //     cnt.delete();
        // }
        card_contours = [];
    }
}

export async function cv2_detect(input, minArea) {
    let src;

    if (input instanceof HTMLImageElement) {
        src = cv.imread(input);
    } else if (input instanceof ImageData) {
        src = new cv.Mat(input.height, input.width, cv.CV_8UC4);
        src.data.set(input.data);
        cv.cvtColor(src, src, cv.COLOR_RGBA2BGR); // Convert from RGBA to BGR
    } else {
        throw new Error("Please input a valid HTMLImageElement or ImageData.");
    }
    // console.log(`Input image size: ${src.cols}x${src.rows}`);

    let result = await cv2_crazy_matching(src);
    src.delete();
    return result;
}
