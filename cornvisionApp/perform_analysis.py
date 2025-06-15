import cv2 as cv
import numpy as np
from stitching.images import Images
from stitching.feature_detector import FeatureDetector
from stitching.feature_matcher import FeatureMatcher
from stitching.subsetter import Subsetter
from stitching.camera_estimator import CameraEstimator
from stitching.camera_adjuster import CameraAdjuster
from stitching.camera_wave_corrector import WaveCorrector
from stitching.warper import Warper
from stitching.seam_finder import SeamFinder
from stitching.exposure_error_compensator import ExposureErrorCompensator
from stitching.blender import Blender
from run_prediction import find_crops
from run_prediction import count_rows
from generate_heatmap import generate_density_heatmap
import gc

def update_progress_bar(self, value):
    self.progress_bar.progress['value'] += value
    self.root.update_idletasks()  # Process events to update UI

def equalize_brightness(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hsv[..., 1] = cv.equalizeHist(hsv[..., 1])  # Equalize Saturation channel
    return cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

def compute_final_megapix(target_total_mp, images):
    total_pixels = sum(img.shape[0] * img.shape[1] for img in images)
    if total_pixels > target_total_mp * 1000000:
        scale = (target_total_mp * 1000000) / total_pixels
    else:
        scale = total_pixels / 10000000  # Scale to 10 megapixels if total pixels are less than 10 million
    return round(scale, 3)

def boxes_to_lines(row_boxes, min_spacing_px=500):
    # 1.  flatten one level  →  list of individual boxes
    flat = [box for sub in row_boxes for box in sub]
    flat.sort(key=lambda b: float(np.ravel(b)[0] + np.ravel(b)[2]) / 2)
    kept_cx = []
    lines = []
    for b in flat:
        x1, y1, x2, y2 = map(float, np.ravel(b)[:4])
        cx = 0.5 * (x1 + x2)
        if any(abs(cx - k) < min_spacing_px for k in kept_cx):
            continue
        kept_cx.append(cx)
        lines.append(((cx, y1), (cx, y2)))
    return lines

def generate_colormap():
    custom_colormap = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(64):
        custom_colormap[i, 0, 0] = 256 * 3 / 4 # Red channel
        custom_colormap[i, 0, 1] = 2.34 * i # Green channel
    for i in range(64, 256):
        custom_colormap[i, 0, 1] = 150 + 0.54 * (i - 64) # Green channel
    for i in range(64, 128):
        custom_colormap[i, 0, 0] = 192 - (3 * i) / 2  # Red channel
    return custom_colormap

#====== Run YOLO on Images ======#
def analyze_images(self, images:list, real_width_m:float):
    ncount = len(images)
    self.progress_text.config(text="...מחפש תירסים")  # Update text if needed
    self.progress_text.update()

    # Perform analysis on images - find crops and rows
    results, _ = find_crops(images, self) # Run YOLO on images to find crops
    row_boxes, _ = count_rows(images)  # Run YOLO on images to find rows
    row_lines = boxes_to_lines(row_boxes)  # list[((x1,y1),(x2,y2))]
    max_val = 0  # Initialize max_val for colorbar later
    heatmaps = []
    my_colormap = generate_colormap()  # Generate custom colormap
    for img, result in zip(images, results):
        mask, max_val_new = generate_density_heatmap(img, result, real_width_m, row_lines=row_lines, cmap=my_colormap)
        heatmaps.append(mask)
        if max_val_new > max_val:
            max_val = max_val_new
        update_progress_bar(self, (25/ncount))
    del row_lines, row_boxes, results
    gc.collect()

    # ====== Resize Images For Stitching ======#
    images = Images.of(images, medium_megapix=0.4, low_megapix=0.05,
                       final_megapix=compute_final_megapix(10, images))  # Pass file list here to create Images object
    medium_imgs = list(images.resize(Images.Resolution.MEDIUM))
    low_imgs = list(images.resize(Images.Resolution.LOW))
    final_imgs = list(images.resize(Images.Resolution.FINAL))

    resized_heatmaps = [
        cv.resize(hm, (final_img.shape[1], final_img.shape[0]), interpolation=cv.INTER_NEAREST)
        for hm, final_img in zip(heatmaps, final_imgs)
    ]
    del heatmaps
    gc.collect()

    # ====== Define Feature Detector, Find Features ======#
    self.progress_text.configure(text="...מחפש התאמות בין תמונות")  # Update text if needed
    self.progress_text.update()
    finder = FeatureDetector(detector="orb", nfeatures=900)  # Initialize detector
    features = [finder.detect_features(equalize_brightness(img)) for img in
                medium_imgs]  # Find features on medium images
    update_progress_bar(self, 10)
    # ====== Define Feature Matcher, Match Features ======#
    matcher = FeatureMatcher(range_width=1,
                             matcher_type="affine")  # Initialize matcher, range_width = how many images each img is compared to (-1 for all)
    matches = matcher.match_features(features)  # Match features
    matcher.get_confidence_matrix(matches)  # Get conf matrix, use later to split un-matched images?

    # ====== Define Subsetter, Create Subset of Relevant Only ======#
    self.progress_text.configure(text="     ...חותך תמונות    ")  # Update text if needed
    self.progress_text.update()
    subsetter = Subsetter(confidence_threshold=0.4, matches_graph_dot_file=None)
    dot_notation = subsetter.get_matches_graph(images.names, matches)
    print(dot_notation)
    indices = subsetter.get_indices_to_keep(features, matches)
    #medium_imgs = subsetter.subset_list(medium_imgs, indices)
    low_imgs = subsetter.subset_list(low_imgs, indices)
    final_imgs = subsetter.subset_list(final_imgs, indices)
    features = subsetter.subset_list(features, indices)
    del medium_imgs
    gc.collect()

    update_progress_bar(self, 10)
    matches = subsetter.subset_matches(matches, indices)
    images.subset(indices)
    # plot_images(final_imgs) # Good till here - all final images resized

    # ====== Define Camera Calibrations ======#
    camera_estimator = CameraEstimator()
    camera_adjuster = CameraAdjuster(confidence_threshold=0.4)
    wave_corrector = WaveCorrector(wave_correct_kind="auto")
    cameras = camera_estimator.estimate(features, matches)
    cameras = camera_adjuster.adjust(features, matches, cameras)
    cameras = wave_corrector.correct(cameras)

    # ====== Define Warper ======#
    warper = Warper(warper_type="cylindrical")
    warper.set_scale(cameras)  # Set scale - medium focal length

    # Warp low resolution images #
    self.progress_text.configure(text="...מעוות תמונות")  # Update text if needed
    self.progress_text.update()
    low_sizes = images.get_scaled_img_sizes(Images.Resolution.LOW)
    camera_aspect = images.get_ratio(Images.Resolution.MEDIUM,
                                     Images.Resolution.LOW)  # since cameras were obtained on medium imgs
    warped_low_imgs = list(warper.warp_images(low_imgs, cameras, camera_aspect))
    warped_low_masks = list(warper.create_and_warp_masks(low_sizes, cameras, camera_aspect))
    low_corners, low_sizes = warper.warp_rois(low_sizes, cameras, camera_aspect)
    del low_imgs
    gc.collect()
    update_progress_bar(self, 10)
    # Warp final resolution images #
    final_sizes = images.get_scaled_img_sizes(Images.Resolution.FINAL)
    camera_aspect = images.get_ratio(Images.Resolution.MEDIUM, Images.Resolution.FINAL)
    warped_final_imgs = list(warper.warp_images(final_imgs, cameras, camera_aspect))
    warped_final_masks = list(warper.create_and_warp_masks(final_sizes, cameras, camera_aspect))
    warped_heatmaps = list(warper.warp_images(resized_heatmaps, cameras, camera_aspect))
    final_corners, final_sizes = warper.warp_rois(final_sizes, cameras, camera_aspect)
    del final_imgs
    del resized_heatmaps
    gc.collect()

    # ====== Define Seam Finder ======#
    seam_finder = SeamFinder()
    seam_masks = seam_finder.find(warped_low_imgs, low_corners, warped_low_masks)
    seam_masks = [seam_finder.resize(seam_mask, mask) for seam_mask, mask in zip(seam_masks, warped_final_masks)]

    seam_masks_plots = [SeamFinder.draw_seam_mask(img, seam_mask) for img, seam_mask in
                        zip(warped_final_imgs, seam_masks)]

    # ====== Define Exposure Compensator ======#
    compensator = ExposureErrorCompensator(compensator='gain_blocks', nr_feeds=1, block_size=32)
    compensator.feed(low_corners, warped_low_imgs, warped_low_masks)
    compensated_imgs = [compensator.apply(idx, corner, img, mask)
                        for idx, (img, mask, corner)
                        in enumerate(zip(warped_final_imgs, warped_final_masks, final_corners))]

    # ====== Blender for images ======#
    self.progress_text.configure(text="...מחבר תמונות")  # Update text if needed
    self.progress_text.update()
    blender_img = Blender(blender_type='multiband', blend_strength=5)
    blender_img.prepare(final_corners, final_sizes)
    for img, mask, corner in zip(compensated_imgs, seam_masks, final_corners):
        blender_img.feed(img, mask, corner)
    panorama_img, _ = blender_img.blend()
    update_progress_bar(self, 10)


    # ====== Blender for heatmap ======#
    blender_heatmap = Blender(blender_type='multiband', blend_strength=5)
    blender_heatmap.prepare(final_corners, final_sizes)
    for heatmap, mask, corner in zip(warped_heatmaps, seam_masks, final_corners):
        blender_heatmap.feed(heatmap, mask, corner)
        update_progress_bar(self, (15 / ncount))
    panorama_heatmap, _ = blender_heatmap.blend()
    update_progress_bar(self, 10)
    del blender_heatmap
    del blender_img
    return panorama_img, panorama_heatmap, max_val, my_colormap

