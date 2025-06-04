from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
from pathlib import Path
from stitching.images import Images
from stitching.feature_detector import FeatureDetector
from stitching.feature_matcher import FeatureMatcher
from stitching.subsetter import Subsetter
from stitching.camera_estimator import CameraEstimator
from stitching.camera_adjuster import CameraAdjuster
from stitching.camera_wave_corrector import WaveCorrector
from stitching.warper import Warper
from stitching.timelapser import Timelapser # Check if needed
from stitching.cropper import Cropper
from stitching.seam_finder import SeamFinder
from stitching.exposure_error_compensator import ExposureErrorCompensator
from stitching.blender import Blender
from run_prediction import run_yolo_on_images
from generate_heatmap import generate_circle_mask
import gc

def plot_image(img, figsize_in_inches=(5,5), grayscale=False):
    fig, ax = plt.subplots(figsize=figsize_in_inches)
    if grayscale:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ax.imshow(gray, cmap='gray')
    else:
        ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def plot_images(imgs, figsize_in_inches=(5,5)):
    fig, axs = plt.subplots(1, len(imgs), figsize=figsize_in_inches)
    for col, img in enumerate(imgs):
        axs[col].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()

def equalize_brightness(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hsv[..., 1] = cv.equalizeHist(hsv[..., 1])  # Equalize Saturation channel
    return cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

#====== Define Sources ======#
folder = Path(r"C:\Users\Boaz\Desktop\University Courses\D\Project\Viktor Images\Stitch")
verbose_folder = Path(r"C:\Users\Boaz\Desktop\University Courses\D\Project\Viktor Images\Stitch\verbose")
imgs = sorted(str(p) for p in folder.glob("*.JPG")) # Create images file list

#====== Run YOLO on Images ======#
images_for_yolo = [cv.imread(str(p)) for p in imgs]
results =  run_yolo_on_images(images_for_yolo)
heatmaps = []
for img, result in zip(images_for_yolo, results):
    mask = generate_circle_mask(img, result)
    heatmaps.append(mask)
del images_for_yolo
del results
gc.collect()

#====== Resize Images For Stitching ======#
images = Images.of(imgs, medium_megapix=0.3, low_megapix=0.02, final_megapix=0.02) # Pass file list here to create Images object
medium_imgs = list(images.resize(Images.Resolution.MEDIUM))
low_imgs = list(images.resize(Images.Resolution.LOW))
final_imgs = list(images.resize(Images.Resolution.FINAL))

#====== Resize Heatmaps to Final Image Size ======#
resized_heatmaps = [
    cv.resize(hm, (final_img.shape[1], final_img.shape[0]), interpolation=cv.INTER_NEAREST)
    for hm, final_img in zip(heatmaps, final_imgs)
]
del heatmaps
gc.collect()

#====== Define Feature Detector, Find Features ======#
finder = FeatureDetector(detector="orb", nfeatures=900) # Initialize detector
features = [finder.detect_features(equalize_brightness(img)) for img in medium_imgs] # Find features on medium images

#====== Define Feature Matcher, Match Features ======#
matcher = FeatureMatcher(range_width=1, matcher_type="affine")  # Initialize matcher, range_width = how many images each img is compared to (-1 for all)
matches = matcher.match_features(features) # Match features
matcher.get_confidence_matrix(matches) # Get conf matrix, use later to split un-matched images?

#====== Define Subsetter, Create Subset of Relevant Only ======#
subsetter = Subsetter(confidence_threshold=0.4, matches_graph_dot_file=None)
dot_notation = subsetter.get_matches_graph(images.names, matches)
print(dot_notation)
indices = subsetter.get_indices_to_keep(features, matches)
medium_imgs = subsetter.subset_list(medium_imgs, indices)
low_imgs = subsetter.subset_list(low_imgs, indices)
final_imgs = subsetter.subset_list(final_imgs, indices)
features = subsetter.subset_list(features, indices)
del medium_imgs
gc.collect()

matches = subsetter.subset_matches(matches, indices)
images.subset(indices)
#plot_images(final_imgs) # Good till here - all final images resized

#====== Define Camera Calibrations ======#
camera_estimator = CameraEstimator()
camera_adjuster = CameraAdjuster(adjuster="no", confidence_threshold=0.9)
wave_corrector = WaveCorrector()
cameras = camera_estimator.estimate(features, matches)
cameras = camera_adjuster.adjust(features, matches, cameras)
cameras = wave_corrector.correct(cameras)

#====== Define Warper ======#
warper = Warper(warper_type="compressedPlaneA2B1")
warper.set_scale(cameras) # Set scale - medium focal length

# Warp low resolution images #
low_sizes = images.get_scaled_img_sizes(Images.Resolution.LOW)
camera_aspect = images.get_ratio(Images.Resolution.MEDIUM, Images.Resolution.LOW)  # since cameras were obtained on medium imgs
warped_low_imgs = list(warper.warp_images(low_imgs, cameras, camera_aspect))
warped_low_masks = list(warper.create_and_warp_masks(low_sizes, cameras, camera_aspect))
low_corners, low_sizes = warper.warp_rois(low_sizes, cameras, camera_aspect)
del low_imgs
gc.collect()
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

#====== Define Seam Finder ======#
seam_finder = SeamFinder()
seam_masks = seam_finder.find(warped_low_imgs, low_corners, warped_low_masks)
seam_masks = [seam_finder.resize(seam_mask, mask) for seam_mask, mask in zip(seam_masks, warped_final_masks)]

seam_masks_plots = [SeamFinder.draw_seam_mask(img, seam_mask) for img, seam_mask in zip(warped_final_imgs, seam_masks)]


#====== Define Exposure Compensator ======#
compensator = ExposureErrorCompensator(compensator='gain_blocks', nr_feeds=1, block_size=32)
compensator.feed(low_corners, warped_low_imgs, warped_low_masks)
compensated_imgs = [compensator.apply(idx, corner, img, mask)
                    for idx, (img, mask, corner)
                    in enumerate(zip(warped_final_imgs, warped_final_masks, final_corners))]

#====== Blender for images ======#
blender_img = Blender(blender_type='multiband', blend_strength=5)
blender_img.prepare(final_corners, final_sizes)
for img, mask, corner in zip(compensated_imgs, seam_masks, final_corners):
    blender_img.feed(img, mask, corner)
panorama_img, _ = blender_img.blend()

#====== Blender for heatmap ======#
blender_heatmap = Blender(blender_type='multiband', blend_strength=5)
blender_heatmap.prepare(final_corners, final_sizes)
for heatmap, mask, corner in zip(warped_heatmaps, seam_masks, final_corners):
    blender_heatmap.feed(heatmap, mask, corner)
panorama_heatmap, _ = blender_heatmap.blend()
del blender_heatmap
del blender_img
#====== Plot Results ======#
cv.imwrite("C:\\Users\Boaz\Desktop\\University Courses\D\Project\Viktor Images\Stitch\\verbose\mheatmap1.jpg", panorama_heatmap)

if len(panorama_heatmap.shape) == 2 or panorama_heatmap.shape[2] == 1:
    panorama_heatmap = cv.cvtColor(panorama_heatmap, cv.COLOR_GRAY2BGR)
gray_img = cv.cvtColor(panorama_img, cv.COLOR_BGR2GRAY)
gray_img_bgr = cv.cvtColor(gray_img, cv.COLOR_GRAY2BGR)
alpha = 0.4  # transparency for heatmap

overlay = cv.addWeighted(panorama_img, 1, panorama_heatmap, alpha, 0)
plot_image(overlay, (20,20), grayscale=False)
