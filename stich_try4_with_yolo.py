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
import stitch_try2


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

#====== Resize Images For Stitching ======#
imgs = sorted(str(p) for p in folder.glob("*.JPG")) # Create images file list
images = Images.of(imgs, medium_megapix=0.45, low_megapix=0.1, final_megapix=0.2) # Pass file list here to create Images object
medium_imgs = list(images.resize(Images.Resolution.MEDIUM))
low_imgs = list(images.resize(Images.Resolution.LOW))
final_imgs = list(images.resize(Images.Resolution.FINAL))

#====== Define Feature Detector, Find Features ======#
finder = FeatureDetector(detector="orb", nfeatures=900) # Initialize detector
features = [finder.detect_features(equalize_brightness(img)) for img in medium_imgs] # Find features on medium images

#====== Define Feature Matcher, Match Features ======#
matcher = FeatureMatcher(range_width=3, matcher_type="affine")  # Initialize matcher, range_width = how many images each img is compared to (-1 for all)
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
matches = subsetter.subset_matches(matches, indices)
images.subset(indices)
#plot_images(final_imgs) # Good till here - all final images resized

#====== Define Camera Calibrations ======#
camera_estimator = CameraEstimator()
camera_adjuster = CameraAdjuster(confidence_threshold=0.4)
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

# Warp final resolution images #
final_sizes = images.get_scaled_img_sizes(Images.Resolution.FINAL)
camera_aspect = images.get_ratio(Images.Resolution.MEDIUM, Images.Resolution.FINAL)
warped_final_imgs = list(warper.warp_images(final_imgs, cameras, camera_aspect))
warped_final_masks = list(warper.create_and_warp_masks(final_sizes, cameras, camera_aspect))
final_corners, final_sizes = warper.warp_rois(final_sizes, cameras, camera_aspect)

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

#====== Run Inference on Compensated Images ======#
# Mask each compensated image with its seam mask
masked_imgs = [
    cv.bitwise_and(img, img, mask=mask.get().astype(np.uint8) if isinstance(mask, cv.UMat) else mask.astype(np.uint8))
    for img, mask in zip(compensated_imgs, seam_masks)
]

# Run inference on masked_imgs as needed
masked_imgs = stitch_try2.run_yolo_on_images(masked_imgs)


#====== Define Blender ======#
blender = Blender(blender_type='multiband', blend_strength=5)
blender.prepare(final_corners, final_sizes)
for img, mask, corner in zip(masked_imgs, seam_masks, final_corners):
    blender.feed(img, mask, corner)
panorama, _ = blender.blend()

plot_image(panorama, (20,20))


"""img_arrays = [cv.imread(p) for p in imgs]
equalized_imgs = [equalize_brightness(img) for img in img_arrays]

stitcher = Stitcher(detector="orb", nfeatures=900, confidence_threshold=0.4, range_width=3, matcher_type="affine",
                    matches_graph_dot_file=True, crop=False, warper_type="compressedPlaneA2B1", wave_correct_kind="no",
                    medium_megapix=0.45, low_megapix=0.08, final_megapix=1)

panorama = stitcher.stitch_verbose(equalized_imgs, verbose_dir=verbose_folder)
#warped = stitcher.warp(img_arrays)

plot_image(panorama, (20, 20), grayscale=True)"""
