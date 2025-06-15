import numpy as np
import cv2 as cv


def generate_density_heatmap(image, boxes, real_width_m=4, row_lines=None, cmap=None):
    global maxval

    h, w = image.shape[:2] # Get image dimensions
    px_per_meter = w / real_width_m # Calculate pixels per meter
    cell_size = int(px_per_meter) # Each cell is 1x1 meter in pixels

    grid_w = w // cell_size # Number of cells in width
    grid_h = h // cell_size # Number of cells in height

    heatmap = np.zeros((grid_h, grid_w), dtype=np.float32) # Initialize heatmap
    rows_in = np.zeros((grid_h, grid_w), dtype=np.float32)  # Initialize rows_in

    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        gx = min(max(cx // cell_size, 0), grid_w - 1)
        gy = min(max(cy // cell_size, 0), grid_h - 1)
        heatmap[gy, gx] += 1

    if row_lines is not None:
        for (x1, y1), (x2, y2) in row_lines:
            # build the axis-aligned bounding box of the line
            xmin, xmax = sorted([x1, x2])
            ymin, ymax = sorted([y1, y2])

            gx0 = int(xmin / cell_size)
            gx1 = int(xmax / cell_size)
            gy0 = int(ymin / cell_size)
            gy1 = int(ymax / cell_size)

            # mark every grid cell the row passes through
            for gy in range(max(gy0, 0), min(gy1, grid_h - 1) + 1):
                for gx in range(max(gx0, 0), min(gx1, grid_w - 1) + 1):
                    here = heatmap[gy, gx]
                    # left / right neighbours (use same value if out of bounds)
                    left = heatmap[gy, gx - 1] if gx - 1 >= 0 else here
                    right = heatmap[gy, gx + 1] if gx + 1 < grid_w else here
                    if here >= 1.5 * left or here >= 1.5 * right and here > 0:
                        rows_in[gy, gx] += 1
        rows_in[rows_in == 0] = 1
        heatmap = heatmap / rows_in # Normalize heatmap by number of rows in each cell
        maxval = np.max(heatmap)  # Get maximum value for color bar later

    # Normalize and colorize
    lo, hi = np.percentile(heatmap, [5, 99])  # Clip extreme values
    heatmap = np.clip(heatmap, min(5, lo), hi)  # Clip values to reduce noise
    heatmap_norm = cv.normalize(heatmap, None, 40, 210, cv.NORM_MINMAX).astype(np.uint8)
    heatmap_colored = cv.applyColorMap(heatmap_norm, cmap)
    heatmap_colored = cv.cvtColor(heatmap_colored, cv.COLOR_BGR2RGB)
    heatmap_resized = cv.resize(heatmap_colored, (w, h), interpolation=cv.INTER_LANCZOS4)
    del heatmap, heatmap_norm, heatmap_colored  # Free memory
    return heatmap_resized, maxval



