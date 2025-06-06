import cv2
import numpy as np
from tqdm import tqdm
from scipy import signal
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# FILL IN YOUR ID
ID1 = 318452364
ID2 = 207767021

PYRAMID_FILTER = 1.0 / 256 * np.array([[1, 4, 6, 4, 1],
                                       [4, 16, 24, 16, 4],
                                       [6, 24, 36, 24, 6],
                                       [4, 16, 24, 16, 4],
                                       [1, 4, 6, 4, 1]])
X_DERIVATIVE_FILTER = np.array([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]])
Y_DERIVATIVE_FILTER = X_DERIVATIVE_FILTER.copy().transpose()

WINDOW_SIZE = 5


def get_video_parameters(capture: cv2.VideoCapture) -> dict:
    """Get an OpenCV capture object and extract its parameters.

    Args:
        capture: cv2.VideoCapture object.

    Returns:
        parameters: dict. Video parameters extracted from the video.

    """
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return {"fourcc": fourcc, "fps": fps, "height": height, "width": width,
            "frame_count": frame_count}


def build_pyramid(image: np.ndarray, num_levels: int) -> list[np.ndarray]:
    """Coverts image to a pyramid list of size num_levels.

    First, create a list with the original image in it. Then, iterate over the
    levels. In each level, convolve the PYRAMID_FILTER with the image from the
    previous level. Then, decimate the result using indexing: simply pick
    every second entry of the result.
    Hint: Use signal.convolve2d with boundary='symm' and mode='same'.

    Args:
        image: np.ndarray. Input image.
        num_levels: int. The number of blurring / decimation times.

    Returns:
        pyramid: list. A list of np.ndarray of images.

    Note that the list length should be num_levels + 1 as the in first entry of
    the pyramid is the original image.
    You are not allowed to use cv2 PyrDown here (or any other cv2 method).
    We use a slightly different decimation process from this function.
    """
    pyramid = [image.copy()]
    for level in range(num_levels):
        current_image = pyramid[level]
        blurred = signal.convolve2d(current_image, PYRAMID_FILTER,
                                    boundary='symm', mode='same')
        decimated = blurred[::2, ::2]
        pyramid.append(decimated)
    return pyramid


def lucas_kanade_step(I1: np.ndarray, I2: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Basic Lucas-Kanade step implementation."""
    w = window_size // 2
    Ix = signal.convolve2d(I2, X_DERIVATIVE_FILTER, mode='same')
    Iy = signal.convolve2d(I2, Y_DERIVATIVE_FILTER, mode='same')
    It = I2 - I1

    du = np.zeros_like(I1, dtype=np.float32)
    dv = np.zeros_like(I1, dtype=np.float32)

    for i in range(w, I1.shape[0] - w):
        for j in range(w, I1.shape[1] - w):
            Ix_window = Ix[i - w:i + w + 1, j - w:j + w + 1].flatten()
            Iy_window = Iy[i - w:i + w + 1, j - w:j + w + 1].flatten()
            It_window = It[i - w:i + w + 1, j - w:j + w + 1].flatten()

            A = np.vstack((Ix_window, Iy_window)).T
            b = -It_window

            try:
                flow = np.linalg.lstsq(A, b, rcond=None)[0]
                du[i, j] = flow[0]
                dv[i, j] = flow[1]
            except:
                pass

    return du, dv


def faster_lucas_kanade_step_optimized(I1: np.ndarray,
                                       I2: np.ndarray,
                                       window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Optimized implementation of Lucas-Kanade Step.

    Key optimizations:
    1. Higher threshold for small images (increased from 6000 to 10000)
    2. More aggressive corner filtering
    3. No interpolation for very sparse corners
    4. Higher corner detection threshold
    """
    # Increased threshold for using full LK
    if I1.shape[0] * I1.shape[1] < 10000:
        return lucas_kanade_step(I1, I2, window_size)

    du = np.zeros_like(I1, dtype=np.float32)
    dv = np.zeros_like(I1, dtype=np.float32)

    # Corner detection with higher threshold
    blockSize = 2
    ksize = 3
    k = 0.04
    harris_response = cv2.cornerHarris(I2.astype(np.float32), blockSize, ksize, k)

    # Higher threshold to get fewer but stronger corners
    threshold = 0.1 * harris_response.max()  # Increased from 0.05
    corners = np.argwhere(harris_response > threshold)

    # Limit to fewer corners for faster processing
    max_corners = 50  # Reduced from 100
    if corners.shape[0] > max_corners:
        strengths = harris_response[corners[:, 0], corners[:, 1]]
        idx = np.argsort(strengths)[-max_corners:]
        corners = corners[idx]

    # If we have very few corners, just return sparse flow without interpolation
    if corners.shape[0] < 10:
        return du, dv

    w = window_size // 2
    Ix = signal.convolve2d(I2, X_DERIVATIVE_FILTER, mode='same')
    Iy = signal.convolve2d(I2, Y_DERIVATIVE_FILTER, mode='same')
    It = I2 - I1

    # Process corners and directly update du, dv without interpolation
    for i, j in corners:
        if i < w or i >= I1.shape[0] - w or j < w or j >= I1.shape[1] - w:
            continue

        Ix_window = Ix[i - w:i + w + 1, j - w:j + w + 1].flatten()
        Iy_window = Iy[i - w:i + w + 1, j - w:j + w + 1].flatten()
        It_window = It[i - w:i + w + 1, j - w:j + w + 1].flatten()

        A = np.vstack((Ix_window, Iy_window)).T
        b = -It_window

        try:
            flow = np.linalg.lstsq(A, b, rcond=None)[0]
            # Directly set the flow at corner locations
            du[i, j] = flow[0]
            dv[i, j] = flow[1]
        except:
            continue

    # Simple dilation to spread corner flow to nearby pixels
    kernel = np.ones((5, 5), np.float32) / 25
    du = cv2.filter2D(du, -1, kernel)
    dv = cv2.filter2D(dv, -1, kernel)

    return du, dv


def faster_lucas_kanade_optical_flow_optimized(
        I1: np.ndarray, I2: np.ndarray, window_size: int, max_iter: int,
        num_levels: int) -> tuple[np.ndarray, np.ndarray]:
    """Optimized version with reduced iterations and faster warping."""
    h_factor = int(np.ceil(I1.shape[0] / (2 ** num_levels)))
    w_factor = int(np.ceil(I1.shape[1] / (2 ** num_levels)))
    IMAGE_SIZE = (w_factor * (2 ** num_levels),
                  h_factor * (2 ** num_levels))

    if I1.shape != IMAGE_SIZE:
        I1 = cv2.resize(I1, IMAGE_SIZE)
    if I2.shape != IMAGE_SIZE:
        I2 = cv2.resize(I2, IMAGE_SIZE)

    pyramid_I1 = build_pyramid(I1, num_levels)
    pyramid_I2 = build_pyramid(I2, num_levels)

    u = np.zeros(pyramid_I2[-1].shape, dtype=np.float32)
    v = np.zeros(pyramid_I2[-1].shape, dtype=np.float32)

    for level in range(num_levels, -1, -1):
        I1_level = pyramid_I1[level]
        I2_level = pyramid_I2[level]

        if level != num_levels:
            u = cv2.resize(u, (I1_level.shape[1], I1_level.shape[0])) * 2
            v = cv2.resize(v, (I1_level.shape[1], I1_level.shape[0])) * 2

        # Reduce iterations for speed (especially at higher resolutions)
        current_max_iter = max_iter if level > 1 else 1

        for _ in range(current_max_iter):
            # Create mesh grid for warping
            h, w = I2_level.shape
            y, x = np.mgrid[0:h, 0:w].astype(np.float32)

            # Warp using cv2.remap which is faster than griddata
            map_x = x + u
            map_y = y + v
            I2_warped = cv2.remap(I2_level, map_x, map_y, cv2.INTER_LINEAR)

            du, dv = faster_lucas_kanade_step_optimized(I1_level, I2_warped, window_size)
            u += du
            v += dv

    return u, v


def extract_moving_objects_from_video_ultra_fast(input_video_path, output_video_path_extracted,
                                                 binary_video_path, scale_factor=0.8, threshold=0.001):
    """Ultra-fast version with frame downsampling and optimized processing.

    Args:
        input_video_path: Path to input video
        output_video_path_extracted: Path for extracted moving objects video
        binary_video_path: Path for binary mask video
        scale_factor: Factor to downscale frames for processing (0.5 = half resolution)
        threshold: Motion threshold
    """

    cap = cv2.VideoCapture(input_video_path)
    video_params = get_video_parameters(cap)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out_extracted = cv2.VideoWriter(output_video_path_extracted, fourcc, video_params["fps"],
                                    (video_params["width"], video_params["height"]))

    out_binary = cv2.VideoWriter(binary_video_path, fourcc, video_params["fps"],
                                 (video_params["width"], video_params["height"]), isColor=False)

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        cap.release()
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Downscale for processing
    prev_gray_small = cv2.resize(prev_gray, None, fx=scale_factor, fy=scale_factor)

    out_extracted.write(prev_frame)
    out_binary.write(np.zeros_like(prev_gray))

    frame_count = 0
    total_frames = video_params["frame_count"]

    print(f"Processing {total_frames} frames at {scale_factor * 100:.0f}% resolution...")

    # Progress bar
    pbar = tqdm(total=total_frames - 1)

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        curr_gray_small = cv2.resize(curr_gray, None, fx=scale_factor, fy=scale_factor)

        # Use optimized optical flow on downscaled frames
        u, v = faster_lucas_kanade_optical_flow_optimized(
            prev_gray_small, curr_gray_small,
            window_size=5, max_iter=5, num_levels=5  # Reduced iterations and levels
        )

        # Upscale flow to original resolution
        u_full = cv2.resize(u, (video_params["width"], video_params["height"])) / scale_factor
        v_full = cv2.resize(v, (video_params["width"], video_params["height"])) / scale_factor

        flow_magnitude = np.sqrt(u_full ** 2 + v_full ** 2)
        moving_mask = (flow_magnitude > threshold).astype(np.uint8) * 255
        """
        # Clean up mask with smaller kernel for speed
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        moving_mask_cleaned = cv2.morphologyEx(moving_mask, cv2.MORPH_CLOSE, kernel)"""

        extracted_frame = curr_frame.copy()
        extracted_frame[moving_mask == 0] = [0, 0, 0]

        out_extracted.write(extracted_frame)
        out_binary.write(moving_mask)

        prev_gray_small = curr_gray_small
        frame_count += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out_extracted.release()
    out_binary.release()
    cv2.destroyAllWindows()

    print(f"Processing complete!")
    print(f"Extracted objects saved to: {output_video_path_extracted}")
    print(f"Binary mask video saved to: {binary_video_path}")
