import cv2
import numpy as np
from scipy import ndimage
from scipy.signal import savgol_filter
import os


def ultra_stabilize_video(input_path, output_path, stabilization_strength=0.95):
    """
    Ultra-stable video stabilization that makes background as still as possible

    Args:
        input_path: Input video path
        output_path: Output video path
        stabilization_strength: 0.0 (no stabilization) to 1.0 (maximum stabilization)
    """
    cap = cv2.VideoCapture(input_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing {total_frames} frames at {fps}fps, resolution: {width}x{height}")

    # Read all frames first for better analysis
    frames = []
    print("Loading all frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

        if len(frames) % 100 == 0:
            print(f"Loaded {len(frames)} frames...")

    cap.release()

    if len(frames) < 2:
        print("Not enough frames to process")
        return

    print("Calculating dense transformations...")

    # Calculate transformations using dense optical flow for better accuracy
    transforms = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    for i in range(1, len(frames)):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

        # Calculate dense optical flow using Farneback method
        flow = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, None, None,
            pyr_scale=0.5,
            levels=5,
            winsize=21,
            iterations=5,
            poly_n=7,
            poly_sigma=1.5,
            flags=0
        )

        # Alternative: Use Farneback dense optical flow
        flow = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, None, None)

        if flow is not None:
            # Sample flow at regular grid points for robust estimation
            h, w = prev_gray.shape
            grid_size = 20
            y_coords = np.arange(grid_size, h - grid_size, grid_size)
            x_coords = np.arange(grid_size, w - grid_size, grid_size)

            prev_pts = []
            curr_pts = []

            # Create grid points and calculate their flow
            for y in y_coords:
                for x in x_coords:
                    prev_pts.append([x, y])

            prev_pts = np.array(prev_pts, dtype=np.float32).reshape(-1, 1, 2)

            # Calculate optical flow for grid points
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, prev_pts, None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            )

            # Filter good points
            good_old = prev_pts[status == 1]
            good_new = curr_pts[status == 1]

            if len(good_old) > 50:
                # Remove outliers using distance threshold
                distances = np.sqrt(np.sum((good_new - good_old) ** 2, axis=1))
                median_dist = np.median(distances)
                inliers = distances < (median_dist * 1.5)  # More conservative

                if np.sum(inliers) > 20:
                    good_old = good_old[inliers]
                    good_new = good_new[inliers]

                    # Robust transformation estimation with RANSAC
                    transform_matrix, inlier_mask = cv2.estimateAffinePartial2D(
                        good_old, good_new,
                        method=cv2.RANSAC,
                        ransacReprojThreshold=2.0,
                        maxIters=3000,
                        confidence=0.995
                    )

                    if transform_matrix is not None and inlier_mask is not None:
                        # Only use if we have enough inliers
                        inlier_ratio = np.sum(inlier_mask) / len(inlier_mask)

                        if inlier_ratio > 0.3:  # At least 30% inliers
                            dx = transform_matrix[0, 2]
                            dy = transform_matrix[1, 2]
                            da = np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])

                            # More aggressive clamping for ultra-stability
                            dx = np.clip(dx, -20, 20)
                            dy = np.clip(dy, -20, 20)
                            da = np.clip(da, -0.05, 0.05)

                            transforms.append([dx, dy, da])
                        else:
                            transforms.append([0, 0, 0])
                    else:
                        transforms.append([0, 0, 0])
                else:
                    transforms.append([0, 0, 0])
            else:
                transforms.append([0, 0, 0])
        else:
            transforms.append([0, 0, 0])

        prev_gray = curr_gray

        if i % 50 == 0:
            print(f"Calculated transforms for {i}/{len(frames) - 1} frames")

    transforms = np.array(transforms)

    print("Ultra-aggressive trajectory smoothing...")

    # Calculate cumulative trajectory
    trajectory = np.cumsum(transforms, axis=0)

    # Multi-stage ultra-smoothing
    smoothed_trajectory = np.copy(trajectory)

    # Stage 1: Very heavy Gaussian smoothing
    sigma_values = [50, 30, 15]  # Multiple passes with decreasing sigma
    for sigma in sigma_values:
        for i in range(3):
            smoothed_trajectory[:, i] = ndimage.gaussian_filter1d(
                smoothed_trajectory[:, i], sigma=sigma, mode='nearest'
            )

    # Stage 2: Savitzky-Golay for ultra-smooth curves
    window_length = min(101, len(smoothed_trajectory) // 3)
    if window_length % 2 == 0:
        window_length += 1
    if window_length >= 5:
        for i in range(3):
            smoothed_trajectory[:, i] = savgol_filter(
                smoothed_trajectory[:, i], window_length, 3
            )

    # Stage 3: Ultra-aggressive low-pass filter
    def ultra_low_pass_filter(data, alpha=0.9):
        filtered = np.copy(data)
        for i in range(1, len(data)):
            filtered[i] = alpha * filtered[i - 1] + (1 - alpha) * data[i]
        return filtered

    for i in range(3):
        smoothed_trajectory[:, i] = ultra_low_pass_filter(smoothed_trajectory[:, i], alpha=0.85)

    # Stage 4: Apply stabilization strength
    difference = smoothed_trajectory - trajectory
    transforms_smooth = transforms + (difference * stabilization_strength)

    print("Creating ultra-stable output...")

    # Smart cropping to eliminate all border artifacts
    crop_percent = 0.15  # Crop 15% for ultra-stability
    crop_w = int(width * crop_percent)
    crop_h = int(height * crop_percent)

    new_width = width - 2 * crop_w
    new_height = height - 2 * crop_h

    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

    # Process first frame
    first_frame = frames[0][crop_h:height - crop_h, crop_w:width - crop_w]
    out.write(first_frame)

    # Apply ultra-stable transforms
    for i in range(len(transforms_smooth)):
        if i + 1 >= len(frames):
            break

        frame = frames[i + 1]
        dx, dy, da = transforms_smooth[i]

        # Create transformation matrix
        transform_matrix = np.array([
            [np.cos(da), -np.sin(da), dx],
            [np.sin(da), np.cos(da), dy]
        ], dtype=np.float32)

        # Apply transformation with border handling
        frame_stabilized = cv2.warpAffine(
            frame, transform_matrix, (width, height),
            borderMode=cv2.BORDER_REFLECT_101
        )

        # Crop to final size
        frame_cropped = frame_stabilized[crop_h:height - crop_h, crop_w:width - crop_w]

        out.write(frame_cropped)

        if i % 50 == 0:
            print(f"Output: {i}/{len(transforms_smooth)} frames")

    out.release()
    cv2.destroyAllWindows()
    print("Ultra-stabilization completed!")


def background_lock_stabilization(input_path, output_path):
    """
    Alternative approach: Lock onto background features specifically
    """
    cap = cv2.VideoCapture(input_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Read reference frame (first frame as background reference)
    ret, reference_frame = cap.read()
    if not ret:
        return

    reference_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)

    # Detect strong background features
    detector = cv2.SIFT_create(
        nfeatures=1500,  # Increase from 1000 for more robust matching
        nOctaveLayers=4,  # Add more octave layers for scale invariance
        contrastThreshold=0.03,  # Lower to detect more features (default 0.04)
        edgeThreshold=15  # Increase to filter out edge-like features
    )
    ref_keypoints, ref_descriptors = detector.detectAndCompute(reference_gray, None)

    print(f"Found {len(ref_keypoints)} reference features in background")

    # FLANN matcher for feature matching
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Create output video with cropping
    crop_percent = 0.1
    crop_w = int(width * crop_percent)
    crop_h = int(height * crop_percent)
    new_width = width - 2 * crop_w
    new_height = height - 2 * crop_h

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

    # Write first frame
    first_cropped = reference_frame[crop_h:height - crop_h, crop_w:width - crop_w]
    out.write(first_cropped)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect features in current frame
        curr_keypoints, curr_descriptors = detector.detectAndCompute(frame_gray, None)

        if curr_descriptors is not None and len(curr_descriptors) > 20:
            # Match features with reference frame
            matches = flann.knnMatch(ref_descriptors, curr_descriptors, k=2)

            # Apply ratio test to get good matches
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.6 * n.distance:
                        good_matches.append(m)

            if len(good_matches) > 20:
                # Extract matched points
                ref_pts = np.array([ref_keypoints[m.queryIdx].pt for m in good_matches])
                curr_pts = np.array([curr_keypoints[m.trainIdx].pt for m in good_matches])

                # Find homography to align current frame to reference
                homography, mask = cv2.findHomography(
                    curr_pts, ref_pts,
                    cv2.RANSAC, 5.0, maxIters=8000, confidence=0.999
                )

                if homography is not None:
                    # Warp current frame to align with reference
                    aligned_frame = cv2.warpPerspective(frame, homography, (width, height))
                else:
                    aligned_frame = frame
            else:
                aligned_frame = frame
        else:
            aligned_frame = frame

        # Crop and write
        cropped_frame = aligned_frame[crop_h:height - crop_h, crop_w:width - crop_w]
        out.write(cropped_frame)

        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Background-locked: {frame_count} frames")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Background lock stabilization completed!")


import cv2
import numpy as np


def stabilize_video(input_path, output_path):
    # Read input video
    cap = cv2.VideoCapture(input_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Pre-define transformation-store array
    transforms = np.zeros((total_frames - 1, 3), np.float32)

    frame_count = 0

    print("Calculating transformations...")
    # Calculate transformations between frames
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Detect feature points in previous frame
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200,
                                           qualityLevel=0.01, minDistance=30,
                                           blockSize=3)

        if prev_pts is not None and len(prev_pts) > 0:
            # Calculate optical flow (movement of points)
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray,
                                                             prev_pts, None)

            # Filter only valid points
            idx = np.where(status == 1)[0]
            if len(idx) > 0:
                prev_pts = prev_pts[idx]
                curr_pts = curr_pts[idx]

                # Find transformation matrix
                m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)

                if m is not None:
                    # Extract translation
                    dx = m[0, 2]
                    dy = m[1, 2]

                    # Extract rotation angle
                    da = np.arctan2(m[1, 0], m[0, 0])

                    transforms[frame_count] = [dx, dy, da]

        prev_gray = curr_gray.copy()
        frame_count += 1

    print("Smoothing trajectory...")
    # Calculate smooth trajectory using moving average
    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = np.copy(trajectory)

    # Smooth the trajectory for the entire video
    smoothing_radius = min(50, len(trajectory) // 4)  # Adaptive smoothing radius

    for i in range(len(trajectory)):
        # Calculate the window bounds
        start_idx = max(0, i - smoothing_radius)
        end_idx = min(len(trajectory), i + smoothing_radius + 1)

        # Apply smoothing
        smoothed_trajectory[i] = np.mean(trajectory[start_idx:end_idx], axis=0)

    # Calculate difference between smooth and original trajectory
    difference = smoothed_trajectory - trajectory
    transforms_smooth = transforms + difference

    print("Applying stabilization...")
    # Reset capture and apply smoothed transforms
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Write the first frame as-is
    ret, frame = cap.read()
    if ret:
        out.write(frame)

    # Apply transforms to remaining frames
    for i in range(len(transforms_smooth)):
        ret, frame = cap.read()
        if not ret:
            break

        # Extract smoothed transformation
        dx, dy, da = transforms_smooth[i]

        # Reconstruct transformation matrix
        m = np.array([[np.cos(da), -np.sin(da), dx],
                      [np.sin(da), np.cos(da), dy]], dtype=np.float32)

        # Apply transformation
        frame_stabilized = cv2.warpAffine(frame, m, (width, height))

        # Fix border artifacts with smaller border to maintain quality
        border_size = 5
        frame_stabilized = cv2.copyMakeBorder(frame_stabilized,
                                              border_size, border_size, border_size, border_size,
                                              cv2.BORDER_REFLECT)

        # Crop back to original size (removes the added border)
        frame_stabilized = frame_stabilized[border_size:height + border_size,
                           border_size:width + border_size]

        out.write(frame_stabilized)

        # Progress indicator
        if i % 30 == 0:
            print(f"Progress: {i}/{len(transforms_smooth)} frames processed")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Stabilization completed!")

wrkdir = os.getcwd()
# Method 3: Background lock method (locks onto first frame background)
background_lock_stabilization(
    fr"{wrkdir}\Inputs\INPUT.avi",
    fr"{wrkdir}\Outputs\background_locked.avi"
)
r"""# Usage
stabilize_video(r'C:\Users\zaita\Downloads\FinalProject\Outputs\background_locked.avi',
                r'C:\Users\zaita\Downloads\FinalProject\Outputs\background_lockedstabilize.avi')"""

