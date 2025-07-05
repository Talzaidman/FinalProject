import cv2
import numpy as np
from collections import defaultdict, deque
import os

# ===== CONFIGURATION PARAMETERS =====
PARAMS = {
    # Training Parameters
    'num_pattern_pairs': 4,  # Number of regular+mirrored pairs to add before final segment
    'learning_rate': 0.001,  # Constant learning rate for background subtraction
    'background_alpha': 0.05,  # Alpha for background accumulator

    # Background Subtractor Parameters
    'knn_history': 500,  # KNN history length
    'knn_dist_threshold': 200.0,  # KNN distance threshold
    'detect_shadows': True,  # Whether to detect shadows

    # Contour Tracking Parameters
    'max_history': 30,  # Maximum history for contour tracking
    'motion_threshold': 2.5,  # Motion threshold for stationary object detection
    'consistency_threshold': 0.8,  # Consistency threshold

    # Morphological Operations
    'morph_kernel_sizes': {
        'small': (3, 3),
        'medium': (5, 5),
        'large': (7, 7)
    },
    'morph_iterations': {
        'open': 2,
        'close_medium': 4,
        'close_large': 4
    },

    # Area Filtering
    'min_contour_area': 800,  # Minimum contour area to consider
    'max_area_ratio': 0.8,  # Maximum area as ratio of frame size

    # Display Parameters
    'display_scale': 0.5,  # Scale factor for display windows
    'progress_update_interval': 20,  # Frame interval for progress updates
}


# ====================================


class ContourTracker:
    def __init__(self):
        self.contour_history = defaultdict(lambda: deque(maxlen=PARAMS['max_history']))
        self.motion_threshold = PARAMS['motion_threshold']
        self.consistency_threshold = PARAMS['consistency_threshold']
        self.max_history = PARAMS['max_history']
        self.prev_frame_gray = None

    def calculate_ssim_patch(self, patch1, patch2):
        """Calculate SSIM between two image patches"""
        if patch1.shape != patch2.shape:
            return 0.0

        # Convert to float
        patch1 = patch1.astype(np.float64)
        patch2 = patch2.astype(np.float64)

        # Constants for SSIM
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        # Calculate means
        mu1 = np.mean(patch1)
        mu2 = np.mean(patch2)

        # Calculate variances and covariance
        sigma1_sq = np.var(patch1)
        sigma2_sq = np.var(patch2)
        sigma12 = np.mean((patch1 - mu1) * (patch2 - mu2))

        # Calculate SSIM
        numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def calculate_optical_flow_in_contour(self, contour, curr_gray):
        """Calculate average optical flow magnitude within a contour"""
        if self.prev_frame_gray is None:
            return float('inf')  # No previous frame for comparison

        # Create mask for the contour
        mask = np.zeros_like(curr_gray)
        cv2.fillPoly(mask, [contour], 255)

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Extract regions
        curr_roi = curr_gray[y:y + h, x:x + w]
        prev_roi = self.prev_frame_gray[y:y + h, x:x + w]
        mask_roi = mask[y:y + h, x:x + w]

        if curr_roi.size == 0 or prev_roi.size == 0:
            return float('inf')

        # Calculate dense optical flow
        try:
            flow = cv2.calcOpticalFlowPyrLK(
                prev_roi, curr_roi,
                self._get_tracking_points(prev_roi, mask_roi),
                None,
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )

            if flow[0] is not None and flow[1] is not None:
                # Calculate motion vectors
                good_old = flow[0][flow[2].ravel() == 1]
                good_new = flow[1][flow[2].ravel() == 1]

                if len(good_old) > 0:
                    motion_vectors = good_new - good_old
                    motion_magnitudes = np.linalg.norm(motion_vectors, axis=1)
                    return np.mean(motion_magnitudes)

        except Exception:
            pass

        return float('inf')

    def _get_tracking_points(self, roi, mask_roi, max_points=20):
        """Get good tracking points within the ROI"""
        # Detect corners
        corners = cv2.goodFeaturesToTrack(
            roi,
            maxCorners=max_points,
            qualityLevel=0.01,
            minDistance=10,
            mask=mask_roi
        )

        if corners is not None:
            return corners
        else:
            # Fallback: create grid points
            h, w = roi.shape
            points = []
            for i in range(5, w - 5, max(w // 4, 8)):
                for j in range(5, h - 5, max(h // 4, 8)):
                    if mask_roi[j, i] > 0:
                        points.append([[i, j]])

            return np.array(points, dtype=np.float32) if points else None

    def calculate_edge_density(self, roi):
        """Calculate edge density in a region"""
        # Apply Canny edge detection
        edges = cv2.Canny(roi, 50, 150)

        # Calculate edge density
        total_pixels = roi.shape[0] * roi.shape[1]
        edge_pixels = np.sum(edges > 0)

        return edge_pixels / total_pixels if total_pixels > 0 else 0

    def is_stationary_object(self, contour, roi, gray_roi, curr_gray, frame_count, background_roi=None):
        """Determine if a contour represents a stationary object using multiple criteria"""

        # Get contour properties
        x, y, w, h = cv2.boundingRect(contour)
        center = (x + w // 2, y + h // 2)
        area = cv2.contourArea(contour)

        # Calculate motion
        motion_magnitude = self.calculate_optical_flow_in_contour(contour, curr_gray)

        # Calculate edge density
        edge_density = self.calculate_edge_density(gray_roi)

        # Create unique identifier for tracking
        contour_id = f"{center[0] // 25}_{center[1] // 25}_{area // 150}"

        # Store current frame data
        frame_data = {
            'center': center,
            'area': area,
            'motion': motion_magnitude,
            'edge_density': edge_density,
            'frame': frame_count
        }

        # Add SSIM comparison with background if available
        if background_roi is not None and background_roi.shape == gray_roi.shape:
            ssim_score = self.calculate_ssim_patch(gray_roi, background_roi)
            frame_data['ssim_bg'] = ssim_score
        else:
            frame_data['ssim_bg'] = 0.0

        self.contour_history[contour_id].append(frame_data)

        # Need enough history to make decision
        if len(self.contour_history[contour_id]) < min(self.max_history, 8):
            return False

        history = list(self.contour_history[contour_id])

        # 1. Check spatial consistency (stationary position)
        centers = [data['center'] for data in history]
        center_variance = np.var(centers, axis=0)
        spatial_threshold = 30  # pixels

        is_spatially_stable = np.max(center_variance) < spatial_threshold

        # 2. Check motion consistency (low motion over time)
        motions = [data['motion'] for data in history if data['motion'] != float('inf')]

        if len(motions) > 0:
            avg_motion = np.mean(motions)
            motion_stability = np.std(motions)
            is_motion_stable = (avg_motion < self.motion_threshold and
                                motion_stability < self.motion_threshold * 0.5)
        else:
            is_motion_stable = False

        # 3. Check edge consistency (similar structure over time)
        edge_densities = [data['edge_density'] for data in history]
        edge_consistency = 1.0 - (np.std(edge_densities) / (np.mean(edge_densities) + 1e-6))
        is_edge_consistent = edge_consistency > 0.7

        # 4. Check SSIM with background (if available)
        ssim_scores = [data['ssim_bg'] for data in history if data['ssim_bg'] > 0]
        is_similar_to_background = False

        if len(ssim_scores) > 0:
            avg_ssim = np.mean(ssim_scores)
            is_similar_to_background = avg_ssim > 0.6  # High structural similarity to background

        # Decision logic: Object is likely background if multiple criteria are met
        criteria_met = sum([
            is_spatially_stable,
            is_motion_stable,
            is_edge_consistent,
            is_similar_to_background
        ])

        # Require at least 3 out of 4 criteria for conservative classification
        is_likely_background = criteria_met >= 3

        return is_likely_background

    def update_previous_frame(self, gray_frame):
        """Update the previous frame for optical flow calculation"""
        self.prev_frame_gray = gray_frame.copy()


def create_multi_pattern_video(all_frames):
    """
    Create extended video with configurable number of regular+mirrored pairs plus final regular segment
    """
    original_length = len(all_frames)
    num_pairs = PARAMS['num_pattern_pairs']

    # Create the pattern segments
    extended_frames = []
    segment_info = []

    # Add regular+mirrored pairs
    for i in range(num_pairs):
        # Regular segment
        extended_frames.extend(all_frames.copy())
        segment_info.append(f"REGULAR-{i + 1}")

        # Mirrored segment
        extended_frames.extend(all_frames[::-1])
        segment_info.append(f"MIRRORED-{i + 1}")

    # Add final regular segment
    final_segment_start = len(extended_frames)
    extended_frames.extend(all_frames.copy())
    segment_info.append("REGULAR-FINAL")

    print(f"Multi-pattern video creation:")
    print(f"  Original video length: {original_length} frames")
    print(f"  Number of regular+mirrored pairs: {num_pairs}")

    current_frame = 0
    for i, segment_name in enumerate(segment_info):
        start_frame = current_frame + 1
        end_frame = current_frame + original_length
        print(f"  Segment {i + 1} ({segment_name}): frames {start_frame}-{end_frame}")
        current_frame = end_frame

    print(f"  Total extended video length: {len(extended_frames)} frames")
    print(f"  Final segment starts at frame: {final_segment_start + 1}")

    return extended_frames, final_segment_start, segment_info


def enhanced_background_subtraction_with_multi_pattern(input_path, binary_output_path=None,
                                                       extracted_output_path=None):
    """
    Enhanced GMM background subtraction with multi-pattern training
    Pattern: regular + mirrored + regular + mirrored + regular
    Only the final regular segment is saved to output
    """

    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found!")
        return

    # Get video properties
    cap_temp = cv2.VideoCapture(input_path)
    if not cap_temp.isOpened():
        print(f"Error: Cannot open video file '{input_path}'")
        return

    fps = int(cap_temp.get(cv2.CAP_PROP_FPS))
    width = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_temp.release()

    print(f"Video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")

    # Create GMM background subtractor
    backSub = cv2.createBackgroundSubtractorKNN(
        history=PARAMS['knn_history'],
        dist2Threshold=PARAMS['knn_dist_threshold'],
        detectShadows=PARAMS['detect_shadows']
    )

    # Load all frames
    print("\nLoading video frames...")
    cap = cv2.VideoCapture(input_path)
    original_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        original_frames.append(frame)
    cap.release()
    print(f"Loaded {len(original_frames)} frames")

    # Create multi-pattern extended video
    print(f"\nCreating multi-pattern video...")
    extended_frames, final_segment_start, segment_info = create_multi_pattern_video(original_frames)

    # Setup video writers for final segment only
    binary_out = None
    extracted_out = None

    if binary_output_path:
        output_dir = os.path.dirname(binary_output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        binary_out = cv2.VideoWriter(binary_output_path, fourcc, fps, (width, height), isColor=False)

    if extracted_output_path:
        output_dir = os.path.dirname(extracted_output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        extracted_out = cv2.VideoWriter(extracted_output_path, fourcc, fps, (width, height), isColor=True)

    # Initialize enhanced tracking
    contour_tracker = ContourTracker()

    # Background accumulator for SSIM comparison
    background_accumulator = None
    background_count = 0
    num_training_segments = len(segment_info) - 1

    print(f"\n=== MULTI-PATTERN TRAINING AND INFERENCE ===")
    print(f"Processing {len(segment_info)} segments ({num_training_segments} training + 1 final)")
    print(f"Training on first {num_training_segments} segments, saving only the final segment")
    print(f"Using constant learning rate: {PARAMS['learning_rate']}")

    frame_count = 0
    final_segment_frame_count = 0
    original_length = len(original_frames)

    for frame in extended_frames:
        frame_count += 1
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Determine current segment
        segment_idx = frame_count // original_length
        if segment_idx >= len(segment_info):
            segment_idx = len(segment_info) - 1
        current_segment = segment_info[segment_idx]
        is_final_segment = frame_count > final_segment_start

        if is_final_segment:
            final_segment_frame_count += 1

        # Apply background subtraction with constant learning rate
        learning_rate = PARAMS['learning_rate']
        fgMask = backSub.apply(frame, learningRate=learning_rate)
        fgMask[fgMask == 127] = 0  # Remove shadows

        # Morphological operations using parameters
        if is_final_segment and final_segment_frame_count > 20:
            # Enhanced operations for final segment
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, PARAMS['morph_kernel_sizes']['small'])
            kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, PARAMS['morph_kernel_sizes']['medium'])
            kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, PARAMS['morph_kernel_sizes']['large'])

            fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel_small,
                                      iterations=PARAMS['morph_iterations']['open'])
            fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel_medium,
                                      iterations=PARAMS['morph_iterations']['close_medium'])
            fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel_large,
                                      iterations=PARAMS['morph_iterations']['close_large'])
        else:
            # Basic operations for training segments
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))

            fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel_small, iterations=1)
            fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel_medium, iterations=2)

        # Build background reference with configurable alpha
        if background_accumulator is None:
            background_accumulator = gray_frame.astype(np.float32)
            background_count = 1
        else:
            alpha = PARAMS['background_alpha']
            background_accumulator = (1 - alpha) * background_accumulator + alpha * gray_frame.astype(np.float32)
            background_count = min(background_count + 1, 300)

        # Find contours
        contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Enhanced mask for final output
        enhanced_mask = np.zeros_like(fgMask)

        for contour in contours:
            area = cv2.contourArea(contour)

            # Area filtering using parameters
            if area < PARAMS['min_contour_area'] or area > width * height * PARAMS['max_area_ratio']:
                continue

            # Extract ROI
            x, y, w, h = cv2.boundingRect(contour)
            roi = frame[y:y + h, x:x + w]
            gray_roi = gray_frame[y:y + h, x:x + w]

            # Get background ROI for comparison
            background_roi = None
            if background_count > 20:
                background_roi = background_accumulator[y:y + h, x:x + w].astype(np.uint8)

            # Use enhanced tracking only for final segment
            if is_final_segment and final_segment_frame_count > 20:
                # Enhanced tracking for stable final frames
                is_background = contour_tracker.is_stationary_object(
                    contour, roi, gray_roi, gray_frame, frame_count, background_roi
                )

                if not is_background:
                    cv2.fillPoly(enhanced_mask, [contour], 255)
            else:
                # Use basic detection for training segments and early final frames
                cv2.fillPoly(enhanced_mask, [contour], 255)

        # Update optical flow reference
        contour_tracker.update_previous_frame(gray_frame)

        # Update final mask
        fgMask = enhanced_mask

        # Create extracted foreground
        extracted_frame = cv2.bitwise_and(frame, frame, mask=fgMask)

        # Save frames ONLY for final segment
        if is_final_segment:
            if binary_out is not None:
                binary_out.write(fgMask)
            if extracted_out is not None:
                extracted_out.write(extracted_frame)

        # Display progress with segment information
        if frame_count % PARAMS['progress_update_interval'] == 0:
            if is_final_segment:
                progress = (final_segment_frame_count / len(original_frames)) * 100
                print(f"🎯 FINAL SEGMENT: {progress:.1f}% ({final_segment_frame_count}/{len(original_frames)}) - "
                      f"Total frame: {frame_count}/{len(extended_frames)}")
            else:
                segment_progress = ((frame_count % original_length) / original_length) * 100
                print(f"🔄 TRAINING - {current_segment}: {segment_progress:.1f}% - "
                      f"Frame: {frame_count}/{len(extended_frames)}")

        # Display frames with segment information
        scale = PARAMS['display_scale']
        frame_resized = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        extracted_resized = cv2.resize(extracted_frame, (0, 0), fx=scale, fy=scale)
        mask_resized = cv2.resize(fgMask, (0, 0), fx=scale, fy=scale)

        # Add segment and frame information
        status_text = f"FINAL - SAVING" if is_final_segment else "TRAINING"
        color = (0, 255, 0) if is_final_segment else (0, 165, 255)  # Green for final, orange for training

        frame_display = frame_resized.copy()
        cv2.putText(frame_display, f"{current_segment} - {status_text}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame_display, f"Frame {frame_count}/{len(extended_frames)}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        extracted_display = extracted_resized.copy()
        cv2.putText(extracted_display, f"{current_segment} - {status_text}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(extracted_display, f"Frame {frame_count}/{len(extended_frames)}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        mask_display = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
        cv2.putText(mask_display, f"{current_segment} - {status_text}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(mask_display, f"Frame {frame_count}/{len(extended_frames)}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imshow('Multi-Pattern Video Frame', frame_display)
        cv2.imshow('Enhanced Binary Mask', mask_display)
        cv2.imshow('Enhanced Extracted Foreground', extracted_display)

        # Add visual separator when transitioning to final segment
        if frame_count == final_segment_start + 1:
            print("\n" + "=" * 80)
            print("🎬 STARTING FINAL SEGMENT - NOW SAVING TO OUTPUT FILES 🎬")
            print("=" * 80)
            print(f"Background model has been extensively trained on {num_training_segments} segments!")
            print("The final segment should have excellent foreground extraction!")
            print("=" * 80 + "\n")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Processing interrupted by user")
            break

    # Cleanup
    if binary_out is not None:
        binary_out.release()
    if extracted_out is not None:
        extracted_out.release()
    cv2.destroyAllWindows()

    print(f"\n=== MULTI-PATTERN PROCESSING COMPLETED ===")
    print(f"Training segments processed: {final_segment_start} frames")
    print(f"Final segment frames saved: {final_segment_frame_count}")
    print(f"Output files contain only the final segment")
    print(f"Background model trained on {num_training_segments * len(original_frames)} frames before final processing!")


def main():
    # Define file paths
    os.chdir('..')
    wrkdir = os.getcwd()
    INPUT_VIDEO = fr"{wrkdir}\Outputs\background_locked.avi"
    BINARY_OUTPUT = fr"{wrkdir}\Outputs\multi_pattern_binary.avi"
    EXTRACTED_OUTPUT = fr"{wrkdir}\Outputs\multi_pattern_extracted.avi"

    print("=== MULTI-PATTERN BACKGROUND SUBTRACTION ===")
    print(f"Configuration:")
    print(f"  Number of regular+mirrored pairs: {PARAMS['num_pattern_pairs']}")
    print(f"  Learning rate: {PARAMS['learning_rate']}")
    print(f"  Background alpha: {PARAMS['background_alpha']}")
    print(f"  Min contour area: {PARAMS['min_contour_area']}")
    print("=" * 50)

    # Run enhanced background subtraction with multi-pattern training
    enhanced_background_subtraction_with_multi_pattern(
        INPUT_VIDEO,
        BINARY_OUTPUT,
        EXTRACTED_OUTPUT
    )


if __name__ == "__main__":
    main()