import cv2
import numpy as np
from collections import defaultdict, deque
import os

class ContourTracker:
    def __init__(self, max_history=15, motion_threshold=2.0, consistency_threshold=0.8):
        self.contour_history = defaultdict(lambda: deque(maxlen=max_history))
        self.motion_threshold = motion_threshold
        self.consistency_threshold = consistency_threshold
        self.max_history = max_history
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

        # Debug output for significant decisions
        if is_likely_background and frame_count % 30 == 0:
            print(f"Frame {frame_count}: Potential background object detected")
            print(f"  Spatial: {is_spatially_stable}, Motion: {is_motion_stable}")
            print(f"  Edge: {is_edge_consistent}, SSIM: {is_similar_to_background}")
            print(f"  Avg motion: {avg_motion:.2f}, Edge consistency: {edge_consistency:.3f}")
            if ssim_scores:
                print(f"  SSIM with background: {avg_ssim:.3f}")

        return is_likely_background

    def update_previous_frame(self, gray_frame):
        """Update the previous frame for optical flow calculation"""
        self.prev_frame_gray = gray_frame.copy()


def enhanced_background_subtraction_with_postprocessing(input_path, binary_output_path=None,
                                                        extracted_output_path=None,
                                                        num_training_passes=1):
    """
    Enhanced GMM background subtraction with intelligent post-processing
    Uses temporal consistency, optical flow, SSIM, and edge density analysis
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
    print(f"  Training passes: {num_training_passes}")

    # Create GMM background subtractor
    backSub = cv2.createBackgroundSubtractorKNN(
        history=1000,
        dist2Threshold=200.0,
        detectShadows=True
    )

    # Load all frames
    print("\nLoading video frames...")
    cap = cv2.VideoCapture(input_path)
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()
    print(f"Loaded {len(all_frames)} frames")

    # Training passes (same as original)
    for pass_num in range(1, num_training_passes + 1):
        print(f"\n=== PASS {pass_num}: Training Pass ===")
        flip_pass = (pass_num % 2 == 0)

        if flip_pass:
            print(f"Training GMM model (pass {pass_num} - FLIPPED)...")
            frames_to_process = all_frames[::-1]
        else:
            print(f"Training GMM model (pass {pass_num} - NORMAL)...")
            frames_to_process = all_frames

        frame_count = 0
        for frame in frames_to_process:
            frame_count += 1
            _ = backSub.apply(frame)

            if frame_count % 50 == 0:
                progress = (frame_count / total_frames) * 100
                direction = "FLIPPED" if flip_pass else "NORMAL"
                print(f"Pass {pass_num} ({direction}) progress: {progress:.1f}%")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Setup video writers
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
    contour_tracker = ContourTracker(max_history=15, motion_threshold=2.5, consistency_threshold=0.8)

    # Background accumulator for SSIM comparison
    background_accumulator = None
    background_count = 0

    print(f"\n=== ENHANCED INFERENCE PASS ===")
    print("Processing with temporal consistency, optical flow, SSIM, and edge analysis...")

    frame_count = 0
    for frame in all_frames:
        frame_count += 1
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Standard GMM processing
        fgMask = backSub.apply(frame, learningRate=0.001)
        fgMask[fgMask == 127] = 0  # Remove shadows

        # Morphological operations
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel_small, iterations=2)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel_medium, iterations=4)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel_large, iterations=4)

        # Build background reference
        if background_accumulator is None:
            background_accumulator = gray_frame.astype(np.float32)
            background_count = 1
        else:
            # Update background with low learning rate
            alpha = 0.02
            background_accumulator = (1 - alpha) * background_accumulator + alpha * gray_frame.astype(np.float32)
            background_count = min(background_count + 1, 100)

        # Find contours
        contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Enhanced mask for final output
        enhanced_mask = np.zeros_like(fgMask)

        for contour in contours:
            area = cv2.contourArea(contour)

            # Basic area filtering
            if area < 600 or area > width * height * 0.8:
                continue

            # Extract ROI
            x, y, w, h = cv2.boundingRect(contour)
            roi = frame[y:y + h, x:x + w]
            gray_roi = gray_frame[y:y + h, x:x + w]

            # Get background ROI for comparison (if available)
            background_roi = None
            if background_count > 10:
                background_roi = background_accumulator[y:y + h, x:x + w].astype(np.uint8)

            # Check if this contour represents a stationary background object
            is_background = contour_tracker.is_stationary_object(
                contour, roi, gray_roi, gray_frame, frame_count, background_roi
            )

            if not is_background:
                # Add valid foreground contour to enhanced mask
                cv2.fillPoly(enhanced_mask, [contour], 255)

        # Update optical flow reference
        contour_tracker.update_previous_frame(gray_frame)

        # Update final mask
        fgMask = enhanced_mask

        # Create extracted foreground
        extracted_frame = cv2.bitwise_and(frame, frame, mask=fgMask)

        # Save frames
        if binary_out is not None:
            binary_out.write(fgMask)
        if extracted_out is not None:
            extracted_out.write(extracted_frame)

        # Display progress
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Enhanced processing: {progress:.1f}% ({frame_count}/{total_frames})")

        # Display frames
        frame_resized = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        extracted_resized = cv2.resize(extracted_frame, (0, 0), fx=0.5, fy=0.5)
        mask_resized = cv2.resize(fgMask, (0, 0), fx=0.5, fy=0.5)

        cv2.imshow('Original Frame', frame_resized)
        cv2.imshow('Enhanced Binary Mask', mask_resized)
        cv2.imshow('Enhanced Extracted Foreground', extracted_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Processing interrupted by user")
            break

    # Cleanup
    if binary_out is not None:
        binary_out.release()
    if extracted_out is not None:
        extracted_out.release()
    cv2.destroyAllWindows()

    print(f"\n=== ENHANCED PROCESSING COMPLETED ===")
    print(f"Used temporal consistency, optical flow, SSIM, and edge density analysis")
    print(f"Total frames processed: {frame_count}")


def main():
    # Define file paths
    wrkdir = os.getcwd()
    INPUT_VIDEO = fr"{wrkdir}\Outputs\background_locked.avi"
    BINARY_OUTPUT = fr"{wrkdir}\Outputs\enhanced_binary.avi"
    EXTRACTED_OUTPUT = fr"{wrkdir}\Outputs\enhanced_extracted.avi"

    # Run enhanced background subtraction
    enhanced_background_subtraction_with_postprocessing(
        INPUT_VIDEO,
        BINARY_OUTPUT,
        EXTRACTED_OUTPUT,
        num_training_passes=5
    )


if __name__ == "__main__":
    main()