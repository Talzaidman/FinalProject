import cv2
import numpy as np
import os

# Configuration parameters
MAX_CORNERS = 300
QUALITY_LEVEL = 0.05
MIN_DISTANCE = 3
BLOCK_SIZE = 5
RANSAC_THRESHOLD = 1.5
MAX_ITERATIONS = 2000
SMOOTHING_WINDOW = 5
MIN_FEATURES_THRESHOLD = 50  # Minimum features to maintain before refreshing

# Input and output file paths
INPUT_VIDEO = r"C:\Users\zaita\Downloads\FinalProject\Inputs\INPUT.avi"
OUTPUT_VIDEO = r"C:\Users\zaita\Downloads\FinalProject\Outputs\stabilize.avi"

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Parameters for feature detection
feature_params = dict(maxCorners=MAX_CORNERS,
                      qualityLevel=QUALITY_LEVEL,
                      minDistance=MIN_DISTANCE,
                      blockSize=BLOCK_SIZE)


def detect_features(frame):
    """Detect good features to track in the frame."""
    corners = cv2.goodFeaturesToTrack(frame, mask=None, **feature_params)
    return corners


def track_features_from_reference(reference_gray, current_gray, reference_pts):
    """Track features directly from reference frame to current frame."""
    if reference_pts is None or len(reference_pts) == 0:
        return None, None

    # Calculate optical flow from reference frame to current frame
    current_pts, status, error = cv2.calcOpticalFlowPyrLK(reference_gray, current_gray,
                                                          reference_pts, None, **lk_params)

    # Select good points
    if current_pts is not None:
        good_ref = reference_pts[status == 1]
        good_curr = current_pts[status == 1]

        # Additional filtering based on error
        if len(good_ref) > 0:
            error = error[status == 1]
            error_threshold = np.median(error) + 2 * np.std(error)
            mask = error.flatten() < error_threshold
            good_ref = good_ref[mask]
            good_curr = good_curr[mask]

        return good_ref, good_curr

    return None, None


def estimate_transformation(ref_pts, curr_pts):
    """Estimate transformation from reference points to current points using RANSAC."""
    if ref_pts is None or curr_pts is None or len(ref_pts) < 4:
        return np.eye(2, 3, dtype=np.float32)

    try:
        # Use estimateAffinePartial2D for rigid transformation
        transform_matrix, inliers = cv2.estimateAffinePartial2D(
            ref_pts, curr_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=RANSAC_THRESHOLD,
            maxIters=MAX_ITERATIONS
        )

        if transform_matrix is not None:
            return transform_matrix
        else:
            return np.eye(2, 3, dtype=np.float32)

    except cv2.error:
        return np.eye(2, 3, dtype=np.float32)


def smooth_transformations(transforms, window_size=SMOOTHING_WINDOW):
    """Smooth the transformation trajectory using a moving average."""
    if len(transforms) < window_size:
        return transforms

    smoothed_transforms = []

    for i in range(len(transforms)):
        # Define window bounds
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(transforms), i + window_size // 2 + 1)

        # Average transformations in the window
        window_transforms = transforms[start_idx:end_idx]
        avg_transform = np.mean(window_transforms, axis=0)
        smoothed_transforms.append(avg_transform)

    return smoothed_transforms


def calculate_centering_offset(transforms, frame_width, frame_height):
    """Calculate the offset needed to center the average of all transformed frames."""
    # Calculate the average translation across all transforms
    total_dx = 0.0
    total_dy = 0.0

    for transform in transforms:
        total_dx += transform[0, 2]  # x translation
        total_dy += transform[1, 2]  # y translation

    avg_dx = total_dx / len(transforms)
    avg_dy = total_dy / len(transforms)

    print(f"Average translation: dx={avg_dx:.2f}, dy={avg_dy:.2f}")

    # Calculate offset to center the average position
    center_x = frame_width / 2.0
    center_y = frame_height / 2.0

    # The centering offset moves the average position back to the center
    offset_x = -avg_dx
    offset_y = -avg_dy

    print(f"Centering offset: offset_x={offset_x:.2f}, offset_y={offset_y:.2f}")

    return offset_x, offset_y


def apply_centering_to_transforms(transforms, offset_x, offset_y):
    """Apply centering offset to all transformation matrices."""
    centered_transforms = []

    for transform in transforms:
        # Create a copy of the transform
        centered_transform = transform.copy()

        # Add the centering offset to the translation component
        centered_transform[0, 2] += offset_x  # x translation
        centered_transform[1, 2] += offset_y  # y translation

        centered_transforms.append(centered_transform)

    return centered_transforms

# stabilize_video() contains the stabilization process detailed below in a single function
def stabilize_video(input_path, output_path):
    """
    Stabilize a video using feature-based tracking and absolute transformation.
    
    Args:
        input_path (str): Path to input video file
        output_path (str): Path to output stabilized video file
    
    Returns:
        bool: True if successful, False otherwise
    """
    
    print(f"Starting video stabilization...")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found!")
        return False
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{input_path}'")
        return False

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not create output video writer")
        cap.release()
        return False

    # Initialize variables for first pass
    reference_gray = None
    reference_features = None
    absolute_transforms = []  # Direct transformations from reference frame
    frame_count = 0
    features_refreshed_count = 0

    print("First pass: Computing absolute transformations from reference frame...")

    # First pass: Extract absolute transformations from reference frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if reference_gray is None:
            # First frame - set as reference
            reference_gray = curr_gray.copy()
            reference_features = detect_features(reference_gray)
            absolute_transforms.append(np.eye(2, 3, dtype=np.float32))  # Identity for first frame
            print(f"Reference frame set with {len(reference_features) if reference_features is not None else 0} features")
        else:
            # Track features directly from reference frame to current frame
            good_ref, good_curr = track_features_from_reference(reference_gray, curr_gray, reference_features)

            if good_ref is not None and len(good_ref) >= MIN_FEATURES_THRESHOLD:
                # Estimate transformation directly from reference frame to current frame
                transform = estimate_transformation(good_ref, good_curr)
                absolute_transforms.append(transform)
            else:
                # Not enough features - try to refresh feature set
                print(f"Frame {frame_count}: Only {len(good_ref) if good_ref is not None else 0} features tracked, refreshing...")

                # Use previous transform if available, otherwise identity
                if len(absolute_transforms) > 0:
                    absolute_transforms.append(absolute_transforms[-1])  # Use previous transform
                else:
                    absolute_transforms.append(np.eye(2, 3, dtype=np.float32))

                # Refresh reference features for better tracking
                new_features = detect_features(curr_gray)
                if new_features is not None and len(new_features) > len(reference_features) if reference_features is not None else True:
                    # Warp new features back to reference frame coordinate system
                    try:
                        if len(absolute_transforms) > 1:
                            inv_transform = np.linalg.inv(np.vstack([absolute_transforms[-1], [0, 0, 1]]))[:2, :]
                            reference_features = cv2.transform(new_features.reshape(-1, 1, 2),
                                                             np.vstack([inv_transform, [0, 0, 1]])).reshape(-1, 1, 2)
                        else:
                            reference_features = new_features
                        features_refreshed_count += 1
                    except:
                        # If transformation fails, keep old features
                        pass

        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Analyzed {frame_count}/{total_frames} frames")

    print(f"Features were refreshed {features_refreshed_count} times during tracking")
    print("Smoothing absolute transformations...")

    # Smooth the absolute transformations
    smoothed_absolute_transforms = smooth_transformations(absolute_transforms, SMOOTHING_WINDOW)

    # Calculate centering offset to center the average of all transformed frames
    print("Calculating centering offset...")
    offset_x, offset_y = calculate_centering_offset(smoothed_absolute_transforms, width, height)

    # Apply centering offset to all transforms
    print("Applying centering offset to transformations...")
    centered_transforms = apply_centering_to_transforms(smoothed_absolute_transforms, offset_x, offset_y)

    # Reset video capture for second pass
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0

    print("Second pass: Applying centered absolute stabilization...")

    # Second pass: Apply centered absolute stabilization
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count == 0:
            # First frame - apply centering offset
            centering_transform = np.array([[1, 0, offset_x], [0, 1, offset_y]], dtype=np.float32)
            stabilized_frame = cv2.warpAffine(frame, centering_transform, (width, height),
                                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        elif frame_count < len(centered_transforms):
            # Get the centered absolute transformation
            absolute_transform = centered_transforms[frame_count]

            # Apply inverse transformation to warp current frame back to reference frame
            try:
                # Create 3x3 matrix for inversion
                full_transform = np.vstack([absolute_transform, [0, 0, 1]])
                inv_transform = np.linalg.inv(full_transform)[:2, :]

                # Apply transformation to align frame with reference frame
                stabilized_frame = cv2.warpAffine(frame, inv_transform, (width, height),
                                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            except np.linalg.LinAlgError:
                # If inversion fails, use original frame with centering
                centering_transform = np.array([[1, 0, offset_x], [0, 1, offset_y]], dtype=np.float32)
                stabilized_frame = cv2.warpAffine(frame, centering_transform, (width, height),
                                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                print(f"Warning: Transform inversion failed for frame {frame_count}")
        else:
            # Apply centering to any remaining frames
            centering_transform = np.array([[1, 0, offset_x], [0, 1, offset_y]], dtype=np.float32)
            stabilized_frame = cv2.warpAffine(frame, centering_transform, (width, height),
                                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Write stabilized frame
        out.write(stabilized_frame)

        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Stabilized {frame_count}/{total_frames} frames")

    # Cleanup
    cap.release()
    out.release()

    print(f"Centered absolute video stabilization completed!")
    print(f"Stabilized video saved as: {output_path}")
    print(f"All {len(absolute_transforms)} frames aligned to reference frame and centered")
    print("Video is now centered - the average position of all stabilized frames is at the center")
    
    return True

# Try a different stabilization approach
def stabilize_video2(input_path, output_path):
    """
    Stabilize video using the course's suggested approach:
    Frame-to-frame feature matching with RANSAC transformation estimation.
    
    Args:
        input_path (str): Path to input video file
        output_path (str): Path to output stabilized video file
    
    Returns:
        bool: True if successful, False otherwise
    """
    
    print(f"Starting video stabilization (Course Approach)...")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found!")
        return False
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{input_path}'")
        return False

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not create output video writer")
        cap.release()
        return False

    # Feature detection parameters
    feature_params = dict(
        maxCorners=300,
        qualityLevel=0.05,
        minDistance=3,
        blockSize=7
    )

    # Lucas-Kanade optical flow parameters
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01)
    )

    # Variables for stabilization
    prev_gray = None
    cumulative_transform = np.eye(3, dtype=np.float32)  # Cumulative transformation
    transforms = []  # Store all transformations for smoothing
    frame_count = 0
    
    print("Processing video frames...")

    # First pass: Calculate frame-to-frame transformations
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            # 1. Find features in previous frame
            prev_features = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
            
            if prev_features is not None and len(prev_features) > 10:
                # 2. Track features to current frame using optical flow
                curr_features, status, error = cv2.calcOpticalFlowPyrLK(
                    prev_gray, curr_gray, prev_features, None, **lk_params
                )
                
                # 3. Filter good matches
                good_prev = prev_features[status == 1]
                good_curr = curr_features[status == 1]
                
                if len(good_prev) >= 5:  # Need at least 5 points for transformation
                    # 4. Estimate transformation using RANSAC
                    # Using estimateAffinePartial2D for rigid transformation (translation + rotation + uniform scale)
                    transform_matrix, inliers = cv2.estimateAffinePartial2D(
                        good_prev, good_curr,
                        method=cv2.RANSAC,
                        ransacReprojThreshold=1.0,
                        maxIters=1000,
                        confidence=0.995
                    )
                    
                    if transform_matrix is not None:
                        # Convert 2x3 matrix to 3x3 for easier manipulation
                        transform_3x3 = np.vstack([transform_matrix, [0, 0, 1]])
                        transforms.append(transform_3x3)
                        
                        # Debug info
                        dx = transform_matrix[0, 2]
                        dy = transform_matrix[1, 2]
                        da = np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])
                        
                        if frame_count % 30 == 0:  # Print every 30 frames
                            print(f"Frame {frame_count}: Features: {len(good_prev)}, "
                                  f"Transform: dx={dx:.1f}, dy={dy:.1f}, angle={np.degrees(da):.1f}°")
                    else:
                        # No good transformation found, use identity
                        transforms.append(np.eye(3, dtype=np.float32))
                        if frame_count % 30 == 0:
                            print(f"Frame {frame_count}: No good transformation found")
                else:
                    # Not enough features, use identity transformation
                    transforms.append(np.eye(3, dtype=np.float32))
                    if frame_count % 30 == 0:
                        print(f"Frame {frame_count}: Only {len(good_prev)} features - insufficient")
            else:
                # No features detected, use identity
                transforms.append(np.eye(3, dtype=np.float32))
                if frame_count % 30 == 0:
                    print(f"Frame {frame_count}: No features detected")
        else:
            # First frame - identity transformation
            transforms.append(np.eye(3, dtype=np.float32))
            print(f"Frame {frame_count}: First frame (reference)")

        prev_gray = curr_gray.copy()
        frame_count += 1

    print(f"Calculated {len(transforms)} transformations")

    # Smooth transformations using moving average
    print("Smoothing transformations...")
    window_size = min(5, len(transforms))  # Use smaller window if video is short
    smoothed_transforms = []
    
    for i in range(len(transforms)):
        # Define smoothing window
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(transforms), i + window_size // 2 + 1)
        
        # Calculate average transformation in window
        window_transforms = transforms[start_idx:end_idx]
        
        # Average translation and rotation components separately
        avg_dx = np.mean([t[0, 2] for t in window_transforms])
        avg_dy = np.mean([t[1, 2] for t in window_transforms])
        avg_rot = np.mean([np.arctan2(t[1, 0], t[0, 0]) for t in window_transforms])
        avg_scale = np.mean([np.sqrt(t[0, 0]**2 + t[1, 0]**2) for t in window_transforms])
        
        # Reconstruct smoothed transformation matrix
        smoothed_transform = np.array([
            [avg_scale * np.cos(avg_rot), -avg_scale * np.sin(avg_rot), avg_dx],
            [avg_scale * np.sin(avg_rot), avg_scale * np.cos(avg_rot), avg_dy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        smoothed_transforms.append(smoothed_transform)

    # Second pass: Apply smoothed transformations
    print("Applying stabilization...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
    frame_count = 0
    cumulative_transform = np.eye(3, dtype=np.float32)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count == 0:
            # First frame - no transformation needed
            stabilized_frame = frame.copy()
        else:
            # Apply cumulative transformation
            if frame_count < len(smoothed_transforms):
                # Get transformation from previous frame to current
                frame_transform = smoothed_transforms[frame_count]
                
                # Invert the transformation to stabilize
                try:
                    inv_transform = np.linalg.inv(frame_transform)
                    cumulative_transform = cumulative_transform @ inv_transform
                    
                    # Apply cumulative transformation (only the 2x3 part for warpAffine)
                    transform_2x3 = cumulative_transform[:2, :]
                    stabilized_frame = cv2.warpAffine(
                        frame, transform_2x3, (width, height),
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0
                    )
                except np.linalg.LinAlgError:
                    # If matrix inversion fails, use original frame
                    stabilized_frame = frame.copy()
                    print(f"Warning: Matrix inversion failed for frame {frame_count}")
            else:
                stabilized_frame = frame.copy()

        # Write stabilized frame
        out.write(stabilized_frame)
        
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Stabilized {frame_count}/{total_frames} frames")

    # Cleanup
    cap.release()
    out.release()

    print(f"Frame-to-frame video stabilization completed!")
    print(f"Stabilized video saved as: {output_path}")
    print(f"Processed {frame_count} frames with smoothed transformations")
    
    return True

# Original code Tal ran. Delete later
if __name__ == "__main__":
    # Main stabilization process
    print("Starting absolute video stabilization with centering...")

    # Open input video
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{INPUT_VIDEO}'")
        exit(1)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    # Initialize variables for first pass
    reference_gray = None
    reference_features = None
    absolute_transforms = []  # Direct transformations from reference frame
    frame_count = 0
    features_refreshed_count = 0

    print("First pass: Computing absolute transformations from reference frame...")

    # First pass: Extract absolute transformations from reference frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if reference_gray is None:
            # First frame - set as reference
            reference_gray = curr_gray.copy()
            reference_features = detect_features(reference_gray)
            absolute_transforms.append(np.eye(2, 3, dtype=np.float32))  # Identity for first frame
            print(f"Reference frame set with {len(reference_features) if reference_features is not None else 0} features")
        else:
            # Track features directly from reference frame to current frame
            good_ref, good_curr = track_features_from_reference(reference_gray, curr_gray, reference_features)

            if good_ref is not None and len(good_ref) >= MIN_FEATURES_THRESHOLD:
                # Estimate transformation directly from reference frame to current frame
                transform = estimate_transformation(good_ref, good_curr)
                absolute_transforms.append(transform)
            else:
                # Not enough features - try to refresh feature set
                print(
                    f"Frame {frame_count}: Only {len(good_ref) if good_ref is not None else 0} features tracked, refreshing...")

                # Use previous transform if available, otherwise identity
                if len(absolute_transforms) > 0:
                    absolute_transforms.append(absolute_transforms[-1])  # Use previous transform
                else:
                    absolute_transforms.append(np.eye(2, 3, dtype=np.float32))

                # Refresh reference features for better tracking
                new_features = detect_features(curr_gray)
                if new_features is not None and len(new_features) > len(
                        reference_features) if reference_features is not None else True:
                    # Warp new features back to reference frame coordinate system
                    try:
                        if len(absolute_transforms) > 1:
                            inv_transform = np.linalg.inv(np.vstack([absolute_transforms[-1], [0, 0, 1]]))[:2, :]
                            reference_features = cv2.transform(new_features.reshape(-1, 1, 2),
                                                            np.vstack([inv_transform, [0, 0, 1]])).reshape(-1, 1, 2)
                        else:
                            reference_features = new_features
                        features_refreshed_count += 1
                    except:
                        # If transformation fails, keep old features
                        pass

        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Analyzed {frame_count}/{total_frames} frames")

    print(f"Features were refreshed {features_refreshed_count} times during tracking")
    print("Smoothing absolute transformations...")

    # Smooth the absolute transformations
    smoothed_absolute_transforms = smooth_transformations(absolute_transforms, SMOOTHING_WINDOW)

    # Calculate centering offset to center the average of all transformed frames
    print("Calculating centering offset...")
    offset_x, offset_y = calculate_centering_offset(smoothed_absolute_transforms, width, height)

    # Apply centering offset to all transforms
    print("Applying centering offset to transformations...")
    centered_transforms = apply_centering_to_transforms(smoothed_absolute_transforms, offset_x, offset_y)

    # Reset video capture for second pass
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0

    print("Second pass: Applying centered absolute stabilization...")

    # Second pass: Apply centered absolute stabilization
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count == 0:
            # First frame - apply centering offset
            centering_transform = np.array([[1, 0, offset_x], [0, 1, offset_y]], dtype=np.float32)
            stabilized_frame = cv2.warpAffine(frame, centering_transform, (width, height),
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        elif frame_count < len(centered_transforms):
            # Get the centered absolute transformation
            absolute_transform = centered_transforms[frame_count]

            # Apply inverse transformation to warp current frame back to reference frame
            try:
                # Create 3x3 matrix for inversion
                full_transform = np.vstack([absolute_transform, [0, 0, 1]])
                inv_transform = np.linalg.inv(full_transform)[:2, :]

                # Apply transformation to align frame with reference frame
                stabilized_frame = cv2.warpAffine(frame, inv_transform, (width, height),
                                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            except np.linalg.LinAlgError:
                # If inversion fails, use original frame with centering
                centering_transform = np.array([[1, 0, offset_x], [0, 1, offset_y]], dtype=np.float32)
                stabilized_frame = cv2.warpAffine(frame, centering_transform, (width, height),
                                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                print(f"Warning: Transform inversion failed for frame {frame_count}")
        else:
            # Apply centering to any remaining frames
            centering_transform = np.array([[1, 0, offset_x], [0, 1, offset_y]], dtype=np.float32)
            stabilized_frame = cv2.warpAffine(frame, centering_transform, (width, height),
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Write stabilized frame
        out.write(stabilized_frame)

        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Stabilized {frame_count}/{total_frames} frames")

    # Cleanup
    cap.release()
    out.release()

    print(f"Centered absolute video stabilization completed!")
    print(f"Stabilized video saved as: {OUTPUT_VIDEO}")
    print(f"All {len(absolute_transforms)} frames aligned to reference frame and centered")
    print("Video is now centered - the average position of all stabilized frames is at the center")