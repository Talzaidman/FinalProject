import cv2
import numpy as np

# Configuration parameters
MAX_CORNERS = 200
QUALITY_LEVEL = 0.1
MIN_DISTANCE = 5
BLOCK_SIZE = 5
RANSAC_THRESHOLD = 1.5
MAX_ITERATIONS = 2000
SMOOTHING_WINDOW = 2
MIN_FEATURES_THRESHOLD = 150  # Minimum features to maintain before refreshing

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