import cv2
import numpy as np
from scipy import ndimage
from scipy.signal import savgol_filter
import os


def background_lock_stabilization(input_path, output_path):
    """
    Alternative approach: Lock onto background features specifically
    Fixed to maintain original frame size with padding and reduced cropping
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

    # Reduced crop percentage to keep person visible (from 0.1 to 0.05)
    crop_percent = 0.05  # Reduced from 10% to 5%
    crop_w = int(width * crop_percent)
    crop_h = int(height * crop_percent)

    # Create output video with ORIGINAL dimensions (not cropped)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))  # Original size!

    # Process first frame - crop and pad back to original size
    first_cropped = reference_frame[crop_h:height - crop_h, crop_w:width - crop_w]
    # Pad back to original dimensions
    first_padded = cv2.copyMakeBorder(
        first_cropped,
        crop_h, crop_h, crop_w, crop_w,  # top, bottom, left, right
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0]  # black padding
    )
    out.write(first_padded)

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

        # Crop to remove unstable borders
        cropped_frame = aligned_frame[crop_h:height - crop_h, crop_w:width - crop_w]

        # Pad back to original dimensions to maintain frame size
        padded_frame = cv2.copyMakeBorder(
            cropped_frame,
            crop_h, crop_h, crop_w, crop_w,  # top, bottom, left, right
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0]  # black padding
        )

        out.write(padded_frame)

        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Background-locked: {frame_count} frames")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Background lock stabilization completed!")


def main():
    print("Video Stabilization Test")
    print("=" * 50)

    # Input and output paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(BASE_DIR)
    INPUTS_DIR = os.path.join(PROJECT_ROOT, 'Inputs')
    OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'Outputs')
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    INPUT_VIDEO = os.path.join(INPUTS_DIR, f'INPUT.avi')
    OUTPUT_VIDEO = os.path.join(OUTPUTS_DIR, f'bg_log_stabilzation_same_dim.avi')


    # Run stabilization
    print("Starting stabilization...")
    background_lock_stabilization(INPUT_VIDEO, OUTPUT_VIDEO)

if __name__ == "__main__":
    main()