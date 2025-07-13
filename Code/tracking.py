import cv2
import numpy as np
import json
import os

class AdaptivePersonTracker:
    def __init__(self, bbox=None):
        self.current_bbox = bbox
        self.target_histogram = None
        self.convergence_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 1)
        self.bbox_history = []  # Track recent positions for stabilization
        self.stability_weight = 0.4
        self.confidence_threshold = 0.6
        self.max_movement_threshold = 120

    def setup_tracking(self, frame, initial_bbox):
        """Set up tracker with initial bounding box"""
        x, y, w, h = initial_bbox
        self.current_bbox = (x, y, w, h)
        self.bbox_history = [initial_bbox] * 3  # Moderate initialization for balance

        # Extract region of interest with padding to get more context
        padding = 10
        roi_x = max(0, x - padding)
        roi_y = max(0, y - padding)
        roi_w = min(frame.shape[1] - roi_x, w + 2 * padding)
        roi_h = min(frame.shape[0] - roi_y, h + 2 * padding)

        roi_region = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        hsv_roi = cv2.cvtColor(roi_region, cv2.COLOR_BGR2HSV)

        # More conservative mask to focus on skin tones and clothing
        roi_mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

        # Compute target histogram with more bins for better discrimination
        self.target_histogram = cv2.calcHist([hsv_roi], [0, 1], roi_mask, [50, 60], [0, 180, 0, 256])
        cv2.normalize(self.target_histogram, self.target_histogram, 0, 255, cv2.NORM_MINMAX)

    def validate_movement(self, new_bbox):
        """Check if movement is reasonable to prevent large jumps"""
        if not self.bbox_history:
            return new_bbox

        last_bbox = self.bbox_history[-1]

        # Calculate movement distance
        center_old = (last_bbox[0] + last_bbox[2] // 2, last_bbox[1] + last_bbox[3] // 2)
        center_new = (new_bbox[0] + new_bbox[2] // 2, new_bbox[1] + new_bbox[3] // 2)

        movement_distance = np.sqrt((center_new[0] - center_old[0]) ** 2 + (center_new[1] - center_old[1]) ** 2)

        # If movement is too large, limit it
        if movement_distance > self.max_movement_threshold:
            # Scale down the movement
            scale_factor = self.max_movement_threshold / movement_distance

            new_center_x = int(center_old[0] + (center_new[0] - center_old[0]) * scale_factor)
            new_center_y = int(center_old[1] + (center_new[1] - center_old[1]) * scale_factor)

            # Keep original size but adjust position
            corrected_bbox = (
                new_center_x - new_bbox[2] // 2,
                new_center_y - new_bbox[3] // 2,
                new_bbox[2],
                new_bbox[3]
            )
            return corrected_bbox

        return new_bbox

    def stabilize_position(self, new_bbox):
        """Apply heavy temporal smoothing to reduce tracking jitter"""
        if len(self.bbox_history) == 0:
            return new_bbox

        # Validate movement first
        validated_bbox = self.validate_movement(new_bbox)

        # Use weighted average of fewer previous frames for responsiveness
        num_history = min(len(self.bbox_history), 3)

        # Calculate weighted average position with more weight on recent frames
        total_weight = 0
        avg_x = avg_y = avg_w = avg_h = 0

        for i in range(num_history):
            weight = (i + 1) ** 2 / num_history  # Exponential weight favoring recent frames
            bbox = self.bbox_history[-(i + 1)]

            avg_x += bbox[0] * weight
            avg_y += bbox[1] * weight
            avg_w += bbox[2] * weight
            avg_h += bbox[3] * weight
            total_weight += weight

        # Normalize
        avg_x /= total_weight
        avg_y /= total_weight
        avg_w /= total_weight
        avg_h /= total_weight

        # Moderate smoothing - more responsive to changes
        stable_x = int(avg_x * self.stability_weight + validated_bbox[0] * (1 - self.stability_weight))
        stable_y = int(avg_y * self.stability_weight + validated_bbox[1] * (1 - self.stability_weight))
        stable_w = int(avg_w * self.stability_weight + validated_bbox[2] * (1 - self.stability_weight))
        stable_h = int(avg_h * self.stability_weight + validated_bbox[3] * (1 - self.stability_weight))

        stabilized_bbox = (stable_x, stable_y, stable_w, stable_h)

        # Maintain shorter history buffer for more responsiveness
        self.bbox_history.append(stabilized_bbox)
        if len(self.bbox_history) > 5:
            self.bbox_history.pop(0)

        return stabilized_bbox

    def track_next_frame(self, frame):
        """Track person in next frame with improved stability"""
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Use both H and S channels for better tracking
        probability_map = cv2.calcBackProject([hsv_frame], [0, 1], self.target_histogram, [0, 180, 0, 256], 1)

        # Light smoothing to the probability map
        probability_map = cv2.GaussianBlur(probability_map, (3, 3), 0)

        # Use meanshift algorithm for tracking with more responsive criteria
        _, raw_bbox = cv2.meanShift(probability_map, self.current_bbox, self.convergence_criteria)

        x, y, w, h = raw_bbox
        raw_position = (x, y, w, h)

        # Apply heavy position stabilization
        final_position = self.stabilize_position(raw_position)

        # Update current position
        self.current_bbox = final_position

        return final_position


def process_video_tracking(input_video, output_video, tracking_json, start_bbox):
    """
    Main function to track person through video with improved stability

    Parameters:
        input_video: Path to input video file
        output_video: Path to save annotated video
        tracking_json: Path to save tracking coordinates
        start_bbox: Initial bounding box (x, y, w, h). Required parameter.
    """
    video_capture = cv2.VideoCapture(input_video)
    if not video_capture.isOpened():
        raise RuntimeError("Failed to open video file")

    # Extract video properties
    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize video writer
    codec = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_video, codec, frame_rate, (frame_width, frame_height))

    # Get first frame
    success, first_frame = video_capture.read()
    if not success:
        raise RuntimeError("Could not read first frame")

    # Create tracker instance
    person_tracker = AdaptivePersonTracker()
    person_tracker.setup_tracking(first_frame, start_bbox)

    # Annotate first frame
    x, y, w, h = start_bbox
    cv2.rectangle(first_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.putText(first_frame, 'Stable Tracking', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    coordinate_data = []
    coordinate_data.append([int(x), int(y), int(w), int(h)])

    video_writer.write(first_frame)
    frame_number = 1

    # Process remaining frames
    while True:
        success, current_frame = video_capture.read()
        if not success:
            break

        # Track person in current frame
        x, y, w, h = person_tracker.track_next_frame(current_frame)

        # Draw stable tracking rectangle with thicker border
        cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Add center point for better visualization
        center_x = x + w // 2
        center_y = y + h // 2
        cv2.circle(current_frame, (center_x, center_y), 3, (0, 255, 0), -1)

        # Save coordinates
        coordinate_data.append([int(x), int(y), int(w), int(h)])

        video_writer.write(current_frame)
        frame_number += 1

    # Cleanup
    video_capture.release()
    video_writer.release()
    cv2.destroyAllWindows()

    # Export tracking data in required format: frame_number -> [ROW, COL, HEIGHT, WIDTH]
    tracking_results = {}
    for frame_idx, bbox in enumerate(coordinate_data):
        # Convert from [x, y, w, h] to [ROW, COL, HEIGHT, WIDTH]
        # ROW = y, COL = x
        tracking_results[str(frame_idx)] = [bbox[1], bbox[0], bbox[3], bbox[2]]  # [y, x, h, w]

    with open(tracking_json, 'w') as json_file:
        json.dump(tracking_results, json_file, indent=2)

    print(f"Stable tracking completed successfully!")
    print(f"Processed {len(coordinate_data)} frames")
    print(f"Annotated video: {output_video}")
    print(f"Tracking data: {tracking_json}")


# Usage example:
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(BASE_DIR)
    OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'Outputs')
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    MATTED_VIDEO = os.path.join(OUTPUTS_DIR, f'multi_pattern_matted_same_dim.avi')
    OUTPUT_VIDEO = os.path.join(OUTPUTS_DIR, f'stable_histogram_tracking.avi')
    TRACKING_JSON = os.path.join(OUTPUTS_DIR, f'stable_tracking.json')

    # Stable tracking with predefined bounding box
    process_video_tracking(MATTED_VIDEO, OUTPUT_VIDEO, TRACKING_JSON,
                           start_bbox=(90, 210, 330, 770))