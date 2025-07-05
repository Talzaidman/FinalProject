import json
import os
import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# SET NUMBER OF PARTICLES
N = 20

# Initial Settings # ADJUST!
s_initial = [95,    # x center
             480,    # y center
              95,    # half width
              325,    # half height
               0,    # velocity x
               0]    # velocity y


def predict_particles(s_prior: np.ndarray) -> np.ndarray:
    """Progress the prior state with time and add noise."""
    s_prior = s_prior.astype(float)
    state_drifted = s_prior.copy()

    # Check if this is our first prediction call
    first_prediction = not getattr(predict_particles, "initialized", False)

    if first_prediction:
        # Initial frame: spread particles around starting position
        startup_noise = np.random.normal(0, 10, s_prior.shape)
        state_drifted = state_drifted + startup_noise
        # Set flag to indicate we've been initialized
        predict_particles.initialized = True
    else:
        # Subsequent frames: apply motion model then add uncertainty
        state_drifted[0, :] += state_drifted[4, :]  # x position update
        state_drifted[1, :] += state_drifted[5, :]  # y position update

        # Add random noise to account for motion uncertainty
        location_noise = np.random.normal(0, 4, (2, s_prior.shape[1]))
        speed_noise = np.random.normal(0, 2, (2, s_prior.shape[1]))

        state_drifted[0:2, :] += location_noise  # Add noise to x,y positions
        state_drifted[4:6, :] += speed_noise  # Add noise to velocities

    state_drifted = state_drifted.astype(int)
    return state_drifted

def compute_normalized_histogram(image: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Compute the normalized histogram using the state parameters."""
    state = np.floor(state)
    state = state.astype(int)
    hist = np.zeros((16, 16, 16))

    # Extract rectangle parameters from state vector
    center_x, center_y = state[0], state[1]
    half_width, half_height = state[2], state[3]

    # Calculate patch boundaries with image bounds checking
    left_edge = max(0, center_x - half_width)
    right_edge = min(image.shape[1], center_x + half_width)
    top_edge = max(0, center_y - half_height)
    bottom_edge = min(image.shape[0], center_y + half_height)

    # Extract the region of interest
    image_patch = image[top_edge:bottom_edge, left_edge:right_edge]

    # Handle empty patch case
    if image_patch.size == 0:
        hist[:] = 1.0 / (16 ** 3)  # Set uniform distribution
    else:
        # Quantize colors from 8-bit to 4-bit and compute histogram
        quantized_colors = (image_patch // 16).reshape(-1, 3)
        color_histogram, _ = np.histogramdd(
            quantized_colors,
            bins=(16, 16, 16),
            range=((0, 16), (0, 16), (0, 16))
        )
        hist = color_histogram

    hist = np.reshape(hist, 16 * 16 * 16)

    # normalize safely
    total_pixels = hist.sum()
    if total_pixels > 0:
        hist = hist / total_pixels
    else:
        hist = np.ones_like(hist) / (16 ** 3)

    return hist

def sample_particles(previous_state: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    """Sample particles from the previous state according to the cdf."""
    s_next = np.zeros(previous_state.shape)

    for particle_idx in range(previous_state.shape[1]):
        random_value = np.random.uniform(0, 1)
        selected_idx = np.searchsorted(cdf, random_value)
        selected_idx = min(selected_idx, previous_state.shape[1] - 1)
        s_next[:, particle_idx] = previous_state[:, selected_idx]

    return s_next

def bhattacharyya_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Calculate Bhattacharyya Distance between two histograms p and q."""
    bc = np.sum(np.sqrt(p * q))
    distance = -np.log(bc)
    return distance

def draw_tracking_rectangles(image: np.ndarray, state: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Draw tracking rectangles on the image and return the annotated image."""
    # Create a copy to avoid modifying the original
    annotated_image = image.copy()
    
    # Calculate weighted average particle position (green rectangle)
    x_avg = np.sum(state[0, :] * W)
    y_avg = np.sum(state[1, :] * W)
    w_avg = 2 * np.sum(state[2, :] * W)  # Convert half-width to full width
    h_avg = 2 * np.sum(state[3, :] * W)  # Convert half-height to full height

    # Convert center coordinates to top-left corner for rectangle
    x_avg_tl = int(x_avg - w_avg / 2)
    y_avg_tl = int(y_avg - h_avg / 2)
    
    # Draw green rectangle for average
    cv2.rectangle(annotated_image, 
                  (x_avg_tl, y_avg_tl), 
                  (x_avg_tl + int(w_avg), y_avg_tl + int(h_avg)), 
                  (0, 255, 0), 2)  # Green color in BGR

    """"
    # Find best particle (highest weight) - red rectangle
    max_particle_idx = np.argmax(W)
    x_max = state[0, max_particle_idx]
    y_max = state[1, max_particle_idx]
    w_max = 2 * state[2, max_particle_idx]
    h_max = 2 * state[3, max_particle_idx]

    # Convert center coordinates to top-left corner
    x_max_tl = int(x_max - w_max / 2)
    y_max_tl = int(y_max - h_max / 2)

    # Draw red rectangle for max
    cv2.rectangle(annotated_image, 
                  (x_max_tl, y_max_tl), 
                  (x_max_tl + w_max, y_max_tl + h_max), 
                  (0, 0, 255), 2)  # Red color in BGR
    """
    return annotated_image

def main_video_tracking(input_video_path: str, output_video_path: str):
    """Main function for video-based particle filter tracking."""
    
    # Open input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Properties of video to be tracked: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' - Which is bwtter?
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Initialize particle filter
    state_at_first_frame = np.matlib.repmat(s_initial, N, 1).T
    S = predict_particles(state_at_first_frame)
    
    # Read first frame
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        cap.release()
        out.release()
        return
    
    # COMPUTE NORMALIZED HISTOGRAM FOR TEMPLATE
    q = compute_normalized_histogram(first_frame, s_initial)
    
    # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
    W = np.zeros(N)
    for i in range(N):
        p = compute_normalized_histogram(first_frame, S[:, i])
        W[i] = np.exp(-9 * bhattacharyya_distance(p, q))
    W = W / np.sum(W)
    C = np.cumsum(W)
    
    # Write first frame with tracking rectangles
    annotated_frame = draw_tracking_rectangles(first_frame, S, W)
    out.write(annotated_frame)
    
    print("Processing video frames...")
    
    # MAIN TRACKING LOOP
    while True:
        ret, current_frame = cap.read()
        if not ret:
            break
            
        S_prev = S
        
        # SAMPLE THE CURRENT PARTICLE FILTERS
        S_next_tag = sample_particles(S_prev, C)
        
        # PREDICT THE NEXT PARTICLE FILTERS
        S = predict_particles(S_next_tag)
        
        # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
        W = np.zeros(N)
        for i in range(N):
            p = compute_normalized_histogram(current_frame, S[:, i])
            W[i] = np.exp(-9 * bhattacharyya_distance(p, q))
        W = W / np.sum(W)
        C = np.cumsum(W)
        
        # Draw tracking rectangles on current frame
        annotated_frame = draw_tracking_rectangles(current_frame, S, W)
        
        # Write frame to output video
        out.write(annotated_frame)
        
    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Tracking complete. Output saved to: {output_video_path}")

if __name__ == "__main__":
    # Example usage
     # Our IDs
    ID1 = '318452364'
    ID2 = '207767021'

    # Path setup
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(BASE_DIR)
    INPUTS_DIR = os.path.join(PROJECT_ROOT, 'Inputs')
    OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'Outputs')
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # File paths
    INPUT_VIDEO = os.path.join(INPUTS_DIR, 'INPUT.avi')

    # Output video files (following project naming requirements)
    STABILIZED_VIDEO = os.path.join(OUTPUTS_DIR, f'stabilize_{ID1}_{ID2}.avi')
    STABILIZED_VIDEO_BG_LOCKED = os.path.join(OUTPUTS_DIR, f'background_locked.avi')
    BINARY_VIDEO = os.path.join(OUTPUTS_DIR, f'binary_{ID1}_{ID2}.avi')
    EXTRACTED_VIDEO = os.path.join(OUTPUTS_DIR, f'extracted_{ID1}_{ID2}.avi')
    MATTED_VIDEO = os.path.join(OUTPUTS_DIR, f'matted.avi')
    #MATTED_VIDEO = os.path.join(OUTPUTS_DIR, f'matted_{ID1}_{ID2}.avi')
    ALPHA_VIDEO = os.path.join(OUTPUTS_DIR, f'alpha_{ID1}_{ID2}.avi')
    OUTPUT_VIDEO = os.path.join(OUTPUTS_DIR, f'OUTPUT_{ID1}_{ID2}.avi')
    TRACKING_VIDEO = os.path.join(OUTPUTS_DIR, f'tracking_test_on_matted.avi')

    main_video_tracking(MATTED_VIDEO, TRACKING_VIDEO)