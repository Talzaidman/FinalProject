import cv2
import numpy as np
import os

import cv2
import numpy as np
import os

def multipass_median_background_subtraction(input_path, binary_output_path=None, extracted_output_path=None):
    """
    Multi-pass median background subtraction:
    Pass 1: Initial rough background estimation
    Pass 2: Detect obvious foreground, mask it out, recalculate background  
    Pass 3: Final processing with clean background
    """
    
    print(f"Starting multi-pass median background subtraction...")
    print(f"Input: {input_path}")
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found!")
        return False
    
    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{input_path}'")
        return False

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
    
    # ============================================================================
    # PASS 1: Initial rough background estimation
    # ============================================================================
    print("\n=== PASS 1: Initial Background Estimation ===")
    
    # Sample frames strategically  
    sample_frames = 60
    early_frames = np.linspace(0, total_frames//4, 20, dtype=int)
    middle_frames = np.linspace(total_frames//3, 2*total_frames//3, 20, dtype=int)  
    late_frames = np.linspace(3*total_frames//4, total_frames-1, 20, dtype=int)
    frame_indices = np.concatenate([early_frames, middle_frames, late_frames])
    
    sampled_frames = []
    
    print(f"Sampling {len(frame_indices)} frames for initial background...")
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            sampled_frames.append(frame.astype(np.float32))
        
        if (i + 1) % 20 == 0 or (i + 1) == len(frame_indices):
            print(f"  Loaded {i + 1}/{len(frame_indices)} frames")
    
    # Calculate initial background
    print("Calculating initial median background...")
    frames_array = np.stack(sampled_frames, axis=0)
    initial_background = np.median(frames_array, axis=0).astype(np.uint8)
    
    del sampled_frames, frames_array
    print("✅ Initial background calculated")
    
    # ============================================================================
    # PASS 2: Detect foreground areas and create clean background
    # ============================================================================
    print("\n=== PASS 2: Detecting Foreground Areas ===")
    
    # Go through all frames and detect obvious foreground
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Collect foreground masks from multiple frames
    foreground_masks = []
    frame_samples_for_clean_bg = []
    
    # Sample every Nth frame for foreground detection
    detection_step = max(1, total_frames // 100)  # Sample ~100 frames for detection
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only process every Nth frame for efficiency
        if frame_count % detection_step == 0:
            # Detect foreground using initial background
            diff = cv2.absdiff(frame, initial_background)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            
            # Use lower threshold to detect obvious foreground
            _, fg_mask = cv2.threshold(diff_gray, 40, 255, cv2.THRESH_BINARY)
            
            # Clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            foreground_masks.append(fg_mask)
            frame_samples_for_clean_bg.append(frame.astype(np.float32))
            
            if len(foreground_masks) % 20 == 0:
                print(f"  Processed {len(foreground_masks)} frames for foreground detection")
        
        frame_count += 1
    
    print(f"Collected {len(foreground_masks)} frames for clean background calculation")
    
    # Create a cumulative foreground mask (areas where person appeared)
    print("Creating cumulative foreground mask...")
    cumulative_fg_mask = np.zeros((height, width), dtype=np.float32)
    
    for mask in foreground_masks:
        cumulative_fg_mask += (mask / 255.0)
    
    # Normalize to get probability of being foreground
    cumulative_fg_mask /= len(foreground_masks)
    
    # Create clean background by excluding foreground-prone areas
    print("Calculating clean background...")
    clean_background = np.zeros((height, width, 3), dtype=np.float32)
    background_counts = np.zeros((height, width), dtype=np.float32)
    
    # For each pixel, only use frames where it's likely background
    for i, (frame, fg_mask) in enumerate(zip(frame_samples_for_clean_bg, foreground_masks)):
        # Create weight mask: lower weight where foreground was detected
        weight_mask = 1.0 - (fg_mask.astype(np.float32) / 255.0)
        
        # Add weighted contribution to background
        for c in range(3):  # For each color channel
            clean_background[:, :, c] += frame[:, :, c] * weight_mask
        background_counts += weight_mask
    
    # Avoid division by zero
    background_counts[background_counts == 0] = 1
    
    # Calculate final clean background
    for c in range(3):
        clean_background[:, :, c] /= background_counts
    
    clean_background = clean_background.astype(np.uint8)
    
    del frame_samples_for_clean_bg, foreground_masks
    print("✅ Clean background calculated")
    
    # ============================================================================
    # PASS 3: Final processing with clean background
    # ============================================================================
    print("\n=== PASS 3: Final Processing ===")
    
    # Setup output videos
    binary_out = None
    extracted_out = None
    
    if binary_output_path:
        os.makedirs(os.path.dirname(binary_output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        binary_out = cv2.VideoWriter(binary_output_path, fourcc, fps, (width, height), isColor=False)

    if extracted_output_path:
        os.makedirs(os.path.dirname(extracted_output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        extracted_out = cv2.VideoWriter(extracted_output_path, fourcc, fps, (width, height), isColor=True)
    
    # Process all frames with clean background
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate difference from clean background
        diff = cv2.absdiff(frame, clean_background)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Use moderate threshold now that background is cleaner
        threshold_value = 60
        _, binary_mask = cv2.threshold(diff_gray, threshold_value, 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # Clean up noise
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        # Connect person parts
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
        
        # Area filtering
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_mask = np.zeros_like(binary_mask)
        
        min_area = 300
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                cv2.fillPoly(clean_mask, [contour], 255)
        
        binary_mask = clean_mask
        
        # Create extracted foreground
        extracted_frame = cv2.bitwise_and(frame, frame, mask=binary_mask)
        
        # Debug output for first few frames
        if frame_count < 5:
            white_pixels = np.sum(binary_mask == 255)
            total_pixels = binary_mask.size
            white_percentage = (white_pixels / total_pixels) * 100
            print(f"Frame {frame_count}: {white_percentage:.1f}% white pixels")
        
        # Save frames
        if binary_out is not None:
            binary_out.write(binary_mask)
        if extracted_out is not None:
            extracted_out.write(extracted_frame)
        
        frame_count += 1
        if frame_count % 50 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")
    
    # Cleanup
    cap.release()
    if binary_out is not None:
        binary_out.release()
    if extracted_out is not None:
        extracted_out.release()
    
    print("✅ Multi-pass median background subtraction completed!")
    return True

def smart_median_background_subtraction(input_path, binary_output_path=None, extracted_output_path=None):
    """
    Smart median background subtraction that handles your specific issues:
    1. Samples strategically to avoid person contamination
    2. Uses higher threshold to reduce noise
    3. Better morphological operations
    """
    
    print(f"Starting smart median background subtraction...")
    print(f"Input: {input_path}")
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found!")
        return False
    
    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{input_path}'")
        return False

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
    
    # Smart sampling strategy: avoid frames where person is likely in center
    # Use more samples but distribute them strategically
    sample_frames = 60  # More samples for better background estimation
    
    # Sample from different parts of the video (20 from each section)
    early_frames = np.linspace(0, total_frames//4, 20, dtype=int)
    middle_frames = np.linspace(total_frames//3, 2*total_frames//3, 20, dtype=int)  
    late_frames = np.linspace(3*total_frames//4, total_frames-1, 20, dtype=int)
    
    frame_indices = np.concatenate([early_frames, middle_frames, late_frames])
    
    sampled_frames = []
    
    print(f"Sampling {len(frame_indices)} frames strategically...")
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            sampled_frames.append(frame.astype(np.float32))
        
        if (i + 1) % 5 == 0 or (i + 1) == len(frame_indices):
            print(f"  Loaded {i + 1}/{len(frame_indices)} frames")
    
    if len(sampled_frames) == 0:
        print("Error: No frames could be loaded!")
        cap.release()
        return False
    
    # Calculate median background
    print("Calculating median background...")
    frames_array = np.stack(sampled_frames, axis=0)
    median_background = np.median(frames_array, axis=0).astype(np.uint8)
    print("✅ Background calculated!")
    
    del sampled_frames, frames_array
    
    # Setup output videos
    binary_out = None
    extracted_out = None
    
    if binary_output_path:
        os.makedirs(os.path.dirname(binary_output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        binary_out = cv2.VideoWriter(binary_output_path, fourcc, fps, (width, height), isColor=False)

    if extracted_output_path:
        os.makedirs(os.path.dirname(extracted_output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        extracted_out = cv2.VideoWriter(extracted_output_path, fourcc, fps, (width, height), isColor=True)
    
    # Process all frames
    print("Processing frames with smart parameters...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate absolute difference from background
        diff = cv2.absdiff(frame, median_background)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Use higher threshold to reduce false positives
        threshold_value = 80  # Much higher than before
        _, binary_mask = cv2.threshold(diff_gray, threshold_value, 255, cv2.THRESH_BINARY)
        
        # Aggressive morphological operations
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        
        # Remove noise aggressively
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
        
        # Connect person parts
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_large, iterations=3)
        
        # Final cleanup
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # Smart area filtering: keep largest component + nearby components
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (likely the person)
            largest_contour = max(contours, key=cv2.contourArea)
            largest_area = cv2.contourArea(largest_contour)
            
            clean_mask = np.zeros_like(binary_mask)
            
            # Keep large contours and contours close to the largest one
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Keep if large enough OR close to largest contour
                if area > 500 or (area > 100 and area > largest_area * 0.1):
                    cv2.fillPoly(clean_mask, [contour], 255)
            
            binary_mask = clean_mask
        
        # Create extracted foreground
        extracted_frame = cv2.bitwise_and(frame, frame, mask=binary_mask)
        
        # Debug output for first few frames
        if frame_count < 5:
            white_pixels = np.sum(binary_mask == 255)
            total_pixels = binary_mask.size
            white_percentage = (white_pixels / total_pixels) * 100
            print(f"Frame {frame_count}: {white_percentage:.1f}% white pixels")
        
        # Save frames
        if binary_out is not None:
            binary_out.write(binary_mask)
        if extracted_out is not None:
            extracted_out.write(extracted_frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")
    
    # Cleanup
    cap.release()
    if binary_out is not None:
        binary_out.release()
    if extracted_out is not None:
        extracted_out.release()
    
    print("✅ Smart median background subtraction completed!")
    return True

def running_average_background_subtraction(input_path, binary_output_path=None, extracted_output_path=None, learning_rate=0.05):
    """
    Background subtraction using running average background model.
    Updates background continuously: new_bg = (1-alpha) * old_bg + alpha * current_frame
    
    Args:
        input_path (str): Path to input video file
        binary_output_path (str): Path to save binary mask video
        extracted_output_path (str): Path to save extracted foreground video
        learning_rate (float): How fast to adapt background (0.01-0.1, lower = slower adaptation)
    
    Returns:
        bool: True if successful, False otherwise
    """
    
    print(f"Starting running average background subtraction...")
    print(f"Input: {input_path}")
    print(f"Learning rate: {learning_rate}")
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found!")
        return False
    
    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{input_path}'")
        return False

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
    
    # Setup output videos
    binary_out = None
    extracted_out = None
    
    if binary_output_path:
        os.makedirs(os.path.dirname(binary_output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        binary_out = cv2.VideoWriter(binary_output_path, fourcc, fps, (width, height), isColor=False)
        if not binary_out.isOpened():
            print(f"Error: Could not create binary output video")
            cap.release()
            return False
        print(f"Binary output will be saved to: {binary_output_path}")

    if extracted_output_path:
        os.makedirs(os.path.dirname(extracted_output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        extracted_out = cv2.VideoWriter(extracted_output_path, fourcc, fps, (width, height), isColor=True)
        if not extracted_out.isOpened():
            print(f"Error: Could not create extracted output video")
            cap.release()
            if binary_out:
                binary_out.release()
            return False
        print(f"Extracted output will be saved to: {extracted_output_path}")
    
    # Initialize background model
    background_model = None
    frame_count = 0
    
    print("Processing frames with adaptive background...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to float for calculations
        frame_float = frame.astype(np.float32)
        
        if background_model is None:
            # Initialize background with first frame
            background_model = frame_float.copy()
            print("Background model initialized with first frame")
            
            # For first frame, create empty mask (no foreground detected yet)
            binary_mask = np.zeros((height, width), dtype=np.uint8)
            extracted_frame = np.zeros_like(frame)
            
        else:
            # Update background model using running average
            # background = (1 - alpha) * old_background + alpha * current_frame
            background_model = (1 - learning_rate) * background_model + learning_rate * frame_float
            
            # Convert background to uint8 for difference calculation
            background_uint8 = background_model.astype(np.uint8)
            
            # Calculate absolute difference
            diff = cv2.absdiff(frame, background_uint8)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to create binary mask
            threshold_value = 50  # Start with moderate threshold
            _, binary_mask = cv2.threshold(diff_gray, threshold_value, 255, cv2.THRESH_BINARY)
            
            # Morphological operations to clean up the mask
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            
            # Remove small noise
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
            
            # Connect nearby components
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
            
            # Filter by area
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            clean_mask = np.zeros_like(binary_mask)
            
            min_area = 300  # Minimum area for person components
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    cv2.fillPoly(clean_mask, [contour], 255)
            
            binary_mask = clean_mask
            
            # Create extracted foreground
            extracted_frame = cv2.bitwise_and(frame, frame, mask=binary_mask)
        
        # Debug output for first few frames
        if frame_count < 5:
            white_pixels = np.sum(binary_mask == 255)
            total_pixels = binary_mask.size
            white_percentage = (white_pixels / total_pixels) * 100
            print(f"Frame {frame_count}: {white_percentage:.1f}% white pixels")
        
        # Save frames
        if binary_out is not None:
            binary_out.write(binary_mask)
        if extracted_out is not None:
            extracted_out.write(extracted_frame)
        
        frame_count += 1
        
        # Progress update
        if frame_count % 50 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")
    
    # Cleanup
    cap.release()
    if binary_out is not None:
        binary_out.release()
    if extracted_out is not None:
        extracted_out.release()
    
    print("✅ Running average background subtraction completed!")
    
    # Verify outputs
    if binary_output_path and os.path.exists(binary_output_path):
        print(f"✅ Binary video saved: {binary_output_path}")
    if extracted_output_path and os.path.exists(extracted_output_path):
        print(f"✅ Extracted video saved: {extracted_output_path}")
    
    return True

def fixed_median_background_subtraction(input_path, binary_output_path=None, extracted_output_path=None):
    """
    Simple and reliable median background subtraction.
    Focus on getting the basics right without over-engineering.
    """
    
    print(f"Starting fixed median background subtraction...")
    print(f"Input: {input_path}")
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found!")
        return False
    
    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{input_path}'")
        return False

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
    
    # Step 1: Sample frames - use fewer frames to avoid person bias
    sample_frames = 30  # Reduced - less chance of person being in each pixel
    frame_indices = np.linspace(0, total_frames-1, min(sample_frames, total_frames), dtype=int)
    sampled_frames = []
    
    print(f"Sampling {len(frame_indices)} frames for background estimation...")
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            sampled_frames.append(frame.astype(np.float32))
        
        if (i + 1) % 10 == 0 or (i + 1) == len(frame_indices):
            print(f"  Loaded {i + 1}/{len(frame_indices)} frames")
    
    if len(sampled_frames) == 0:
        print("Error: No frames could be loaded!")
        cap.release()
        return False
    
    # Step 2: Calculate median background (simple approach)
    print("Calculating median background...")
    frames_array = np.stack(sampled_frames, axis=0)
    median_background = np.median(frames_array, axis=0).astype(np.uint8)
    print("✅ Background calculated!")
    
    # Debug: Let's see what the background looks like
    print(f"Background stats - Min: {median_background.min()}, Max: {median_background.max()}, Mean: {median_background.mean():.1f}")
    
    del sampled_frames, frames_array
    
    # Setup output videos
    binary_out = None
    extracted_out = None
    
    if binary_output_path:
        os.makedirs(os.path.dirname(binary_output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        binary_out = cv2.VideoWriter(binary_output_path, fourcc, fps, (width, height), isColor=False)

    if extracted_output_path:
        os.makedirs(os.path.dirname(extracted_output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        extracted_out = cv2.VideoWriter(extracted_output_path, fourcc, fps, (width, height), isColor=True)
    
    # Step 3: Process all frames
    print("Processing frames...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate absolute difference from background
        diff = cv2.absdiff(frame, median_background)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Simple fixed threshold - start conservative
        threshold_value = 70
        _, binary_mask = cv2.threshold(diff_gray, threshold_value, 255, cv2.THRESH_BINARY)
        
        # Light morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Remove small noise
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Connect nearby components
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_connect, iterations=1)
        
        # Filter by area - be more permissive
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_mask = np.zeros_like(binary_mask)
        
        min_area = 200  # Lower threshold
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                cv2.fillPoly(clean_mask, [contour], 255)
        
        binary_mask = clean_mask
        
        # Create extracted foreground
        extracted_frame = cv2.bitwise_and(frame, frame, mask=binary_mask)
        
        # Debug: Print some stats for first few frames
        if frame_count < 3:
            white_pixels = np.sum(binary_mask == 255)
            total_pixels = binary_mask.size
            white_percentage = (white_pixels / total_pixels) * 100
            print(f"Frame {frame_count}: {white_percentage:.1f}% white pixels")
        
        # Save frames
        if binary_out is not None:
            binary_out.write(binary_mask)
        if extracted_out is not None:
            extracted_out.write(extracted_frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")
    
    # Cleanup
    cap.release()
    if binary_out is not None:
        binary_out.release()
    if extracted_out is not None:
        extracted_out.release()
    
    print("✅ Fixed median background subtraction completed!")
    return True

def median_background_subtraction(input_path, binary_output_path=None, extracted_output_path=None, sample_frames=50):
    """
    Memory-efficient background subtraction using temporal median filter.
    Samples frames evenly throughout the video to estimate background.
    
    Args:
        input_path (str): Path to input video file
        binary_output_path (str): Path to save binary mask video
        extracted_output_path (str): Path to save extracted foreground video
        sample_frames (int): Number of frames to sample for background estimation
    
    Returns:
        bool: True if successful, False otherwise
    """
    
    print(f"Starting median-based background subtraction...")
    print(f"Input: {input_path}")
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found!")
        return False
    
    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{input_path}'")
        return False

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
    print(f"Using {sample_frames} sampled frames for background estimation")
    
    # Step 1: Sample frames evenly throughout the video
    frame_indices = np.linspace(0, total_frames-1, min(sample_frames, total_frames), dtype=int)
    sampled_frames = []
    
    print("Loading sampled frames...")
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            sampled_frames.append(frame.astype(np.float32))
        
        if (i + 1) % 10 == 0 or (i + 1) == len(frame_indices):
            print(f"  Loaded {i + 1}/{len(frame_indices)} sampled frames")
    
    if len(sampled_frames) == 0:
        print("Error: No frames could be loaded!")
        cap.release()
        return False
    
    print(f"Successfully loaded {len(sampled_frames)} frames for background estimation")
    
    # Step 2: Calculate median background
    print("Calculating median background...")
    try:
        frames_array = np.stack(sampled_frames, axis=0)
        median_background = np.median(frames_array, axis=0).astype(np.uint8)
        print("✅ Median background calculated successfully!")
        
        # Clear memory
        del sampled_frames, frames_array
        
    except MemoryError:
        print("❌ Still not enough memory. Try reducing sample_frames parameter.")
        cap.release()
        return False
    
    # Step 3: Setup output videos
    binary_out = None
    extracted_out = None
    
    if binary_output_path:
        output_dir = os.path.dirname(binary_output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        binary_out = cv2.VideoWriter(binary_output_path, fourcc, fps, (width, height), isColor=False)
        
        if not binary_out.isOpened():
            print(f"Error: Could not create binary output video")
            cap.release()
            return False
        print(f"Binary output will be saved to: {binary_output_path}")

    if extracted_output_path:
        output_dir = os.path.dirname(extracted_output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        extracted_out = cv2.VideoWriter(extracted_output_path, fourcc, fps, (width, height), isColor=True)
        
        if not extracted_out.isOpened():
            print(f"Error: Could not create extracted output video")
            cap.release()
            if binary_out:
                binary_out.release()
            return False
        print(f"Extracted output will be saved to: {extracted_output_path}")
    
    # Step 4: Process all frames for foreground extraction
    print("Processing frames for foreground extraction...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate absolute difference from background
        diff = cv2.absdiff(frame, median_background)
        
        # Convert to grayscale for thresholding
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to create binary mask
        threshold_value = 50  # Adjust if needed
        _, binary_mask = cv2.threshold(diff_gray, threshold_value, 255, cv2.THRESH_BINARY)
        
        # Clean up the mask with morphological operations
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Remove noise
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        # Fill holes
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
        
        # Remove small components
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_mask = np.zeros_like(binary_mask)
        
        min_area = 800  # Minimum area for person components
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                cv2.fillPoly(clean_mask, [contour], 255)
        
        binary_mask = clean_mask
        
        # Create extracted foreground
        extracted_frame = cv2.bitwise_and(frame, frame, mask=binary_mask)
        
        # Save frames
        if binary_out is not None:
            binary_out.write(binary_mask)
        
        if extracted_out is not None:
            extracted_out.write(extracted_frame)
        
        frame_count += 1
        
        # Progress update
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")
    
    # Cleanup
    cap.release()
    if binary_out is not None:
        binary_out.release()
    if extracted_out is not None:
        extracted_out.release()
    
    print("✅ Median background subtraction completed successfully!")
    
    # Verify outputs
    if binary_output_path and os.path.exists(binary_output_path):
        print(f"✅ Binary video saved: {binary_output_path}")
    if extracted_output_path and os.path.exists(extracted_output_path):
        print(f"✅ Extracted video saved: {extracted_output_path}")
    
    return True

def gmm_background_subtraction_multi_pass(input_path, binary_output_path=None,
                                          extracted_output_path=None, num_training_passes=5):
    """
    Perform background subtraction using GMM with multiple training passes.
    Multiple passes: Train the GMM model (no output saved)
    Final pass: Inference with trained model (outputs saved)

    Args:
        input_path (str): Path to input video file
        binary_output_path (str): Path to save binary mask video (None to disable saving)
        extracted_output_path (str): Path to save extracted foreground video (None to disable saving)
        num_training_passes (int): Number of training passes before inference
    """

    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found!")
        return

    # Get video properties first
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
        history=1000,  # Large history for better learning
        dist2Threshold=150.0,
        detectShadows=True
    )

    # Read all frames once to avoid repeated file reading
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

    # Multiple training passes
    for pass_num in range(1, num_training_passes + 1):
        print(f"\n=== PASS {pass_num}: Training Pass ===")

        # Determine if this pass should be flipped
        flip_pass = (pass_num % 2 == 0)  # Flip even-numbered passes

        if flip_pass:
            print(f"Training GMM model (pass {pass_num} - FLIPPED)...")
            frames_to_process = all_frames[::-1]  # Reverse the frames
        else:
            print(f"Training GMM model (pass {pass_num} - NORMAL)...")
            frames_to_process = all_frames

        frame_count = 0
        for frame in frames_to_process:
            frame_count += 1

            # Apply GMM for training (discard the mask)
            _ = backSub.apply(frame)

            # Show training progress
            if frame_count % 50 == 0:
                progress = (frame_count / total_frames) * 100
                direction = "FLIPPED" if flip_pass else "NORMAL"
                print(f"Pass {pass_num} ({direction}) progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")

            # Optional: show training frames
            if frame_count % 20 == 0:  # Show every 20th frame
                train_frame_small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                window_title = f'Training Pass {pass_num} {"(FLIPPED)" if flip_pass else "(NORMAL)"}'
                cv2.imshow(window_title, train_frame_small)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Training interrupted by user")
                break

        direction = "flipped" if flip_pass else "normal"
        print(f"Pass {pass_num} ({direction}) completed: {frame_count} frames processed")

    # Setup video writers for inference pass
    binary_out = None
    extracted_out = None

    if binary_output_path:
        output_dir = os.path.dirname(binary_output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        binary_out = cv2.VideoWriter(binary_output_path, fourcc, fps, (width, height), isColor=False)
        print(f"Binary output will be saved to: {binary_output_path}")

    if extracted_output_path:
        output_dir = os.path.dirname(extracted_output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        extracted_out = cv2.VideoWriter(extracted_output_path, fourcc, fps, (width, height), isColor=True)
        print(f"Extracted output will be saved to: {extracted_output_path}")

    # Final pass: Inference with minimal learning
    final_pass = num_training_passes + 1
    print(f"\n=== PASS {final_pass}: Inference Pass ===")
    print("Processing video with fully trained model (saving outputs)...")

    frame_count = 0
    for frame in all_frames:
        frame_count += 1

        # Apply GMM with no learning for pure inference
        fgMask = backSub.apply(frame, learningRate=0.001)  # No adaptation during inference

        # Remove detected shadows from the mask
        fgMask[fgMask == 127] = 0

        # Enhanced noise reduction and morphological operations
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        # Multi-stage morphological operations for better results
        # Remove small noise
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel_small, iterations=2)
        # Fill small holes
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel_medium, iterations=4)
        # Final smoothing
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel_large, iterations=4)

        # Find contours and filter by area
        contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create clean mask with area filtering
        clean_mask = np.zeros_like(fgMask)
        min_area = 600  # Minimum area for objects
        max_area = width * height * 0.8  # Maximum 80% of frame

        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                cv2.fillPoly(clean_mask, [contour], 255)

        fgMask = clean_mask

        # Create extracted foreground image
        extracted_frame = cv2.bitwise_and(frame, frame, mask=fgMask)

        # Optional: Add white background instead of black
        # Uncomment for white background:
        # background = np.ones_like(frame) * 255
        # background_mask = cv2.bitwise_not(fgMask)
        # background = cv2.bitwise_and(background, background, mask=background_mask)
        # extracted_frame = cv2.add(extracted_frame, background)

        # Save frames
        if binary_out is not None:
            binary_out.write(fgMask)

        if extracted_out is not None:
            extracted_out.write(extracted_frame)

        # Display progress
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Inference progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")

        # Display the frames
        frame_resized = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        extracted_resized = cv2.resize(extracted_frame, (0, 0), fx=0.5, fy=0.5)
        mask_resized = cv2.resize(fgMask, (0, 0), fx=0.5, fy=0.5)

        cv2.imshow('Original Frame', frame_resized)
        cv2.imshow('Binary Mask', mask_resized)
        cv2.imshow('Extracted Foreground', extracted_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Processing interrupted by user")
            break

    # Cleanup
    if binary_out is not None:
        binary_out.release()
    if extracted_out is not None:
        extracted_out.release()
    cv2.destroyAllWindows()

    print(f"\n=== PROCESSING COMPLETED ===")
    print(f"Total training passes: {num_training_passes}")
    print(f"Frames processed in each pass: {len(all_frames)}")

    for i in range(1, num_training_passes + 1):
        direction = "flipped" if (i % 2 == 0) else "normal"
        print(f"Pass {i}: Training ({direction}) - no output")

    print(f"Pass {final_pass}: Inference - outputs saved")

    if binary_output_path and os.path.exists(binary_output_path):
        print(f"Binary output saved to: {binary_output_path}")
    if extracted_output_path and os.path.exists(extracted_output_path):
        print(f"Extracted output saved to: {extracted_output_path}")


def main():
    # Define file paths
    INPUT_VIDEO = r"C:\Users\zaita\Downloads\FinalProject\Outputs\background_locked.avi"
    BINARY_OUTPUT = r"C:\Users\zaita\Downloads\FinalProject\Outputs\binary.avi"
    EXTRACTED_OUTPUT = r"C:\Users\zaita\Downloads\FinalProject\Outputs\extracted.avi"

    # Run background subtraction with multiple training passes
    # Odd passes (1,3,5...) = normal direction
    # Even passes (2,4,6...) = flipped direction
    gmm_background_subtraction_multi_pass(
        INPUT_VIDEO,
        BINARY_OUTPUT,
        EXTRACTED_OUTPUT,
        num_training_passes = 3  # Total number of training passes before inference-was 5 changed to 3 for testing
    )


if __name__ == "__main__":
    main()