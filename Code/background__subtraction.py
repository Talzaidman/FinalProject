import cv2
import numpy as np
import os


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
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
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
        num_training_passes=5  # Total number of training passes before inference
    )


if __name__ == "__main__":
    main()