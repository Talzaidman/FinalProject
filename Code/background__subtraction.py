import cv2
import numpy as np
import os


def gmm_background_subtraction(input_path, binary_output_path=None, extracted_output_path=None):
    """
    Perform background subtraction using Gaussian Mixture Model (GMM) method.
    Saves both binary mask and extracted foreground videos.

    Args:
        input_path (str): Path to input video file
        binary_output_path (str): Path to save binary mask video (None to disable saving)
        extracted_output_path (str): Path to save extracted foreground video (None to disable saving)
    """

    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found!")
        return

    # Open video capture
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video file '{input_path}'")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")

    # Create GMM background subtractor
    backSub = cv2.createBackgroundSubtractorKNN(
        history=200,  # Increase for more stable background model
        dist2Threshold=220.0,  # Lower for more sensitive detection
        detectShadows=True
    )

    # Setup video writers
    binary_out = None
    extracted_out = None

    if binary_output_path:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(binary_output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Binary video writer (grayscale)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        binary_out = cv2.VideoWriter(binary_output_path, fourcc, fps, (width, height), isColor=False)
        print(f"Binary output will be saved to: {binary_output_path}")

    if extracted_output_path:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(extracted_output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Extracted video writer (color)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        extracted_out = cv2.VideoWriter(extracted_output_path, fourcc, fps, (width, height), isColor=True)
        print(f"Extracted output will be saved to: {extracted_output_path}")

    frame_count = 0

    print("Processing video...")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # Apply GMM background subtraction
        fgMask = backSub.apply(frame)

        # Remove detected shadows from the mask
        fgMask[fgMask == 127] = 0

        # Noise reduction and morphological operations
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        # Remove small objects
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel_small, iterations=3)
        # Fill holes in detected objects
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel_medium, iterations=2)

        # Find contours and filter by area
        contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create clean mask
        clean_mask = np.zeros_like(fgMask)
        min_area = 1500  # Adjust based on your needs

        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                cv2.fillPoly(clean_mask, [contour], 255)

        fgMask = clean_mask

        # Create extracted foreground image
        # Apply mask to original frame to extract only foreground objects
        extracted_frame = cv2.bitwise_and(frame, frame, mask=fgMask)

        # Optional: Set background to black or white
        # For black background (current approach - already done by bitwise_and)
        # For white background, uncomment the following lines:
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
        if frame_count % 30 == 0:  # Print every 30 frames
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")

        # Optional: Display the frames (uncomment to see real-time processing)
        # Press 'q' to quit early
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Binary Mask', fgMask)
        cv2.imshow('Extracted Foreground', extracted_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Processing interrupted by user")
            break

    # Cleanup
    cap.release()
    if binary_out is not None:
        binary_out.release()
    if extracted_out is not None:
        extracted_out.release()
    cv2.destroyAllWindows()

    print(f"Processing completed! Processed {frame_count} frames.")
    if binary_output_path and os.path.exists(binary_output_path):
        print(f"Binary output saved to: {binary_output_path}")
    if extracted_output_path and os.path.exists(extracted_output_path):
        print(f"Extracted output saved to: {extracted_output_path}")


def main():
    # Define file paths
    INPUT_VIDEO = r"C:\Users\zaita\Downloads\FinalProject\Outputs\background_locked.avi"
    BINARY_OUTPUT = r"C:\Users\zaita\Downloads\FinalProject\Outputs\binary.avi"
    EXTRACTED_OUTPUT = r"C:\Users\zaita\Downloads\FinalProject\Outputs\extracted.avi"

    # Run background subtraction with dual output
    gmm_background_subtraction(INPUT_VIDEO, BINARY_OUTPUT, EXTRACTED_OUTPUT)


if __name__ == "__main__":
    main()