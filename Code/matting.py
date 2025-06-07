import cv2
import numpy as np
from pathlib import Path


def composite_video_with_background(background_path, colored_mask_path, binary_mask_path,
                                    output_matted_path, output_alpha_path):
    """
    Composite a masked object video with a new background using alpha matting.

    Args:
        background_path: Path to background image
        colored_mask_path: Path to colored mask AVI (object with 0 background)
        binary_mask_path: Path to binary mask AVI
        output_matted_path: Path for output matted video
        output_alpha_path: Path for output alpha channel video
    """

    # Load background image
    background = cv2.imread(background_path)
    if background is None:
        raise ValueError(f"Could not load background image: {background_path}")

    # Open video captures
    colored_cap = cv2.VideoCapture(colored_mask_path)
    binary_cap = cv2.VideoCapture(binary_mask_path)

    if not colored_cap.isOpened():
        raise ValueError(f"Could not open colored mask video: {colored_mask_path}")
    if not binary_cap.isOpened():
        raise ValueError(f"Could not open binary mask video: {binary_mask_path}")

    # Get video properties
    fps = colored_cap.get(cv2.CAP_PROP_FPS)
    width = int(colored_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(colored_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(colored_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing video: {width}x{height}, {fps} FPS, {total_frames} frames")

    # Resize background to match video dimensions
    background_resized = cv2.resize(background, (width, height))

    # Define codec and create VideoWriter objects
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # Create output video writers
    matted_writer = cv2.VideoWriter(output_matted_path, fourcc, fps, (width, height))
    alpha_writer = cv2.VideoWriter(output_alpha_path, fourcc, fps, (width, height))

    if not matted_writer.isOpened():
        raise ValueError(f"Could not create output video: {output_matted_path}")
    if not alpha_writer.isOpened():
        raise ValueError(f"Could not create alpha video: {output_alpha_path}")

    frame_count = 0

    try:
        while True:
            # Read frames from both videos
            ret_colored, colored_frame = colored_cap.read()
            ret_binary, binary_frame = binary_cap.read()

            if not ret_colored or not ret_binary:
                break

            # Convert binary mask to grayscale if it's colored
            if len(binary_frame.shape) == 3:
                binary_mask = cv2.cvtColor(binary_frame, cv2.COLOR_BGR2GRAY)
            else:
                binary_mask = binary_frame

            # Normalize binary mask to 0-1 range
            alpha = binary_mask.astype(np.float32) / 255.0

            # Create 3-channel alpha for broadcasting
            alpha_3ch = np.stack([alpha, alpha, alpha], axis=2)

            # Perform alpha compositing
            # result = foreground * alpha + background * (1 - alpha)
            colored_frame_float = colored_frame.astype(np.float32)
            background_float = background_resized.astype(np.float32)

            # Composite the image
            matted_frame = (colored_frame_float * alpha_3ch +
                            background_float * (1 - alpha_3ch))

            # Convert back to uint8
            matted_frame = np.clip(matted_frame, 0, 255).astype(np.uint8)

            # Create alpha visualization (convert single channel alpha to 3-channel)
            alpha_vis = np.stack([binary_mask, binary_mask, binary_mask], axis=2)

            # Write frames
            matted_writer.write(matted_frame)
            alpha_writer.write(alpha_vis)

            frame_count += 1
            if frame_count % 30 == 0:  # Progress update every 30 frames
                print(f"Processed {frame_count}/{total_frames} frames ({frame_count / total_frames * 100:.1f}%)")

    finally:
        # Release everything
        colored_cap.release()
        binary_cap.release()
        matted_writer.release()
        alpha_writer.release()

    print(f"Successfully created matted video: {output_matted_path}")
    print(f"Successfully created alpha video: {output_alpha_path}")
    print(f"Total frames processed: {frame_count}")


def matting_main(background_path, colored_mask_path, binary_mask_path, output_matted_path, output_alpha_path):

    # Check if input files exist
    for path in [background_path, colored_mask_path, binary_mask_path]:
        if not Path(path).exists():
            print(f"Warning: File does not exist: {path}")

    # Create output directory if it doesn't exist
    Path(output_matted_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        composite_video_with_background(
            background_path=background_path,
            colored_mask_path=colored_mask_path,
            binary_mask_path=binary_mask_path,
            output_matted_path=output_matted_path,
            output_alpha_path=output_alpha_path
        )

        print("\n✅ Video compositing completed successfully!")

    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        raise

