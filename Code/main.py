from background__subtraction import enhanced_background_subtraction_with_multi_pattern
# from stabilize import lucas_kanade_faster_video_stabilization
from stabilize import background_lock_stabilization # Change the file name
from tracking import process_video_tracking
from matting import matting_main
import os
import time
import json

os.chdir('..')

# FILL IN YOUR ID
ID1 = '318452364'
ID2 = '207767021'

# Set parameters
WINDOW_SIZE_TAU = 5
MAX_ITER_TAU = 7
NUM_LEVELS_TAU = 5

if __name__ == "__main__":
    print("Starting video processing pipeline...")
    start_time = time.time()

    # Setup working directory
    wrkdir = os.getcwd()
    outputs_dir = os.path.join(wrkdir, 'Outputs')
    os.makedirs(outputs_dir, exist_ok=True)

    # ===== ALL FILE PATHS =====
    # Input files
    INPUT_VIDEO = rf"{wrkdir}\Inputs\INPUT.avi"
    BACKGROUND_IMAGE = rf"{wrkdir}\Inputs\background.jpg"

    # Intermediate files
    STABILIZED_VIDEO = rf"{wrkdir}\Outputs\stabilize_{ID1}_{ID2}.avi"

    # Background subtraction outputs
    BINARY_VIDEO = rf"{wrkdir}\Outputs\binary_{ID1}_{ID2}.avi"
    EXTRACTED_VIDEO = rf"{wrkdir}\Outputs\extracted_{ID1}_{ID2}.avi"

    # Matting outputs
    MATTED_VIDEO = rf"{wrkdir}\Outputs\matted_{ID1}_{ID2}.avi"
    ALPHA_VIDEO = rf"{wrkdir}\Outputs\alpha_{ID1}_{ID2}.avi"

    # Final tracking outputs
    OUTPUT_VIDEO = rf"{wrkdir}\Outputs\OUTPUT_{ID1}_{ID2}.avi"

    # JSON files
    TIMING_JSON = rf"{wrkdir}\Outputs\timing.json"
    TRACKING_JSON = rf"{wrkdir}\Outputs\tracking.json"

    # Initialize timing dictionary with required structure
    timing_data = {}

    # ===== PIPELINE EXECUTION =====

    # Step 1: Video Stabilization
    print("Step 1: Background Lock Stabilization...")
    step_start = time.time()
    background_lock_stabilization(INPUT_VIDEO, STABILIZED_VIDEO)
    stabilize_time = time.time()
    timing_data["time_to_stabilize"] = stabilize_time - start_time
    print(f"Background lock stabilization completed in {stabilize_time - step_start:.2f} seconds")

    # Step 2: Background Subtraction
    print("Step 2: Background Subtraction...")
    enhanced_background_subtraction_with_multi_pattern(
        STABILIZED_VIDEO,
        BINARY_VIDEO,
        EXTRACTED_VIDEO,
    )
    binary_time = time.time()
    timing_data["time_to_binary"] = binary_time - start_time
    print(f"Background subtraction completed in {binary_time - stabilize_time:.2f} seconds")

    # Step 3: Matting
    print("Step 3: Image Matting...")
    matting_main(BACKGROUND_IMAGE, EXTRACTED_VIDEO, BINARY_VIDEO, MATTED_VIDEO, ALPHA_VIDEO)
    matted_time = time.time()
    timing_data["time_to_matted"] = matted_time - start_time
    timing_data["time_to_alpha"] = matted_time - start_time  # Same process creates both
    print(f"Matting completed in {matted_time - binary_time:.2f} seconds")

    # Step 4: Tracking
    print("Step 4: Person Tracking...")
    if os.path.exists(MATTED_VIDEO):
        process_video_tracking(MATTED_VIDEO, OUTPUT_VIDEO, TRACKING_JSON,
                               start_bbox=(90, 210, 330, 770))
        output_time = time.time()
        timing_data["time_to_output"] = output_time - start_time
        print(f"Tracking completed in {output_time - matted_time:.2f} seconds")
        print(f"Final output: {OUTPUT_VIDEO}")
        print(f"Tracking data: {TRACKING_JSON}")
    else:
        print(f"Error: Matted video not found at {MATTED_VIDEO}")
        timing_data["time_to_output"] = 0

    # Save timing data
    print("Saving timing data...")
    with open(TIMING_JSON, 'w') as f:
        json.dump(timing_data, f, indent=2)

    # Final summary
    total_time = time.time() - start_time
    print(f"\n=== PIPELINE COMPLETED ===")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Timing data saved to: {TIMING_JSON}")
    print(f"All outputs saved in: {outputs_dir}")
    print("Pipeline finished successfully!")