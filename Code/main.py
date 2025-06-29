from background__subtraction import enhanced_background_subtraction_with_postprocessing
# from stabilize import lucas_kanade_faster_video_stabilization
from temp_stieb import background_lock_stabilization
import tracking
from matting import matting_main
import os

os.chdir('..')

# FILL IN YOUR ID
ID1 = '318452364'
ID2 = '207767021'

# Choose parameters
WINDOW_SIZE_TAU = 5  # Add your value here!
MAX_ITER_TAU = 7  # Add your value here!
NUM_LEVELS_TAU = 5  # Add your value here!


if __name__ == "__main__":
    print("start")
    wrkdir = os.getcwd()
    """lucas_kanade_faster_video_stabilization(rf"{wrkdir}\Inputs\INPUT.avi",
                                        rf"{wrkdir}\Outputs\stabilize.avi",
                                        WINDOW_SIZE_TAU,
                                        MAX_ITER_TAU,
                                        NUM_LEVELS_TAU)"""
    """background_lock_stabilization(rf"{wrkdir}\Inputs\INPUT.avi",
                                            rf"{wrkdir}\Outputs\stabilize.avi" # Stabilization threshold
                                            )"""

    INPUT_VIDEO = rf"{wrkdir}\Outputs\background_locked.avi"
    BINARY_OUTPUT = rf"{wrkdir}\Outputs\binary.avi"
    EXTRACTED_OUTPUT = rf"{wrkdir}\Outputs\extracted.avi"

    # Run background subtraction with flipped training
    enhanced_background_subtraction_with_postprocessing(
        INPUT_VIDEO,
        BINARY_OUTPUT,
        EXTRACTED_OUTPUT,
        num_training_passes=5  # Total number of training passes before inference
    )
    # Define file paths
    background_path = fr"{wrkdir}\Inputs\background.jpg"
    colored_mask_path = fr"{wrkdir}\Outputs\extracted.avi"
    binary_mask_path = fr"{wrkdir}\Outputs\binary.avi"
    output_matted_path = fr"{wrkdir}\Outputs\matted.avi"
    output_alpha_path = fr"{wrkdir}\Outputs\alpha.avi"

    matting_main(background_path, colored_mask_path, binary_mask_path, output_matted_path, output_alpha_path)
    #tracking()

    print("end")
