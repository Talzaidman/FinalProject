from background__subtraction import gmm_background_subtraction_multi_pass
#from stabilize import lucas_kanade_faster_video_stabilization
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
    
    r"""lucas_kanade_faster_video_stabilization(r"C:\Users\zaita\Downloads\FinalProject\Inputs\INPUT.avi",
                                        r"C:\Users\zaita\Downloads\FinalProject\Outputs\stabilize.avi",
                                        WINDOW_SIZE_TAU,
                                        MAX_ITER_TAU,
                                        NUM_LEVELS_TAU)"""
                                        
    INPUT_VIDEO = r"C:\Users\zaita\Downloads\FinalProject\Outputs\background_locked.avi"
    BINARY_OUTPUT = r"C:\Users\zaita\Downloads\FinalProject\Outputs\binary.avi"
    EXTRACTED_OUTPUT = r"C:\Users\zaita\Downloads\FinalProject\Outputs\extracted.avi"

    # Run background subtraction with flipped training
    r"""gmm_background_subtraction_multi_pass(
        INPUT_VIDEO,
        BINARY_OUTPUT,
        EXTRACTED_OUTPUT,
        num_training_passes=5  # Total number of training passes before inference
    )"""
    # Define file paths
    background_path = r"C:\Users\zaita\Downloads\FinalProject\Inputs\background.jpg"
    colored_mask_path = r"C:\Users\zaita\Downloads\FinalProject\Outputs\extracted.avi"
    binary_mask_path = r"C:\Users\zaita\Downloads\FinalProject\Outputs\binary.avi"
    output_matted_path = r"C:\Users\zaita\Downloads\FinalProject\Outputs\matted.avi"
    output_alpha_path = r"C:\Users\zaita\Downloads\FinalProject\Outputs\alpha.avi"

    matting_main(background_path, colored_mask_path, binary_mask_path, output_matted_path, output_alpha_path)
    #tracking()

    print("end")
