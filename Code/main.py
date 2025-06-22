# Imports
from background__subtraction import gmm_background_subtraction_multi_pass
# from stabilize import lucas_kanade_faster_video_stabilization  # Tal's comment
from stabilize import stabilize_video, stabilize_video2
import tracking
from matting import matting_main
import os
import time
import json

# Our IDs
ID1 = '318452364'
ID2 = '207767021'

# Parameters
WINDOW_SIZE_TAU = 5
MAX_ITER_TAU = 7
NUM_LEVELS_TAU = 5

# Path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
INPUTS_DIR = os.path.join(PROJECT_ROOT, 'Inputs')
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'Outputs')
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# File paths
INPUT_VIDEO = os.path.join(INPUTS_DIR, 'INPUT.avi')
BACKGROUND_IMAGE = os.path.join(INPUTS_DIR, 'background.jpg')

# Output video files (following project naming requirements)
STABILIZED_VIDEO = os.path.join(OUTPUTS_DIR, f'stabilize_{ID1}_{ID2}.avi')
BINARY_VIDEO = os.path.join(OUTPUTS_DIR, f'binary_{ID1}_{ID2}.avi')
EXTRACTED_VIDEO = os.path.join(OUTPUTS_DIR, f'extracted_{ID1}_{ID2}.avi')
MATTED_VIDEO = os.path.join(OUTPUTS_DIR, f'matted_{ID1}_{ID2}.avi')
ALPHA_VIDEO = os.path.join(OUTPUTS_DIR, f'alpha_{ID1}_{ID2}.avi')
OUTPUT_VIDEO = os.path.join(OUTPUTS_DIR, f'OUTPUT_{ID1}_{ID2}.avi')

# JSON output files
TIMING_JSON = os.path.join(OUTPUTS_DIR, 'timing.json')
TRACKING_JSON = os.path.join(OUTPUTS_DIR, 'tracking.json')

if __name__ == "__main__":
    print("start")
    
       # Test Stabilization
    print("\n" + "="*50)
    print("TESTING: VIDEO STABILIZATION")
    print("="*50)
    
    # Check if input exists
    if not os.path.exists(INPUT_VIDEO):
        print(f"❌ Input video not found: {INPUT_VIDEO}")
        exit(1)
    
    print("Starting video stabilization...")
    start_time = time.time()
    
    success = stabilize_video2(INPUT_VIDEO, STABILIZED_VIDEO)
    
    elapsed = time.time() - start_time
    
    if success:
        print(f"✅ Stabilization completed successfully in {elapsed:.2f} seconds!")
        print(f"✅ Stabilized video saved: {STABILIZED_VIDEO}")
    else:
        print(f"❌ Stabilization failed!")
    

    """lucas_kanade_faster_video_stabilization(
        INPUT_VIDEO,
        STABILIZED_VIDEO,
        WINDOW_SIZE_TAU,
        MAX_ITER_TAU,
        NUM_LEVELS_TAU
    )"""                                       

    # Run background subtraction with flipped training
    """gmm_background_subtraction_multi_pass(
    INPUT_VIDEO,            # or STABILIZED_VIDEO once stabilization works
    BINARY_VIDEO,
    EXTRACTED_VIDEO,
    num_training_passes=5   # Total number of training passes before inference
    )"""


    """matting_main(
    BACKGROUND_IMAGE,
    EXTRACTED_VIDEO,
    BINARY_VIDEO,
    MATTED_VIDEO,
    ALPHA_VIDEO
    )"""
    
    #tracking()

    print("end")
