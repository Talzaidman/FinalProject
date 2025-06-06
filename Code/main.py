from background__subtraction import extract_moving_objects_from_video_ultra_fast
from stabilize import lucas_kanade_faster_video_stabilization
import tracking
import matting
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
                                        
    extract_moving_objects_from_video_ultra_fast(r"C:\Users\zaita\Downloads\FinalProject\Outputs\ultra_stable.avi",
                                       r"C:\Users\zaita\Downloads\FinalProject\Outputs\extracted.avi",
                                        r"C:\Users\zaita\Downloads\FinalProject\Outputs\binary.avi",   threshold=3.0)

    #tracking()
    #matting()
    print("end")
