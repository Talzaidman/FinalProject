from background__subtraction import gmm_background_subtraction
#from stabilize import lucas_kanade_faster_video_stabilization
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
                                        
    gmm_background_subtraction(r"C:\Users\zaita\Downloads\FinalProject\Outputs\background_locked.avi",
                                       r"C:\Users\zaita\Downloads\FinalProject\Outputs\binary.avi",
                                        r"C:\Users\zaita\Downloads\FinalProject\Outputs\extracted.avi")

    #tracking()
    #matting()
    print("end")
