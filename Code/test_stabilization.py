#!/usr/bin/env python3
"""
Testing script for video stabilization functions
Usage: python test_stabilization.py [function_name]
"""

import sys
import os
import time
from stabilize import stabilize_video, stabilize_video2

# IDs
ID1 = '318452364'
ID2 = '207767021'

# Path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
INPUTS_DIR = os.path.join(PROJECT_ROOT, 'Inputs')
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'Outputs')
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# File paths
INPUT_VIDEO = os.path.join(INPUTS_DIR, 'INPUT.avi')

def test_stabilize_video():
    """Test the reference frame stabilization approach"""
    output_path = os.path.join(OUTPUTS_DIR, f'test_stabilize_{ID1}_{ID2}.avi')
    
    print("Testing stabilize_video() - Reference Frame Approach")
    print(f"Input: {INPUT_VIDEO}")
    print(f"Output: {output_path}")
    
    if not os.path.exists(INPUT_VIDEO):
        print(f"❌ Input video not found: {INPUT_VIDEO}")
        return False
    
    start_time = time.time()
    success = stabilize_video(INPUT_VIDEO, output_path)
    elapsed = time.time() - start_time
    
    if success:
        print(f"✅ Reference stabilization completed in {elapsed:.2f} seconds!")
        print(f"✅ Output saved: {output_path}")
    else:
        print(f"❌ Reference stabilization failed!")
    
    return success

def test_stabilize_video2():
    """Test the frame-to-frame stabilization approach"""
    output_path = os.path.join(OUTPUTS_DIR, f'test_stabilize2_{ID1}_{ID2}.avi')
    
    print("Testing stabilize_video2() - Frame-to-Frame Approach")
    print(f"Input: {INPUT_VIDEO}")
    print(f"Output: {output_path}")
    
    if not os.path.exists(INPUT_VIDEO):
        print(f"❌ Input video not found: {INPUT_VIDEO}")
        return False
    
    start_time = time.time()
    success = stabilize_video2(INPUT_VIDEO, output_path)
    elapsed = time.time() - start_time
    
    if success:
        print(f"✅ Frame-to-frame stabilization completed in {elapsed:.2f} seconds!")
        print(f"✅ Output saved: {output_path}")
    else:
        print(f"❌ Frame-to-frame stabilization failed!")
    
    return success

def compare_both():
    """Test both stabilization methods and compare"""
    print("="*60)
    print("COMPARING BOTH STABILIZATION METHODS")
    print("="*60)
    
    print("\n1. Testing Reference Frame Method...")
    success1 = test_stabilize_video()
    
    print("\n2. Testing Frame-to-Frame Method...")
    success2 = test_stabilize_video2()
    
    print("\n" + "="*60)
    print("COMPARISON RESULTS:")
    print(f"Reference Frame Method: {'✅ SUCCESS' if success1 else '❌ FAILED'}")
    print(f"Frame-to-Frame Method: {'✅ SUCCESS' if success2 else '❌ FAILED'}")
    print("="*60)
    
    if success1 and success2:
        print("\n💡 Both methods worked! Compare the output videos to see which is better.")
        print("   Look for smoother motion and less jitter.")
    elif success1:
        print("\n💡 Reference frame method worked better!")
    elif success2:
        print("\n💡 Frame-to-frame method worked better!")
    else:
        print("\n❌ Both methods failed. Check your parameters and input video.")

def main():
    """Main function with command line argument support"""
    if len(sys.argv) > 1:
        function_name = sys.argv[1].lower()
        
        if function_name in ['ref', 'reference', '1']:
            test_stabilize_video()
        elif function_name in ['frame', 'f2f', '2']:
            test_stabilize_video2()
        elif function_name in ['both', 'compare', 'all']:
            compare_both()
        else:
            print("Unknown function. Use:")
            print("  python test_stabilization.py ref      # Test reference method")
            print("  python test_stabilization.py frame    # Test frame-to-frame method")
            print("  python test_stabilization.py both     # Test both and compare")
    else:
        # Default: show menu
        print("Video Stabilization Tester")
        print("="*30)
        print("1. Test Reference Frame Method")
        print("2. Test Frame-to-Frame Method")
        print("3. Compare Both Methods")
        print("q. Quit")
        
        while True:
            choice = input("\nEnter your choice (1/2/3/q): ").strip().lower()
            
            if choice == '1':
                test_stabilize_video()
                break
            elif choice == '2':
                test_stabilize_video2()
                break
            elif choice == '3':
                compare_both()
                break
            elif choice == 'q':
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1, 2, 3, or q.")

if __name__ == "__main__":
    main()