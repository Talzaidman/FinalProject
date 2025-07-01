#!/usr/bin/env python3
"""
Debug script for video stabilization issues
This will help identify what's going wrong
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from stabilize import detect_features, track_features_from_reference, estimate_transformation

def analyze_video_properties(video_path):
    """Analyze basic video properties"""
    print(f"\n📹 ANALYZING VIDEO: {video_path}")
    print("="*50)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Cannot open video!")
        return None
    
    # Get properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"📊 Resolution: {width}x{height}")
    print(f"📊 Frame rate: {fps:.2f} FPS")
    print(f"📊 Total frames: {total_frames}")
    print(f"📊 Duration: {duration:.2f} seconds")
    
    # Analyze first few frames for motion
    frames = []
    for i in range(min(10, total_frames)):
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        else:
            break
    
    cap.release()
    
    if len(frames) >= 2:
        # Calculate frame differences to see motion
        diffs = []
        for i in range(1, len(frames)):
            diff = cv2.absdiff(frames[i-1], frames[i])
            mean_diff = np.mean(diff)
            diffs.append(mean_diff)
        
        avg_motion = np.mean(diffs)
        max_motion = np.max(diffs)
        
        print(f"📊 Average motion between frames: {avg_motion:.2f}")
        print(f"📊 Maximum motion between frames: {max_motion:.2f}")
        
        if avg_motion < 5:
            print("⚠️  WARNING: Very low motion detected - video might be stable already")
        elif avg_motion > 50:
            print("⚠️  WARNING: Very high motion detected - might be challenging to stabilize")
    
    return {
        'width': width, 'height': height, 'fps': fps, 
        'total_frames': total_frames, 'frames': frames[:5]
    }

def test_feature_detection(frames, max_corners_list=[100, 300, 500], quality_levels=[0.01, 0.03, 0.05]):
    """Test different feature detection parameters"""
    print(f"\n🔍 TESTING FEATURE DETECTION")
    print("="*50)
    
    if not frames:
        print("❌ No frames to analyze!")
        return
    
    test_frame = frames[0]
    
    print("Testing different parameter combinations...")
    
    best_params = None
    best_count = 0
    
    for max_corners in max_corners_list:
        for quality_level in quality_levels:
            feature_params = dict(
                maxCorners=max_corners,
                qualityLevel=quality_level,
                minDistance=3,
                blockSize=7
            )
            
            corners = cv2.goodFeaturesToTrack(test_frame, mask=None, **feature_params)
            count = len(corners) if corners is not None else 0
            
            print(f"  Max: {max_corners:3d}, Quality: {quality_level:.2f} → {count:3d} features")
            
            if count > best_count:
                best_count = count
                best_params = feature_params.copy()
    
    print(f"\n🏆 Best parameters found {best_count} features:")
    for key, value in best_params.items():
        print(f"   {key}: {value}")
    
    return best_params

def test_tracking_quality(frames, feature_params):
    """Test how well features track between frames"""
    print(f"\n🎯 TESTING FEATURE TRACKING")
    print("="*50)
    
    if len(frames) < 2:
        print("❌ Need at least 2 frames for tracking!")
        return
    
    # Detect features in first frame
    ref_frame = frames[0]
    features = cv2.goodFeaturesToTrack(ref_frame, mask=None, **feature_params)
    
    if features is None or len(features) == 0:
        print("❌ No features detected in reference frame!")
        return
    
    print(f"📊 Reference frame has {len(features)} features")
    
    # Test tracking to subsequent frames
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    
    tracking_success = []
    
    for i, curr_frame in enumerate(frames[1:], 1):
        # Track features
        tracked_features, status, error = cv2.calcOpticalFlowPyrLK(
            ref_frame, curr_frame, features, None, **lk_params
        )
        
        if tracked_features is not None:
            good_features = tracked_features[status == 1]
            success_rate = len(good_features) / len(features) * 100
            tracking_success.append(success_rate)
            
            print(f"  Frame {i}: {len(good_features)}/{len(features)} features tracked ({success_rate:.1f}%)")
        else:
            tracking_success.append(0)
            print(f"  Frame {i}: 0/{len(features)} features tracked (0.0%)")
    
    avg_success = np.mean(tracking_success) if tracking_success else 0
    print(f"\n📊 Average tracking success rate: {avg_success:.1f}%")
    
    if avg_success < 30:
        print("⚠️  WARNING: Poor tracking performance - try different parameters")
        print("   Suggestions:")
        print("   - Increase window size in LK params")
        print("   - Decrease quality level for more features")
        print("   - Increase maxLevel for pyramid tracking")
    
    return avg_success

def test_transformation_estimation(frames, feature_params):
    """Test transformation estimation"""
    print(f"\n🔄 TESTING TRANSFORMATION ESTIMATION")
    print("="*50)
    
    if len(frames) < 2:
        print("❌ Need at least 2 frames!")
        return
    
    ref_frame = frames[0]
    curr_frame = frames[1]
    
    # Detect and track features
    ref_features = cv2.goodFeaturesToTrack(ref_frame, mask=None, **feature_params)
    
    if ref_features is None:
        print("❌ No features in reference frame!")
        return
    
    good_ref, good_curr = track_features_from_reference(ref_frame, curr_frame, ref_features)
    
    if good_ref is None or len(good_ref) < 4:
        print(f"❌ Only {len(good_ref) if good_ref is not None else 0} features tracked - need at least 4!")
        return
    
    print(f"📊 {len(good_ref)} feature pairs for transformation estimation")
    
    # Test transformation
    transform = estimate_transformation(good_ref, good_curr)
    
    if transform is not None:
        # Analyze transformation
        dx = transform[0, 2]
        dy = transform[1, 2]
        rotation = np.arctan2(transform[1, 0], transform[0, 0]) * 180 / np.pi
        scale = np.sqrt(transform[0, 0]**2 + transform[0, 1]**2)
        
        print(f"📊 Estimated transformation:")
        print(f"   Translation: ({dx:.2f}, {dy:.2f}) pixels")
        print(f"   Rotation: {rotation:.2f} degrees")
        print(f"   Scale: {scale:.3f}")
        
        # Check if transformation seems reasonable
        translation_magnitude = np.sqrt(dx**2 + dy**2)
        
        if translation_magnitude > 100:
            print("⚠️  WARNING: Very large translation - might indicate poor feature matching")
        if abs(rotation) > 10:
            print("⚠️  WARNING: Large rotation - check if this is expected")
        if scale < 0.8 or scale > 1.2:
            print("⚠️  WARNING: Significant scaling - might indicate poor estimation")
    else:
        print("❌ Failed to estimate transformation!")

def suggest_parameters(video_info, tracking_success):
    """Suggest optimal parameters based on analysis"""
    print(f"\n💡 PARAMETER SUGGESTIONS")
    print("="*50)
    
    width, height = video_info['width'], video_info['height']
    
    # Suggest feature detection parameters
    if width * height > 1920 * 1080:  # High resolution
        suggested_corners = 800
        suggested_quality = 0.02
    elif width * height > 1280 * 720:  # Medium resolution
        suggested_corners = 500
        suggested_quality = 0.03
    else:  # Lower resolution
        suggested_corners = 300
        suggested_quality = 0.05
    
    print("🔍 Feature Detection:")
    print(f"   MAX_CORNERS = {suggested_corners}")
    print(f"   QUALITY_LEVEL = {suggested_quality}")
    print(f"   MIN_DISTANCE = 2  # For denser features")
    print(f"   BLOCK_SIZE = 7    # For stability")
    
    # Suggest optical flow parameters
    if tracking_success < 50:
        win_size = 25
        max_level = 4
    else:
        win_size = 21
        max_level = 3
    
    print("\n🎯 Optical Flow:")
    print(f"   winSize = ({win_size}, {win_size})")
    print(f"   maxLevel = {max_level}")
    print(f"   criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)")
    
    # Suggest RANSAC parameters
    print("\n🔄 RANSAC:")
    print(f"   RANSAC_THRESHOLD = 0.8")
    print(f"   MAX_ITERATIONS = 5000")
    print(f"   MIN_FEATURES_THRESHOLD = {max(50, suggested_corners // 10)}")

def main():
    """Main diagnostic function"""
    # Setup paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(BASE_DIR)
    INPUT_VIDEO = os.path.join(PROJECT_ROOT, 'Inputs', 'INPUT.avi')
    
    print("🔧 VIDEO STABILIZATION DIAGNOSTICS")
    print("="*60)
    
    if not os.path.exists(INPUT_VIDEO):
        print(f"❌ Input video not found: {INPUT_VIDEO}")
        return
    
    # Step 1: Analyze video
    video_info = analyze_video_properties(INPUT_VIDEO)
    if not video_info:
        return
    
    # Step 2: Test feature detection
    best_feature_params = test_feature_detection(video_info['frames'])
    if not best_feature_params:
        return
    
    # Step 3: Test tracking
    tracking_success = test_tracking_quality(video_info['frames'], best_feature_params)
    
    # Step 4: Test transformation
    test_transformation_estimation(video_info['frames'], best_feature_params)
    
    # Step 5: Provide suggestions
    suggest_parameters(video_info, tracking_success)
    
    print(f"\n🎯 NEXT STEPS:")
    print("1. Update your stabilize.py with the suggested parameters")
    print("2. Try both stabilize_video() and stabilize_video2() with new parameters")
    print("3. If still poor, the video might be too challenging or already stable")
    print("4. Consider trying a different approach (e.g., deep learning methods)")

if __name__ == "__main__":
    main()