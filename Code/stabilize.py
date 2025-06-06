import cv2
import numpy as np
from tqdm import tqdm
from scipy import signal
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


# FILL IN YOUR ID
ID1 = 318452364
ID2 = 207767021


PYRAMID_FILTER = 1.0 / 256 * np.array([[1, 4, 6, 4, 1],
                                       [4, 16, 24, 16, 4],
                                       [6, 24, 36, 24, 6],
                                       [4, 16, 24, 16, 4],
                                       [1, 4, 6, 4, 1]])
X_DERIVATIVE_FILTER = np.array([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]])
Y_DERIVATIVE_FILTER = X_DERIVATIVE_FILTER.copy().transpose()

WINDOW_SIZE = 5


def get_video_parameters(capture: cv2.VideoCapture) -> dict:
    """Get an OpenCV capture object and extract its parameters.

    Args:
        capture: cv2.VideoCapture object.

    Returns:
        parameters: dict. Video parameters extracted from the video.

    """
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return {"fourcc": fourcc, "fps": fps, "height": height, "width": width,
            "frame_count": frame_count}


def build_pyramid(image: np.ndarray, num_levels: int) -> list[np.ndarray]:
    """Coverts image to a pyramid list of size num_levels.

    First, create a list with the original image in it. Then, iterate over the
    levels. In each level, convolve the PYRAMID_FILTER with the image from the
    previous level. Then, decimate the result using indexing: simply pick
    every second entry of the result.
    Hint: Use signal.convolve2d with boundary='symm' and mode='same'.

    Args:
        image: np.ndarray. Input image.
        num_levels: int. The number of blurring / decimation times.

    Returns:
        pyramid: list. A list of np.ndarray of images.

    Note that the list length should be num_levels + 1 as the in first entry of
    the pyramid is the original image.
    You are not allowed to use cv2 PyrDown here (or any other cv2 method).
    We use a slightly different decimation process from this function.
    """
    pyramid = [image.copy()]
    for level in range(num_levels):
        current_image = pyramid[level]
        blurred = signal.convolve2d(current_image, PYRAMID_FILTER, 
                                   boundary='symm', mode='same')
        decimated = blurred[::2, ::2]
        pyramid.append(decimated)
    return pyramid


def lucas_kanade_step(I1: np.ndarray,
                      I2: np.ndarray,
                      window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Perform one Lucas-Kanade Step.

    This method receives two images as inputs and a window_size. It
    calculates the per-pixel shift in the x-axis and y-axis. That is,
    it outputs two maps of the shape of the input images. The first map
    encodes the per-pixel optical flow parameters in the x-axis and the
    second in the y-axis.

    (1) Calculate Ix and Iy by convolving I2 with the appropriate filters (
    see the constants in the head of this file).
    (2) Calculate It from I1 and I2.
    (3) Calculate du and dv for each pixel:
      (3.1) Start from all-zeros du and dv (each one) of size I1.shape.
      (3.2) Loop over all pixels in the image (you can ignore boundary pixels up
      to ~window_size/2 pixels in each side of the image [top, bottom,
      left and right]).
      (3.3) For every pixel, pretend the pixel's neighbors have the same (u,
      v). This means that for NxN window, we have N^2 equations per pixel.
      (3.4) Solve for (u, v) using Least-Squares solution. When the solution
      does not converge, keep this pixel's (u, v) as zero.
    For detailed Equations reference look at slides 4 & 5 in:
    http://www.cse.psu.edu/~rtc12/CSE486/lecture30.pdf

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.

    Returns:
        (du, dv): tuple of np.ndarray-s. Each one is of the shape of the
        original image. dv encodes the optical flow parameters in rows and du
        in columns.
    """
    Ix = signal.convolve2d(I2, X_DERIVATIVE_FILTER, mode='same')
    Iy = signal.convolve2d(I2, Y_DERIVATIVE_FILTER, mode='same')
    It = I2 - I1
    
    du = np.zeros_like(I1)
    dv = np.zeros_like(I1)
    
    w = window_size // 2

    for i in range(w, I1.shape[0]-w):
        for j in range(w, I1.shape[1]-w):
            
            # Extract window around current pixel
            Ix_window = Ix[i-w:i+w+1, j-w:j+w+1].flatten()
            Iy_window = Iy[i-w:i+w+1, j-w:j+w+1].flatten()
            It_window = It[i-w:i+w+1, j-w:j+w+1].flatten()

            A = np.vstack((Ix_window, Iy_window)).T             #  A matrix and b vector
            b = -It_window         
            # Solve least squares
            try:
                flow = np.linalg.lstsq(A, b, rcond=None)[0]
                # Check if flow values are valid before assignment
                if np.isfinite(flow[0]) and np.isfinite(flow[1]):
                    du[i,j] = flow[0]
                    dv[i,j] = flow[1]
            except:
                continue
    return du, dv


def warp_image(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Warp image using the optical flow parameters in u and v.

    Note that this method needs to support the case where u and v shapes do
    not share the same shape as of the image. We will update u and v to the
    shape of the image. The way to do it, is to:
    (1) cv2.resize to resize the u and v to the shape of the image.
    (2) Then, normalize the shift values according to a factor. This factor
    is the ratio between the image dimension and the shift matrix (u or v)
    dimension (the factor for u should take into account the number of columns
    in u and the factor for v should take into account the number of rows in v).

    As for the warping, use `scipy.interpolate`'s `griddata` method. Define the
    grid-points using a flattened version of the `meshgrid` of 0:w-1 and 0:h-1.
    The values here are simply image.flattened().
    The points you wish to interpolate are, again, a flattened version of the
    `meshgrid` matrices - don't forget to add them v and u.
    Use `np.nan` as `griddata`'s fill_value.
    Finally, fill the nan holes with the source image values.
    Hint: For the final step, use np.isnan(image_warp).

    Args:
        image: np.ndarray. Image to warp.
        u: np.ndarray. Optical flow parameters corresponding to the columns.
        v: np.ndarray. Optical flow parameters corresponding to the rows.

    Returns:
        image_warp: np.ndarray. Warped image.
    """
    h, w = image.shape
    # If u and v are not the same size as image, scale before resizing
    if u.shape != image.shape:
        orig_h, orig_w = u.shape
        u = u * (w / orig_w)
        v = v * (h / orig_h)
        u = cv2.resize(u, (w, h), interpolation=cv2.INTER_LINEAR)
        v = cv2.resize(v, (w, h), interpolation=cv2.INTER_LINEAR)

    # Create meshgrid of coordinates
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + u).astype(np.float32)
    map_y = (y + v).astype(np.float32)

    # Use cv2.remap for efficient warping
    image_warp = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Ensure output dtype matches input
    if image_warp.dtype != image.dtype:
        image_warp = np.clip(image_warp, 0, 255).astype(image.dtype)
    return image_warp


def lucas_kanade_optical_flow(I1: np.ndarray,
                              I2: np.ndarray,
                              window_size: int,
                              max_iter: int,
                              num_levels: int) -> tuple[np.ndarray, np.ndarray]:
    """Calculate LK Optical Flow for max iterations in num-levels.

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        (u, v): tuple of np.ndarray-s. Each one of the shape of the
        original image. v encodes the optical flow parameters in rows and u in
        columns.
    """
    h_factor = int(np.ceil(I1.shape[0] / (2 ** (num_levels - 1))))
    w_factor = int(np.ceil(I1.shape[1] / (2 ** (num_levels - 1))))
    target_size = (w_factor * (2 ** (num_levels - 1)), 
                  h_factor * (2 ** (num_levels - 1)))
    
    
    if I1.shape != target_size:     # Resize images if needed
        I1 = cv2.resize(I1, target_size)
    if I2.shape != target_size:
        I2 = cv2.resize(I2, target_size)
    
    pyramid_I1 = build_pyramid(I1, num_levels)
    pyramid_I2 = build_pyramid(I2, num_levels)
    

    u = np.zeros(pyramid_I2[-1].shape)
    v = np.zeros(pyramid_I2[-1].shape)
    
    for level in range(num_levels-1, -1, -1):
        I1_level = pyramid_I1[level]
        I2_level = pyramid_I2[level]
        
        # Resize u and v to current level
        if level != num_levels-1:
            u = cv2.resize(u, (I1_level.shape[1], I1_level.shape[0])) * 2
            v = cv2.resize(v, (I1_level.shape[1], I1_level.shape[0])) * 2

        rows, cols = I2_level.shape
        y_coords, x_coords = np.mgrid[0:rows, 0:cols]
        
        for _ in range(max_iter):   #does itterations
            if u.shape != I2_level.shape:
                u = cv2.resize(u, (I2_level.shape[1], I2_level.shape[0]))
            if v.shape != I2_level.shape:
                v = cv2.resize(v, (I2_level.shape[1], I2_level.shape[0]))
            
            coords = np.stack([x_coords + u, y_coords + v], axis=-1)
            I2_warp = griddata((x_coords.flatten(), y_coords.flatten()), 
                             I2_level.flatten(),
                             coords.reshape(-1, 2),
                             method='linear',
                             fill_value=0).reshape(I2_level.shape)
            
            du, dv = lucas_kanade_step(I1_level, I2_warp, window_size)
            
            if du.shape != u.shape:     # make sure size are good
                du = cv2.resize(du, (u.shape[1], u.shape[0]))
            if dv.shape != v.shape:
                dv = cv2.resize(dv, (v.shape[1], v.shape[0]))
            
            u += du
            v += dv
    
    
    if u.shape != I1.shape:     ## Resize  u and v if not the same size as i1
        u = cv2.resize(u, (I1.shape[1], I1.shape[0]))
    if v.shape != I1.shape:
        v = cv2.resize(v, (I1.shape[1], I1.shape[0]))
    
    return u, v


def lucas_kanade_video_stabilization(input_video_path: str,
                                     output_video_path: str,
                                     window_size: int,
                                     max_iter: int,
                                     num_levels: int) -> None:
    """Use LK Optical Flow to stabilize the video and save it to file.

    Args:
        input_video_path: str. path to input video.
        output_video_path: str. path to output stabilized video.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        None.

    Recipe:
        (1) Open a VideoCapture object of the input video and read its
        parameters.
        (2) Create an output video VideoCapture object with the same
        parameters as in (1) in the path given here as input.
        (3) Convert the first frame to grayscale and write it as-is to the
        output video.
        (4) Resize the first frame as in the Full-Lucas-Kanade function to
        K * (2^(num_levels - 1)) X M * (2^(num_levels - 1)).
        Where: K is the ceil(h / (2^(num_levels - 1)),
        and M is ceil(h / (2^(num_levels - 1)).
        (5) Create a u and a v which are og the size of the image.
        (6) Loop over the frames in the input video (use tqdm to monitor your
        progress) and:
          (6.1) Resize them to the shape in (4).
          (6.2) Feed them to the lucas_kanade_optical_flow with the previous
          frame.
          (6.3) Use the u and v maps obtained from (6.2) and compute their
          mean values over the region that the computation is valid (exclude
          half window borders from every side of the image).
          (6.4) Update u and v to their mean values inside the valid
          computation region.
          (6.5) Add the u and v shift from the previous frame diff such that
          frame in the t is normalized all the way back to the first frame.
          (6.6) Save the updated u and v for the next frame (so you can
          perform step 6.5 for the next frame.
          (6.7) Finally, warp the current frame with the u and v you have at
          hand.
          (6.8) We highly recommend you to save each frame to a directory for
          your own debug purposes. Erase that code when submitting the exercise.
       (7) Do not forget to gracefully close all VideoCapture and to destroy
       all windows.
    """
    # Open input video and read parameters
    cap = cv2.VideoCapture(input_video_path)
    # Get video properties
    video_parameters = get_video_parameters(cap)
    w = frame_width = video_parameters["width"] # Corrected here
    h = frame_height = video_parameters["height"]
    fps = video_parameters["fps"]

    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Read first frame
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return

    # Write first frame as-is to output
    out.write(first_frame)

    # Convert to grayscale for calculations
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # Resize to dimensions divisible by 2^num_levels
    h_factor = int(np.ceil(first_gray.shape[0] / (2 ** num_levels)))
    w_factor = int(np.ceil(first_gray.shape[1] / (2 ** num_levels)))
    new_h = h_factor * (2 ** num_levels)
    new_w = w_factor * (2 ** num_levels)
    
    # Initialize accumulated flow
    u_accum = np.zeros((new_h, new_w), dtype=np.float32)
    v_accum = np.zeros((new_h, new_w), dtype=np.float32)
    
    prev_gray_resized = cv2.resize(first_gray, (new_w, new_h))

    # Process each frame
    frame_count = video_parameters["frame_count"]
    pbar = tqdm(total=frame_count - 1)

    while True:
        # Read next frame
        ret, curr_frame = cap.read()
        if not ret:
            break

        # Convert to grayscale for calculations
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        curr_gray_resized = cv2.resize(curr_gray, (int(new_w), int(new_h)))

        # Calculate optical flow
        u, v = lucas_kanade_optical_flow(prev_gray_resized,
                                         curr_gray_resized,
                                         window_size,
                                         max_iter,
                                         num_levels)

        # Calculate mean motion in valid region (excluding borders)
        half_window = window_size // 2
        valid_region_u = u[half_window:-half_window, half_window:-half_window]
        valid_region_v = v[half_window:-half_window, half_window:-half_window]

        u_mean = np.mean(valid_region_u)
        v_mean = np.mean(valid_region_v)

        # Create uniform motion maps with mean values
        u_uniform = np.full((new_h, new_w), u_mean)
        v_uniform = np.full((new_h, new_w), v_mean)

        # Accumulate motion - this is a critical step for stabilization
        # We want to stabilize relative to the first frame
        u_accum = u_accum + u_uniform  # Add current displacement to accumulated
        v_accum = v_accum + v_uniform

        # Need to resize to match frame dimensions before warping
        u_resized = cv2.resize(u_accum, (w, h))
        v_resized = cv2.resize(v_accum, (w, h))

        # Scale the flow values after resizing
        u_resized = u_resized * (w / new_w)
        v_resized = v_resized * (h / new_h)

        # Warp the current frame using accumulated flow
        curr_gray_stabilized = warp_image(curr_gray, u_resized, v_resized)

        # Ensure output is in the correct format
        curr_gray_stabilized = np.clip(curr_gray_stabilized, 0, 255).astype(np.uint8)

        # Convert back to BGR for output
        curr_frame_stabilized = cv2.cvtColor(curr_gray_stabilized, cv2.COLOR_GRAY2BGR)

        # Write stabilized frame to output
        out.write(curr_frame_stabilized)

        # Update previous frame
        prev_gray_resized = curr_gray_resized
        pbar.update(1)

    # Clean up
    pbar.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def faster_lucas_kanade_step(I1: np.ndarray,
                             I2: np.ndarray,
                             window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Faster implementation of a single Lucas-Kanade Step.

    (1) If the image is small enough (you need to design what is good
    enough), simply return the result of the good old lucas_kanade_step
    function.
    (2) Otherwise, find corners in I2 and calculate u and v only for these
    pixels.
    (3) Return maps of u and v which are all zeros except for the corner
    pixels you found in (2).

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.

    Returns:
        (du, dv): tuple of np.ndarray-s. Each one of the shape of the
        original image. dv encodes the shift in rows and du in columns.
    """
    # If image is small enough, use regular lucas_kanade_step
    if I1.shape[0] * I1.shape[1] < 6000:  # Threshold of 50000 pixels
        return lucas_kanade_step(I1, I2, window_size)
    
    
    du = np.zeros_like(I1, dtype=np.float32)
    dv = np.zeros_like(I1, dtype=np.float32)
    
    # Use cv2.cornerHarris for corner detection
    blockSize = 2
    ksize = 3
    k = 0.04
    harris_response = cv2.cornerHarris(I2.astype(np.float32), blockSize, ksize, k)
    # Threshold for an optimal value, it may vary depending on the image.
    threshold = 0.05 * harris_response.max()
    corners = np.argwhere(harris_response > threshold)
    # corners is a list of (row, col) = (y, x)
    # Optionally, limit to 100 strongest corners
    if corners.shape[0] > 100:
        # Sort by response strength
        strengths = harris_response[corners[:,0], corners[:,1]]
        idx = np.argsort(strengths)[-100:]
        corners = corners[idx]
    # Create a visualization of corners on I2
    """corner_vis = cv2.cvtColor(I2.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    corner_vis[corners[:,0], corners[:,1]] = [0, 0, 255]  # Mark corners in red
    plt.figure()
    plt.imshow(corner_vis)
    plt.title('Detected Corners')
    plt.axis('off')
    plt.show()"""
    w = window_size // 2
    Ix = signal.convolve2d(I2, X_DERIVATIVE_FILTER, mode='same')
    Iy = signal.convolve2d(I2, Y_DERIVATIVE_FILTER, mode='same')
    It = I2 - I1
    corner_points = []
    du_corners = []
    dv_corners = []
    for i, j in corners:
        if i < w or i >= I1.shape[0]-w or j < w or j >= I1.shape[1]-w:
            continue
        Ix_window = Ix[i-w:i+w+1, j-w:j+w+1].flatten()
        Iy_window = Iy[i-w:i+w+1, j-w:j+w+1].flatten()
        It_window = It[i-w:i+w+1, j-w:j+w+1].flatten()
        A = np.vstack((Ix_window, Iy_window)).T
        b = -It_window
        try:
            flow = np.linalg.lstsq(A, b, rcond=None)[0]
            corner_points.append([i, j])
            du_corners.append(flow[0])
            dv_corners.append(flow[1])
        except:
            continue
    # Interpolate sparse flow to dense flow
    if len(corner_points) > 0:
        corner_points = np.array(corner_points)
        from scipy.interpolate import griddata
        grid_y, grid_x = np.mgrid[0:I1.shape[0], 0:I1.shape[1]]
        du_dense = griddata(corner_points, du_corners, (grid_y, grid_x), method='linear', fill_value=0)
        dv_dense = griddata(corner_points, dv_corners, (grid_y, grid_x), method='linear', fill_value=0)
        du = du_dense.astype(np.float32)
        dv = dv_dense.astype(np.float32)
    return du, dv


def faster_lucas_kanade_optical_flow(
        I1: np.ndarray, I2: np.ndarray, window_size: int, max_iter: int,
        num_levels: int) -> tuple[np.ndarray, np.ndarray]:
    """Calculate LK Optical Flow for max iterations in num-levels .

    Use faster_lucas_kanade_step instead of lucas_kanade_step.

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        (u, v): tuple of np.ndarray-s. Each one of the shape of the
        original image. v encodes the shift in rows and u in columns.
    """
    h_factor = int(np.ceil(I1.shape[0] / (2 ** num_levels)))
    w_factor = int(np.ceil(I1.shape[1] / (2 ** num_levels)))
    IMAGE_SIZE = (w_factor * (2 ** num_levels),
                  h_factor * (2 ** num_levels))
    if I1.shape != IMAGE_SIZE:
        I1 = cv2.resize(I1, IMAGE_SIZE)
    if I2.shape != IMAGE_SIZE:
        I2 = cv2.resize(I2, IMAGE_SIZE)
    pyramid_I1 = build_pyramid(I1, num_levels)  # create levels list for I1
    pyarmid_I2 = build_pyramid(I2, num_levels)  # create levels list for I1
    u = np.zeros(pyarmid_I2[-1].shape)  # create u in the size of smallest image
    v = np.zeros(pyarmid_I2[-1].shape)  # create v in the size of smallest image
    """INSERT YOUR CODE HERE.
    Replace u and v with their true value."""

    
    for level in range(num_levels-1, -1, -1):
        I1_level = pyramid_I1[level]
        I2_level = pyarmid_I2[level]
        
        # Resize u and v to current level
        if level != num_levels-1:
            u = cv2.resize(u, (I1_level.shape[1], I1_level.shape[0])) * 2
            v = cv2.resize(v, (I1_level.shape[1], I1_level.shape[0])) * 2

        rows, cols = I2_level.shape
        y_coords, x_coords = np.mgrid[0:rows, 0:cols]
        
        for _ in range(max_iter):   #does itterations
            if u.shape != I2_level.shape:
                u = cv2.resize(u, (I2_level.shape[1], I2_level.shape[0]))
            if v.shape != I2_level.shape:
                v = cv2.resize(v, (I2_level.shape[1], I2_level.shape[0]))
            
            coords = np.stack([x_coords + u, y_coords + v], axis=-1)
            I2_warp = griddata((x_coords.flatten(), y_coords.flatten()), 
                             I2_level.flatten(),
                             coords.reshape(-1, 2),
                             method='linear',
                             fill_value=0).reshape(I2_level.shape)
            
            du, dv = faster_lucas_kanade_step(I1_level, I2_warp, window_size)
            
            if du.shape != u.shape:     # make sure size are good
                du = cv2.resize(du, (u.shape[1], u.shape[0]))
            if dv.shape != v.shape:
                dv = cv2.resize(dv, (v.shape[1], v.shape[0]))
            
            u += du
            v += dv
    
    
    if u.shape != I1.shape:     ## Resize  u and v if not the same size as i1
        u = cv2.resize(u, (I1.shape[1], I1.shape[0]))
    if v.shape != I1.shape:
        v = cv2.resize(v, (I1.shape[1], I1.shape[0]))
    
    return u, v


def lucas_kanade_faster_video_stabilization(
        input_video_path: str, output_video_path: str, window_size: int,
        max_iter: int, num_levels: int) -> None:
    """Calculate LK Optical Flow to stabilize the video and save it to file.

    Args:
        input_video_path: str. path to input video.
        output_video_path: str. path to output stabilized video.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        None.
    """
    # Open input video and read parameters
    cap = cv2.VideoCapture(input_video_path)
    # Get video properties
    video_parameters = get_video_parameters(cap)
    frame_width = video_parameters["width"]
    frame_height = video_parameters["height"]
    fps = video_parameters["fps"]

    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Read first frame
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return

    # Write first frame as-is to output
    out.write(first_frame)

    # Convert to grayscale for calculations
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # Resize to dimensions divisible by 2^num_levels
    h_factor = int(np.ceil(first_gray.shape[0] / (2 ** num_levels)))
    w_factor = int(np.ceil(first_gray.shape[1] / (2 ** num_levels)))
    new_h = h_factor * (2 ** num_levels)
    new_w = w_factor * (2 ** num_levels)
    
    # Initialize accumulated flow
    u_accum = np.zeros((new_h, new_w), dtype=np.float32)
    v_accum = np.zeros((new_h, new_w), dtype=np.float32)
    
    prev_gray_resized = cv2.resize(first_gray, (new_w, new_h))

    # Process each frame
    frame_count = video_parameters["frame_count"]
    pbar = tqdm(total=frame_count - 1)

    while True:
        # Read next frame
        ret, curr_frame = cap.read()
        if not ret:
            break

        # Convert to grayscale for calculations
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        curr_gray_resized = cv2.resize(curr_gray, (int(new_w), int(new_h)))

        # Calculate optical flow
        u, v = faster_lucas_kanade_optical_flow(prev_gray_resized,
                                         curr_gray_resized,
                                         window_size,
                                         max_iter,
                                         num_levels)

        # Calculate mean motion in valid region (excluding borders)
        half_window = window_size // 2
        valid_region_u = u[half_window:-half_window, half_window:-half_window]
        valid_region_v = v[half_window:-half_window, half_window:-half_window]

        u_mean = np.mean(valid_region_u)
        v_mean = np.mean(valid_region_v)

        # Create uniform motion maps with mean values
        u_uniform = np.full((new_h, new_w), u_mean)
        v_uniform = np.full((new_h, new_w), v_mean)

        # Accumulate motion - this is a critical step for stabilization
        # We want to stabilize relative to the first frame
        u_accum = u_accum + u_uniform  # Add current displacement to accumulated
        v_accum = v_accum + v_uniform

        # Resize to match frame dimensions
        h, w = curr_gray.shape
        u_resized = cv2.resize(u_accum, (w, h))
        v_resized = cv2.resize(v_accum, (w, h))

        # Scale the flow values after resizing
        u_resized = u_resized * (w / new_w)
        v_resized = v_resized * (h / new_h)

        # Warp the current frame using accumulated flow
        curr_gray_stabilized = warp_image(curr_gray, u_resized, v_resized)

        # Ensure output is in the correct format
        curr_gray_stabilized = np.clip(curr_gray_stabilized, 0, 255).astype(np.uint8)

        # Convert back to BGR for output
        curr_frame_stabilized = cv2.cvtColor(curr_gray_stabilized, cv2.COLOR_GRAY2BGR)

        # Write stabilized frame to output
        out.write(curr_frame_stabilized)

        # Update previous frame
        prev_gray_resized = curr_gray_resized
        pbar.update(1)

    # Clean up
    pbar.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def lucas_kanade_faster_video_stabilization_fix_effects(
        input_video_path: str, output_video_path: str, window_size: int,
        max_iter: int, num_levels: int, start_rows: int = 10,
        start_cols: int = 2, end_rows: int = 30, end_cols: int = 30) -> None:
    """Calculate LK Optical Flow to stabilize the video and save it to file.

    Args:
        input_video_path: str. path to input video.
        output_video_path: str. path to output stabilized video.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.
        start_rows: int. The number of lines to cut from top.
        end_rows: int. The number of lines to cut from bottom.
        start_cols: int. The number of columns to cut from left.
        end_cols: int. The number of columns to cut from right.

    Returns:
        None.
    """
    cap = cv2.VideoCapture(input_video_path)
    video_parameters = get_video_parameters(cap)
    frame_width = video_parameters["width"]
    frame_height = video_parameters["height"]
    fps = video_parameters["fps"]

    cropped_width = frame_width - start_cols - end_cols
    cropped_height = frame_height - start_rows - end_rows

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (cropped_width, cropped_height))

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return

    first_frame_cropped = first_frame[start_rows:frame_height-end_rows, 
                                     start_cols:frame_width-end_cols]
    out.write(first_frame_cropped)

    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    h_factor = int(np.ceil(first_gray.shape[0] / (2 ** num_levels)))
    w_factor = int(np.ceil(first_gray.shape[1] / (2 ** num_levels)))
    new_h = h_factor * (2 ** num_levels)
    new_w = w_factor * (2 ** num_levels)
    
    u_accum = np.zeros((new_h, new_w), dtype=np.float32)
    v_accum = np.zeros((new_h, new_w), dtype=np.float32)
    
    prev_gray_resized = cv2.resize(first_gray, (new_w, new_h))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    pbar = tqdm(total=frame_count, desc="Stabilizing Video")

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        curr_gray_resized = cv2.resize(curr_gray, (new_w, new_h))
        u, v = faster_lucas_kanade_optical_flow(prev_gray_resized, 
                                               curr_gray_resized, 
                                               window_size, 
                                               max_iter, 
                                               num_levels)
        half_window = window_size // 2
        valid_region_u = u[half_window:-half_window, half_window:-half_window]
        valid_region_v = v[half_window:-half_window, half_window:-half_window]

        u_mean = np.mean(valid_region_u)
        v_mean = np.mean(valid_region_v)

        u_uniform = np.full((new_h, new_w), u_mean)
        v_uniform = np.full((new_h, new_w), v_mean)
        u_accum = u_accum + u_uniform
        v_accum = v_accum + v_uniform
        h, w = curr_gray.shape
        u_resized = cv2.resize(u_accum, (w, h))
        v_resized = cv2.resize(v_accum, (w, h))
        u_resized = u_resized * (w / new_w)
        v_resized = v_resized * (h / new_h)
        curr_gray_stabilized = warp_image(curr_gray, u_resized, v_resized)
        curr_gray_stabilized = np.clip(curr_gray_stabilized, 0, 255).astype(np.uint8)
        curr_frame_stabilized = cv2.cvtColor(curr_gray_stabilized, cv2.COLOR_GRAY2BGR)
        curr_frame_cropped = curr_frame_stabilized[start_rows:frame_height-end_rows, 
                                                  start_cols:frame_width-end_cols]
        out.write(curr_frame_cropped)
        prev_gray_resized = curr_gray_resized
        pbar.update(1)

    # Clean up
    pbar.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()


