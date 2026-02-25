import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import map_coordinates
from utils import *



def convIx(im):
    """
    Convolves on the input image - horizontally, along the X axis, simulating partial derivatives w.r.t X.
    """
    K = np.array([[1, 0, -1]])
    return convolve2d(im, K, mode='same', boundary='symm')

def convIy(im):
    """
    Convolves on the input image - vertically, along the Y axis, simulating partial derivatives w.r.t Y.
    """
    K = np.array([[1], [0], [-1]])
    return convolve2d(im, K, mode='same', boundary='symm')


def harris_corner_detector(im):
    """
    Implements the harris corner detection algorithm.
    :param im: A 2D array representing a grayscale image.
    :return: An array with shape (N,2), where its ith entry is the [x,y] coordinates of the ith corner point.
    """
    # Obtaining image derivatives in both X & Y axes using convolution with the appropriate kernels
    Ix = convIx(im)
    Iy = convIy(im)
    IxIy = blur_spatial(Ix * Iy,3)
    IyIx = blur_spatial(Iy * Ix, 3)
    Ix2 = blur_spatial(Ix * Ix, 3)
    Iy2 = blur_spatial(Iy * Iy, 3)

    corner_out = np.zeros((im.shape[0],im.shape[1]))
    # Specify alpha parameter
    alpha = 0.04
    detM = Ix2 * Iy2 - IxIy ** 2
    traceM = Ix2 + Iy2
    # Harris corner detector metric on image
    corner_out = detM - alpha * traceM ** 2
    # Non-Max Suppression on output image
    corner_out = non_maximum_suppression(corner_out)
    return np.argwhere(corner_out == 1)[:, ::-1]


def feature_descriptor(im, points, desc_rad=3):
    """
    Samples descriptors at the given feature points.
    :param im: A 2D array representing a grayscale image.
    :param points: An array with shape (N,2) representing feature points coordinates in the image.
    :param desc_rad: "Radius" of descriptors to compute.
    :return: An array of 2D patches, each patch i representing the descriptor of point i.
    """
    ret_arr_patches = []
    for i in range(len(points)):
        y = points[i,1]
        x = points[i,0]

        yy, xx = np.mgrid[y - desc_rad:y + desc_rad + 1, x-desc_rad:x + desc_rad + 1]

        patch = map_coordinates(im, [yy, xx], mode='constant', cval=0)

        patch = patch - np.mean(patch)

        norm = np.linalg.norm(patch)

        # Prevent division by zero
        if norm > 0:
            patch = patch / norm
        else:
            patch = np.zeros_like(patch)
        ret_arr_patches.append(patch)
    return ret_arr_patches


def find_features(im):
    """
    Detects and extracts feature points from a specific pyramid level.
    :param im: A 2D array representing a grayscale image.
    :return: A list containing:
             1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                These coordinates are provided at the original image level.
            2) A feature descriptor array with shape (N,K,K)
    """
    pyrs = build_gaussian_pyramid(im, 3, 3)
    pyr3 = pyrs[2]
    cor = spread_out_corners(im, m = 7, n = 7, radius = 12, harris_corner_detector = harris_corner_detector)
    fet_des = feature_descriptor(pyr3, cor/4, 3)
    return [cor, fet_des]


def match_features(desc1, desc2, min_score):
    """
    Return indices of matching descriptors.
    :param desc1: A feature descriptor array with shape (N1,K,K).
    :param desc2: A feature descriptor array with shape (N2,K,K).
    :param min_score: Minimal match score.
    :return: A list containing:
                1) An array with shape (M,) and dtype int of matching indices in desc1.
                2) An array with shape (M,) and dtype int of matching indices in desc2.
    """
    # Flatten descriptors to shape (N, K*K)
    D1 = np.array([d.flatten() for d in desc1])  # shape: (N1, K*K)
    D2 = np.array([d.flatten() for d in desc2])  # shape: (N2, K*K)
    # Compute feature descriptors' dot product
    scores = D1 @ D2.T
    # Absolute thresholding
    mask_score = scores > min_score
    
    # Find top-2 matches per row
    row_top2 = np.argpartition(scores, -2, axis=1)[:, -2:]
    mask_row = np.zeros_like(scores, dtype=bool)
    np.put_along_axis(mask_row, row_top2, True, axis=1)

    # Find top-2 matches per column
    col_top2 = np.argpartition(scores, -2, axis=0)[-2:, :]
    mask_col = np.zeros_like(scores, dtype=bool)
    np.put_along_axis(mask_col, col_top2, True, axis=0)

    # Composite mask for matches that are top-2 in both their row and column, and above the minimum score threshold
    final_mask = mask_score & mask_row & mask_col
    # Identify & extract indices of matches from the final mask
    idx1, idx2 = np.where(final_mask)
    return [idx1.astype(int), idx2.astype(int)]
  

def apply_homography(pos1, H12):
    """
    Apply homography to inhomogenous points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
    """
    N = pos1.shape[0]

    # convert to homogeneous coordinates (N,3)
    homog = np.hstack([pos1, np.ones((N, 1))])

    # apply homography
    warped = (H12 @ homog.T).T   # (N,3)

    # normalize & avoid division by zero
    warped /= (warped[:, 2:3] + 1e-10)
    # return projected points
    return warped[:, :2]


def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only = False):
    """
    Computes homography between two sets of points using RANSAC.
    :param points1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
    :param points2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
    :param num_iter: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :param translation_only: see estimate rigid transform
    :return: A list containing:
                1) A 3x3 normalized homography matrix.
                2) An Array with shape (S,) where S is the number of inliers,
                    containing the indices in pos1/pos2 of the maximal set of inlier matches found.
    """
    max_inline = 0
    idx = None
    # Handle degenerate case of too few points to estimate homography
    if points1.shape[0] < 2 or points2.shape[0] != points1.shape[0]:
      return [np.eye(3), np.array([])]
    for _ in range(num_iter):
        # Sample 2 indices from the matched points
        first_point_rand = np.random.choice(len(points1))
        sec_point_rand = np.random.choice(len(points1))
        # Ensure indices are different to avoid degenerate homography estimation
        while(sec_point_rand == first_point_rand):
            sec_point_rand = np.random.choice(len(points1))
        # Using sampled indices, estimate homography
        hom_mat = estimate_rigid_transform(points1[[first_point_rand,sec_point_rand]],points2[[first_point_rand,sec_point_rand]])
        # Apply estimated homography on all points1
        trans_po = apply_homography(points1, hom_mat)
        # Compute distance (euclidean) between transformed points1 and points2
        dis_calc = np.linalg.norm(trans_po - points2, axis=1)
        # Compare distance to threshold using mask - inliers are points with distance below threshold
        temp_in_count = sum(dis_calc<inlier_tol)
        # Check for max inliers and save if found
        if temp_in_count > max_inline:
            # Save indices of max inliers & their total count
            idx = np.where(dis_calc < inlier_tol)[0]
            max_inline = temp_in_count
    if idx is None: # Handling degenerate case of no inliers found:
      return [np.eye(3), np.array([])]
    # Recomputing homography using all inliers, returning it along with inliers' indices
    return [estimate_rigid_transform(points1[idx], points2[idx]), idx]


def display_matches(im1, im2, points1, points2, inliers):
    """
    Display matching points.
    :param im1: A grayscale image.
    :param im2: A grayscale image.
    :param points1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
    :param points2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
    :param inliers: An array with shape (S,) of inlier matches.
    """
    H, W = im1.shape

    # concatenate images horizontally
    canvas = np.hstack([im1, im2])

    plt.figure(figsize=(12, 6))
    plt.imshow(canvas, cmap='gray')
    plt.axis('off')

    inliers = set(inliers)

    for i in range(points1.shape[0]):
        x1, y1 = points1[i]
        x2, y2 = points2[i]

        # shift x coordinate of second image
        x2_shifted = x2 + W

        # choose color for inliers & outliers accordingly
        if i in inliers:
            color = 'b'  # inlier → blue
        else:
            color = 'y'  # outlier → yellow

        # draw line
        plt.plot([x1, x2_shifted], [y1, y2], color=color, linewidth=0.5)

        # draw points
        plt.scatter([x1, x2_shifted], [y1, y2], c='r', s=10)

    plt.show()


def accumulate_homographies(H_successive, m):
    """
    Convert a list of successive homographies to a list of homographies to a common reference frame.
    :param H_successive: A list of M-1 3x3 homography
      matrices where H_successive[i] is a homography which transforms points
      from coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system towards which we would like to
      accumulate the given homographies.
    :return: A list of M 3x3 homography matrices,
      where H2m[i] transforms points from coordinate system i to coordinate system m
    """
    H2m_right = []
    till_now = np.eye(3)
    # RIGHT of m (backward accumulation - product of INVERSE homographies)
    for i in range(m,len(H_successive)):
        till_now = till_now @ np.linalg.inv(H_successive[i])
        # Normalize homographies w.r.t the last element
        till_now = till_now / till_now[2,2]
        H2m_right.append(till_now)
    
    # LEFT of m (forward accumulation - product of homographies)
    H2m_left = []
    till_now = np.eye(3)
    for i in range(m-1, -1, -1):
        till_now = H_successive[i] @ till_now
        # Normalize homographies w.r.t the last element
        till_now = till_now / till_now[2,2]
        H2m_left.append(till_now)
    H2m_left = H2m_left[::-1] # reverse order of left homographies to match coordinate systems order
    return H2m_left + [np.eye(3)] + H2m_right


def compute_bounding_box(homography, w, h):
    """
    computes bounding box of warped image under homography, without actually warping the image
    :param homography: homography
    :param w: width of the image
    :param h: height of the image
    :return: 2x2 array, where the first row is [x,y] of the top left corner,
     and the second row is the [x,y] of the bottom right corner
    """
    # Initializing image corners:
    corners = np.array([[0, 0],[0, h - 1],[w - 1,0], [w - 1, h - 1]])
    # Applying the homography on the corners of the source image -> destination image corners, thereby warping the image
    cor = apply_homography(corners, homography)
    # Computing the bounding box of the warped image by taking the min and max of the warped corners
    xmin, ymin = np.floor(np.min(cor, axis = 0))
    xmax, ymax = np.ceil(np.max(cor, axis = 0))
    # return rectangle in panorama space containing the warped image
    return np.array([[xmin, ymin], [xmax, ymax]])


def warp_channel(image, homography):
    """
    Warps a 2D image with a given homography.
    :param image: a 2D image.
    :param homography: homography.
    :return: A 2D warped image.
    """
    # Bounding box coordinates that fully cover the warped source image, in the coordinate system of the panorama destination image
    bound = compute_bounding_box(homography,image.shape[1],image.shape[0])
    # Integer grid limit pixels that cover the bounding box
    x0, y0 = np.floor(bound[0]).astype(int)
    x1, y1 = np.ceil(bound[1]).astype(int)

    yy, xx = np.mgrid[y0:y1, x0:x1]
    # Stack destination pixel coordinates into (N,2) points in (x,y) order
    points = np.stack([xx.ravel(), yy.ravel()], axis = 1)
    # Backward mapping: destination_panorama -> source_image using inverse homography
    points_src = apply_homography(points, np.linalg.inv(homography))
    # Interpolate source image at these continuous coordinates
    patch = map_coordinates(image, [points_src[:, 1], points_src[:, 0]], mode='constant', cval=0)
    # Reshape sampled values back to the destination grid shape
    return patch.reshape(yy.shape)


def warp_image(image, homography):
    """
    Warps an RGB image with a given homography.
    :param image: an RGB image.
    :param homography: homograhpy.
    :return: A warped image.
    """
    # Warp each channel separately and stack them back together, return result:
    return np.clip(np.stack([warp_channel(image[:, :, 0], homography), warp_channel(image[:, :, 1], homography), warp_channel(image[:, :, 2], homography)], axis = 2), 0, 1)



##################################################################################################


def align_images(files, translation_only=False):
    """
    compute homographies between all images to a common coordinate system
    :param translation_only: see estimte_rigid_transform
    """
    # Extract feature point locations and descriptors.
    points_and_descriptors = []
    for file in files:
        image = read_image(file, 1)
        points_and_descriptors.append(find_features(image))

    # Compute homographies between successive pairs of images.
    Hs = []
    for i in range(len(points_and_descriptors) - 1):
        points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
        desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

        # Find matching feature points.
        ind1, ind2 = match_features(desc1, desc2, .7)
        points1, points2 = points1[ind1, :], points2[ind2, :]

        # Compute homography using RANSAC.
        H12, inliers = ransac_homography(points1, points2, 100, 6, translation_only)

        Hs.append(H12)

    # Compute composite homographies from the central coordinate system.
    accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
    homographies = np.stack(accumulated_homographies)
    frames_for_panoramas = filter_homographies_with_translation(homographies, minimum_right_translation=5)
    homographies = homographies[frames_for_panoramas]
    return frames_for_panoramas, homographies


def generate_panoramic_images(data_dir, file_prefix, num_images, out_dir, number_of_panoramas, translation_only=False):
    """
    combine slices from input images to panoramas.
    The naming convention for a sequence of images is file_prefixN.jpg, where N is a running number 001, 002, 003...
    :param data_dir: path to input images.
    :param file_prefix: see above.
    :param num_images: number of images to produce the panoramas with.
    :param out_dir: path to output panoramas.
    :param number_of_panoramas: how many different slices to take from each input image
    """

    file_prefix = file_prefix
    files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
    files = list(filter(os.path.exists, files))
    print('found %d images' % len(files))
    image = read_image(files[0], 1)
    h, w = image.shape

    frames_for_panoramas, homographies = align_images(files, translation_only)

    # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
    bounding_boxes = np.zeros((frames_for_panoramas.size, 2, 2))
    for i in range(frames_for_panoramas.size):
        bounding_boxes[i] = compute_bounding_box(homographies[i], w, h)

    # change our reference coordinate system to the panoramas
    # all panoramas share the same coordinate system
    global_offset = np.min(bounding_boxes, axis=(0, 1))
    bounding_boxes -= global_offset

    slice_centers = np.linspace(0, w, number_of_panoramas + 2, endpoint=True, dtype=np.int32)[1:-1]
    warped_slice_centers = np.zeros((number_of_panoramas, frames_for_panoramas.size))
    # every slice is a different panorama, it indicates the slices of the input images from which the panorama
    # will be concatenated
    for i in range(slice_centers.size):
        slice_center_2d = np.array([slice_centers[i], h // 2])[None, :]
        # homography warps the slice center to the coordinate system of the middle image
        warped_centers = [apply_homography(slice_center_2d, h) for h in homographies]
        # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
        warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

    panorama_size = np.max(bounding_boxes, axis=(0, 1)).astype(np.int32) + 1

    # boundary between input images in the panorama
    x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
    x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                  x_strip_boundary,
                                  np.ones((number_of_panoramas, 1)) * panorama_size[0]])
    x_strip_boundary = x_strip_boundary.round().astype(np.int32)

    panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
    for i, frame_index in enumerate(frames_for_panoramas):
        # warp every input image once, and populate all panoramas
        image = read_image(files[frame_index], 2)
        warped_image = warp_image(image, homographies[i])
        x_offset, y_offset = bounding_boxes[i][0].astype(np.int32)
        y_bottom = y_offset + warped_image.shape[0]

        for panorama_index in range(number_of_panoramas):
            # take strip of warped image and paste to current panorama
            boundaries = x_strip_boundary[panorama_index, i:i + 2]
            image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
            x_end = boundaries[0] + image_strip.shape[1]
            panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

    os.makedirs(out_dir, exist_ok=True)
    for i, panorama in enumerate(panoramas):
        plt.imsave('%s/panorama%02d.png' % (out_dir, i + 1), panorama)


if __name__ == "__main__":
    import ffmpeg
    video_name = "mt_cook.mp4"
    video_name_base = video_name.split('.')[0]
    os.makedirs(f"dump/{video_name_base}", exist_ok=True)
    ffmpeg.input(f"videos/{video_name}").output(f"dump/{video_name_base}/{video_name_base}%03d.jpg").run()
    num_images = len(os.listdir(f"dump/{video_name_base}"))
    print(f"Generated {num_images} images")

    # Visualize feature points on two sample images
    print("Extracting and visualizing feature points...")
    image1 = read_image(f"dump/{video_name_base}/{video_name_base}001.jpg", 1)
    image2 = read_image(f"dump/{video_name_base}/{video_name_base}002.jpg", 1)

    # Extract feature points and descriptors
    points1, desc1 = find_features(image1)
    points2, desc2 = find_features(image2)

    # Visualize points on first image
    print(f"Found {len(points1)} feature points in image 1")
    visualize_points(image1, points1)

    # Visualize points on second image
    print(f"Found {len(points2)} feature points in image 2")
    visualize_points(image2, points2)

    # Match features between the two images
    print("Matching features between images...")
    ind1, ind2 = match_features(desc1, desc2, 0.9)
    matched_points1 = points1[ind1]
    matched_points2 = points2[ind2]
    print(f"Found {len(ind1)} matches")
    visualize_points(image1, matched_points1)
    visualize_points(image2, matched_points2)

    # Run RANSAC to find inliers
    H12, inliers = ransac_homography(matched_points1, matched_points2, 100, 6, translation_only=False)
    print(f"Found {len(inliers)} inliers out of {len(matched_points1)} matches")

    # Display matches with inliers and outliers
    display_matches(image1, image2, matched_points1, matched_points2, inliers)

    # Generate panoramic images
    print("\nGenerating panoramic images...")
    generate_panoramic_images(f"dump/{video_name_base}/", video_name_base,
                              num_images=num_images, out_dir=f"out/{video_name_base}", number_of_panoramas=3)

