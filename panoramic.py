import cv2
import numpy as np
import matplotlib.pyplot as plt


def calculate_matches(image1, image2):
    # Load the images
    img1 = cv2.imread(image1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2, cv2.IMREAD_GRAYSCALE)

    # Initialize the SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors for both images
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # Initialize the FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Perform matching
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Apply Lowe's ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return good_matches, keypoints1, keypoints2


def dlt(matrix1, matrix2):
    """
    Get two sets of (x,y) coordinates from matches between images and calculate the homography by using SVD. return homography
    :param matrix1: coordinates of img1 from matches between images
    :param matrix2: coordinates of img2 from matches between images
    :return: homography 3X3 matrix
    """
    # matrix "a" as we learned - we will define values for each point
    a = np.zeros((8, 9))

    # run over all point and build "a" matrix
    for i in range(4):
        x1, y1 = matrix1[i]
        x2, y2 = matrix2[i]
        a[i * 2] = [0, 0, 0, -x1, -y1, -1, y2 * x1, y2 * y1, y2]
        a[i * 2 + 1] = [x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2]

    # svd on matrix "a"
    u, d, v = np.linalg.svd(a)

    # make sure that sigma_8 > 0 ( in svd d is diagonal and sigma_1 > sigma_2 > ... > sigma_8)
    if np.min(d) <= 0:
        print("ERROR: negative number in d")
    else:
        # homography equals to the last column of v matrix in svd (as we learned)
        h = v[-1]

        # return homography as 3x3 matrix
        h = h.reshape((3, 3))
        return h


def RANSAC(coordinates1, coordinates2, threshold, max_iterations=1000):
    """
    :param coordinates1: coordinates of matching points in image1
    :param coordinates2:coordinates of matching points in image2
    :param threshold: threshold for the distance
    :param max_iterations:(max_iteration times) until the "best homography" will be found (the one that gives maximal count of inliners)
    :return: best homography and number of inliners
    """
    best_inliners = 0
    best_homography = np.zeros((3, 3), dtype=np.float32)

    for i in range(max_iterations):
        print("iteration:",i)
        # choose 4 random points from each coordinate
        samples = np.random.choice(np.arange(len(coordinates1)), size=4, replace=False)
        new_coordinates1 = coordinates1[samples]
        new_coordinates2 = coordinates2[samples]

        # find the homography of the chosen points
        h = dlt(new_coordinates1, new_coordinates2)

        # apply homography on all points in image1
        after_homography = cv2.perspectiveTransform(coordinates1.reshape(-1, 1, 2), h)

        # find distance of points in image2 from image1 after homography using the norm
        distance = np.linalg.norm(after_homography.squeeze(axis=1) - coordinates2, axis=1)

        # find number of inliner using a known threshold
        inliners = np.count_nonzero(distance < threshold)

        # Update best homography and inlier
        if inliners > best_inliners:
            best_homography = h
            best_inliners = inliners

    return best_homography, best_inliners

# return homography matrix 3X3, inliers

def stitch_images(image1, image2, homography):
    # Get images height and width , turn them to 1X2 vectors
    h1, w1 = cv2.imread(image1).shape[:2]
    h2, w2 = cv2.imread(image2).shape[:2]
    corners1 = np.array([[0, 0], [0, h1], [w1, h1], [w1, 0]], dtype=np.float32).reshape(-1, 1, 2)
    corners2 = np.array([[0, 0], [0, h2], [w2, h2], [w2, 0]], dtype=np.float32).reshape(-1, 1, 2)
    # Translate the homography to take under account the field of view of the second image
    corners2_transformed = cv2.perspectiveTransform(corners2, np.linalg.inv(homography).astype(np.float32))
    all_corners = np.concatenate((corners1, corners2_transformed), axis=0)
    x, y, w, h = cv2.boundingRect(all_corners)

    # Adjust the homography matrix to map from img2 to img1
    H_adjusted = np.linalg.inv(homography)

    # Warp the images
    img1_warped = cv2.warpPerspective(cv2.imread(image1), np.eye(3), (w, h))
    img2_warped = cv2.warpPerspective(cv2.imread(image2), H_adjusted, (w, h))

    # Combine the warped images into a single output image
    output = cv2.addWeighted(img1_warped, 0.5, img2_warped, 0.5, 0)
    plt.imshow(img1_warped)
    plt.figure()
    plt.imshow(img2_warped)

    # Create a mask for the overlapping region
    mask1 = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask1, [np.int32(corners1)], (255))
    mask2 = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask2, [np.int32(corners2_transformed)], (255))
    overlap_mask = cv2.bitwise_and(mask1, mask2)
    not_overlap_img2_mask = cv2.bitwise_and(cv2.bitwise_not(overlap_mask), mask2)

    # Blend only the overlapping region

    blended = cv2.addWeighted(img1_warped, 0.5, img2_warped, 0.5, 0)

    # Copy img1_warped and img2_warped to blended using the overlap_mask
    blended = cv2.bitwise_and(blended, blended, mask=overlap_mask)
    blended += cv2.bitwise_and(img1_warped, img1_warped, mask=cv2.bitwise_not(overlap_mask))
    blended += cv2.bitwise_and(img2_warped, img2_warped, mask=not_overlap_img2_mask)

    plt.figure()
    plt.imshow(blended, cmap='gray')
    cv2.imwrite('panoramic_image.jpg', blended)


# Main
if __name__ == '__main__':

    image1 = 'pic/Hanging1.png'
    image2 = 'pic/Hanging2.png'

    # Calculate good matches between the images and obtain keypoints
    matches, keypoints1, keypoints2 = calculate_matches(image1, image2)

    # Extract coordinates of the keypoints
    coordinates1 = np.float32([keypoints1[match.queryIdx].pt for match in matches])
    coordinates2 = np.float32([keypoints2[match.trainIdx].pt for match in matches])

    # RANSAC to find the best homography
    homography, inliers = RANSAC(coordinates1, coordinates2, threshold=1)
    #homography, inliers = RANSAC(coordinates1, coordinates2, threshold=10000)

    # Stitch the images together
    stitch_images(image1, image2, homography)
