import cv2
import numpy as np

def stitch_images(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)  

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:  
            good.append(m)

    MIN_MATCH_COUNT = 4 

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)


        height1, width1 = img1.shape[:2]
        height2, width2 = img2.shape[:2]
        img1_dims = np.float32([[0, 0], [0, height1], [width1, height1], [width1, 0]]).reshape(-1, 1, 2)
        img2_dims = np.float32([[0, 0], [0, height2], [width2, height2], [width2, 0]]).reshape(-1, 1, 2)
        img2_transformed_dims = cv2.perspectiveTransform(img2_dims, H)
        combined_dims = np.concatenate((img1_dims, img2_transformed_dims), axis=0)

        [x_min, y_min] = np.int32(combined_dims.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(combined_dims.max(axis=0).ravel() + 0.5)
        translation_dist = [-x_min, -y_min]
        H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

        output_img = cv2.warpPerspective(img1, H_translation.dot(H), (x_max-x_min, y_max-y_min))
        output_img[translation_dist[1]:translation_dist[1] + height2, translation_dist[0]:translation_dist[0] + width2] = img2

        return output_img
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        return None

image1 = cv2.imread('img/a3.JPG')
image2 = cv2.imread('img/b3.JPG')

result = stitch_images(image1, image2)

if result is not None:
    cv2.imshow('Stitched Image', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('result.jpg', result)
else:
    print("Image stitching failed.")
