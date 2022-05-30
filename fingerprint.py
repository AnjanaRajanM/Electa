import cv2
import numpy as np
import os

test_original = cv2.imread("SOCOFing\Anju12.BMP ")
cv2.imshow("Original", cv2.resize(test_original, None, fx=1, fy=1))

#test_original = cv2.imread("SOCOFing\Real/try.BMP ")
#600__M_Right_middle_finger

best_score = 0
filename = None
image = None
kp1, kp2, mp = None, None, None

for file in [file for file in os.listdir("SOCOFing\Real")]:
    fingerprint_database_image = cv2.imread("SOCOFing\Real/" + file)

    sift = cv2.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(test_original, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_database_image, None)

    matches = cv2.FlannBasedMatcher(dict(algorithm=1, trees=10),
                                    dict()).knnMatch(descriptors_1, descriptors_2, k=2)

    match_points = []

    for p, q in matches:
        if p.distance < 0.1 * q.distance:
            match_points.append(p)

    keypoints = 0
    if len(keypoints_1) < len(keypoints_2):
        keypoints = len(keypoints_1)
    else:
        keypoints = len(keypoints_2)

    if (len(match_points) / keypoints)>0.95:            #if len(match_points) / keypoints * 100 > best_score:
        best_score = len(match_points) / keypoints * 100
        filename = file
        image = fingerprint_database_image
        kp1, kp2, mp = keypoints_1, keypoints_2, match_points



print("Best Match: " +filename[:-4])
print("Score " + str(best_score))

#result = cv2.drawMatches(test_original, kp1, image, kp2, mp, None)
#result = cv2.resize(result, None, fx=2.5, fy=2.5)

#cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
