import cv2
import numpy as np
import os
import face_recognition

test_original = cv2.imread("Fingerprintdataset/a.BMP ")
cv2.imshow("Original", cv2.resize(test_original, None, fx=1, fy=1))

#test_original = cv2.imread("SOCOFing\Real/try.BMP ")
#600__M_Right_middle_finger

best_score = 0
filename = None
image = None
kp1, kp2, mp = None, None, None

for file in [file for file in os.listdir("Fingerprintdataset\Real")]:
    fingerprint_database_image = cv2.imread("Fingerprintdataset\Real/" + file)

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



print(filename[:-4])
print("Score " + str(best_score))



KNOWN_FACES_DIR = 'known_faces'

TOLERANCE = 0.5
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'hog'  # default: 'hog', other one can be 'cnn' - CUDA accelerated (if available) deep-learning pretrained model
video = cv2.VideoCapture(0)




known_faces = []
known_names = []

'''

    # Next we load every file of faces of known person
for filename in os.listdir(f'{KNOWN_FACES_DIR}'):
    image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{filename}')
    encoding = face_recognition.face_encodings(image)[0]

    known_faces.append(encoding)
    known_names.append(filename)
'''

image = face_recognition.load_image_file('known_faces/'+ filename[:-4]+'.jpg')
encoding = face_recognition.face_encodings(image)[0]

known_faces.append(encoding)
known_names.append(filename)


while True:

    # Load face
    ret,image = video.read()

    # This time we first grab face locations - we'll need them to draw boxes
    locations = face_recognition.face_locations(image, model=MODEL)

    # Now since we know loctions, we can pass them to face_encodings as second argument
    # Without that it will search for faces once again slowing down whole process
    encodings = face_recognition.face_encodings(image, locations)

    # But this time we assume that there might be more faces in an image - we can find faces of dirrerent people
    print(f', found {len(encodings)} face(s)')
    match = None
    for face_encoding, face_location in zip(encodings, locations):

        # We use compare_faces (but might use face_distance as well)
        # Returns array of True/False values in order of passed known_faces
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)

        # Since order is being preserved, we check if any face was found then grab index
        # then label (name) of first matching known face withing a tolerance

        if True in results:  # If at least one is true, get a name of first of found labels
            match = known_names[results.index(True)]
            print(f' - {match[:-4]} from {results}')
            break

    if match:
        break



