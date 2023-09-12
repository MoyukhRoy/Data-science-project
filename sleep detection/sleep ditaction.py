

import cv2
import dlib
from scipy.spatial import distance
from imutils import face_utils
import playsound
import threading

def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of vertical eye landmarks
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])

    # Compute the euclidean distance between the horizontal eye landmark
    C = distance.euclidean(eye[0], eye[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def sound_alarm(path):
    # Play an alert sound to alert the driver of drowsiness
    playsound.playsound(path)

# Initialize dlib's face detector (HOG-based) and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/content/drive/MyDrive/shape_predictor_68_face_landmarks.dat")

def main():
    # Load the alarm sound (download a sound file and provide its path)
    alarm_sound_path = ""

    # Initialize variables for drowsiness detection
    EYE_AR_THRESHOLD = 0.3  # Adjust this threshold based on your requirements
    EYE_AR_CONSEC_FRAMES = 48
    COUNTER = 0
    ALARM_ON = False

    # Start video stream
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture a frame from the camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ... (rest of the code remains the same)
        # Your existing code for drowsiness detection

        # Display the eye aspect ratio on the frame
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow("Drowsiness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video stream and close the OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
