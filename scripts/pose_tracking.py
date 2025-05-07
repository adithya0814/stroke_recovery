import cv2
import numpy as np
from cvzone.PoseModule import PoseDetector
import os

def collect_landmarks(save_path, max_frames=100):
    cap = cv2.VideoCapture(0)  # Start video capture
    detector = PoseDetector()
    frames = []

    while len(frames) < max_frames:
        success, img = cap.read()
        if not success:
            break

        img = detector.findPose(img)
        lmList, _ = detector.findPosition(img)

        if lmList:
            # Extract the 3D coordinates (x, y, z) of each keypoint
            landmarks = [lm[1:] for lm in lmList]  # Exclude the index
            frames.append(landmarks)

        cv2.imshow("Pose Tracking", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save the sequence of landmarks as a numpy array
    np.save(save_path, np.array(frames))  # Save landmarks sequence to file

# Example usage:
save_path = 'data/landmark_sequences/session_1.npy'
collect_landmarks(save_path)
