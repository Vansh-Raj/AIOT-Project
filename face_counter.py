import cv2
import os
import argparse
import face_recognition
import numpy as np

# -------------------------------
# Module 1: Pre-processing
# -------------------------------
def preprocess_frame(frame):
    """Convert BGR to RGB and return."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def load_video(video_path):
    """Load video and return VideoCapture object."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    return cap

# -------------------------------
# Module 2: Model Building
# -------------------------------
class FaceTracker:
    def __init__(self):
        self.known_encodings = []
        self.face_ids = []
        self.next_id = 0

    def find_or_add_face(self, encoding):
        """Assign a unique ID to a new face or return an existing one."""
        if not self.known_encodings:
            self.known_encodings.append(encoding)
            self.face_ids.append(self.next_id)
            self.next_id += 1
            return self.face_ids[-1]

        distances = face_recognition.face_distance(self.known_encodings, encoding)
        best_match_index = np.argmin(distances)

        if distances[best_match_index] < 0.5:
            return self.face_ids[best_match_index]
        else:
            self.known_encodings.append(encoding)
            self.face_ids.append(self.next_id)
            self.next_id += 1
            return self.face_ids[-1]

# -------------------------------
# Optional Module 3: Optimization
# -------------------------------
# (Basic optimization via caching + distance thresholding already built-in)

# -------------------------------
# Main Processing Function
# -------------------------------
def process_video(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = load_video(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

    out_path = os.path.join(output_dir, 'annotated_output.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    tracker = FaceTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = preprocess_frame(frame)
        face_locations = face_recognition.face_locations(rgb_frame, model='hog')
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            face_id = tracker.find_or_add_face(face_encoding)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {face_id}", (left, top - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Total unique faces detected: {tracker.next_id}")

# -------------------------------
# Entry Point
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("--output", default="results", help="Directory to save output video")
    args = parser.parse_args()

    process_video(args.video_path, args.output)
