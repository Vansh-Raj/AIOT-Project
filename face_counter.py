import cv2
import os
import argparse
import face_recognition
import numpy as np

class FaceTracker:
    def __init__(self):
        self.known_encodings = []
        self.face_ids = []
        self.next_id = 0

    def find_or_add_face(self, encoding):
        if len(self.known_encodings) == 0:
            self.known_encodings.append(encoding)
            self.face_ids.append(self.next_id)
            self.next_id += 1
            return self.face_ids[-1]

        # Use face distance and find the best match
        distances = face_recognition.face_distance(self.known_encodings, encoding)
        best_match_index = np.argmin(distances)
        if distances[best_match_index] < 0.5:
            return self.face_ids[best_match_index]
        else:
            self.known_encodings.append(encoding)
            self.face_ids.append(self.next_id)
            self.next_id += 1
            return self.face_ids[-1]

def process_video(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_path = os.path.join(output_dir, 'annotated_output.avi')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    tracker = FaceTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model='hog')
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            face_id = tracker.find_or_add_face(face_encoding)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {face_id}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Total unique faces detected: {tracker.next_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("--output", default="results", help="Directory to save output video")
    args = parser.parse_args()

    process_video(args.video_path, args.output)
