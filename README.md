
# 🎥 Face Tracker Video using OpenCV and face_recognition

This project tracks and uniquely identifies human faces from a CCTV video using Python, OpenCV, and the `face_recognition` library.

---

## 🚀 Features

- Detects and tracks faces in real-time  
- Assigns unique ID to every face  
- Annotates and saves the output video  
- Supports HOG-based face detection  
- Stores results in a specified output directory  

---

## 📦 Requirements

- Python 3.8 – 3.10 (recommended)  
- `face_recognition`  
- `opencv-python`  
- `dlib`  
- `numpy`  
- `Pillow`  

---

## 🧑‍💻 Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/face_tracker_video.git
   cd face_tracker_video
2. Create and activate a virtual environment
   
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
3. Install dependencies
   ```bash
   pip install -r requirements.txt
4. (Optional) Manually install face_recognition_models if needed
   ```bash
   pip install git+https://github.com/ageitgey/face_recognition_models
5. (Linux only) Fix OpenCV GUI issues (if cv2.imshow fails)
   ```bash
   sudo apt install libgtk2.0-dev pkg-config
   pip uninstall opencv-python
   pip install opencv-python-headless
6.📂 Usage
   Put your input video file (e.g., cctv.mp4) in the project directory.

   Run the tracker:
   bash
   python face_counter.py cctv.mp4 --output results_dir

Press q to quit early

Output will be saved as annotated_output.avi in results_dir

🧠 How It Works
Loads video frame-by-frame

Detects and encodes faces

Compares new encodings with stored ones using cosine similarity

Assigns and tracks unique IDs

Saves annotated video with tracking overlays

🗂 Directory Structure
bash
Copy
Edit
face_tracker_video/
│
├── face_counter.py        # Main script
├── requirements.txt       # Dependencies
├── cctv.mp4               # Example input video
├── results_dir/           # Output folder
└── README.md              # This file
📄 License
This project is licensed under the MIT License.

👨‍💻 Author
Vansh – https://github.com/Vansh-Raj

📬 Contact
For feedback or queries, feel free to open an issue or reach out!
