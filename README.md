# Gesture Image Viewer

Gesture‑controlled image viewer that uses **MediaPipe Hand Landmarker** to detect pinch gestures and zoom an image in real time.

## Demo Video

[![demoVideo](https://img.youtube.com/vi/oZMJVzAkSbs/0.jpg)](https://www.youtube.com/watch?v=oZMJVzAkSbs)  
**Click the image to see Demo video!!**

## Overview

`gestureImageViewer.py` opens a webcam stream, tracks up to two hands, and lets you scale an on‑screen image simply by pinching your thumb and index finger together. A logarithmic mapping makes the zoom feel natural and responsive.

## Features

* **Real‑time hand tracking** with MediaPipe Tasks
* Detects both **single‑hand** and **two‑hand** pinch gestures
* **Smooth, logarithmic zoom** proportional to the distance change between hands
* Landmark overlay window for visual debugging
* Easy‑to‑tweak thresholds and paths at the top of the script

## Requirements

| Package         | Version      |
| --------------- | ------------ |
| Python          | 3.10 or newer|
| OpenCV‑Python   | >= 4.8       |
| MediaPipe       | >= 0.10      |
| NumPy           | latest       |

Install everything with:

```bash
pip install mediapipe opencv-python numpy
```

## Getting Started

1. Place the image you want to view in `media/` (default is `baboon.png`).
2. Download `hand_landmarker.task` from the MediaPipe release page and put it in `mediapipeModel/`.
3. Make sure your webcam is connected (or change `video_path`).
4. Run:

```bash
python gestureImageViewer.py
```

The script will open two windows:

* **LIVE\_STREAM video** – your webcam feed with hand landmarks drawn.
* **test** – the image that zooms in/out as you pinch.

## Controls

| Action                | Gesture                                                |
| --------------------- | ------------------------------------------------------ |
| Start zoom (two‑hand) | Pinch thumb & index fingertips on **both** hands       |
| Zoom in/out           | Move hands apart / together while pinched              |
| End zoom              | Release pinch on either hand                           |
| Single‑hand zoom      | Pinch with one hand; move hand toward/away from camera |

Press **Esc** to quit, or **Space** to pause/resume the video.

## How It Works

1. **Hand Tracking** – MediaPipe extracts 21 landmarks per hand at each frame.

2. **Pinch Detection** – `is_pinch()` measures the normalised thumb‑to‑index distance and compares it with `pinch_threshold` (0.8–1.0 works well).

3. **Scaling** – `scale_img_w_diag()` computes the diagonal distance between the two pinched hands at the start (`l_i`) and current frame (`l_c`) and applies:

   ```python
   scale_factor = math.log(l_c / l_i, 2)
   ```

4. **Rendering** – `draw_landmarks_on_image()` overlays landmarks, and OpenCV shows both the annotated video and the scaled image.

## Customisation

| Parameter         | Location      | Purpose                    |
| ----------------- | ------------- | -------------------------- |
| `image_path`      | top of script | Source image file          |
| `model_path`      | top of script | MediaPipe model location   |
| `video_path`      | top of script | Webcam index or video file |
| `pinch_threshold` | main loop     | Pinch sensitivity          |
| `INTER_NEAREST`   | `cv.resize`   | Interpolation method       |

## Project Structure

```text
project/
├── gestureImageViewer.py
├── mediapipeModel/
│   └── hand_landmarker.task
└── media/
    └── baboon.png
```

## Troubleshooting

* **Laggy video** – Lower your webcam resolution or comment out landmark drawing.
* **False pinches** – Increase `pinch_threshold` slightly.
* **No hands detected** – Check lighting; MediaPipe needs clear contrast to pick up fingers.

## License

The code is licensed under the GNU AGPL v3.0.

For more details, see the `LICENSE` file or visit [GNU AGPL v3.0 License](https://www.gnu.org/licenses/agpl-3.0.html)
