# Camera-Based Conveyor Inspection & Tracking

A modular computer-vision project for detecting and tracking boxes in conveyor footage. The system estimates object dimensions and motion, and classifies boxes using labelled image crops. It is implemented with classical computer-vision techniques and does not rely on pretrained detection models.

## Project Objectives
- Detect box regions in video frames using image segmentation.
- Track detected objects across frames, including short detection gaps.
- Estimate object size and speed using camera calibration and optical flow.
- Classify objects into box types using image-based features.
- Combine the modules in one pipeline and export results for inspection.

## Modules
| Module | Main approach | Result |
|---|---|---|
| A - Segmentation | Compares snake, split/merge, watershed, Felzenszwalb, mean shift, and normalized-cut methods. | Segmentation comparison grid and box detections. |
| B - Calibration | Uses checkerboard images with `cv2.calibrateCamera`; estimates the belt plane and supports box-size measurement. | Camera parameters, calibration data, and visualisation. |
| C - Optical flow | Implements Lucas-Kanade flow and fits an affine motion model with RANSAC. | Motion visualisation and speed estimate. |
| D - Tracking | Uses a constant-velocity Kalman filter and Hungarian assignment for multi-object association. | Track plot and annotated tracking video. |
| E - Recognition | Compares ORB alignment, PCA eigenboxes, and Hu-moment descriptors. | Recognition metrics and eigenbox visualisation. |

The integrated `pipeline.py` connects detection, tracking, calibration, motion estimation, and recognition. It reports per-track measurements and saves an annotated video and CSV summary.

## Data
Input data is not included. Provide data in the following layout:
```text
data/
	video.mp4
	calib/                 # Checkerboard calibration images
	boxes/
		typeA/               # Labelled crop images for each box type
		typeB/
		typeC/
```

Use a fixed camera and include some frames where the belt is visible without boxes. Calibration images should use the same camera resolution as the video; one checkerboard image should show the board flat on the belt. For recognition, create one folder per class and add representative cropped images. The included `E_recognition/collect_crops.py` can help collect crops from video.

To combine source clips or image frames stored in `data/raw/`:
```bash
python data/prepare_video.py --source-dir data/raw --out data/video.mp4
```

For a code-only smoke test, `python make_test_data.py` generates a synthetic dataset. Synthetic data is useful for checking the pipeline, but is not evidence of real-world accuracy.

## Installation
Python 3.10 or newer is recommended. Install dependencies from the repository root:
```bash
python -m pip install -r requirements.txt
```

## Running the Project
Run the complete pipeline:
```bash
python pipeline.py --video data/video.mp4 --box_h 4.0
```

Run a module separately for development or inspection:
```bash
python A_segmentation/segmentation.py --video data/video.mp4
python B_calibration/calibrate.py --images "data/calib/*.jpg" --cols 9 --rows 6 --square_cm 2.5 --belt_idx 0
python C_optical_flow/optical_flow.py --video data/video.mp4 --frame 150 --box_idx 0
python D_tracking/kalman_tracker.py --video data/video.mp4
python E_recognition/recognize.py
```

Set `--cols` and `--rows` to the checkerboard's inner-corner count, and `--square_cm` to the physical square size. Set `--box_h` to the box-top height above the belt.

## Outputs
Generated results are written to `output/`, including:
- `pipeline_output.mp4` - annotated pipeline example video (synthetic validation data).
- `pipeline_results.csv` - per-track size, speed, and predicted class.
- Module plots and visualisations, such as segmentation, calibration, optical-flow, tracking, and recognition results.

See [`output/README.md`](output/README.md) for the source of each checked-in result and the real-video versus synthetic-validation distinction.

## Scope and Limitations
- Results depend on camera stability, lighting, calibration quality, and representative training crops.
- Perfectly touching boxes without a visible boundary may be detected as one object.
- Synthetic data validates code execution only; evaluate accuracy using real footage and real calibration data.
