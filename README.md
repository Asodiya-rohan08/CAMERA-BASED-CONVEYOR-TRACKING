# Camera-Based Conveyor Inspection & Tracking

A classical computer-vision project that detects and tracks boxes in conveyor video, estimates their size and speed, and recognises box types. It uses OpenCV and does not require pretrained AI models.

## Modules
- **A - Segmentation:** compares segmentation methods and detects boxes.
- **B - Calibration:** calibrates the camera and estimates box dimensions.
- **C - Optical flow:** estimates object motion and speed.
- **D - Tracking:** follows boxes with a Kalman filter.
- **E - Recognition:** classifies boxes from labelled image crops.
- **Pipeline:** combines the modules and saves annotated video and results.

## Setup
Use Python 3.10 or newer. From the project folder, install dependencies:
```bash
python -m pip install -r requirements.txt
```

## Input data
Add your own files; video and sample datasets are not included in this repository.
- Conveyor video: `data/video.mp4`
- Checkerboard photos: `data/calib/`
- Labelled box crops: `data/boxes/<type>/`

To create one video from clips or image frames in `data/raw/`:
```bash
python data/prepare_video.py --source-dir data/raw --out data/video.mp4
```

You can generate synthetic data for a code-only test with `python make_test_data.py`. Synthetic results are not real-world measurements.

## Run
Run the complete project from the repository root:
```bash
python pipeline.py --video data/video.mp4 --box_h 4.0
```

Run individual modules when needed:
```bash
python A_segmentation/segmentation.py --video data/video.mp4
python B_calibration/calibrate.py --images "data/calib/*.jpg" --cols 9 --rows 6 --square_cm 2.5 --belt_idx 0
python C_optical_flow/optical_flow.py --video data/video.mp4 --frame 150 --box_idx 0
python D_tracking/kalman_tracker.py --video data/video.mp4
python E_recognition/recognize.py
```

For calibration, set `--cols`, `--rows`, and `--square_cm` to match your checkerboard. Set `--box_h` to the height of the box above the belt.

## Outputs
Generated files are saved in `output/`, including the annotated pipeline video, a CSV of tracked boxes, and module visualisations.

## Notes
- Keep the camera fixed and use consistent lighting.
- Calibration photos and video should have the same resolution.
- Results depend on the quality of the video, calibration, and labelled box crops.
