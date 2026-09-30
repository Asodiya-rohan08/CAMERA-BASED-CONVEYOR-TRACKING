# Camera-Based Conveyor Inspection & Tracking (classical CV)

Python 3.10+, `pip install -r requirements.txt`. All scripts are run **from the project root**.

## Data you must provide (real footage)
```
data/video.mp4          your conveyor / table video (fixed camera, boxes moving, empty belt visible at some moments)
data/calib/*.jpg        10-15 checkerboard photos (one of them lying FLAT on the belt, camera as when filming)
data/boxes/<type>/*.png >=3 box types, ~10 crops each (helper: E_recognition/collect_crops.py)
```
`python make_test_data.py` makes a **synthetic** set only to check the code runs (then run `B_calibration/calibrate.py` on it to get a calib.npz).
**Never report synthetic results as real-world results.** Replace generated media, calibration images, box crops, and `B_calibration/calib.npz` with your own data before evaluating the system.

## Prepare video from source clips
Put source video clips or ordered image frames in `data/raw/`, then combine them into the input video and extract a few preview frames:
```bash
python data/prepare_video.py --source-dir data/raw --out data/video.mp4 --extract-frames 5
```
For one existing video, pass it directly with `--source path/to/clip.mp4`. The prepared video is written to `data/video.mp4` and preview frames go to `output/representative_frames/`.

## How to run
| Module | Command |
|---|---|
| A Segmentation | `python A_segmentation/segmentation.py --video data/video.mp4` -> `output/segmentation_grid.png` |
| B Calibration | `python B_calibration/calibrate.py --images "data/calib/*.jpg" --cols 9 --rows 6 --square_cm 2.5 --belt_idx 0` then `python B_calibration/measure_box.py --frame 100 --ref_idx 0 --unk_idx 1 --ref_cm 8.3 5.0 --unk_cm 7.3 5.5` |
| C Optical flow | `python C_optical_flow/optical_flow.py --video data/video.mp4 --frame 150 --box_idx 0` |
| D Kalman | `python D_tracking/kalman_tracker.py --video data/video.mp4 --drop_len 5` |
| E Recognition | `python E_recognition/recognize.py` |
| Pipeline | `python pipeline.py --video data/video.mp4 [--box_h 4.0]` |

`--cols/--rows` = inner corners of your checkerboard; `--square_cm` = printed square size. `--box_h` = height of box tops above the belt.

## What each module demonstrates
**A** - Eight segmentation methods (snake, quadtree split, merge, split&merge, watershed, Felzenszwalb, mean shift, N-Cut) on the same 5 frames in one grid. Split/merge are written from scratch. Watershed with distance-transform markers is used as the production detector (`detect_boxes`) because it separates touching boxes.

**B** - `cv2.calibrateCamera` gives K and distortion; the belt-plane pose comes from a checkerboard lying on the belt. P = K[R|t] is decomposed manually (RQ) and checked against OpenCV. `measure_box.py` estimates an unknown box size under orthographic, weak-perspective, affine and full-perspective models. `camera_model.py` back-projects pixels to belt-plane cm for the rest of the project.

**C** - Coarse-to-fine Lucas-Kanade is implemented from scratch (2x2 normal equations per window, min-eigenvalue test for the aperture problem). It is compared with OpenCV sparse LK on corners. A 6-parameter affine flow model is fitted with RANSAC on one box, and belt speed in cm/s comes from the fitted flow plus calibration.

**D** - A 4-state constant-velocity Kalman filter from scratch, with Hungarian association for several boxes. The console prints predicted / measured / corrected positions for 10+ frames, including a simulated 5-frame occlusion where only prediction is used. Outputs a plot and overlay video.

**E** - Three recognisers: ORB+homography alignment, PCA eigenboxes with nearest neighbour, and Hu-moment invariants. Accuracy is reported on held-out crops under clean, rotated and lighting-changed conditions. Top eigenvectors are saved to `output/eigenboxes.png`.

## Pipeline
`pipeline.py` chains A -> D -> B -> C -> E and prints, per tracked box: ID, size (cm), speed from optical flow and from the Kalman velocity, and recognised type (majority vote across frames). It also writes `output/pipeline_out.mp4` and `output/pipeline_results.csv`.

## Practical notes / known limitations
- **Same resolution everywhere.** Film the video and take the checkerboard photos at the same camera resolution (e.g. 1280x720); `pipeline.py` warns if they differ.
- **Empty-belt background.** The background is the median of 40 frames, so boxes must be moving and the camera must not move. Constant lighting helps a lot.
- **Touching boxes** are split using notches in the silhouette plus distance-transform watershed. Two boxes of identical height pressed perfectly flush (no notch, no visible seam) cannot be separated by shape alone and are reported as one box.
- **Size accuracy** on the synthetic test was within ~2%; on real footage expect a few percent (mask edge blur, box height `--box_h`).
- **Module D** adds `--noise` px of Gaussian noise to the measurements (default 3) so smoothing is visible; use `--noise 0` for the raw segmentation output.
- **Module C, panel (c)** deliberately corrupts 10% of the vectors (`--inject_outliers`) to demonstrate RANSAC; the printed speed uses the clean fit.
- **Module E** reports accuracy on rotated / re-lit copies of the held-out crops as well as the clean ones; the commentary in `recognize.py` should be checked against your own table.
