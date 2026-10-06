# ScrewDetect
Real-time YOLO based detection of screw, plug, weld, and adhesive connections and whether they're new or worn. A fine tuned YOLO model trained on a dataset containing images of Allen bolts of multiple sizes including M4, M6, M8, M10, M14, M16 trained to detect New and Rusted bolts and classify them according to their sizes


## Features

- YOLO object detection for the classes `Schraube_Neu` and `Schraube_Ver`
- Dataset preparation and train/validation/test splitting
- Model training and evaluation
- Live camera inference
- RealSense depth-based screw size estimation


### Summary

This model configuration reliably distinguishes new and worn screws with high confidence and is suitable for real-time detection tasks and downstream size estimation with a RealSense camera.

## Project Structure

```text
proki/
├── configs/
│   ├── data.yaml                 # Dataset paths and class names
│   └── train_config.yaml         # Default training parameters
├── data/
│   ├── cvat_exports/             # CVAT annotation exports
│   ├── processed/
│   │   ├── all/                  # Prepared dataset before splitting
│   │   ├── images/{train,val,test}/
│   │   └── labels/{train,val,test}/
│   └── raw/
│       ├── schraube/             # Raw captures by screw category
│       └── ver_schraube/
├── models/
│   ├── eval/                     # Evaluation results and plots
│   └── train/                    # Training runs and model weights
├── scripts/
│   ├── capture.py                # Capture color images with RealSense
│   ├── prepare_dataset.py        # Convert/copy a CVAT export
│   ├── split_dataset.py          # Split data into train/val/test
│   ├── train.py                  # Train a YOLO model
│   ├── evaluate.py               # Evaluate trained weights
│   ├── inference_rt.py           # Live camera object detection
│   └── measure.py                # Detection and depth-based size estimate
├── Dockerfile
├── docker-compose.yaml
├── yolo11n.pt                    # YOLO starting weights
├── yolo26n.pt                    # YOLO starting weights
├── README.md
└── LICENSE
```

## Requirements

### Software

- Linux environment with Docker and Docker Compose
- NVIDIA GPU and NVIDIA Container Toolkit for GPU training with the provided container configuration
- Container image based on Ultralytics, with PyTorch, CUDA 12.6 wheels, OpenCV, and `pyrealsense2`
- Python packages used by the scripts: `ultralytics`, `torch`, `opencv-python`, `numpy`, `PyYAML`, and `pyrealsense2`
- X11 display access for the OpenCV camera preview windows

### Hardware

- Intel RealSense D435 or D455 RGB-D camera for image capture and depth-based measurement
- A V4L-compatible camera can be used with live object detection (`inference_rt.py`)
- An NVIDIA GPU is recommended for training; the default training configuration selects GPU device `0`

The Docker Compose configuration passes through the host GPU, camera devices, and X11 display. Camera access and display forwarding may need adjustment for your machine.


## Dataset

You can find the exported data from CVAT in the `cvat_exports` folder in `.zip` format. Currently there are 250 labelled images for new Allen bolts with the label `Schraube_Neu` and 80 images with mixed rusted and new bolts with the rusted labelled with `Schraube_Ver`. For training, a classic split for `70-20-10` for `train-validation-test`.

## Setup

1. Clone the Environment

```sh
git clone https://git-ce.rwth-aachen.de/wzl-mq-ms/forschung-lehre/proki.git
cd proki/
```

2. Build the docker image

```sh
docker compose up -d
```
Wait till the image gets built and the container starts

3. Enter the container 

```sh
docker compose exec proki bash
```

## Workflow

1. Capture or collect images.
2. Prepare the dataset from its annotation export.
3. Split the prepared data into training, validation, and test sets.
4. Train a model.
5. If needed, Evaluate the trained weights.
6. Run live inference or depth-based size measurement.

## Usage

#### 1. Capture images

```sh
python3 scripts/capture.py --count <NUMBER>
```
Args:
- `--out`: Location for saving captured images | default: `data/raw`
- `--tag`: prefix for the filenames            | default: `capture`
- `--count`: number to start counting from     | required
- `--width`, `--height`: image size            | default: `1280` x `720`)

#### 2. Prepare Dataset

```sh
python3 scripts/prepare_dataset.py --source <NUMBER>
```
Args:
- `--source`: Folder path containing the images                                     | Required
- `--destination`: Folder location where you want to save the split data            | default: `capture`

#### 3. Split Dataset

```sh
python3 scripts/split_dataset.py --source data/processed/all
```

Args:
- `--source`: folder with `images/` and `labels/` subfolders to split | required
- `--destination`: folder where the split data is saved                             | default: `data/processed`
- `--train`: fraction of data used for training                                     | default: `0.7`
- `--val`: fraction of data used for validation                                     | default: `0.2`
- `--test`: fraction of data used for testing                                       | default: `0.1`
- `--seed`: random seed for a reproducible split                                    | default: `93`

### 4. Train Model
```sh
python3 scripts/train.py --name <NAME>
```
Replace `NAME` with a your desired name. Training outputs are saved under `/proki/models/train/<NAME>`.

### 5. Evaluate Model

```sh
python3 scripts/evaluate.py --weights /proki/models/train/<NAME>/weights/best.pt
``` 
Args:
- `--weights`: path to the trained YOLO model weights (`.pt`) | required
- `--data`: path to the dataset configuration YAML file | default: `configs/data.yaml`
- `--split`: dataset split used for evaluation (`val` or `test`) | default: `test`
- `--project`: folder where evaluation results and plots are saved | default: `/proki/models/eval`
- `--name`: name of the evaluation run | default: `exp001`


### 6. Run Inference

```sh
python3 scripts/inference_rt.py --weights /proki/models/train/<NAME>/weights/best.pt
```
Args:
- `--weights`: path to the trained YOLO model weights (`.pt`) | required
- `--source`: camera device or video source used for inference | default: `/dev/video6`
- `--confidence`: minimum confidence threshold for detections | default: `0.6`
- `--width`: requested camera frame width in pixels | default: `1920`
- `--height`: requested camera frame height in pixels | default: `1080`
- `--imgsz`: image size used by YOLO during inference | default: `640`

> **Note:** The camera device path may differ between systems. To identify the available camera devices and their corresponding video sources, run:
>
> ```sh
> v4l2-ctl --list-devices
> ```
>
> You should see a list similar to:
>
> ```text
> Intel RealSense Depth Camera (usb-0000:00:14.0-2):
>     /dev/video2
>     /dev/video3
>     /dev/video4
>     /dev/video5
>     /dev/video6
>     /dev/video7
>     /dev/media1
>
> Integrated Camera (usb-0000:00:14.0-6):
>     /dev/video0
>     /dev/video1
>     /dev/media0
> ```
>
> The available `/dev/video*` sources may vary between systems. Test the listed video sources to determine which one corresponds to the desired camera stream. The working source can then be passed to the inference script using the `--source` argument.

### 7. Run Measurement inference

```sh
python3 scripts/measurement.py --weights /proki/models/train/<NAME>/weights/best.pt
```
- `--weights`: path to the trained YOLO model weights (`.pt`) | required
- `--source`: camera device or video source used for inference | default: `/dev/video6`
- `--confidence`: minimum confidence threshold for detections | default: `0.6`
- `--width`: requested camera frame width in pixels | default: `1920`
- `--height`: requested camera frame height in pixels | default: `1080`
- `--imgsz`: image size used by YOLO during inference | default: `640`

## Configuration

### Dataset configuration (`configs/data.yaml`)

- `path`: dataset root directory, set to `../data/processed` relative to the configuration file.
- `train`, `val`, `test`: image directories for each split, relative to `path`. The matching labels are expected in the corresponding `labels/` directories.
- `names`: mapping from class IDs to class names: `0` is `Schraube_Neu` and `1` is `Schraube_Ver`. Keep these IDs and names consistent with the dataset annotations.

### Training configuration (`configs/train_config.yaml`)

- `model`: starting YOLO weights (`yolo11n.pt` by default).
- `data`: dataset configuration file (`configs/data.yaml`).
- `epochs`: maximum number of training epochs (`100`).
- `imgsz`: input image size in pixels (`640`).
- `batch`: number of images processed per batch (`16`); reduce this if GPU memory is insufficient.
- `patience`: stop early if validation results do not improve for this many epochs (`40`).
- `device`: training device (`0` selects the first CUDA GPU).
- `save`: save training checkpoints and outputs (`True`).
- `project`: parent directory for training runs (`/proki/models/train` in the provided container).
- `name`: default run directory name (`exp001`). The `--name` argument to `scripts/train.py` overrides this value.

Edit these YAML values to change the default training setup. For example, select another starting checkpoint by changing `model`, or reduce `batch` to fit a smaller GPU.


## Results

The current best model run is saved in `models/train/exp006`. It reaches a strong classification performance for detecting both screw states in the validation set.

### Final metrics

| Metric | Value |
| --- | ---: |
| Precision | 0.9646 |
| Recall | 0.9764 |
| mAP@0.5 | 0.9845 |
| mAP@0.5:0.95 | 0.8273 |
| `Schraube_Neu` AP | 0.967 |
| `Schraube_Ver` AP | 0.995 |

The training curves and evaluation plots from this run are stored under `models/train/exp006/` and include the box loss, precision/recall, F1-confidence, and confusion-matrix visualizations.
<!-- 
### Example training and validation outputs

![Training batch 0](models/train/exp006/train_batch0.jpg)
![Training batch 1](models/train/exp006/train_batch1.jpg)
![Validation predictions](models/train/exp006/val_batch0_pred.jpg)
![Validation predictions 2](models/train/exp006/val_batch1_pred.jpg)
![Validation predictions 3](models/train/exp006/val_batch2_pred.jpg)

### Evaluation plots

![F1 confidence curve](models/train/exp006/BoxF1_curve.png)
![Precision-recall curve](models/train/exp006/BoxPR_curve.png)
![Precision curve](models/train/exp006/BoxP_curve.png)
![Recall curve](models/train/exp006/BoxR_curve.png)
![Confusion matrix](models/train/exp006/confusion_matrix.png)
![Normalized confusion matrix](models/train/exp006/confusion_matrix_normalized.png) -->

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
