# Lesson 3 - Inference

The objective of the lesson is to use several pre-trained models to run inference on a video file or webcam feed.

The following types of inferences are supported:
- Detection of vehicles in video files
- Detection of human pose in a webcam feed

## 1. Download the Model

The Intel Openvino Model Downloader tool is used to download all the required pretrained models. The commands to do this shown for each pretrained model type, where:

- `<INSTALL_DIR>` is the path to the install location of the OpenVino Toolkit (`/intel/openvino/`).
- `<MODEL_DIR>` is the directory path where all pre-trained models are stored (ex: `./models/`)

### Vehicle Detection

```sh 
<INSTALL_DIR>/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name vehicle-detection-0202 -o <MODEL_DIR>
```

### Human Pose Estimation
```sh 
<INSTALL_DIR>/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name human-pose-estimation-0001 -o <MODEL_DIR>
```

## 2. Run the Inference
