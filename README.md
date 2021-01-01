# Edge Computing with Computer Vision

A repo for doing edge computing demonstrations using [Intel's Openvino Toolkit](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html). 

## Requirements

In order to run code in this repository you will need:

- Python 3.6 or 3.7 (Currently 3.8 and above is not supported by Intel)
- [Intel OpenVino Toolkit 2021](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html)

Install project dependencies via `pip`:

```sh
pip install -r requirements.txt
```

That's it! Now you're ready to run Intel OpenVino toolkit for running inferences for ML models at edge.

## Download a Pretrained Model

Intel's Openvino Toolkit provides a utility class for downloading pretrained ML models from [Intel's Model Zoo](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/pretrained-models.html).

Here is documentation on the [downloader utility class](https://docs.openvinotoolkit.org/latest/omz_tools_downloader_README.html)

For convenience the downloader has been copied into the [utils](./utils/downloader) folder of this project.

Here is an example of running the [downloader.py](./utils/downloader/downloader.py) script to download a desired model from the Model Zoo:

Note that the variable `<OPENVINO_DIR>` should be replaced with the full path to the `openvino` installation. 

```sh

# Find the downloader tool provided by OpenVino
cd <OPENVINO_DIR>/openvino/deployment_tools/open_model_zoo/tools/downloader

# Download a model to the `models` dir
python3 ./downloader.py --name human-pose-estimation-0001 --precisions FP16,FP16-INT8
-o <MODEL_DIR>
```

Where `<MODEL_DIR>` is the full path to where the pre-trained model should be downloaded. 
