# Summarize Videos Using OpenVINO-GenAI, Langchain, and MiniCPM-V-2_6

## Installation

Install Intel Client GPU, Conda, and Set Up Python Environment

```
# Validated on Ubuntu 24.04 and 22.04
./install.sh
```

Note: if this script has already been performed and you'd like to re-install the sample project only, the following
command can be used to skip the re-install of dependencies.

```
./install.sh --skip
```

## Convert and Save Optimized MiniCPM-V-2_6

First, follow the steps on the [MiniCPM-V-2_6 HuggingFace Page](https://huggingface.co/openbmb/MiniCPM-V-2_6) to gain
access to the model. For more information on user access tokens for access to gated models
see [here](https://huggingface.co/docs/hub/en/security-tokens).

Next, convert and save the optimized model.

```
optimum-cli export openvino -m openbmb/MiniCPM-V-2_6 --trust-remote-code --weight-format int8 MiniCPM_INT8 # int4 also available 
```

## Run Video Summarization

Summarize [this sample video](https://github.com/intel-iot-devkit/sample-videos/raw/master/one-by-one-person-detection.mp4)
using `video_summarizer.py`.

```
./run-demo.sh 
```

Note: if the demo has already been run, you can use the following command to skip the video download.

```
./run-demo.sh --skip
```



################### NEED TO SORT: vertexai install stuff
# Install gcloud
https://cloud.google.com/sdk/docs/install

# Login and authenticate (To do: dermine how to install via vertex) 
gcloud init (follow log in)
gcloud auth application-default login (credentials saved at C:\Users\zmeicler\AppData\Roaming\gcloud\application_default_credentials.json])

# Create conda env
conda create -n vertex python=3.10 -y
conda activate vertex
pip install --upgrade google-cloud-aiplatform