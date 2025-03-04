# Summarize Videos Using OpenVINO-GenAI, Langchain, and MiniCPM-V-2_6

## Installation
Follow [link](https://cloud.google.com/sdk/docs/install) to create google cloud account and install gcloud 

Next, login and authenticate your account
```
# Initialize gcloud and create login credentials. Tip: Save path to .json credential file.
gcloud init

# Use credentials to login and authorize account
gcloud auth application-default login 
```

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