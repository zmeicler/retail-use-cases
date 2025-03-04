#!/bin/bash

# source activate-conda.sh
# activate_conda
# conda activate ovlangvidsumm

# if [ "$1" == "--skip" ]; then
# 	echo "Skipping sample video download"
# else
#     # Download sample video
#     wget https://github.com/intel-iot-devkit/sample-videos/raw/master/one-by-one-person-detection.mp4
# fi

INPUT_FILE="one-by-one-person-detection.mp4"
DEVICE="GPU"
RESOLUTION_X=480
RESOLUTION_Y=270
PROMPT='As an expert investigator, please analyze this video. Summarize the video, highlighting any shoplifting or suspicious activity. The output must contain the following 3 sections: Overall Summary, Activity Observed, Potential Suspicious Activity. It should be formatted similar to the following example:

**Overall Summary**
Here is a detailed description of the video.

**Activity Observed**
1) Here is a bullet point list of the activities observed. If nothing is observed, say so, and the list should have no more than 10 items.

**Potential Suspicious Activity**
1) Here is a bullet point list of suspicious behavior (if any) to highlight.
'

echo "Starting FastAPI app"
START /B uvicorn api.app:app
APP_PID=$!

echo "Running Video Summarizer"
python summarizer/video_summarizer.py $INPUT_FILE MiniCPM_INT8/ -d $DEVICE -r $RESOLUTION_X $RESOLUTION_Y -p "$PROMPT"

# terminate fastapi app after video summarization concludes
kill $APP_PID
