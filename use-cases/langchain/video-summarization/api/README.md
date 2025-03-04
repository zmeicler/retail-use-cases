# FastAPI App for Merging Summaries and Assignment of Anomaly Scores

Please follow the instructions below for running just the Summary Merger/Anomaly Score Assignment API for standalone testing.
These are not required for running the full video summarization pipeline via `run-demo.sh`. (which includes Summary Merging)


## Installation

Ensure `install.sh` has been run to Python packages/requirements to be installed. If it has been run already, move to the next section.

```shell
# Validated on Ubuntu 24.04 and 22.04
./install.sh
```

Note: if this script has already been performed and you'd like to re-install the sample project only, the following
command can be used to skip the re-install of dependencies.

```shell
./install.sh --skip
```

## FastAPI Endpoints

The app currently exposes one endpoint:

1. `POST /merge_summaries` - Merges summaries and assigns anomaly scores to the merged summary.

### Request Body
```json
{
    "summaries": {
        "chunk_0": "text1",
        "chunk_1": "text2",
        ...
    }
}
```

### Response Body

```json
{
    "overall_summary": "string",
    "anomaly_score": 0.7
}
```

## Run Summary Merger/Anomaly Score Assignment API

```shell
Open a terminal:
conda activate ovlangvidsumm
# API will be running at http://localhost:8000
uvicorn api.app:app

Open another terminal:
conda activate ovlangvidsumm
python3 test_api.py
```

## Additional: Instantiate Summary Merger without FastAPI wrapper

Summary Merger can also be run without the FastAPI wrapper by directly instantiating the `SummaryMerger` class.
This option is useful if one wants to use the miniCPM runtime object without having to instantiate a new LLM.

```python
from merger.summary_merger import SummaryMerger

# use existing chain object
summary_merger = SummaryMerger(chain=chain, device="GPU")
ret = summary_merger.merge_summaries(chunk_summaries)

# OR
# uses LLAMA3.2 by default
summary_merger = SummaryMerger(device="GPU")
ret = summary_merger.merge_summaries(chunk_summaries)
```

