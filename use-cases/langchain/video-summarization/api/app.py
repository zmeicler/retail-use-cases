from fastapi import FastAPI
from merger.summary_merger import SummaryMerger
from pydantic import BaseModel


class SummaryMergerRequest(BaseModel):
    """
    Pydantic model for the request body of the merge_summaries endpoint. It expects a dictionary of summaries.
    {
        "summaries": {
            "chunk_0": "text1",
            "chunk_1": "text2",
            ...
        }
    }
    """
    summaries: dict


class SummaryMergerResponse(BaseModel):
    """
    Pydantic model for the response body of the merge_summaries endpoint.
    It will return the overall summary and the anomaly score.
    """
    overall_summary: str
    anomaly_score: float


app = FastAPI()
summary_merger = SummaryMerger(device="GPU")


@app.get("/")
def root():
    """
    Root path for the application
    """
    return {
        "message": "Hello from App. Use /merge_summaries/<summaries dict {'summaries': {'chunk0': 'text'...}}> to perform summary merging/assign anomaly score"}


@app.post("/merge_summaries")
def merge_summaries(request: SummaryMergerRequest):
    """
    Endpoint for calling summary merger. Input should be in this format:
    {
        "summaries": {
            "chunk_0": "text1",
            "chunk_1": "text2",
            ...
        }
    }"""
    output = summary_merger.merge_summaries(request.summaries)
    return SummaryMergerResponse(**output)
