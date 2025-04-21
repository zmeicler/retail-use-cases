import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from fastapi import FastAPI
from merger.summary_merger import SummaryMerger
from pydantic import BaseModel
from summarizer.MultiRTSPChunkLoader import MultiRTSPChunkLoader
from typing import Dict, List
from summarizer.vertex_extension import VertexWrapper
from summarizer.ov_lvm_wrapper import OVMiniCPMV26Worker

# Hardcoded parameters
endpoint_uri = "127.0.0.1"
endpoint_port = "8000"
vertex_model_name = "gemini-2.0-flash-exp"
summary_merger_device = "GPU"
ov_model_dir = "MiniCPM_INT8"
ov_device = "GPU"
ov_max_new_tokens = 200
ov_max_num_frames = 1
ov_resolution = [210, 190]
use_merger = True

# Initialize ToxicBERTParser. Not fully functional yet.
# from summarizer.purge import ToxicBERTParser
# purger = ToxicBERTParser(model_path="path/to/toxic-bert-openvino-model.xml")

# Initialize OVMiniCPMV26Worker
ov_minicpm = OVMiniCPMV26Worker(
    model_dir=ov_model_dir,
    device=ov_device,
    max_new_tokens=ov_max_new_tokens,
    max_num_frames=ov_max_num_frames,
    resolution=ov_resolution
)

# Initialize SummaryMerger / VertexWrapper if specified
summary_merger = None
vertex_wrapper = None
if use_merger:
    summary_merger = SummaryMerger(device=summary_merger_device)
    vertex_wrapper = VertexWrapper(model_name=vertex_model_name)

# Start FastAPI app
app = FastAPI()

@app.get("/")
def root():
    """
    Root path for the application
    """
    return {
        "message": "Welcome to the Video Summarization API. The following endpoints are available:",
        "endpoints": {
            "/merge_summaries": "POST - Merge summaries and assign an anomaly score (available if --use_merger is enabled).",
            "/generate_summary": "POST - Generate a summary for a video file using OVMiniCPMV26.",
            "/start_chunk_loader": "POST - Start the MultiRTSPChunkLoader to process video streams.",
            "/vertex_generate": "POST - Generate content using VertexWrapper (available if --use_merger is enabled)."
        }
    }

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

@app.post("/merge_summaries")
def merge_summaries(request: SummaryMergerRequest):
    """
    Endpoint for calling summary merger.
    """
    if summary_merger is None:
        return {"error": "SummaryMerger is not initialized."}

    output = summary_merger.merge_summaries(request.summaries)
    return SummaryMergerResponse(**output)

class OVMiniCPMRequest(BaseModel):
    """
    Pydantic model for the request body of the OVMiniCPMV26Wrapper endpoint.
    It expects a video file path and a question for summarization.
    {
        "video_file": "path/to/video.mp4",
        "question": "What is happening in the video?"
    }
    """
    video_file: str
    question: str


class OVMiniCPMResponse(BaseModel):
    """
    Pydantic model for the response body of the OVMiniCPMV26Wrapper endpoint.
    It will return the generated summary.
    """
    summary: str

@app.post("/generate_summary")
def generate_summary(request: OVMiniCPMRequest):
    """
    Endpoint for generating a summary using OVMiniCPMV26Wrapper.
    """
    if ov_minicpm is None:
        return {"error": "OVMiniCPMV26Worker is not initialized."}

    prompt = f"{request.video_file},{request.question}"
    summary = ov_minicpm._call(prompt)
    return OVMiniCPMResponse(summary=summary)

class ChunkLoaderRequest(BaseModel):
    """
    Pydantic model for the request body of the start_chunk_loader endpoint.
    It expects camera sources, chunk type, chunk arguments, and an optional output directory.
    {
        "camera_sources": {"cam01": "rtsp://...", "cam02": "rtsp://..."},
        "chunk_type": "sliding_window",
        "chunk_args": {"window_size": 300, "fps": 15, "overlap": 30},
        "output_dir": "output_chunks"
    }
    """
    camera_sources: Dict[str, str]
    chunk_type: str
    chunk_args: Dict
    output_dir: str = "output_chunks"

@app.post("/start_chunk_loader")
def start_chunk_loader(request: ChunkLoaderRequest):
    """
    Endpoint to start an instance of MultiRTSPChunkLoader and call its lazy_load() method.
    Input should be in this format:
    {
        "camera_sources": {"cam01": "rtsp://...", "cam02": "rtsp://..."},
        "chunk_type": "sliding_window",
        "chunk_args": {"window_size": 300, "fps": 15, "overlap": 30},
        "output_dir": "output_chunks"
    }
    """
    loader = MultiRTSPChunkLoader(
        camera_sources=request.camera_sources,
        chunk_type=request.chunk_type,
        chunk_args=request.chunk_args,
        output_dir=request.output_dir
    )
    loader.lazy_load()
    return {"message": "Chunk loader started successfully."}

class VertexRequest(BaseModel):
    """
    Pydantic model for the request body of the VertexWrapper endpoint.
    It expects a text prompt and an optional list of video paths.
    {
        "text_prompt": "Analyze this video.",
        "video_paths": ["path/to/video1.mp4", "path/to/video2.mp4"]
    }
    """
    text_prompt: str
    video_paths: List[str] = None

class VertexResponse(BaseModel):
    """
    Pydantic model for the response body of the VertexWrapper endpoint.
    It will return the generated content and anomaly score (if applicable).
    """
    content: str
    anomaly_score: float = None

@app.post("/vertex_generate")
def vertex_generate(request: VertexRequest):
    """
    Endpoint for generating content using VertexWrapper.
    """
    if vertex_wrapper is None:
        return {"error": "VertexWrapper is not initialized."}

    response_text = vertex_wrapper.generate(request.text_prompt, request.video_paths)
    anomaly_score = VertexWrapper.extract_anomaly_score(response_text)
    return VertexResponse(content=response_text, anomaly_score=anomaly_score)

# uvicorn.run("api.app:app", host=endpoint_uri, port=endpoint_port, reload=reload)
