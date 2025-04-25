import os
import json
import re
from typing import Dict
import requests
from langchain_core.documents import Document

os.environ["no_proxy"] = "localhost,127.0.0.1"

def post_request(input_data, endpoint="merge_summaries"):
    """
    Sends a POST request to the specified endpoint with the given input data.

    :param input_data: Data to be sent in the POST request.
    :param endpoint: API endpoint to communicate with. Defaults to 'merge_summaries'.
    :return: Response content from the API.
    """
    if endpoint == "merge_summaries":
        formatted_req = {"summaries": input_data}
    elif endpoint == "generate_summary":
        formatted_req = {
            "video_file": input_data["video"],
            "question": input_data["question"]
        }
    elif endpoint == "start_chunk_loader":
        formatted_req = input_data
    elif endpoint == "vertex_generate":
        formatted_req = {
            "text_prompt": input_data["text_prompt"],
            "video_paths": input_data["video_paths"]
        }
    else:
        raise ValueError(f"Unsupported endpoint: {endpoint}")

    url = f"http://127.0.0.1:8000/{endpoint}"
    response = requests.post(url=url, json=formatted_req)
    print(f"Request payload: {formatted_req}")
    print(f"Response status: {response.status_code}, Response content: {response.content}")
    return response.content

def load_chunk_metadata(metadata_file_path: str, chunk_path: str, camera_id: str) -> Document:
    """
    Load metadata for a specific chunk from the metadata file.

    :param metadata_file_path: Path to the metadata JSON file.
    :param chunk_path: Path to the video chunk.
    :param camera_id: ID of the camera.
    :return: Document containing metadata and content.
    """
    with open(metadata_file_path, "r") as metadata_file:
        for line in metadata_file:
            metadata = json.loads(line)
            if metadata["chunk_path"] == chunk_path:
                return Document(
                    page_content=f"[{camera_id}] Chunk metadata loaded.",
                    metadata=metadata
                )
    return None

def all_metadata_written(timestamp, rtsp_sources, chunk_dir):
    """
    Check if metadata for all chunks corresponding to the given timestamp has been written.
    """
    checks = [False] * len(rtsp_sources)
    for idx, camera_id in enumerate(rtsp_sources.keys()):
        # Check to see if metadata file exists
        camera_dir = os.path.join(chunk_dir, camera_id)
        metadata_file_path = os.path.join(camera_dir, f"{camera_id}_metadata.json")
        if not os.path.exists(metadata_file_path):
            print(f"Metadata file for {camera_id} does not exist: {metadata_file_path}")
            return False
        
        # If so, load it and see if the timestamp exists for this chunk
        with open(metadata_file_path, "r") as metadata_file:
            for line in metadata_file:
                metadata = json.loads(line)
                if metadata["timestamp"] ==timestamp:
                    # This camera has metadata for the given timestamp, set check to True
                    checks[idx] = True
                    break
            print(f"Metadata for {camera_id} does not contain timestamp {timestamp}.")

    return all(checks)

def all_chunks_exist(timestamp: str, rtsp_sources: Dict[str, str], chunk_dir: str) -> bool:
    """
    Check if all cameras have the chunk for the given timestamp.

    :param timestamp: The timestamp to check.
    :param rtsp_sources: Dictionary of camera IDs and their RTSP sources.
    :param chunk_dir: Directory where chunks are stored.
    :return: True if all chunks exist, False otherwise.
    """
    print([os.path.join(chunk_dir, cam_id, f"{cam_id}_chunk_{timestamp}.avi")
        for cam_id in rtsp_sources])
    return all(
        os.path.exists(os.path.join(chunk_dir, cam_id, f"{cam_id}_chunk_{timestamp}.avi"))
        for cam_id in rtsp_sources
    )


def is_valid_rtsp_url(url: str) -> bool:
    """
    Check if a string is a valid RTSP URL.
    """
    rtsp_pattern = re.compile(r"^rtsp://[^\s]+$")
    return bool(rtsp_pattern.match(url))


def extract_timestamp_from_chunk(chunk_filename: str) -> str:
    """
    Extract the timestamp from a chunk filename.

    :param chunk_filename: The filename of the chunk.
    :return: The extracted timestamp as a string.
    """
    return chunk_filename.split("_chunk_")[-1].replace(".avi", "")
