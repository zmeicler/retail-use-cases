import os
import sys
import time
import argparse
from concurrent.futures.thread import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from pydantic import BaseModel, Field
import json
import threading
from MultiRTSPChunkLoader import MultiRTSPChunkLoader
from utils import post_request, is_valid_rtsp_url, load_chunk_metadata, all_metadata_written, extract_timestamp_from_chunk

class SummarizerConfig(BaseModel):
    rtsp_sources: list[str]
    prompt: str = Field(default="Please summarize this video.")
    chunk_duration: int = Field(default=30, ge=1, description="Chunk duration in seconds.")
    chunk_overlap: int = Field(default=2, ge=0, description="Overlap between chunks in seconds.")
    framerate: int = Field(default=10, ge=1, description="Framerate for video processing.")
    outfile: str = Field(default="", description="Output file for generated summaries.")
    chunk_dir: str = Field(default="multi_cam_chunks", description="Directory to store video chunks.")
    anomaly_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Threshold for anomaly detection.")
    merge_cadence: int = Field(default=60, ge=1, description="Cadence (in seconds) to merge summaries.")
    use_merger: bool = Field(default=True, description="Whether to use the merger/vertex for overall summaries.")
    merge_cadence_mode: str = Field(default="real_time", description="Mode for merge cadence: 'real_time' or 'timestamp'.")

class VideoSummarizer:
    def __init__(self, config: SummarizerConfig):
        self.config = config

        # Initialize RTSP Feeds (Doesn't need to be a call to endpoint. Initilize it here, call lazy_load in a thread))
        self.rtsp_sources = {f"cam{str(i+1).zfill(2)}": source for i, source in enumerate(config.rtsp_sources)}
        self.validate_sources()
        self.chunk_loader_params = {"window_size": config.chunk_duration * config.framerate,
                                "fps": config.framerate,
                                "overlap": config.chunk_overlap * config.framerate}
        self.loader = MultiRTSPChunkLoader(
            camera_sources=self.rtsp_sources,
            chunk_type="sliding_window",
            chunk_args=self.chunk_loader_params,
            output_dir=config.chunk_dir)

        # Run lazy_load in a separate thread
        threading.Thread(target=self.loader.lazy_load, daemon=True).start()

        # Set up variables for video summarization and anomaly detection
        self.cloud_prompt = (
            config.prompt +
            " Please analyze all attached videos as if they were combined into a single video. "
            "In addition, the last information produced must be a score between 0 and 1 to represent how suspicious the video is. "
            "The score should be a float rounded to the tenth decimal and formatted as the following example: \n **anomaly score**: 0.0"
        )
        self.chunk_summaries = {}
        self.last_merge_time = time.time()

    def validate_sources(self):
        for source in self.rtsp_sources.values():
            if not (is_valid_rtsp_url(source) or os.path.exists(source)):
                print(f"Invalid RTSP URL or file path: {source}")
                sys.exit()

    def generate_chunk_summary(self, file, timestamp):
        for camera_id in self.rtsp_sources.keys():
            # Load metadata (May want to refactor this to be returned in summarize when we check for metadata existance)
            camera_dir = os.path.join(self.config.chunk_dir, camera_id)
            metadata_file_path = os.path.join(camera_dir, f"{camera_id}_metadata.json")
            doc = load_chunk_metadata(metadata_file_path, os.path.join(camera_dir, file), camera_id)

            # Generate summary
            inputs = {"video": doc.metadata["chunk_path"], "question": self.config.prompt}
            with ThreadPoolExecutor() as pool:
                future = pool.submit(post_request, inputs, endpoint="generate_summary")
                output = future.result().decode("utf-8")

            # Update storage (TO DO: Milvis integration)
            start_time = datetime.strptime(doc.metadata["timestamp"], "%Y-%m-%d_%H-%M-%S")
            chunk_args = self.chunk_loader_params #["chunk_args"]
            window_size = chunk_args["window_size"]
            fps = chunk_args["fps"]
            end_time = start_time + timedelta(seconds=window_size / fps)
            self.chunk_summaries[Path(doc.metadata["chunk_path"]).stem] = {
                "start_time": start_time,
                "end_time": end_time,
                "summary": output
            }

    def generate_overall_summary(self):
        # Generate overall summary from available chunk summaries
        with ThreadPoolExecutor() as pool:
            future = pool.submit(post_request, 
                            {key: value["summary"] for key, value in self.chunk_summaries.items()})
            res = eval(future.result().decode("utf-8"))
        
        # Check for anomaly score and extend to Vertex if needed
        if res["anomaly_score"] >= self.config.anomaly_threshold:
            print("Anomaly score exceeds threshold. Extending to Vertex.")

            # Gather required chunks
            chunks = [os.path.join(self.config.chunk_dir, camera_id, file)
                      for camera_id in self.rtsp_sources.keys()
                      for file in os.listdir(os.path.join(self.config.chunk_dir, camera_id))
                      if file.endswith(".avi")]

            # Limit the number of chunks to 10
            if len(chunks) > 10:
                step = len(chunks) // 10
                chunks = chunks[::step][:10]
            
            # Post request to Vertex
            vertex_request = {"text_prompt": self.cloud_prompt, "video_paths": chunks}
            vertex_response = post_request(vertex_request, endpoint="vertex_generate")
            vertex_response = eval(vertex_response.decode("utf-8"))
            res = {
                "overall_summary": vertex_response.get("content", ""),
                "anomaly_score": vertex_response.get("anomaly_score", 0.0)
            }
                    
        return res
    
    def cleanup_chunks(self):
        """Delete processed chunks that have been summarized and reset chunk summaries."""
        for camera_id in self.rtsp_sources.keys():
            camera_dir = os.path.join(self.config.chunk_dir, camera_id)
            if os.path.exists(camera_dir):
                for file in os.listdir(camera_dir):
                    chunk_name = Path(file).stem
                    if file.endswith(".avi") and chunk_name in self.chunk_summaries:
                        os.remove(os.path.join(camera_dir, file))
                        del self.chunk_summaries[chunk_name]

    def summarize(self):
        while True:
            # Check if videos have been created by the chunk loader
            camera_dir = os.path.join(self.config.chunk_dir, "cam01")
            if not os.path.exists(camera_dir):
                continue

            # Find the oldest chunk based on timestamp (BIGGEST AREA FOR IMPROVEMENT)
            chunks = [vfile for vfile in os.listdir(camera_dir) if vfile.endswith(".avi")]
            if chunks:
                oldest_chunk = min(chunks, key=lambda f: datetime.strptime(extract_timestamp_from_chunk(f), "%Y-%m-%d_%H-%M-%S"))

                # If metadata for all chunks for this timestamp has been written, process them
                timestamp = extract_timestamp_from_chunk(oldest_chunk)
                if all_metadata_written(timestamp, self.rtsp_sources, self.config.chunk_dir):
                    self.generate_chunk_summary(oldest_chunk, timestamp)

            # Call generate_overall_summary at the specified cadence
            if self.config.use_merger:
                # Use real time for keeping track of the last merge time, if asked
                current_time = time.time()
                if self.config.merge_cadence_mode == "real_time":
                    if current_time - self.last_merge_time >= self.config.merge_cadence and self.chunk_summaries:
                        self.last_merge_time = current_time
                        res = self.generate_overall_summary()

                # Use timestamp for keeping track of the last merge time, if asked
                elif self.config.merge_cadence_mode == "timestamp":
                    if self.chunk_summaries:
                        newest_chunk_end_time = max(
                            summary["end_time"] for summary in self.chunk_summaries.values()
                        )
                        if (newest_chunk_end_time - datetime.fromtimestamp(self.last_merge_time)).total_seconds() >= self.config.merge_cadence:
                            self.last_merge_time = newest_chunk_end_time.timestamp()
                            res = self.generate_overall_summary()

                # Determine start and end times for the overall summary
                oldest_chunk_start_time = min(
                    summary["start_time"] for summary in self.chunk_summaries.values()
                )
                newest_chunk_end_time = max(
                    summary["end_time"] for summary in self.chunk_summaries.values()
                )

                # Write to JSON file
                json_output = {
                    "start_time": oldest_chunk_start_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "end_time": newest_chunk_end_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "summary": res.get("overall_summary", ""),
                    "anomaly_score": res.get("anomaly_score", 0.0)
                }
                json_file_path = self.config.outfile
                with open(json_file_path, "a") as json_file:
                    json.dump(json_output, json_file, indent=4)
                    json_file.write("\n")

                # Cleanup processed chunks
                self.cleanup_chunks()
            else:
                # Write individual chunk summaries to JSON file
                for chunk_name, summary in self.chunk_summaries.items():
                    json_output = {
                        "chunk_name": chunk_name,
                        "start_time": summary["start_time"].strftime("%Y-%m-%d %H:%M:%S"),
                        "end_time": summary["end_time"].strftime("%Y-%m-%d %H:%M:%S"),
                        "summary": summary["summary"]
                    }
                    json_file_path = self.config.outfile
                    with open(json_file_path, "a") as json_file:
                        json.dump(json_output, json_file, indent=4)
                        json_file.write("\n")

                # Cleanup processed chunks
                self.cleanup_chunks()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate video summarization using LangChain, OpenVINO-genai, and MiniCPM-V-2_6.")
    parser.add_argument("rtsp_sources", type=str, nargs="+", help="All RTSP sources to be summarized.")
    parser.add_argument("-p", "--prompt", type=str, default="Please summarize this video.", help="Text prompt.")
    parser.add_argument("-c", "--chunk_duration", type=int, default=30, help="Maximum length in seconds for each chunk.")
    parser.add_argument("-v", "--chunk_overlap", type=int, default=2, help="Overlap in seconds between chunks.")
    parser.add_argument("-fps", "--framerate", type=int, default=10, help="Framerate for processing video chunks.")
    parser.add_argument("-od", "--chunk_dir", type=str, default="multi_cam_chunks", help="Directory to store RTSP chunks.")
    parser.add_argument("-at", "--anomaly_threshold", type=float, default=0.7, help="Threshold for anomaly detection.")
    parser.add_argument("-mc", "--merge_cadence", type=int, default=60, help="Cadence (in seconds) to merge summaries.")
    parser.add_argument("-jf", "--json_file", type=str, default="summary_output.json", help="JSON file to append summaries.")
    parser.add_argument("-um", "--use_merger", action="store_true", help="Whether to use the merger/vertex for overall summaries.")
    parser.add_argument("-mcm", "--merge_cadence_mode", type=str, default="real_time", help="Mode for merge cadence: 'real_time' or 'timestamp'.")
    args = parser.parse_args()

    config = SummarizerConfig(
        rtsp_sources=args.rtsp_sources,
        prompt=args.prompt,
        chunk_duration=args.chunk_duration,
        chunk_overlap=args.chunk_overlap,
        framerate=args.framerate,
        outfile=args.json_file,
        chunk_dir=args.chunk_dir,
        anomaly_threshold=args.anomaly_threshold,
        merge_cadence=args.merge_cadence,
        use_merger=args.use_merger,
        merge_cadence_mode=args.merge_cadence_mode,
    )

    # Initialize and run the video summarizer
    summarizer = VideoSummarizer(config)
    summarizer.summarize()
