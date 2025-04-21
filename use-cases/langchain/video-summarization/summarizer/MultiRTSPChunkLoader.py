from typing import Iterator, Dict, List, Tuple
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
import cv2
import time
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading
import json

def run_vlm_inference(chunk_path: str, formatted_time: str, camera_id: str):
    try:
        print(f"[{camera_id}] LVLM Inference done on chunk at {formatted_time}!")
        # Optionally clean up
        # os.remove(chunk_path)
    except Exception as e:
        print(f"[{camera_id}][ERROR] Inference failed: {e}")

executor = ThreadPoolExecutor(max_workers=4)  # for async inference

class MultiRTSPChunkLoader(BaseLoader):
    def __init__(self, camera_sources: Dict[str, str], chunk_type: str, chunk_args: Dict, output_dir: str = "output_chunks"):
        self.camera_sources = camera_sources  # {camera_id: rtsp_url}
        self.chunk_type = chunk_type
        self.chunk_args = chunk_args
        self.output_dir = output_dir

        self.fps = self.chunk_args.get("fps", 15)
        self.window_size = self.chunk_args.get("window_size", 85)
        self.overlap_frames = self.chunk_args.get("overlap", 0)

        os.makedirs(self.output_dir, exist_ok=True)

    def _process_camera_stream(self, camera_id: str, rtsp_url: str):
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            print(f"[{camera_id}][ERROR] Failed to open RTSP stream.")
            return

        buffer = []
        buffer_start_time = None
        camera_dir = os.path.join(self.output_dir, camera_id)
        os.makedirs(camera_dir, exist_ok=True)
        metadata_file_path = os.path.join(camera_dir, f"{camera_id}_metadata.json")

        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"[{camera_id}][INFO] Stream ended or frame read error.")
                break

            current_time = time.time()
            if not buffer:
                buffer_start_time = current_time

            buffer.append(frame)

            if len(buffer) >= self.window_size:
                # Round start time to aligned window start
                interval_start = int(buffer_start_time // (self.window_size / self.fps)) * (self.window_size / self.fps)
                formatted_time = datetime.utcfromtimestamp(interval_start).strftime('%Y-%m-%d_%H-%M-%S')
                chunk_filename = f"{camera_id}_chunk_{formatted_time}.avi"
                chunk_path = os.path.join(camera_dir, chunk_filename)

                self._save_video_chunk(buffer[:self.window_size], chunk_path, self.fps)

                frames_to_remove = self.window_size - self.overlap_frames
                if frames_to_remove > 0:
                    buffer = buffer[frames_to_remove:]
                    if buffer:
                        buffer_start_time += frames_to_remove / self.fps

                # You can yield the document here or pass to callback/inference
                doc = Document(
                    page_content=f"[{camera_id}] Chunk saved at {chunk_path}",
                    metadata={
                        "chunk_path": chunk_path,
                        "timestamp": formatted_time,
                        "camera_id": camera_id,
                        "window_size": self.window_size,
                        "fps": self.fps,
                        "source": rtsp_url,
                    },
                )
                print(f"[{camera_id}] Metadata: {doc.metadata}")

                # Save metadata to JSON file
                with open(metadata_file_path, "a") as metadata_file:
                    json.dump(doc.metadata, metadata_file)
                    metadata_file.write("\n")

                executor.submit(run_vlm_inference, chunk_path, formatted_time, camera_id)

        cap.release()
        print(f"[{camera_id}] Stream processing complete.")

    def lazy_load(self) -> Iterator[Document]:
        threads = []
        for camera_id, rtsp_url in self.camera_sources.items():
            t = threading.Thread(target=self._process_camera_stream, args=(camera_id, rtsp_url))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

    def _save_video_chunk(self, frames: List, output_path: str, fps: int):
        if not frames:
            return
        height, width, _ = frames[0].shape
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))
        for frame in frames:
            out.write(frame)
        out.release()
        print(f"[SAVE] Chunk written to {output_path}")

# if __name__ == "__main__":
#     # rtsp_sources = {
#     #     "cam08": "rtsp://admin:intel123!@192.168.2.108",
#     #     "cam09": "rtsp://admin:intel123!@192.168.2.109",
#     #     "cam07": "rtsp://admin:intel123!@192.168.2.107",
#     #     "cam12": "rtsp://admin:intel123!@192.168.2.112",
#     # }
#     rtsp_sources = {
#         "cam08": "one-by-one-person-detection.mp4",
#         "cam09": "one-by-one-person-detection.mp4",
#     }

#     chunk_loader = MultiRTSPChunkLoader(
#         camera_sources=rtsp_sources,
#         chunk_type="sliding_window",
#         chunk_args={
#             "window_size": 85,
#             "fps": 15,
#             "overlap": 15,
#         },
#         output_dir="multi_cam_chunks"
#     )

#     print("[INFO] Starting multi-camera RTSP processing with VLM...")
#     chunk_loader.lazy_load()
