from typing import List, Optional

import openvino_genai
from decord import VideoReader, cpu
from langchain.llms.base import LLM
from openvino import Tensor as ovTensor

import torch
from torch import Tensor
import torch.nn.functional as F

def encode_video(video_path: str,
                 max_num_frames: int = 64,
                 resolution: list = []) -> list:

    # Create cross entopy closure functionality
    def compute_cross_entropy(frames: torch.Tensor) -> torch.Tensor:
        if frames.shape[0] < 2:
            return torch.tensor([])  # Return empty tensor if not enough frames
        
        frames = frames.float() / 255.0  # Normalize pixel values
        
        probs = frames[:-1]  # Previous frame as input
        targets = frames[1:]  # Next frame as target labels
        
        b, h, w, c = probs.shape
        probs = probs.permute(0, 3, 1, 2)  # Change to (batch, channels, height, width)
        targets = targets.permute(0, 3, 1, 2)  # Same for targets
        
        cross_entropy = F.cross_entropy(probs, targets, reduction='none')
        return cross_entropy.mean(dim=-1)  # Aggregate per-frame entropy scores

    # Load Video
    if len(resolution) != 0:
        vr = VideoReader(video_path, width=resolution[0],
                         height=resolution[1], ctx=cpu(0))
    else:
        vr = VideoReader(video_path, ctx=cpu(0))

    # Calculate cross entropy
    frames = vr.get_batch(range(len(vr))).asnumpy()
    frames_tensor = torch.stack([Tensor(v.astype('uint8')) for v in frames])    
    cross_entropy_scores = compute_cross_entropy(frames_tensor)

    # Sample frames
    num_available_frames = cross_entropy_scores.shape[0]
    num_frames_to_sample = min(max_num_frames, num_available_frames)    
    top_entropy_indices = torch.topk(cross_entropy_scores, num_frames_to_sample).indices.sort().values.tolist()        
    sampled_frames = frames_tensor[top_entropy_indices]
    
    print('Num frames sampled:', len(sampled_frames))
    return [ovTensor(v.numpy().astype('uint8')) for v in sampled_frames]
    # return [v for v in sampled_frames]

# def encode_video(video_path: str,
#                  max_num_frames: int = 64,
#                  resolution: list = []) -> list:
#     def uniform_sample(l: list, n: int) -> list:
#         gap = len(l) / n
#         idxs = [int(i * gap + gap / 2) for i in range(n)]
#         return [l[i] for i in idxs]

#     if len(resolution) != 0:
#         vr = VideoReader(video_path, width=resolution[0],
#                          height=resolution[1], ctx=cpu(0))
#     else:
#         vr = VideoReader(video_path, ctx=cpu(0))

#     frame_idx = [i for i in range(0, len(vr), max(1, int(len(vr) / max_num_frames)))]
#     if len(frame_idx) > max_num_frames:
#         frame_idx = uniform_sample(frame_idx, max_num_frames)
#     frames = vr.get_batch(frame_idx).asnumpy()

#     frames = [Tensor(v.astype('uint8')) for v in frames]
#     print('Num frames sampled:', len(frames))
#     return frames


def streamer(subword: str) -> bool:
    '''

    Args:
        subword: sub-word of the generated text.

    Returns: Return flag corresponds whether generation should be stopped.

    '''
    print(subword, end='', flush=True)

    # No value is returned as in this example we don't want to stop the generation in this method.
    # "return None" will be treated the same as "return False".


class OVMiniCPMV26Wrapper(LLM):
    ovpipe: object
    generation_config: object
    max_num_frames: int
    resolution: list[int]

    @property
    def _llm_type(self) -> str:
        return "Custom OV MiniCPM-V-2_6"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
    ) -> str:

        # Parse prompt
        video_fh, question = prompt.split(',', 1)

        # Process text only
        if video_fh == '':
            self.ovpipe.start_chat()
            generated_text = self.ovpipe.generate(question,
                                                  generation_config=self.generation_config,
                                                  streamer=streamer)

        # Process video and text
        else:
            frames = encode_video(video_fh, self.max_num_frames,
                                  resolution=self.resolution)
            self.ovpipe.start_chat()
            generated_text = self.ovpipe.generate(question,
                                                  images=frames,
                                                  generation_config=self.generation_config,
                                                  streamer=streamer)

        self.ovpipe.finish_chat()
        return str(generated_text)


def OVMiniCPMV26Worker(model_dir: str,
                       device: str,
                       max_new_tokens: int,
                       max_num_frames: int,
                       resolution: list[int]) -> object:
    # Start ov genai pipeline
    enable_compile_cache = dict()
    if "GPU" == device:
        # Cache compiled models on disk for GPU to save time on the
        # next run. It's not beneficial for CPU.
        enable_compile_cache["CACHE_DIR"] = "vlm_cache"

    pipe = openvino_genai.VLMPipeline(model_dir, device, **enable_compile_cache)

    # Set variables for inference 
    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = max_new_tokens

    # Wrap for langchain integration
    ovminicpm_wrapper = OVMiniCPMV26Wrapper(ovpipe=pipe,
                                            generation_config=config,
                                            max_num_frames=max_num_frames,
                                            resolution=resolution)
    return ovminicpm_wrapper
