from typing import List, Optional

import openvino_genai
from decord import VideoReader, cpu
from langchain.llms.base import LLM
from openvino import Tensor


def encode_video(video_path: str,
                 max_num_frames: int = 64,
                 resolution: list = []) -> list:
    def uniform_sample(l: list, n: int) -> list:
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    if len(resolution) != 0:
        vr = VideoReader(video_path, width=resolution[0],
                         height=resolution[1], ctx=cpu(0))
    else:
        vr = VideoReader(video_path, ctx=cpu(0))

    frame_idx = [i for i in range(0, len(vr), max(1, int(len(vr) / max_num_frames)))]
    if len(frame_idx) > max_num_frames:
        frame_idx = uniform_sample(frame_idx, max_num_frames)
    frames = vr.get_batch(frame_idx).asnumpy()

    frames = [Tensor(v.astype('uint8')) for v in frames]
    print('Num frames sampled:', len(frames))
    return frames


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
