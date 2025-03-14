import argparse
import ast
import os
import sys
import time
import json
from itertools import tee
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path

import requests
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders.video import VideoChunkLoader

from ov_lvm_wrapper import OVMiniCPMV26Worker

from vertex_extension import VertexWrapper

os.environ["no_proxy"] = "localhost,127.0.0.1"

def post_request(input_data):
    formatted_req = {
        "summaries": input_data
    }
    response = requests.post(url="http://127.0.0.1:8000/merge_summaries", json=formatted_req)
    return response.content

def tag_last(generator):
    # Create two independent iterators from the original generator    
    gen1, gen2 = tee(generator)

    # Advance gen2 by one step to look ahead at the next item    
    next(gen2, None)

    # Iterate through both simultaneously
    for current, next_item in zip(gen1, gen2):
        yield current, False

    # Exit when next_item is the last item, and yield it with True        
    yield next_item, True
    
if __name__ == '__main__':
    # Parse inputs
    parser_txt = "Generate video summarization using LangChain, OpenVINO-genai, and MiniCPM-V-2_6."
    parser = argparse.ArgumentParser(parser_txt)
    parser.add_argument("video_file", type=str,
                        help='Path to video you want to summarize.')
    parser.add_argument("model_dir", type=str,
                        help="Path to openvino-genai optimized model")
    parser.add_argument("-p", "--prompt", type=str,
                        help="Text prompt. By default set to: `Please summarize this video.`",
                        default="Please summarize this video.")
    parser.add_argument("-d", "--device", type=str,
                        help="Target device for running ov MiniCPM-v-2_6",
                        default="CPU")
    parser.add_argument("-t", "--max_new_tokens", type=int,
                        help="Maximum number of tokens to be generated.",
                        default=500)
    parser.add_argument("-f", "--max_num_frames", type=int,
                        help="Maximum number of frames to be sampled per chunk for inference. Set to a smaller number if OOM.",
                        default=32)
    parser.add_argument("-c", "--chunk_duration", type=int,
                        help="Maximum length in seconds for each chunk of video.",
                        default=30)
    parser.add_argument("-v", "--chunk_overlap", type=int,
                        help="Overlap in seconds between chunks of input video.",
                        default=2)
    parser.add_argument("-C", "--merge_cadence", type=int,
                        help="Cadence (sec) in which to merge chunk summaries")
    parser.add_argument("-r", "--resolution", type=int, nargs=2,
                        help="Desired spatial resolution of input video if different than original. Width x Height")
    parser.add_argument("-o", "--outfile", type=str,
                        help="File to write generated text.", default='')
    parser.add_argument("-e", "--extend_to_vertex", action="store_true", help='Extend pipeline to vertexai to analayze anomalous videos.')
    parser.add_argument("-a", "--anomaly_thresh", type=float, default=0.7,
                        help="If pipeline has been extended to the cloud, set a threshold to determine if video is anomalous and should be reevaluated by cloud model.")
    parser.add_argument("-m", "--cloud_model", type=str, default="gemini-2.0-flash-exp",
                        help="Name of google model to use if the pipeline has been extended to the cloud.")
    
    init_st_time = time.time()
    args = parser.parse_args()
    if not os.path.exists(args.video_file):
        print(f"{args.video_file} does not exist.")
        exit()
            
    # Create template for inputs
    prompt = PromptTemplate(
        input_variables=["video", "question"],
        template="{video},{question}"
    )

    # Wrap OpenVINO-GenAI optimized model in custom langchain wrapper
    resolution = [] if not args.resolution else args.resolution
    ov_minicpm = OVMiniCPMV26Worker(model_dir=args.model_dir,
                                    device=args.device,
                                    max_new_tokens=args.max_new_tokens,
                                    max_num_frames=args.max_num_frames,
                                    resolution=resolution)
    
    # Create pipeline
    chain = prompt | ov_minicpm

    # Initialize cloud model
    if args.extend_to_vertex:
        cloud_model = VertexWrapper(args.cloud_model)
        cloud_prompt = args.prompt + 'Please analyze all attached videos as if they were combined into a single video. In addition, the last information produced must be a score between 0 and 1 to represent how suspicious the the video is. The score should be a float rounded to the tenth decimal and formatted as the following example: \n **anomaly score**: 0.0'
        
    # Initialize video chunk loader
    loader = VideoChunkLoader(
        video_path=args.video_file,
        chunking_mechanism="sliding_window",
        chunk_duration=args.chunk_duration,
        chunk_overlap=args.chunk_overlap)

    # Define cadence at which we'll merge chunk summaries. 
    merge_cadence = max(1, int(args.merge_cadence / args.chunk_duration)) if \
        args.merge_cadence else float('inf')
    print("\nInitialization Time: {} sec\n".format(time.time() - init_st_time))
            
    # Loop through docs and generate chunk summaries
    mode = "w"    
    chunk_summaries = {}
    merge_start_time = 0    
    last_chunk_processed = 0
    tot_inf_st_time = time.time()
    for doc, is_last in tag_last(loader.lazy_load()):
        
        # Generate chunk summary
        chunk_st_time = time.time()
        video_name = Path(doc.metadata['chunk_path'])
        inputs = {"video": video_name, "question": args.prompt}
        output = chain.invoke(inputs)
        chunk_summaries[Path(doc.metadata['chunk_path']).stem] = f"Start time: {doc.metadata['start_time']} End time: {doc.metadata['end_time']}\n" + output
        print("Chunk Summary Time: {} sec\n".format(time.time() - chunk_st_time))
        
        # Merge chunk summaries at specified cadence
        call_merger = (doc.metadata['chunk_id']+1) % merge_cadence == 0
        if merge_cadence == float('inf') and not is_last:
            # If we're asked to make a summary of full input video, don't merge yet.
            continue
        if call_merger or (not call_merger and is_last):            
            # Get merged summary and anomoly score
            merge_st_time = time.time()
            with ThreadPoolExecutor() as pool:
                future = pool.submit(post_request, chunk_summaries)
                merge_res = ast.literal_eval(future.result().decode("utf-8"))
            print("Merge Chunks Time: {} sec\n".format(time.time() - merge_st_time))
            
            # Extend to cloud, if asked
            if args.extend_to_vertex and merge_res['anomaly_score'] >= args.anomaly_thresh:
                cloud_st_time = time.time()                
                chunk_ids_to_process = range(last_chunk_processed,
                                             doc.metadata['chunk_id'] + 1)
                merged_chunks = [os.path.join(loader.output_dir,
                                              "chunk_{}.mp4".format(ch_id))\
                                 for ch_id in chunk_ids_to_process]
                cloud_response = cloud_model.generate(cloud_prompt,
                                                      video_paths=merged_chunks)
                anomaly_score = cloud_model.extract_anomaly_score(cloud_response)
                merge_res = {'overall_summary': cloud_response,
                             'anomaly_score': anomaly_score}                
                cloud_model.cleanup()
                print("Cloud Summary Time: {} sec\n\n".format(time.time() - cloud_st_time))
                last_chunk_processed = doc.metadata['chunk_id'] + 1
                
            # Write output to JSON file, if asked
            if args.outfile:
                merge_res["start_time"] = merge_start_time
                merge_res["end_time"] = doc.metadata['end_time']
                FH = open(args.outfile, mode)
                json.dump(merge_res, FH, indent=4)
                FH.write("\n")
                FH.close()                
                if mode == "w":
                    mode = "a"
            merge_start_time = doc.metadata['end_time'] - args.chunk_overlap
            
    # Report full inference time
    print("\nTotal Inference Time: {} sec\n".format(time.time() - tot_inf_st_time))
