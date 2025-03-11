import argparse
import ast
import os
import sys
import time
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path

import requests
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders.video import VideoChunkLoader

from ov_lvm_wrapper import OVMiniCPMV26Worker

from vertex_extension import VertexWrapper

os.environ["no_proxy"] = "localhost,127.0.0.1"

def output_handler(text: str,
                   filename: str = '',
                   mode: str = 'w',
                   verbose: bool = True):
    # Print to terminal
    if verbose:
        print(text)

    # Write to file, if requested
    if filename != '':
        with open(filename, mode) as FH:
            print(text, file=FH)

def post_request(input_data):
    formatted_req = {
        "summaries": input_data
    }
    response = requests.post(url="http://127.0.0.1:8000/merge_summaries", json=formatted_req)
    return response.content

def call_merger(chunk_summaries, chain=None):
    if not chain:
        with ThreadPoolExecutor() as pool:
            future = pool.submit(post_request, chunk_summaries)
            res = ast.literal_eval(future.result().decode("utf-8"))
    else:
        summary_merger = SummaryMerger(chain=chain, device="GPU")
        res = summary_merger.merge_summaries(chunk_summaries)
    print(f"Overall Summary: {res['overall_summary']}")
    print(f"Anomaly Score: {res['anomaly_score']}")        
    return res

def call_vertex(prompt, videos=[]):
    cloud_st_time = time.time()
    cloud_response = cloud_model.generate(prompt, video_paths=videos)
    output_handler(cloud_response, 
                   filename=args.outfile,
                   mode='a')                          
    output_handler("\nVertex Inference time: {} sec\n".format(time.time() - cloud_st_time),
                   filename=args.outfile,
                   mode='a')    
    cloud_model.cleanup()
    return cloud_response

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
    
    tot_st_time = time.time()
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
        merge_prompt = args.prompt + 'Please analyze all attached videos as if they were combined into a single video. In addition, the last information produced must be a score between 0 and 1 to represent how suspicious the the video is. The score should be a float rounded to the tenth decimal and formatted as the following example: \n **anomaly score**: 0.0'
        validation_prompt = "Please determine if the following summaries agree. The summaries will be separated by the delimiter: <>. The last thing produced must be a value of either 0 (the summaries do not agree) or 1 (the summaries do agree). It should be formatted as the following example: **Validation Score**: 1\n"
        
    # Initialize video chunk loader
    loader = VideoChunkLoader(
        video_path=args.video_file,
        chunking_mechanism="sliding_window",
        chunk_duration=args.chunk_duration,
        chunk_overlap=args.chunk_overlap)

    # Define cadence at which we'll merge chunk summaries
    merge_cadence = max(1, int(args.merge_cadence / args.chunk_duration)) if \
        args.merge_cadence else None
    
    # Start log
    output_handler("python " + " ".join(sys.argv),
                   filename=args.outfile, mode='w',
                   verbose=False)
        
    # Loop through docs and generate chunk summaries
    chunk_summaries = {}
    last_chunk_processed = 0
    for doc in loader.lazy_load():
        
        # Generate chunk summaries
        chunk_st_time = time.time()
        video_name = Path(doc.metadata['chunk_path'])
        inputs = {"video": video_name, "question": args.prompt}
        output = chain.invoke(inputs)
        chunk_summaries[Path(doc.metadata['chunk_path']).stem] = f"Start time: {doc.metadata['start_time']} End time: {doc.metadata['end_time']}\n" + output

        # Log output
        output_handler("\nChunk Inference time: {} sec\n".format(time.time() - chunk_st_time), filename=args.outfile, mode='a')
        
        # Merge chunk summaries if asked to do so at a cadence
        if merge_cadence and (doc.metadata['chunk_id'] + 1) % merge_cadence == 0:
            merge_res = call_merger(chunk_summaries)
            
            # Extend to cloud, if asked
            if args.extend_to_vertex and merge_res['anomaly_score'] >= args.anomaly_thresh:
                output_handler(f"\n--Vertex AI Evaluation:{doc.metadata['chunk_id']}--\n",
                               filename=args.outfile,
                               mode='a')
                chunk_ids_to_process = range(max(0, doc.metadata['chunk_id'] - merge_cadence),
                                             doc.metadata['chunk_id'] + 1)
                merged_chunks = [os.path.join(loader.output_dir, "chunk_{}.mp4".format(ch_id))\
                                 for chidx in chunk_ids_to_process]
                vertex_summary, vertex_score = call_vertex(merge_prompt,
                                                           videos=merged_chunks).split("**anomaly score**:")
                vertex_res = {"overall_summary": vertex_summary, "anomaly_score": float(vertex_score)}
                                
                # Call vertex to validate mergeed output                
                validation_res = call_vertex(validation_prompt + merge_res["overall_summary"] + "<>" + vertex_res["overall_summary"])
                _, validation_score = validation_res.split("**validation score**:")
                if int(validation_score) == 0:
                    merge_res = vertex_res
                last_chunk_processed = doc.metadata['chunk_id']
                
    # Process any leftover chunks, if asked
    if not merge_cadence or ((doc.metadata['chunk_id'] + 1) % merge_cadence != 0):
        # merge_res = call_merger(chunk_summaries)
        merge_res = {'anomaly_score': 0.0,
                     'overall_summary': 'Not suspicious'}

        # Extend to cloud, if asked        
        if args.extend_to_vertex and merge_res['anomaly_score'] >= args.anomaly_thresh:
            output_handler(f"\n--Vertex AI Evaluation:{doc.metadata['chunk_id']}--\n",
                           filename=args.outfile,
                           mode='a')              
            chunk_ids_to_process=range(last_chunk_processed + 1,
                                       doc.metadata['chunk_id'] + 1)
            merged_chunks=[os.path.join(loader.output_dir, "chunk_{}.mp4".format(ch_id)) for ch_id in chunk_ids_to_process]
            vertex_summary, vertex_score = call_vertex(merge_prompt,
                                                       videos=merged_chunks).split("**anomaly score**:")
            vertex_res = {"overall_summary": vertex_summary, "anomaly_score": float(vertex_score)}

            # Call vertex to validate mergeed output                
            validation_res = call_vertex(validation_prompt + merge_res["overall_summary"] + "<>" + vertex_res["overall_summary"])
            _, validation_score = validation_res.split("**validation score**:")
            if int(validation_score) == 0:
                merge_res = vertex_res
            last_chunk_processed = doc.metadata['chunk_id']
    
    # Report full inference time
    output_handler("\nTotal Inference time: {} sec\n".format(time.time() - tot_st_time), filename=args.outfile,
                   mode='a')
    output_handler(output, filename=args.outfile, mode='a', verbose=False)
