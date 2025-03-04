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


def post_request(input_data):
    formatted_req = {
        "summaries": input_data
    }
    response = requests.post(url="http://127.0.0.1:8000/merge_summaries", json=formatted_req)
    return response.content


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
    parser.add_argument("-r", "--resolution", type=int, nargs=2,
                        help="Desired spatial resolution of input video if different than original. Width x Height")
    parser.add_argument("-o", "--outfile", type=str,
                        help="File to write generated text.", default='')
    parser.add_argument("-a", "--anomaly_thresh", type=float, default=0.7,
                        help="If pipeline has been extended to the cloud, set a threshold to determine if video is anamolous and shoul be evaluated by cloud model.")
    parser.add_argument("-e", "--extend_to_vertex", action="store_true", help='Use vertex ai to analayze anomalous videos.')
    
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
        cloud_model_name = "gemini-2.0-flash-exp"
        cloud_model = VertexWrapper(cloud_model_name)
    
    # Initialize video chunk loader
    loader = VideoChunkLoader(
        video_path=args.video_file,
        chunking_mechanism="sliding_window",
        chunk_duration=args.chunk_duration,
        chunk_overlap=args.chunk_overlap)

    # Start log
    output_handler("python " + " ".join(sys.argv),
                   filename=args.outfile, mode='w',
                   verbose=False)

    # Loop through docs and generate chunk summaries    
    chunk_summaries = {}
    cnt = 0
    for doc in loader.lazy_load():
        # if cnt == 1:
        #     break
        # Log metadata
        output_handler(str(f"Chunk Metadata: {doc.metadata}"),
                       filename=args.outfile, mode='a')
        output_handler(str(f"Chunk Content: {doc.page_content}"),
                       filename=args.outfile, mode='a')

        # Generate summaries
        chunk_st_time = time.time()
        video_name = Path(doc.metadata['chunk_path'])
        inputs = {"video": video_name, "question": args.prompt}
        output = chain.invoke(inputs)
        
        # Log output
        output_handler(output, filename=args.outfile, mode='a', verbose=False)
        chunk_summaries[Path(doc.metadata['chunk_path']).stem] = f"Start time: {doc.metadata['start_time']} End time: {doc.metadata['end_time']}\n" + output
        output_handler("\nChunk Inference time: {} sec\n".format(time.time() - chunk_st_time), filename=args.outfile,
                       mode='a')
        cnt += 1
    # Summarize the full video, using the subsections summaries from each chunk    
    overall_summ_st_time = time.time()
    
    # two ways to get overall_summary and anomaly score:
    # 1. refer to test_api.py to use post an HTTP request to call API wrapper summary merger (uses llama3.2)
    with ThreadPoolExecutor() as pool:
        future = pool.submit(post_request, chunk_summaries)
        res = ast.literal_eval(future.result().decode("utf-8"))

        print(f"Overall Summary: {res['overall_summary']}")
        print(f"Anomaly Score: {res['anomaly_score']}")

        if args.extend_to_vertex and res['anomaly_score'] >= args.anomaly_thresh:
            cloud_st_time = time.time()
            cloud_prompt = 'Please determine if this video is anomalous given the following summary: {}'.format(res['overall_summary'])
            cloud_response = cloud_model.generate(cloud_prompt)
            output_handler("\n--Vertex AI Evaluation--\n", 
                           filename=args.outfile,
                           mode='a')                                      
            output_handler(cloud_response.text, 
                           filename=args.outfile,
                           mode='a')                          
            output_handler("\nVertex Inference time: {} sec\n".format(time.time() - cloud_st_time),
                           filename=args.outfile,
                           mode='a')
            
    output_handler("\nOverall video summary inference time: {} sec\n".format(time.time() - overall_summ_st_time),
                   filename=args.outfile, mode="a")

    # 2. pass existing minicpm based chain, this does not use the FastAPI route and calls the class functions directly
    # summary_merger = SummaryMerger(chain=chain, device="GPU")
    # ret = summary_merger.merge_summaries(chunk_summaries)
    # print(ret)

    output_handler("\nTotal Inference time: {} sec\n".format(time.time() - tot_st_time), filename=args.outfile,
                   mode='a')
    output_handler(output, filename=args.outfile, mode='a', verbose=False)
