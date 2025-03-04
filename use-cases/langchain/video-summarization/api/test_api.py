import ast
import os
from concurrent.futures.thread import ThreadPoolExecutor

import requests

# ignore proxies for localhost
os.environ["no_proxy"] = "localhost,127.0.0.1"


def post_request(input_data):
    # convert to required format before posting request
    formatted_req = {
        "summaries": input_data
    }
    response = requests.post(url="http://127.0.0.1:8000/merge_summaries", json=formatted_req)
    return response.content


if __name__ == "__main__":
    summaries = {
        "chunk_0": "Start time: 0 End time: 30\n**Overall Summary**\nThe video captures a sequence of moments inside a "
                   "retail store, focusing on the checkout area and the surrounding aisles. The timestamp indicates the "
                   "footage was taken on Tuesday, May 21, 2024, at 06:42:42.\n\n**Activity Observed**\n1. The video shows "
                   "a relatively empty store with no visible customers at the checkout counter.\n2. The shelves on the "
                   "right side are stocked with various products, and the floor is clean and clear of obstructions.\n3. "
                   "There is a green waste bin placed near the checkout counter.\n4. The store appears well-lit, "
                   "and the checkout area is equipped with modern electronic devices.\n\n**Potential Suspicious "
                   "Activity**\n1. There is no visible evidence of shoplifting or any suspicious behavior in the provided "
                   "frames. The store appears orderly, and there are no signs of tampering or "
                   "theft.\n\n**Conclusion**\nBased on the analysis, the video shows a typical scene in a retail store "
                   "with no immediate signs of shoplifting or suspicious activity. The store is clean, organized, "
                   "and operational without any disturbances.",
        "chunk_1": "Start time: 28 End time: 30.002969\n**Overall Summary**\nThe video captures a sequence of moments "
                   "inside a retail store, focusing on the checkout area and the surrounding aisles. The timestamp "
                   "indicates the footage was taken on Tuesday, May 21, 2024, at 06:42:52.\n\n**Activity Observed**\n1. "
                   "The video shows a cashier's station with a computer monitor and a cash drawer.\n2. The aisles are "
                   "stocked with various products, including snacks and beverages.\n3. There is a visible customer "
                   "interaction area near the checkout counter.\n4. The floor is clean and well-maintained.\n5. The store "
                   "appears to be open and operational during the time the video was recorded.\n\n**Potential Suspicious "
                   "Activity**\n1. No overt signs of shoplifting or suspicious behavior are observed in the provided "
                   "frames. The cashier and the customer interaction area remain empty throughout the "
                   "sequence.\n\n**Conclusion**\nBased on the analysis, there is no evidence of shoplifting or suspicious "
                   "activity in the provided video frames. The store appears to be functioning normally without any "
                   "immediate concerns."
    }

    # send the request to the FastAPI endpoint using a ThreadPoolExecutor for async processing
    with ThreadPoolExecutor() as pool:
        print("Processing...")
        future = pool.submit(post_request, summaries)
        res = ast.literal_eval(future.result().decode("utf-8"))

        print(f"Overall Summary: {res['overall_summary']}")
        print(f"Anomaly Score: {res['anomaly_score']}")
