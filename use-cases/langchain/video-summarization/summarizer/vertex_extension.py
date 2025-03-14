import os
import re
import vertexai
from vertexai.generative_models import GenerativeModel, Part

from google.oauth2 import service_account
from google.cloud import storage
import uuid
    
def generate_unique_bucket_name(prefix="sample-vidsumm-bucket"):
    bucket_id = uuid.uuid4().hex
    return f"{prefix}-{bucket_id}"

def upload_to_gcs(source_file_path, bucket_name, destination_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    destination = bucket.blob(destination_path)

    destination.upload_from_filename(source_file_path)
    gcs_url = f"gs://{bucket_name}/{destination_path}"
    print(f"Uploaded video to {gcs_url}")

    return gcs_url

def create_gc_bucket(bucket_name, location="US"):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    bucket.location = location
    bucket = client.create_bucket(bucket)
    print(f"Created bucket {bucket.name}")
    return bucket.name

class VertexWrapper(object):
    def __init__(self, model_name, bucket_name=None, gcloud_key=None):
        # Authenticate account (Maybe to do: authenticate in python)
        # self.credentials = service_account.Credentials.from_service_account_file(gcloud_key)
        
        # Load Model
        self.model_name = model_name
        self.model = GenerativeModel(model_name)
        
        # Set up gc storage bucket
        available_buckets = [b.name for b in storage.Client().list_buckets()]
        if len(available_buckets) == 0 and not bucket_name:
            bucket_name = generate_unique_bucket_name()
            self.bucket_name = create_gc_bucket(bucket_name)
        elif len(available_buckets) > 0 and not bucket_name:
            self.bucket_name = available_buckets[0]
        elif bucket_name in available_buckets:
            self.bucket_name = bucket_name
        else:
            self.bucket_name = create_gc_bucket(bucket_name)
            
    def generate(self, text_prompt, video_paths=None):
        if video_paths:
            parts = [text_prompt]
            for video_path in video_paths:
                video_url = upload_to_gcs(video_path, self.bucket_name,
                                os.path.join("tmp/{}".format(os.path.basename(video_path))))
                parts.append(Part.from_uri(uri=video_url, mime_type="video/mp4"))
            response = self.model.generate_content(parts)
        else:
            response = self.model.generate_content(text_prompt)
        return response.text

    def cleanup(self):
        client = storage.Client()
        bucket = client.bucket(self.bucket_name)
        blobs = list(bucket.list_blobs())
        
        for blob in blobs:
            if blob.name.startswith('tmp/'):
                blob.delete()
                
    @staticmethod
    def extract_anomaly_score(summary):
        # matching based on multiple scenarios observed; goal is to match floating point or integer after Anomaly Score
        # Anomaly Score sometimes is encapsulated within ** and sometimes LLM omits
        match = re.search(r"\*?\*?Anomaly Score\*?\*?:?\s*(-?\d+(\.\d+)?)", summary, re.DOTALL)
        if match:
            return float(match.group(1)) if match.group(1) else 0.0
        return 0.0
    
