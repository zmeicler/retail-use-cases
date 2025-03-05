import os
import vertexai
from vertexai.generative_models import GenerativeModel, Part
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
    def __init__(self, model_name, bucket_name=None):
        # Load Model
        self.model_name = model_name
        self.model =  GenerativeModel(model_name)

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
            
    def generate(self, text_prompt, video_path=None):
        if video_path:
            video_url = upload_to_gcs(video_path, self.bucket_name,
                                os.path.join("tmp/{}".format(os.path.basename(video_path))))
            response = self.model.generate_content(
                [
                    Part.from_uri(uri=video_url, mime_type="video/mp4"),
                    text_prompt, 
                ]
            )        
        else:
            response = self.model.generate_content(text_prompt)
        return response

    def cleanup(self):
        client = storage.Client()
        bucket = client.bucket(self.bucket_name)
        blobs = list(bucket.list_blobs())
        
        for blob in blobs:
            if blob.name.startswith('tmp/'):
                blob.delete()
