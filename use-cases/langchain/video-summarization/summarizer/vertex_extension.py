import vertexai
from vertexai.generative_models import GenerativeModel, Part

# TO DO: Create dataset

# TO DO: Upload to dataset

class VertexWrapper(object):
    def __init__(self, model_name):
        self.model_name = model_name
        self.model =  GenerativeModel(model_name)
        
    def generate(self, text, video_path=None):
        if video_path:
            response = self.model.generate_content(
                [
                    Part(video_path, mime_type="video/mp4"),
                    "Summarize this video please.", 
                ]
            )        
        else:
            response = self.model.generate_content(text)
        return response
