from sentence_transformers import SentenceTransformer, util
import numpy as np
import json
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Init is ran on server startup
# Load your model to GPU as a global variable here.
def init():
    global model
    global tokenizer
    model = SentenceTransformer('msmarco-distilbert-base-tas-b')
    # conditionally load to GPU
    if device == "cuda:0":
        print("loading to GPU...")
        model.cuda()
        print("done")

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    chunks = model_inputs.get('chunks', None)
    content_vector = np.array([chunk['text'] for chunk in chunks])
    
    vectors = model.encode(content_vector, show_progress_bar=True)

    for idx, chunk in enumerate(chunks):
        chunk['vector'] = json.dumps(vectors[idx], cls=NumpyEncoder)
        

    return chunks