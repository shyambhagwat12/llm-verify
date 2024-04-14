from openai import OpenAI
import openai
from datasets import load_dataset

class EmbeddingGenerator:
    def __init__(self, api_key,model):
        self.api_key = api_key
        openai.api_key = self.api_key
        self.model=model

    def generate_embedding(self, text):
        try:
            client = OpenAI(
            api_key=self.api_key, 
            )
            response = client.embeddings.create(
                input=[text],
                model=self.model
            )
          
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
        

def load_dataset_entries(limit=50):
    dataset = load_dataset("evidence_infer_treatment", '2.0', split='train')
    entries = []
    for i, entry in enumerate(dataset):
        if i >= limit:
            break
        example_text = entry['Text']
        entries.append(example_text)
    return entries