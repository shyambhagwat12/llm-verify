import os
from dotenv import load_dotenv
from graphdb import Neo4jDatabase
from utils import EmbeddingGenerator, load_dataset_entries
from evidenceagent import EvidenceAutoAgent
from graphrct import GraphRCT
import dspy
from openai import OpenAI
import openai


load_dotenv()

if __name__ == "__main__":

    model = os.getenv('MODEL')
    api_key = os.getenv('API_KEY')
    api_base_url = os.getenv('API_BASE_URL')
    api_version = os.getenv('API_VERSION')

    openai_key = os.getenv('OPENAI_KEY')
    openai_embeddings_model = os.getenv('OPENAI_EMBEDDINGS_MODEL')

    turbo = dspy.OpenAI(model='gpt-3.5-turbo', api_key=openai_key)

    dspy.settings.configure(lm=turbo)

    openai.api_key = openai_key

    client = OpenAI(api_key=openai.api_key)

    uri = os.getenv('NEO4J_URI')
    user = os.getenv('NEO4J_USER')
    password = os.getenv('NEO4J_PASSWORD')
    neo4j_db = Neo4jDatabase(uri=uri, user=user, password=password)
    try:
        embedding_generator = EmbeddingGenerator(api_key=openai_key, model=openai_embeddings_model)
        evidence_agent = EvidenceAutoAgent(model, api_key, api_base_url, api_version)
        graphRCT = GraphRCT(neo4j_db, embedding_generator, evidence_agent)

        dataset_entries = load_dataset_entries(limit=50)
        

        for example_text in dataset_entries:
            result = graphRCT.forward(example_text)
            print(result)
    finally:
        neo4j_db.close()
