import os
from dotenv import load_dotenv
from dspy import Module, Predict
import dspy
from graphrct import ICOExtraction
from utils import EmbeddingGenerator, load_dataset_entries, Neo4jDatabase
import openai

#THIS NEEDS MORE WORK. - REASONING,FEW SHOT SAMPLES,INTEGRATION WITH METRIC

class Verify(dspy.Signature):
    """Verify the  evidence regarding the outcome based on ICO and related article content."""
    ico_data = dspy.InputField()
    article_content = dspy.InputField()
    significant_change = dspy.OutputField(desc="Conclusion about the outcome: significant increase, significant decrease, or no significant difference")

class SemanticSearch(Module):
    def __init__(self, neo4j_db):
        super().__init__()
        self.neo4j_db = neo4j_db

    def search_related_articles(self, embedding):
        query = """
        MATCH (i:Intervention)
        WITH i, $embedding AS query_embedding
        RETURN i AS Intervention, gds.alpha.similarity.cosine(i.embedding, query_embedding) AS similarity
        ORDER BY similarity DESC
        LIMIT 10
        """
        parameters = {'embedding': embedding}
        return self.neo4j_db.execute_query(query, parameters)

class VerifyRCT(Module):
    def __init__(self, neo4j_db, embedding_generator, Verify):
        super().__init__()
        self.neo4j_db = neo4j_db
        self.embedding_generator = embedding_generator
        self.extract_ico = Predict(ICOExtraction)
        self.evidence_inference = Predict(Verify)
        self.semantic_search = SemanticSearch(neo4j_db)

    def forward(self, text):
        ico_data = self.extract_ico(text=text)
        ico_string = f"{ico_data.intervention} {ico_data.comparator} {ico_data.outcome}"
        ico_embedding = self.embedding_generator.generate_embedding(ico_string)

        related_articles = self.semantic_search.search_related_articles(ico_embedding)
        evidence_output = self.evidence_inference(ico_data=ico_string, article_content=related_articles)
        print("Inferred Conclusion:", evidence_output.significant_change)

