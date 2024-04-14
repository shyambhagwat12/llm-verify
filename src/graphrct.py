from dspy import Module, Predict
from utils import EmbeddingGenerator
import dspy


class ICOExtraction(dspy.Signature):
    """Extract ICO (Intervention, Comparator, Outcome) from medical texts."""
    text = dspy.InputField()
    intervention = dspy.OutputField(desc="Medical intervention described")
    comparator = dspy.OutputField(desc="Comparator in the study")
    outcome = dspy.OutputField(desc="Outcome of the intervention")

class ReferenceExtraction(dspy.Signature):
    """Extracts references from the given text."""
    text = dspy.InputField()
    references = dspy.OutputField(desc="Extracted references from the article")

class GraphRCT(Module):
    def __init__(self, neo4j_db, embedding_generator, evidence_agent):
        super().__init__()
        self.neo4j_db = neo4j_db
        self.embedding_generator = embedding_generator
        self.extract_ico = Predict(ICOExtraction)
        self.extract_references = Predict(ReferenceExtraction)
        self.evidence_agent = evidence_agent

    def extract_evidence(self, text, ico_data):
        return self.evidence_agent.submit_task(text, ico_data)

    def forward(self, text):
        ico_data = self.extract_ico(text=text)
        references = self.extract_references(text=text)
        references_text = '\n'.join([ref['text'] for ref in references if 'text' in ref])
        

        cleaned_content = self.extract_evidence(text, ico_data)
        references = [{'id': 'ref0', 'text': cleaned_content}]
        print("Formatted References:", references) 
        ico_embedding = self.embedding_generator.generate_embedding(ico_data.intervention + " " + ico_data.comparator + " " + ico_data.outcome)

        cypher_query = self.generate_cypher(ico_data, references, ico_embedding)
        self.neo4j_db.execute_query(cypher_query)

        return f"Data updated in Neo4j: {cypher_query}"

    def generate_cypher(self, ico_data, references, ico_embedding):
        embedding_string = ', '.join(str(e) for e in ico_embedding)
        merge_statements = [
            f"MERGE (i:Intervention {{name: '{ico_data.intervention}', embedding: [{embedding_string}]}})",
            f"MERGE (c:Comparator {{name: '{ico_data.comparator}'}})",
            f"MERGE (o:Outcome {{description: '{ico_data.outcome}'}})",
            f"MERGE (i)-[:COMPARED_WITH]->(c)",
            f"MERGE (i)-[:LEADS_TO]->(o)"
        ]

        for ref in references:
            if isinstance(ref, dict) and 'text' in ref:
                ref_text = ref['text'].replace("'", "\\'")
                ref_id = ref.get('id', 'unknown_id')
                merge_statements.append(
                    f"MERGE (r:Reference {{id: '{ref_id}', text: '{ref_text}'}}) MERGE (i)-[:CITED_IN]->(r)"
                )

        return " ".join(merge_statements)
