import neo4j


class Neo4jDatabase:
    def __init__(self, uri, user, password):
        self.driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def execute_query(self, query, parameters=None):
        with self.driver.session() as session:
            return session.write_transaction(lambda tx: tx.run(query, parameters).data())

    def read_query(self, query, parameters=None):
        with self.driver.session() as session:
            return session.read_transaction(lambda tx: tx.run(query, parameters).data())

    def fetch_interventions_and_references(self):
        query = """
        MATCH (i:Intervention)-[:COMPARED_WITH]->(c:Comparator),
              (i)-[:LEADS_TO]->(o:Outcome),
              (i)-[:CITED_IN]->(r:Reference)
        RETURN i.name AS Intervention, c.name AS Comparator, o.description AS Outcome, r.text AS ReferenceText
        """
        return self.read_query(query)
