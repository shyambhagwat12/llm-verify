# Verify-RCT Project

## Overview
Verify-RCT is designed to address  Evidence verification

## Components
- **Neo4j Database**: Graph database for managing clinical trial information.
- **Docker**: Used for containerizing the Neo4j database.
- **OpenAI Models**: Utilized for generating text embeddings.
- **Python**: Serves as the scripting language for processing data and interfacing with Neo4j and OpenAI.
- **Autogen** For Agentic/LLM RCT Evidence extraction
- **Dspy** Auto prompting

## Prerequisites
- Docker
- Python 3.9 or newer
- OpenAI API access
- Neo4j Docker Image

## Installation

### Setting up the Environment
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd your-project-directory
Install required Python libraries:

pip install -r requirements.txt
Configure Environment Variables
Create a .env file at the root of your project directory and add your credentials:

(Neo4j wil run in password no pass mode - so no need to configue neo4j pass)
OPENAI_API_KEY=your_openai_api_key

### Running Neo4j with Docker
Use Docker to run a Neo4j container with the following command:

docker run \
    --name neo4j-graphrct \
    --publish 7474:7474 \
    --publish 7687:7687 \
    --env NEO4J_AUTH=neo4j/test \
    neo4j:latest
This command sets up Neo4j and exposes it on ports 7474 (HTTP) and 7687 (Bolt), allowing for web interface access and programmatic database interactions, respectively.

### Usage
With the environment set up, you can now run your application scripts:

python main.py

This script will connect to the Neo4j database, process data using the OpenAI API, and populate the database with enriched content from clinical trials.

