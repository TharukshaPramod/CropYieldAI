from crewai import Agent
from crewai.tools import BaseTool
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import spacy
from utils.security import sanitize_input, encrypt
from utils.db import Session, CropData

nlp = spacy.load('en_core_web_sm')
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_vectorstore():
    session = Session()
    docs = [Document(page_content=f"{d.crop} {d.location} {d.data}") for d in session.query(CropData).all()]
    session.close()
    if not docs:
        raise ValueError("No data found in CropData table. Please populate the database with load_sample_data().")
    return FAISS.from_documents(docs, embeddings)

vectorstore = load_vectorstore()

class RetrieveDataTool(BaseTool):
    name: str = "RetrieveData"  # Add type annotation for name
    description: str = "Vector-based IR for crop info"  # Add type annotation for description

    def _run(self, query: str) -> str:
        query = sanitize_input(query)
        doc = nlp(query)
        entities = {ent.label_: ent.text for ent in doc.ents}
        refined_query = f"{entities.get('PRODUCT', 'crop')} {entities.get('GPE', 'location')}"

        results = vectorstore.similarity_search(refined_query, k=3)
        raw_results = [res.page_content for res in results]

        summary = f"Retrieved data: {raw_results}" if raw_results else "No data found."
        print(f"Retrieval log: {summary}")
        return encrypt(summary)

retriever_agent = Agent(
    role="Retriever Agent",
    goal="Fetch and refine crop data securely and ethically",
    backstory="Integrates IR, NLP, LLM with Responsible AI checks",
    tools=[RetrieveDataTool()]  # Pass a list with an instance of the custom tool
)

if __name__ == "__main__":
    from utils.security import decrypt
    print(decrypt(retriever_agent.tools[0]._run("wheat yield in California")))