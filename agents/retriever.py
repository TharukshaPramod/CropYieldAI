from crewai import Agent, Tool
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
import spacy
from openai import OpenAI
from utils.security import sanitize_input, encrypt
from utils.db import Session, CropData

nlp = spacy.load('en_core_web_sm')
embeddings = OpenAIEmbeddings()
client = OpenAI()

def load_vectorstore():
    session = Session()
    docs = [Document(page_content=f"{d.crop} {d.location} {d.data}") for d in session.query(CropData).all()]
    return FAISS.from_documents(docs, embeddings)

vectorstore = load_vectorstore()

def retrieve_tool(query: str) -> str:
    query = sanitize_input(query)
    doc = nlp(query)
    entities = {ent.label_: ent.text for ent in doc.ents}
    refined_query = f"{entities.get('PRODUCT', 'crop')} {entities.get('GPE', 'location')}"

    results = vectorstore.similarity_search(refined_query, k=3)
    raw_results = [res.page_content for res in results]

    prompt = f"Summarize crop data: {raw_results}. Explain relevance. Check bias (e.g., only US data?)."
    response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}])
    summary = response.choices[0].message.content

    print(f"Retrieval log: {summary}")
    return encrypt(summary)

retriever_agent = Agent(
    role="Retriever Agent",
    goal="Fetch and refine crop data securely and ethically",
    backstory="Integrates IR, NLP, LLM with Responsible AI checks",
    tools=[Tool(name="RetrieveData", func=retrieve_tool, description="Vector-based IR for crop info")]
)

if __name__ == "__main__":
    from utils.security import decrypt
    print(decrypt(retrieve_tool("wheat yield in California")))