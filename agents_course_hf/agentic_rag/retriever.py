import datasets 
from langchain.docstore.document import Document
from langchain_community.retrievers import BM25Retriever
from langchain.tools import Tool


# Load the dataset
guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")

# Convert dataset entries into Document objects 
docs = [
    Document(
        page_content="\n".join([
            f"Name: {guest['name']}",
            f"Relation: {guest['relation']}",
            f"Description: {guest['description']}",
            f"Email: {guest['email']}"
        ]),
        metadata={"name": guest["name"]}
    )
    for guest in guest_dataset
]

##TODO: work on a better retriver (sentence splitter)
bm25_retriever = BM25Retriever.from_documents(docs)

def extract_text(query: str) -> str:
    """Retrieves detailed information about gala guests based on their name or relations."""
    result = bm25_retriever.invoke(query)
    if result:
        return "\n\n".join([doc.page_content for doc in result])
    else: 
        return "No matching guest information found."
    
guest_info_tool = Tool(
    name="guest_info_retriever",
    func=extract_text,
    description="Retrieves detailed information about gala guests based on their name or relation."
)



