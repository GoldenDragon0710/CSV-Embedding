import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders.csv_loader import CSVLoader
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_KEY = os.getenv("PINECONE_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

if __name__ == '__main__':
    # Load CSV data into Langchain
    csv_loader = CSVLoader(
        'dataset1.csv', 
        encoding="utf-8",
        csv_args={
            # 'quotechar': '"',
            'delimiter': ',',
            'fieldnames': ['furniture','type','url','rate','delivery','sale','price']
        },
    )
    documents = csv_loader.load()

    # Split documents into smaller chunks using Langchain's CharacterTextSplitter
    text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # embed the text using LangChain
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # initialize pinecone
    pinecone.init(
        api_key = PINECONE_KEY,
        environment = PINECONE_ENV
    )
    index_name = "langchain-openai"
    namespace = "whatsapp"

    docsearch = Pinecone.from_texts(
    [t.page_content for t in texts], embeddings, index_name=index_name, namespace=namespace)
