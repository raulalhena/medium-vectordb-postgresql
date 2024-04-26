from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.embeddings.sentence_transformer import ( SentenceTransformerEmbeddings )

CONNECTION_STRING = 'postgresql+psycopg2://postgres:postgres@localhost:5432/medium_vectordb'
COLLECTION_NAME= 'art_of_war'

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

loader = TextLoader('./art_of_war.txt', encoding='utf-8')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=80)
texts = text_splitter.split_documents(documents)

db = PGVector.from_documents(embedding=embedding_function, documents=texts, collection_name=COLLECTION_NAME, connection_string=CONNECTION_STRING)

search_result = db.similarity_search("generales", k=10)

print(search_result)