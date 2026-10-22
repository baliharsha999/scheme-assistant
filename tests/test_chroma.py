import chromadb

client = chromadb.PersistentClient(
    path="./data/ChromaDB"
)

print("Available collections:")

for collection in client.list_collections():
    print(collection.name)