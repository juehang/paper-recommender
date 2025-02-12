import chromadb

class VectorStore:
    def __init__(self, embedding):
        self.embedding = embedding

class ChromaVectorStore(VectorStore):
    """
    A vector store that uses ChromaDB.
    """
    def __init__(self, embedding, path=None):
        self.client = chromadb.Client(path=path)
        super().__init__(embedding)
        self.collection = self.client.get_or_create_collection(
            name="papers",
            embedding_function=self.embedding.get_embedding,
            metadata={"hnsw_space": "cosine"},
            )

    def add_document(self, document, links, rating):
        """
        Adds a document to the vector store.
        Args:
           document (string): The document to add.
            links (list): A list of URLs associated with the document.
        Returns:
            None
        """
        self.collection.add(
            documents=[document],
            metadatas={"links": links, "rating": rating},
            )
        return None
    
    def search(self, query_texts, num_results=10):
        """
        Searches for a query in the vector store.
        Args:
           query_texts (list): A list of queries to search for.
           Returns:
           list: A list of results.
        """
        results = self.collection.query(
            query_texts=query_texts,
            n_results=num_results,
            include=["metadatas", "distances"],
            )
        return results