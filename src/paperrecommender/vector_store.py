import chromadb
import uuid
import time

class VectorStore:
    def __init__(self, embedding):
        self.embedding = embedding

class ChromaVectorStore(VectorStore):
    """
    A vector store that uses ChromaDB.
    """
    def __init__(self, embedding, path=None):
        if path:
            self.client = chromadb.PersistentClient(path=path)
        else:
            self.client = chromadb.Client()
        super().__init__(embedding)

        # Define a class with exactly the interface ChromaDB expects
        class EmbeddingFunctionWrapper:
            def __init__(self, embedding_model):
                self.embedding_model = embedding_model
                
            def __call__(self, input):
                # This method has exactly the signature ChromaDB expects
                return self.embedding_model.get_embedding(input)
        
        # Create an instance of our wrapper class
        embedding_function = EmbeddingFunctionWrapper(embedding)

        self.collection = self.client.get_or_create_collection(
            name="papers",
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"},
            )

    def add_document(self, document, link, rating):
        """
        Adds a document to the vector store.
        Args:
           document (string): The document to add.
            link (string): URL associated with the document.
            rating (int/float): Rating associated with the document.
        Returns:
            None
        """
        doc_id = str(uuid.uuid4())
        # Get current Unix timestamp (seconds since epoch)
        timestamp = time.time()
        self.collection.add(
            ids=[doc_id],
            documents=[document],
            metadatas={"link": link, "rating": rating, "timestamp": timestamp},
            )
        return None
    
    def search(self, query_texts, num_results=10, where_filter=None):
        """
        Searches for a query in the vector store.
        Args:
           query_texts (list): A list of queries to search for.
           num_results (int): Number of results to return.
           where_filter (dict, optional): Filter to apply to the query.
           
        Returns:
           dict: A dictionary containing search results.
        """
        results = self.collection.query(
            query_texts=query_texts,
            n_results=num_results,
            where=where_filter,
            include=["metadatas", "distances"],
            )
        return results
        
    def search_by_time(self, query_texts, num_results=10, days=0):
        """
        Searches for a query in the vector store with time filtering.
        
        Args:
           query_texts (list): A list of queries to search for.
           num_results (int): Number of results to return.
           days (int): Number of days to look back (0 for all documents).
           
        Returns:
           dict: A dictionary containing search results.
        """
        where_filter = None
        if days > 0:
            import time
            from datetime import datetime, timedelta
            # Calculate timestamp for X days ago
            cutoff_date = datetime.now() - timedelta(days=days)
            cutoff_timestamp = cutoff_date.timestamp()
            where_filter = {"timestamp": {"$gte": cutoff_timestamp}}
        
        return self.search(query_texts, num_results, where_filter)
        
    def get_all_documents(self):
        """
        Retrieve all documents from the collection with their metadata.
        
        Returns:
            dict: A dictionary containing all documents, their IDs, embeddings, and metadata
        """
        return self.collection.get(
            include=["embeddings", "metadatas", "documents"]
        )
        
    def get_documents_by_time(self, days=0):
        """
        Retrieve documents from the collection filtered by time.
        
        Args:
            days (int): Number of days to look back (0 for all documents)
            
        Returns:
            dict: A dictionary containing filtered documents, their IDs, embeddings, and metadata
        """
        where_filter = None
        if days > 0:
            import time
            from datetime import datetime, timedelta
            # Calculate timestamp for X days ago
            cutoff_date = datetime.now() - timedelta(days=days)
            cutoff_timestamp = cutoff_date.timestamp()
            where_filter = {"timestamp": {"$gte": cutoff_timestamp}}
        
        return self.collection.get(
            where=where_filter,
            include=["embeddings", "metadatas", "documents"]
        )
        
    def document_exists(self, document):
        """
        Check if a document already exists in the vector store.
        
        Args:
            document (str): The document to check
            
        Returns:
            bool: True if the document exists, False otherwise
        """
        # Get all documents
        all_docs = self.get_all_documents()
        
        # Check if the document exists
        return document in all_docs["documents"]
