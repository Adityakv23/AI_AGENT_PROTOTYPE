"""
RAG (Retrieval-Augmented Generation) system for enhanced document understanding
"""

import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

from config import RAG_CONFIG, DATA_DIR, VECTOR_STORE_PATH


class RAGSystem:
    """
    Retrieval-Augmented Generation system for academic paper knowledge base
    """
    
    def __init__(self):
        self.embedding_model_name = RAG_CONFIG["embedding_model"]
        self.chunk_size = RAG_CONFIG["chunk_size"]
        self.chunk_overlap = RAG_CONFIG["chunk_overlap"]
        self.top_k = RAG_CONFIG["top_k"]
        
        # Initialize embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=str(VECTOR_STORE_PATH))
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="academic_papers",
            metadata={"description": "Academic paper knowledge base"}
        )
        
        print("✓ RAG system initialized")
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            chunks.append(chunk)
            if i + self.chunk_size >= len(words):
                break
        
        return chunks
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the knowledge base"""
        print(f"\nAdding {len(documents)} documents to knowledge base...")
        
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        for doc_idx, doc in enumerate(documents):
            chunks = self.chunk_text(doc.get('content', doc.get('input', '')))
            
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadatas.append({
                    'doc_id': doc_idx,
                    'chunk_id': chunk_idx,
                    'type': doc.get('type', 'paper')
                })
                all_ids.append(f"doc_{doc_idx}_chunk_{chunk_idx}")
        
        # Add to collection
        self.collection.add(
            documents=all_chunks,
            metadatas=all_metadatas,
            ids=all_ids
        )
        
        # Persist
        self.client.persist()
        
        print(f"✓ Added {len(all_chunks)} chunks to knowledge base")
    
    def retrieve(self, query: str, n_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks for a query"""
        n_results = n_results or self.top_k
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        retrieved = []
        if results['documents'] and len(results['documents']) > 0:
            for i, doc in enumerate(results['documents'][0]):
                retrieved.append({
                    'content': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else None
                })
        
        return retrieved
    
    def augment_prompt(
        self, 
        query: str, 
        retrieved_docs: List[Dict[str, Any]]
    ) -> str:
        """Create an augmented prompt with retrieved context"""
        context = "\n\n".join([
            f"[Context {i+1}] {doc['content']}"
            for i, doc in enumerate(retrieved_docs)
        ])
        
        augmented_prompt = f"""Based on the following context, answer the query:

{context}

Query: {query}

Response:"""
        
        return augmented_prompt
    
    def search_and_augment(self, query: str) -> str:
        """Retrieve relevant context and create augmented prompt"""
        retrieved = self.retrieve(query)
        augmented_prompt = self.augment_prompt(query, retrieved)
        return augmented_prompt
    
    def load_training_documents(self) -> List[Dict[str, Any]]:
        """Load training documents to populate knowledge base"""
        from config import TRAINING_DATA_PATH
        
        with open(TRAINING_DATA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        for item in data:
            documents.append({
                'content': item['input'],
                'type': 'academic_paper',
                'summary': item['output']
            })
        
        return documents
    
    def initialize_knowledge_base(self) -> None:
        """Initialize the knowledge base with training documents"""
        documents = self.load_training_documents()
        self.add_documents(documents)


if __name__ == "__main__":
    # Initialize RAG system
    rag = RAGSystem()
    
    # Populate knowledge base
    rag.initialize_knowledge_base()
    
    # Test retrieval
    test_query = "What methods are used for few-shot learning in language models?"
    retrieved = rag.retrieve(test_query)
    
    print(f"\nQuery: {test_query}")
    print(f"Retrieved {len(retrieved)} relevant chunks:")
    for i, result in enumerate(retrieved[:2], 1):
        print(f"\n{i}. {result['content'][:200]}...")

