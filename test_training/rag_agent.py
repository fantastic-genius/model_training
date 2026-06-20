# rag_agent.py
import ollama
from pathlib import Path
import numpy as np
from typing import List, Dict
import json

class SimpleRAG:
    """Simple Retrieval-Augmented Generation without external dependencies"""
    
    def __init__(self, model: str = "llama3.2:3b"):
        self.model = model
        self.documents = []
        self.embeddings = []
        
    def add_document(self, text: str, metadata: Dict = None):
        """Add a document to the knowledge base"""
        # Get embedding from Ollama
        response = ollama.embeddings(model=self.model, prompt=text)
        embedding = response['embedding']
        
        self.documents.append({
            'text': text,
            'metadata': metadata or {},
            'embedding': embedding
        })
        
    def add_documents_from_directory(self, directory: str, extensions: List[str] = ['.txt', '.md']):
        """Load all text files from a directory"""
        path = Path(directory)
        for ext in extensions:
            for file in path.rglob(f'*{ext}'):
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Split into chunks
                        chunks = self._chunk_text(content, chunk_size=500)
                        for i, chunk in enumerate(chunks):
                            self.add_document(
                                chunk,
                                metadata={'source': str(file), 'chunk': i}
                            )
                    print(f"Loaded: {file}")
                except Exception as e:
                    print(f"Error loading {file}: {e}")
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for relevant documents"""
        # Get query embedding
        response = ollama.embeddings(model=self.model, prompt=query)
        query_embedding = response['embedding']
        
        # Calculate similarities
        similarities = []
        for doc in self.documents:
            sim = self._cosine_similarity(query_embedding, doc['embedding'])
            similarities.append((sim, doc))
        
        # Sort and return top k
        similarities.sort(reverse=True, key=lambda x: x[0])
        return [
            {
                'text': doc['text'],
                'metadata': doc['metadata'],
                'score': score
            }
            for score, doc in similarities[:top_k]
        ]
    
    def query(self, question: str, top_k: int = 3) -> str:
        """Answer a question using RAG"""
        # Retrieve relevant documents
        relevant_docs = self.search(question, top_k=top_k)
        
        # Build context
        context = "\n\n".join([
            f"[Source: {doc['metadata'].get('source', 'Unknown')}]\n{doc['text']}"
            for doc in relevant_docs
        ])
        
        # Generate answer
        prompt = f"""Answer the question based on the context below. If the answer is not in the context, say so.

          Context:
          {context}

          Question: {question}

          Answer:"""
        
        response = ollama.chat(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        return response['message']['content']
    
    def save_index(self, filepath: str = "rag_index.json"):
        """Save the document index"""
        with open(filepath, 'w') as f:
            json.dump(self.documents, f)
    
    def load_index(self, filepath: str = "rag_index.json"):
        """Load the document index"""
        with open(filepath, 'r') as f:
            self.documents = json.load(f)

# Usage
rag = SimpleRAG(model="llama3.2:3b")

# Add documents from a directory
# rag.add_documents_from_directory("./docs")

# Or add manually
rag.add_document(
    "SameDayCustom is a platform for custom printing with same-day delivery through partner print shops.",
    metadata={'source': 'company_info'}
)

rag.add_document(
    "The M2 chip has 8 CPU cores, up to 10 GPU cores, and 16-core Neural Engine.",
    metadata={'source': 'tech_specs'}
)

# Query the knowledge base
answer = rag.query("What does SameDayCustom do?")
print(answer)

# Save for later
rag.save_index()