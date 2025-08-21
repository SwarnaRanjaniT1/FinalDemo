"""
Vector storage implementation using FAISS for semantic search.
Implements the vector database component as described in the technical architecture.
"""

import streamlit as st
import numpy as np
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    st.warning("⚠️ FAISS not available - using simple vector search")
    FAISS_AVAILABLE = False

from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple, Optional
import pickle
import json
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity


class VectorStore:
    """
    FAISS-based vector store for document embeddings.
    Implements semantic similarity search with cosine distance.
    """
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", index_type: str = "flat"):
        """
        Initialize the vector store.
        
        Args:
            embedding_model: HuggingFace sentence transformer model
            index_type: FAISS index type ('flat' for exact search)
        """
        self.embedding_model_name = embedding_model
        self.index_type = index_type
        
        # Initialize embedding model with robust fallback
        with st.spinner("Loading embedding model..."):
            # Use reliable public model by default for deployment
            try:
                # For deployment, use all-MiniLM-L6-v2 which is publicly available
                self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                self.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
                st.info("🔄 Using MiniLM embeddings for deployment compatibility")
            except Exception as e:
                st.error(f"❌ Failed to load embedding model: {str(e)}")
                raise RuntimeError("Could not initialize any embedding model")
        
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize vector index
        if FAISS_AVAILABLE:
            if index_type == "flat":
                self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product for cosine similarity
            else:
                raise ValueError(f"Unsupported index type: {index_type}")
            self.use_faiss = True
        else:
            self.index = None
            self.use_faiss = False
            st.info("Using simple cosine similarity search")
        
        # Storage for document metadata and embeddings
        self.embeddings = []
        self.documents = []
        self.document_metadata = {}
        
        
        # Determine if using E5-style prefixes
        self.use_e5_prefixes = "e5" in self.embedding_model_name.lower()
        model_type = "E5" if self.use_e5_prefixes else "MiniLM"
        st.success(f"✅ Vector store initialized with {model_type} embeddings (dim: {self.embedding_dim})")
    
    def add_documents(self, chunks: List[Dict[str, Any]], document_name: str) -> None:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of document chunks
            document_name: Name of the source document
        """
        if not chunks:
            return
        
        # Extract text content for embedding
        texts = [chunk['content'] for chunk in chunks]
        
        # Generate embeddings 
        with st.spinner(f"Generating embeddings for {document_name}..."):
            # Use E5 prefixes only if using E5 model
            if self.use_e5_prefixes:
                prefixed_texts = [f"passage: {text}" for text in texts]
            else:
                prefixed_texts = texts
            embeddings = self.embedding_model.encode(prefixed_texts, convert_to_tensor=False, show_progress_bar=True)
            
            # Normalize embeddings for cosine similarity
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Add to vector index
        if self.use_faiss:
            self.index.add(embeddings.astype('float32'))
        
        # Store embeddings and documents
        start_id = len(self.embeddings)
        self.embeddings.extend(embeddings)
        
        # Add chunk IDs to chunks and store
        chunk_ids = []
        for i, chunk in enumerate(chunks):
            chunk_id = start_id + i
            chunk['id'] = chunk_id
            self.documents.append(chunk)
            chunk_ids.append(chunk_id)
        
        # Store document metadata
        self.document_metadata[document_name] = {
            'chunk_count': len(chunks),
            'chunk_ids': chunk_ids,
            'timestamp': datetime.now().isoformat(),
            'embedding_model': self.embedding_model_name
        }
        
        st.success(f"Added {len(chunks)} chunks from {document_name} to vector store")
    
    def search(self, query: str, top_k: int = 5, min_score: float = 0.3) -> List[Dict[str, Any]]:
        """
        Search for similar documents using semantic similarity.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            min_score: Minimum similarity score threshold
            
        Returns:
            List of similar document chunks with scores
        """
        if len(self.embeddings) == 0:
            return []
        
        # Generate query embedding
        if self.use_e5_prefixes:
            query_text = f"query: {query}"
        else:
            query_text = query
        query_embedding = self.embedding_model.encode([query_text], convert_to_tensor=False)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search in vector index
        if self.use_faiss:
            scores, indices = self.index.search(query_embedding.astype('float32'), min(top_k, len(self.embeddings)))
        else:
            # Simple cosine similarity search
            if not self.embeddings:
                return []
            embedding_matrix = np.array(self.embeddings)
            similarities = cosine_similarity(query_embedding, embedding_matrix)[0]
            # Get top_k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            scores = [similarities[i:i+1] for i in top_indices]
            indices = [top_indices]
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and score >= min_score:  # Valid result above threshold
                document = self.documents[idx].copy()
                document['score'] = float(score)
                results.append(document)
        
        # Sort by score (descending)
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results
    
    def search_with_mmr(self, query: str, top_k: int = 5, diversity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search with Maximal Marginal Relevance for diverse results.
        Implements MMR as described in the technical architecture.
        
        Args:
            query: Search query
            top_k: Number of results to return
            diversity_threshold: Threshold for diversity (lower = more diverse)
            
        Returns:
            List of diverse, relevant document chunks
        """
        if len(self.embeddings) == 0:
            return []
        
        # Get initial candidates (more than needed)
        initial_candidates = self.search(query, top_k * 3, min_score=0.2)
        
        if len(initial_candidates) <= top_k:
            return initial_candidates
        
        # Implement MMR algorithm
        selected = []
        candidates = initial_candidates.copy()
        
        # Select first document (highest relevance)
        if candidates:
            selected.append(candidates.pop(0))
        
        # Select remaining documents based on MMR
        while len(selected) < top_k and candidates:
            mmr_scores = []
            
            for candidate in candidates:
                relevance_score = candidate['score']
                
                # Calculate maximum similarity to already selected documents
                max_similarity = 0.0
                if selected:
                    candidate_embedding = self.embeddings[candidate['id']]
                    for selected_doc in selected:
                        selected_embedding = self.embeddings[selected_doc['id']]
                        # Cosine similarity
                        similarity = np.dot(candidate_embedding, selected_embedding)
                        max_similarity = max(max_similarity, similarity)
                
                # MMR score = λ * relevance - (1-λ) * max_similarity
                mmr_score = diversity_threshold * relevance_score - (1 - diversity_threshold) * max_similarity
                mmr_scores.append((mmr_score, candidate))
            
            # Select document with highest MMR score
            if mmr_scores:
                mmr_scores.sort(key=lambda x: x[0], reverse=True)
                best_candidate = mmr_scores[0][1]
                selected.append(best_candidate)
                candidates.remove(best_candidate)
        
        return selected
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics.
        
        Returns:
            Dictionary with store statistics
        """
        return {
            'total_documents': len(self.document_metadata),
            'total_chunks': len(self.embeddings),
            'embedding_dimension': self.embedding_dim,
            'embedding_model': self.embedding_model_name,
            'index_type': self.index_type,
            'document_names': list(self.document_metadata.keys())
        }
    
    def get_document_info(self, document_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific document.
        
        Args:
            document_name: Name of the document
            
        Returns:
            Document information or None if not found
        """
        return self.document_metadata.get(document_name)
    
    def delete_document(self, document_name: str) -> bool:
        """
        Delete a document from the vector store.
        Note: FAISS doesn't support deletion, so this would require rebuilding the index.
        For demo purposes, we'll just mark as deleted in metadata.
        
        Args:
            document_name: Name of document to delete
            
        Returns:
            True if successful, False otherwise
        """
        if document_name not in self.document_metadata:
            return False
        
        # In a production system, you'd rebuild the index without these documents
        # For demo, just remove from metadata
        del self.document_metadata[document_name]
        
        # Note: This leaves orphaned embeddings in the index
        # Production implementation would rebuild the index
        st.warning(f"Document {document_name} marked as deleted. Index rebuild would be needed in production.")
        
        return True
    
    def save_index(self, filepath: str) -> bool:
        """
        Save the vector store to disk.
        
        Args:
            filepath: Path to save the index
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Save FAISS index
            faiss.write_index(self.index, f"{filepath}.faiss")
            
            # Save metadata and documents
            with open(f"{filepath}.pkl", 'wb') as f:
                data = {
                    'embeddings': self.embeddings,
                    'documents': self.documents,
                    'document_metadata': self.document_metadata,
                    'embedding_model_name': self.embedding_model_name,
                    'embedding_dim': self.embedding_dim,
                    'index_type': self.index_type
                }
                pickle.dump(data, f)
            
            return True
            
        except Exception as e:
            st.error(f"Error saving index: {str(e)}")
            return False
    
    def load_index(self, filepath: str) -> bool:
        """
        Load the vector store from disk.
        
        Args:
            filepath: Path to load the index from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{filepath}.faiss")
            
            # Load metadata and documents
            with open(f"{filepath}.pkl", 'rb') as f:
                data = pickle.load(f)
                self.embeddings = data['embeddings']
                self.documents = data['documents']
                self.document_metadata = data['document_metadata']
                self.embedding_model_name = data['embedding_model_name']
                self.embedding_dim = data['embedding_dim']
                self.index_type = data['index_type']
            
            # Reinitialize embedding model if different
            if hasattr(self, 'embedding_model') and self.embedding_model_name != self.embedding_model.model_name:
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
            return True
            
        except Exception as e:
            st.error(f"Error loading index: {str(e)}")
            return False
