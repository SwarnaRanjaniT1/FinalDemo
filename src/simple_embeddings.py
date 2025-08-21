"""
Simple embeddings implementation for deployment without heavy ML dependencies.
Uses TF-IDF vectorization as a fallback when sentence transformers are unavailable.
"""

import streamlit as st
import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from datetime import datetime


class SimpleEmbeddingStore:
    """
    Simple vector store using TF-IDF for text similarity.
    Fallback implementation for deployment environments.
    """
    
    def __init__(self):
        """Initialize the simple embedding store."""
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        self.documents = []
        self.document_metadata = {}
        self.vectors = None
        self.is_fitted = False
        
        st.success("✅ Simple text similarity store initialized")
    
    def add_documents(self, chunks: List[Dict[str, Any]], document_name: str) -> None:
        """Add document chunks to the store."""
        if not chunks:
            return
        
        # Extract text content
        texts = [chunk['content'] for chunk in chunks]
        
        # Add to documents
        start_id = len(self.documents)
        for i, (chunk, text) in enumerate(zip(chunks, texts)):
            doc = chunk.copy()
            doc['doc_id'] = start_id + i
            doc['document_name'] = document_name
            self.documents.append(doc)
        
        # Refit vectorizer with all documents
        with st.spinner(f"Processing {document_name} with TF-IDF..."):
            all_texts = [doc['content'] for doc in self.documents]
            self.vectors = self.vectorizer.fit_transform(all_texts)
            self.is_fitted = True
        
        # Store metadata
        self.document_metadata[document_name] = {
            'chunk_count': len(chunks),
            'chunk_ids': list(range(start_id, start_id + len(chunks))),
            'timestamp': datetime.now().isoformat()
        }
        
        st.success(f"Added {len(chunks)} chunks from {document_name}")
    
    def search(self, query: str, top_k: int = 5, min_score: float = 0.1) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        if not self.is_fitted or len(self.documents) == 0:
            return []
        
        # Transform query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.vectors)[0]
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = similarities[idx]
            if score >= min_score:
                doc = self.documents[idx].copy()
                doc['score'] = float(score)
                results.append(doc)
        
        return results
    
    def search_with_mmr(self, query: str, top_k: int = 5, diversity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search with basic diversity (simplified MMR)."""
        # Get more candidates than needed
        candidates = self.search(query, top_k * 2, min_score=0.05)
        
        if len(candidates) <= top_k:
            return candidates
        
        # Simple diversity: skip very similar documents
        selected = []
        for candidate in candidates:
            if len(selected) >= top_k:
                break
                
            # Simple diversity check - skip if too similar to already selected
            is_diverse = True
            for selected_doc in selected:
                if abs(candidate['score'] - selected_doc['score']) < 0.1:
                    is_diverse = False
                    break
            
            if is_diverse:
                selected.append(candidate)
        
        return selected[:top_k]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        return {
            'total_documents': len(self.documents),
            'total_files': len(self.document_metadata),
            'vectorizer_vocab_size': len(self.vectorizer.vocabulary_) if self.is_fitted else 0,
            'is_fitted': self.is_fitted
        }