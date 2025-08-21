"""
Simplified vector store implementation for the Enterprise Copilot demo.
Uses basic text similarity when advanced ML libraries are not available.
Provides fallback functionality while maintaining the same interface.
"""

import streamlit as st
import json
import re
import math
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from collections import Counter


class SimpleVectorStore:
    """
    Simplified vector store using TF-IDF and cosine similarity.
    Provides basic semantic search when FAISS and sentence-transformers are not available.
    """
    
    def __init__(self, embedding_model: str = "e5-fallback-tfidf"):
        """Initialize the simple vector store."""
        self.embedding_model_name = embedding_model
        self.documents = []
        self.document_metadata = {}
        self.vocabulary = set()
        self.idf_scores = {}
        
        st.info("📊 Using TF-IDF fallback (install sentence-transformers for E5 embeddings)")
    
    def add_documents(self, chunks: List[Dict[str, Any]], document_name: str) -> None:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of document chunks
            document_name: Name of the source document
        """
        if not chunks:
            return
        
        start_id = len(self.documents)
        chunk_ids = []
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            chunk_id = start_id + i
            chunk['id'] = chunk_id
            
            # Preprocess text for better matching
            processed_content = self._preprocess_text(chunk['content'])
            chunk['processed_content'] = processed_content
            chunk['words'] = processed_content.split()
            
            # Update vocabulary
            self.vocabulary.update(chunk['words'])
            
            self.documents.append(chunk)
            chunk_ids.append(chunk_id)
        
        # Store document metadata
        self.document_metadata[document_name] = {
            'chunk_count': len(chunks),
            'chunk_ids': chunk_ids,
            'timestamp': datetime.now().isoformat(),
            'embedding_model': self.embedding_model_name
        }
        
        # Update IDF scores
        self._calculate_idf_scores()
        
        st.success(f"✅ Added {len(chunks)} chunks from {document_name}")
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better matching."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
        
        words = [word for word in text.split() if word not in stop_words and len(word) > 2]
        
        return ' '.join(words)
    
    def _calculate_idf_scores(self):
        """Calculate IDF (Inverse Document Frequency) scores."""
        doc_count = len(self.documents)
        if doc_count == 0:
            return
        
        # Count document frequency for each word
        word_doc_count = {}
        for doc in self.documents:
            unique_words = set(doc['words'])
            for word in unique_words:
                word_doc_count[word] = word_doc_count.get(word, 0) + 1
        
        # Calculate IDF scores
        self.idf_scores = {}
        for word, doc_freq in word_doc_count.items():
            self.idf_scores[word] = math.log(doc_count / doc_freq)
    
    def _calculate_tfidf_vector(self, text: str) -> Dict[str, float]:
        """Calculate TF-IDF vector for given text."""
        processed_text = self._preprocess_text(text)
        words = processed_text.split()
        
        if not words:
            return {}
        
        # Calculate term frequency
        word_count = Counter(words)
        max_count = max(word_count.values()) if word_count else 1
        
        # Calculate TF-IDF
        tfidf_vector = {}
        for word, count in word_count.items():
            tf = count / max_count  # Normalized term frequency
            idf = self.idf_scores.get(word, 1.0)
            tfidf_vector[word] = tf * idf
        
        return tfidf_vector
    
    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Calculate cosine similarity between two TF-IDF vectors."""
        if not vec1 or not vec2:
            return 0.0
        
        # Calculate dot product
        dot_product = 0.0
        for word in vec1:
            if word in vec2:
                dot_product += vec1[word] * vec2[word]
        
        # Calculate magnitudes
        mag1 = math.sqrt(sum(val ** 2 for val in vec1.values()))
        mag2 = math.sqrt(sum(val ** 2 for val in vec2.values()))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)
    
    def search(self, query: str, top_k: int = 5, min_score: float = 0.1) -> List[Dict[str, Any]]:
        """
        Search for similar documents using TF-IDF cosine similarity.
        
        Args:
            query: Search query
            top_k: Number of top results
            min_score: Minimum similarity score
            
        Returns:
            List of similar document chunks
        """
        if not self.documents:
            return []
        
        # Calculate query vector
        query_vector = self._calculate_tfidf_vector(query)
        
        if not query_vector:
            return []
        
        # Calculate similarities
        results = []
        for doc in self.documents:
            doc_vector = self._calculate_tfidf_vector(doc['content'])
            
            if doc_vector:
                similarity = self._cosine_similarity(query_vector, doc_vector)
                
                # Add keyword matching bonus
                query_words = set(self._preprocess_text(query).split())
                doc_words = set(doc.get('words', []))
                keyword_overlap = len(query_words.intersection(doc_words)) / len(query_words) if query_words else 0
                
                # Combined score
                final_score = 0.7 * similarity + 0.3 * keyword_overlap
                
                if final_score >= min_score:
                    result = doc.copy()
                    result['score'] = final_score
                    results.append(result)
        
        # Sort by score and return top-k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def search_with_mmr(self, query: str, top_k: int = 5, diversity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search with Maximal Marginal Relevance for diverse results.
        
        Args:
            query: Search query
            top_k: Number of results
            diversity_threshold: Threshold for diversity
            
        Returns:
            List of diverse, relevant documents
        """
        # Get initial candidates
        candidates = self.search(query, top_k * 2, min_score=0.05)
        
        if len(candidates) <= top_k:
            return candidates
        
        # MMR selection
        selected = []
        remaining = candidates.copy()
        
        # Select first (most relevant)
        if remaining:
            selected.append(remaining.pop(0))
        
        # Select remaining based on MMR
        while len(selected) < top_k and remaining:
            mmr_scores = []
            
            for candidate in remaining:
                relevance = candidate['score']
                
                # Calculate max similarity to selected documents
                max_similarity = 0.0
                if selected:
                    candidate_words = set(candidate.get('words', []))
                    for selected_doc in selected:
                        selected_words = set(selected_doc.get('words', []))
                        if candidate_words and selected_words:
                            similarity = len(candidate_words.intersection(selected_words)) / len(candidate_words.union(selected_words))
                            max_similarity = max(max_similarity, similarity)
                
                # MMR score
                mmr_score = diversity_threshold * relevance - (1 - diversity_threshold) * max_similarity
                mmr_scores.append((mmr_score, candidate))
            
            # Select best MMR score
            if mmr_scores:
                mmr_scores.sort(key=lambda x: x[0], reverse=True)
                best_candidate = mmr_scores[0][1]
                selected.append(best_candidate)
                remaining.remove(best_candidate)
        
        return selected
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return {
            'total_documents': len(self.document_metadata),
            'total_chunks': len(self.documents),
            'vocabulary_size': len(self.vocabulary),
            'embedding_model': self.embedding_model_name,
            'search_method': 'TF-IDF + Cosine Similarity',
            'document_names': list(self.document_metadata.keys())
        }
    
    def get_document_info(self, document_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific document."""
        return self.document_metadata.get(document_name)
    
    def delete_document(self, document_name: str) -> bool:
        """Delete a document from the vector store."""
        if document_name not in self.document_metadata:
            return False
        
        # Remove document chunks
        chunk_ids = self.document_metadata[document_name]['chunk_ids']
        self.documents = [doc for doc in self.documents if doc.get('id') not in chunk_ids]
        
        # Remove from metadata
        del self.document_metadata[document_name]
        
        # Recalculate vocabulary and IDF scores
        self.vocabulary = set()
        for doc in self.documents:
            self.vocabulary.update(doc.get('words', []))
        
        self._calculate_idf_scores()
        
        st.success(f"✅ Document {document_name} removed")
        return True