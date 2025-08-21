"""
RAG (Retrieval-Augmented Generation) pipeline implementation.
Combines vector retrieval with OpenAI generation as described in the technical architecture.
"""

import streamlit as st
import time
import os
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from .vector_store import VectorStore


class RAGPipeline:
    """
    Implements the complete RAG pipeline: retrieval + generation.
    """
    
    def __init__(self, vector_store: VectorStore, max_context_length: int = 4000):
        """
        Initialize the RAG pipeline.
        
        Args:
            vector_store: Initialized vector store instance
            max_context_length: Maximum context length in characters
        """
        self.vector_store = vector_store
        self.max_context_length = max_context_length
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key)
    
    def query(self, question: str, use_mmr: bool = True, top_k: int = 5) -> Tuple[str, List[Dict[str, Any]], Dict[str, float]]:
        """
        Process a user query through the RAG pipeline.
        
        Args:
            question: User's question
            use_mmr: Whether to use MMR for diverse retrieval
            top_k: Number of documents to retrieve
            
        Returns:
            Tuple of (answer, source_documents, metrics)
        """
        if not self.client:
            raise Exception("OpenAI client not initialized. Check API key configuration.")
        
        # Start timing
        start_time = time.time()
        
        # Step 1: Retrieve relevant documents
        retrieval_start = time.time()
        
        if use_mmr:
            relevant_docs = self.vector_store.search_with_mmr(question, top_k=top_k)
        else:
            relevant_docs = self.vector_store.search(question, top_k=top_k)
        
        retrieval_time = (time.time() - retrieval_start) * 1000  # Convert to milliseconds
        
        if not relevant_docs:
            return "I couldn't find any relevant information to answer your question. Please try rephrasing or check if documents have been uploaded.", [], {
                'retrieval_time': retrieval_time,
                'generation_time': 0,
                'total_time': retrieval_time
            }
        
        # Step 2: Prepare context
        context = self._prepare_context(relevant_docs)
        
        # Step 3: Generate answer
        generation_start = time.time()
        answer = self._generate_answer(question, context, relevant_docs)
        generation_time = (time.time() - generation_start) * 1000
        
        total_time = (time.time() - start_time) * 1000
        
        # Prepare source information
        sources = self._prepare_sources(relevant_docs)
        
        metrics = {
            'retrieval_time': retrieval_time,
            'generation_time': generation_time,
            'total_time': total_time,
            'num_sources': len(relevant_docs)
        }
        
        return answer, sources, metrics
    
    def _prepare_context(self, relevant_docs: List[Dict[str, Any]]) -> str:
        """
        Prepare context string from retrieved documents.
        
        Args:
            relevant_docs: List of relevant document chunks
            
        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(relevant_docs):
            doc_text = f"[Source {i+1}] {doc['content']}"
            
            # Check if adding this document would exceed max context length
            if current_length + len(doc_text) > self.max_context_length:
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        return "\n\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str, sources: List[Dict[str, Any]]) -> str:
        """
        Generate answer using OpenAI API.
        
        Args:
            question: User's question
            context: Retrieved context
            sources: Source documents for citation
            
        Returns:
            Generated answer
        """
        # Create source reference mapping
        source_refs = {}
        for i, doc in enumerate(sources):
            source_refs[f"Source {i+1}"] = f"{doc['document']} (relevance: {doc['score']:.3f})"
        
        # Construct prompt
        prompt = f"""You are an enterprise AI assistant helping employees find information from company documents. 
Use the provided context to answer the user's question accurately and comprehensively.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Provide a clear, comprehensive answer based on the context
2. If the context doesn't contain sufficient information, state this clearly
3. Include relevant source citations in your response using [Source X] format
4. Maintain a professional, helpful tone
5. If you're unsure about any information, express appropriate uncertainty

ANSWER:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful enterprise AI assistant that provides accurate information based on company documents. Always cite your sources and be transparent about the limitations of your knowledge."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=1000,
                temperature=0.1,  # Low temperature for more factual responses
                top_p=0.9
            )
            
            answer = response.choices[0].message.content
            
            # Add source reference information if not already included
            if "[Source" not in answer and source_refs:
                answer += "\n\n**Source References:**\n"
                for ref, info in source_refs.items():
                    answer += f"- {ref}: {info}\n"
            
            return answer
            
        except Exception as e:
            error_msg = str(e)
            if "api_key" in error_msg.lower():
                return "Error: OpenAI API key is invalid or not configured. Please check your API key setup."
            elif "rate_limit" in error_msg.lower():
                return "Error: API rate limit exceeded. Please try again in a moment."
            elif "quota" in error_msg.lower():
                return "Error: API quota exceeded. Please check your OpenAI account billing."
            else:
                return f"Error generating response: {error_msg}"
    
    def _prepare_sources(self, relevant_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prepare source information for display.
        
        Args:
            relevant_docs: Retrieved document chunks
            
        Returns:
            List of source information dictionaries
        """
        sources = []
        
        for i, doc in enumerate(relevant_docs):
            source = {
                'id': doc.get('id', i),
                'document': doc['document'],
                'content': doc['content'],
                'score': doc['score'],
                'chunk_id': doc.get('chunk_id', 'unknown'),
                'tokens': doc.get('tokens', 'unknown')
            }
            sources.append(source)
        
        return sources
    
    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch.
        
        Args:
            questions: List of questions to process
            
        Returns:
            List of results for each question
        """
        results = []
        
        for question in questions:
            try:
                answer, sources, metrics = self.query(question)
                results.append({
                    'question': question,
                    'answer': answer,
                    'sources': sources,
                    'metrics': metrics,
                    'success': True
                })
            except Exception as e:
                results.append({
                    'question': question,
                    'answer': None,
                    'sources': [],
                    'metrics': {},
                    'error': str(e),
                    'success': False
                })
        
        return results
    
    def evaluate_answer_quality(self, question: str, answer: str, ground_truth: str = None) -> Dict[str, float]:
        """
        Evaluate the quality of a generated answer.
        This is a simplified evaluation - production systems would use more sophisticated metrics.
        
        Args:
            question: Original question
            answer: Generated answer
            ground_truth: Ground truth answer (if available)
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['answer_length'] = len(answer)
        metrics['has_sources'] = 1.0 if '[Source' in answer else 0.0
        
        # Word overlap with question (relevance indicator)
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        word_overlap = len(question_words.intersection(answer_words)) / len(question_words) if question_words else 0
        metrics['question_relevance'] = word_overlap
        
        # If ground truth is provided, calculate similarity
        if ground_truth:
            # Simple word-based similarity (could be improved with semantic similarity)
            ground_truth_words = set(ground_truth.lower().split())
            similarity = len(answer_words.intersection(ground_truth_words)) / len(answer_words.union(ground_truth_words))
            metrics['ground_truth_similarity'] = similarity
        
        return metrics
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG pipeline.
        
        Returns:
            Pipeline statistics
        """
        vector_stats = self.vector_store.get_stats()
        
        return {
            'vector_store': vector_stats,
            'openai_model': "gpt-4o",
            'max_context_length': self.max_context_length,
            'api_configured': self.client is not None
        }
