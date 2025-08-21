"""
Utility helper functions for the Enterprise Copilot demo.
"""

import streamlit as st
import time
import re
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import mimetypes


def format_time(milliseconds: float) -> str:
    """
    Format time in milliseconds to human-readable string.
    
    Args:
        milliseconds: Time in milliseconds
        
    Returns:
        Formatted time string
    """
    if milliseconds < 1000:
        return f"{milliseconds:.1f}ms"
    else:
        seconds = milliseconds / 1000
        if seconds < 60:
            return f"{seconds:.2f}s"
        else:
            minutes = int(seconds // 60)
            remaining_seconds = seconds % 60
            return f"{minutes}m {remaining_seconds:.1f}s"


def validate_file_type(filename: str, allowed_types: List[str] = None) -> bool:
    """
    Validate if a file type is allowed.
    
    Args:
        filename: Name of the file
        allowed_types: List of allowed file extensions
        
    Returns:
        True if file type is allowed, False otherwise
    """
    if allowed_types is None:
        allowed_types = ['pdf', 'txt', 'docx']
    
    file_extension = filename.lower().split('.')[-1]
    return file_extension in allowed_types


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to specified length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length of text
        suffix: Suffix to add if text is truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def clean_query(query: str) -> str:
    """
    Clean and normalize user query.
    
    Args:
        query: Raw user query
        
    Returns:
        Cleaned query
    """
    # Remove extra whitespace
    query = re.sub(r'\s+', ' ', query.strip())
    
    # Remove potentially harmful characters
    query = re.sub(r'[<>{}]', '', query)
    
    return query


def extract_keywords(text: str, min_length: int = 3) -> List[str]:
    """
    Extract keywords from text (simple implementation).
    
    Args:
        text: Input text
        min_length: Minimum length of keywords
        
    Returns:
        List of keywords
    """
    # Simple keyword extraction - could be improved with NLP libraries
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Filter out common stop words and short words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
    
    keywords = [word for word in words if len(word) >= min_length and word not in stop_words]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for keyword in keywords:
        if keyword not in seen:
            seen.add(keyword)
            unique_keywords.append(keyword)
    
    return unique_keywords


def format_document_metadata(metadata: Dict[str, Any]) -> str:
    """
    Format document metadata for display.
    
    Args:
        metadata: Document metadata dictionary
        
    Returns:
        Formatted metadata string
    """
    formatted_parts = []
    
    if 'chunk_count' in metadata:
        formatted_parts.append(f"Chunks: {metadata['chunk_count']}")
    
    if 'timestamp' in metadata:
        try:
            timestamp = datetime.fromisoformat(metadata['timestamp'])
            formatted_parts.append(f"Added: {timestamp.strftime('%Y-%m-%d %H:%M')}")
        except:
            formatted_parts.append(f"Added: {metadata['timestamp']}")
    
    if 'embedding_model' in metadata:
        formatted_parts.append(f"Model: {metadata['embedding_model']}")
    
    return " | ".join(formatted_parts)


def calculate_similarity_score_color(score: float) -> str:
    """
    Get color for similarity score visualization.
    
    Args:
        score: Similarity score (0-1)
        
    Returns:
        Color name or hex code
    """
    if score >= 0.8:
        return "green"
    elif score >= 0.6:
        return "blue"
    elif score >= 0.4:
        return "orange"
    else:
        return "red"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe storage.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove or replace dangerous characters
    sanitized = re.sub(r'[^\w\-_\.]', '_', filename)
    
    # Remove multiple consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # Ensure it doesn't start with a dot or dash
    sanitized = sanitized.lstrip('.-')
    
    # Ensure it's not too long
    if len(sanitized) > 100:
        name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
        max_name_length = 95 - len(ext)
        sanitized = name[:max_name_length] + ('.' + ext if ext else '')
    
    return sanitized or 'document'


def create_download_link(data: str, filename: str, link_text: str = "Download") -> str:
    """
    Create a download link for text data.
    
    Args:
        data: Text data to download
        filename: Suggested filename
        link_text: Text for the download link
        
    Returns:
        HTML download link
    """
    import base64
    
    # Encode the data
    b64_data = base64.b64encode(data.encode()).decode()
    
    # Create download link
    href = f'<a href="data:text/plain;base64,{b64_data}" download="{filename}">{link_text}</a>'
    
    return href


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes to human readable format.
    
    Args:
        bytes_value: Number of bytes
        
    Returns:
        Formatted byte string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def validate_query_length(query: str, max_length: int = 1000) -> bool:
    """
    Validate query length.
    
    Args:
        query: User query
        max_length: Maximum allowed length
        
    Returns:
        True if valid, False otherwise
    """
    return len(query.strip()) > 0 and len(query) <= max_length


def get_file_icon(filename: str) -> str:
    """
    Get an emoji icon for a file based on its extension.
    
    Args:
        filename: Name of the file
        
    Returns:
        Emoji icon string
    """
    extension = filename.lower().split('.')[-1]
    
    icons = {
        'pdf': '📄',
        'txt': '📝',
        'docx': '📰',
        'doc': '📰',
        'xlsx': '📊',
        'xls': '📊',
        'pptx': '📈',
        'ppt': '📈',
        'csv': '📋'
    }
    
    return icons.get(extension, '📄')


def estimate_reading_time(text: str, words_per_minute: int = 200) -> str:
    """
    Estimate reading time for text.
    
    Args:
        text: Text content
        words_per_minute: Average reading speed
        
    Returns:
        Formatted reading time
    """
    word_count = len(text.split())
    minutes = max(1, round(word_count / words_per_minute))
    
    if minutes < 60:
        return f"{minutes} min read"
    else:
        hours = minutes // 60
        remaining_minutes = minutes % 60
        if remaining_minutes == 0:
            return f"{hours} hr read"
        else:
            return f"{hours}h {remaining_minutes}m read"


def create_progress_bar(current: int, total: int, width: int = 20) -> str:
    """
    Create a text-based progress bar.
    
    Args:
        current: Current progress
        total: Total items
        width: Width of progress bar
        
    Returns:
        Progress bar string
    """
    if total == 0:
        return "█" * width
    
    progress = min(current / total, 1.0)
    filled_width = int(progress * width)
    bar = "█" * filled_width + "░" * (width - filled_width)
    percentage = int(progress * 100)
    
    return f"{bar} {percentage}%"


def highlight_keywords(text: str, keywords: List[str], max_length: int = 200) -> str:
    """
    Highlight keywords in text excerpt.
    
    Args:
        text: Source text
        keywords: Keywords to highlight
        max_length: Maximum length of excerpt
        
    Returns:
        Highlighted text excerpt
    """
    # Find the best excerpt that contains most keywords
    text_lower = text.lower()
    best_start = 0
    best_score = 0
    
    # Simple sliding window to find best excerpt
    for start in range(0, max(1, len(text) - max_length), max_length // 4):
        excerpt = text[start:start + max_length].lower()
        score = sum(1 for keyword in keywords if keyword.lower() in excerpt)
        if score > best_score:
            best_score = score
            best_start = start
    
    # Extract excerpt
    excerpt = text[best_start:best_start + max_length]
    if best_start > 0:
        excerpt = "..." + excerpt
    if best_start + max_length < len(text):
        excerpt = excerpt + "..."
    
    # Highlight keywords (simple approach)
    for keyword in keywords:
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        excerpt = pattern.sub(f"**{keyword}**", excerpt)
    
    return excerpt
