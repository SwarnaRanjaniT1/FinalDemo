# Sample Documents Directory

This directory is intended for storing sample enterprise documents for testing the RAG-based copilot system.

## Supported Document Types

- **PDF files (.pdf)**: Policy documents, reports, manuals
- **Text files (.txt)**: Plain text documents, procedures, guidelines  
- **Word documents (.docx)**: Formatted documents, procedures, policies

## Document Processing Pipeline

1. **Upload**: Documents are uploaded through the Streamlit interface
2. **Text Extraction**: Content is extracted based on file type
3. **Chunking**: Documents are split into overlapping chunks (512 tokens, 50 token overlap)
4. **Embedding**: Each chunk is converted to a 384-dimensional vector using sentence-transformers
5. **Indexing**: Vectors are stored in FAISS for semantic similarity search

## Usage Instructions

1. Navigate to the "Document Management" page
2. Upload your enterprise documents using the file uploader
3. Click "Process Documents" to add them to the knowledge base
4. Use the "Copilot Chat" page to query the documents

## Demo Data

For demonstration purposes, you can use sample enterprise documents such as:

- Employee handbook excerpts
- IT policy documents
- Standard operating procedures
- Technical documentation
- FAQ documents

## Performance Considerations

- Optimal chunk size: 512 tokens with 50 token overlap
- Maximum context length: 4000 characters
- Target retrieval time: <500ms (p95)
- Target end-to-end response: <700ms (p95)

## Security Notes

- All document processing happens locally
- No data is sent to external services except for OpenAI generation
- Documents are stored in memory only during the session
- Implement proper access controls in production environments
