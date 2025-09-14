# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
# Quick start (recommended)
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Dependencies
```bash
# Install dependencies
uv sync

# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Environment Setup
Create `.env` file in root:
```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### Access Points
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Architecture Overview

This is a **Retrieval-Augmented Generation (RAG) system** for querying course materials with AI-powered responses.

### Core Architecture Pattern
The system follows a **layered architecture** with tool-augmented AI:

```
Frontend (Vanilla JS) ←→ FastAPI ←→ RAG System ←→ AI Generator (Claude) ←→ Search Tools ←→ Vector Store (ChromaDB)
```

### Key Components and Interactions

#### Document Processing Pipeline
- **DocumentProcessor** (`document_processor.py`): Converts course text files into structured chunks
  - Parses course metadata (title, instructor, lessons)
  - Uses sentence-aware chunking with configurable overlap
  - Adds contextual prefixes to chunks for better retrieval

#### RAG Orchestration
- **RAGSystem** (`rag_system.py`): Main coordinator that orchestrates the entire flow
  - Manages document processing and storage
  - Coordinates AI generation with tool usage
  - Handles session management and conversation history

#### AI Integration
- **AIGenerator** (`ai_generator.py`): Handles Claude API integration
  - Uses tool-calling pattern where Claude can invoke search tools
  - Maintains conversation context and history
  - Configured for educational content with specific system prompts

#### Search Architecture
- **CourseSearchTool** (`search_tools.py`): Semantic search tool that Claude can call
  - Provides course name matching and lesson filtering
  - Returns formatted results for AI consumption
- **VectorStore** (`vector_store.py`): ChromaDB integration for vector storage
  - Handles embedding generation using Sentence Transformers
  - Supports filtered search by course and lesson
  - Manages course metadata alongside content chunks

#### Session Management
- **SessionManager** (`session_manager.py`): Tracks conversation history
  - Maintains user sessions with configurable history limits
  - Provides context for multi-turn conversations

### Data Models
- **Course**: Contains title, instructor, lessons, and links
- **Lesson**: Individual lesson with number, title, and optional link
- **CourseChunk**: Text chunk with course/lesson context and metadata

### Configuration System
All settings are centralized in `config.py`:
- AI model settings (using Claude Sonnet 4)
- Document processing parameters (chunk size: 800, overlap: 100)
- Vector search configuration (max results: 5)
- Database paths and session limits

### Query Flow Pattern
1. **Frontend** captures user input and sends to `/api/query`
2. **FastAPI** validates request and calls RAG system
3. **RAG System** retrieves conversation history and calls AI generator
4. **AI Generator** sends query to Claude API with available tools
5. **Claude** decides to use search tool and calls CourseSearchTool
6. **Search Tool** performs semantic search in ChromaDB
7. **Results** flow back through layers with sources tracked
8. **Frontend** renders response with sources in collapsible sections

### Course Document Format
Documents should follow this structure:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [instructor]

Lesson 0: Introduction
[content...]

Lesson 1: [title]
Lesson Link: [optional url]
[content...]
```

### Vector Storage Strategy
- Course content is chunked with sentence boundaries
- Each chunk includes course and lesson context
- Embeddings are generated using `all-MiniLM-L6-v2`
- ChromaDB provides similarity search with metadata filtering

## Important Development Notes

### No Test Framework
This codebase currently has no test infrastructure. When adding features, manual testing through the web interface is required.

### Document Loading
Documents are automatically loaded from the `docs/` folder on application startup. New documents require a server restart to be indexed.

### Session State
User sessions are managed in-memory and will be lost on server restart. Session IDs are generated automatically and persist for the browser session.

### Error Handling
The system has error handling at multiple layers but relies on manual debugging. Check browser console and server logs for issues.

### Configuration Changes
Changes to `config.py` require server restart. Key settings include chunk sizes, search limits, and AI model parameters.

### Additional Memories
- always use `uv` to run the server, never use `pip` directly
- make sure to use `uv` to manage all dependencies
- use `uv` to run Python files