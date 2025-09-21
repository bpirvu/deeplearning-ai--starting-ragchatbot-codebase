# RAG System Query Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                FRONTEND                                     │
│                               (script.js)                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    1. User types query & clicks send
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  sendMessage()                                                              │
│  ├── Disable input/button                                                  │
│  ├── Add user message to UI                                                │
│  ├── Show loading animation                                                │
│  └── POST /api/query                                                       │
│      Body: { query: "...", session_id: "..." }                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                               BACKEND                                      │
│                              FastAPI (app.py)                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    2. HTTP Request received
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  @app.post("/api/query")                                                   │
│  ├── Create session if needed                                              │
│  ├── Call rag_system.query(query, session_id)                             │
│  └── Return QueryResponse                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            RAG SYSTEM                                      │
│                          (rag_system.py)                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    3. Query processing begins
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  rag_system.query()                                                        │
│  ├── Get conversation history ─────────┐                                   │
│  ├── Prepare AI prompt                 │                                   │
│  ├── Call ai_generator                 │                                   │
│  ├── Get sources from tool_manager     │                                   │
│  ├── Update conversation history ◄─────┘                                   │
│  └── Return (response, sources)                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          AI GENERATOR                                      │
│                        (ai_generator.py)                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    4. Send to Claude API with tools
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  generate_response()                                                       │
│  ├── Build system prompt                                                   │
│  ├── Send to Anthropic API                                                 │
│  │   ├── Query + history + tools                                          │
│  │   └── Claude decides to search ─────────┐                              │
│  └── Return AI response                    │                              │
└─────────────────────────────────────────────┼─────────────────────────────────┘
                                            │
                    5. Claude calls search tool
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       SEARCH TOOLS                                         │
│                     (search_tools.py)                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  CourseSearchTool.execute()                                                │
│  ├── Parse search parameters                                               │
│  ├── Call vector_store.search() ─────────┐                                 │
│  ├── Format results for Claude           │                                 │
│  ├── Track sources                       │                                 │
│  └── Return formatted results            │                                 │
└─────────────────────────────────────────────┼─────────────────────────────────┘
                                            │
                    6. Semantic search in vector DB
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        VECTOR STORE                                        │
│                       (vector_store.py)                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  vector_store.search()                                                     │
│  ├── Query embedding generation                                            │
│  ├── ChromaDB similarity search                                            │
│  ├── Apply course/lesson filters                                           │
│  ├── Rank and score results                                                │
│  └── Return SearchResults                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            CHROMADB                                        │
│                     (Course content chunks)                                │
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │   Course1   │  │   Course2   │  │   Course3   │  │   Course4   │      │
│  │   Lesson1   │  │   Lesson1   │  │   Lesson1   │  │   Lesson1   │      │
│  │   Chunk1    │  │   Chunk1    │  │   Chunk1    │  │   Chunk1    │      │
│  │   Vector    │  │   Vector    │  │   Vector    │  │   Vector    │      │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    7. Results flow back up
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          RESPONSE JOURNEY                                  │
│                                                                             │
│  Search Results ──► CourseSearchTool ──► Claude ──► AI Generator           │
│                                           │                                 │
│                           ┌───────────────┘                                 │
│                           ▼                                                 │
│  Final AI Response ──► RAG System ──► FastAPI ──► Frontend                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            FRONTEND                                        │
│                          (Response Handling)                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    8. UI Updates
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Response Processing                                                       │
│  ├── Remove loading animation                                              │
│  ├── Parse markdown response                                               │
│  ├── Display AI message                                                    │
│  ├── Show sources (collapsible)                                            │
│  ├── Re-enable input                                                       │
│  └── Update session state                                                  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW                                     │
│                                                                             │
│  User Query ──► Session Management ──► Tool-Augmented AI ──► Semantic      │
│                                                              Search         │
│       │                                      ▲                   │         │
│       ▼                                      │                   ▼         │
│  UI Response ◄── Formatted Response ◄── AI Response ◄── Search Results     │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Components:

### **Frontend Layer**
- **User Interface**: HTML form with chat messages
- **JavaScript**: Event handling, API calls, UI updates
- **State Management**: Session tracking, loading states

### **API Layer**
- **FastAPI**: HTTP endpoint routing
- **Request/Response**: Pydantic models for validation
- **Error Handling**: HTTP status codes and error messages

### **RAG Orchestration**
- **Session Management**: Conversation history tracking
- **Tool Coordination**: Managing AI tool usage
- **Response Assembly**: Combining AI output with sources

### **AI Processing**
- **Claude Integration**: Anthropic API calls
- **Tool Usage**: Semantic search tool execution
- **Context Awareness**: History and prompt engineering

### **Vector Search**
- **Embedding Generation**: Query vectorization
- **Similarity Search**: ChromaDB queries
- **Filtering**: Course/lesson constraints
- **Result Ranking**: Relevance scoring

### **Data Storage**
- **ChromaDB**: Vector database for semantic search
- **Document Chunks**: Preprocessed course content
- **Metadata**: Course, lesson, and chunk information