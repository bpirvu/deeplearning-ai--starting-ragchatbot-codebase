import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from models import Course, CourseChunk
from sentence_transformers import SentenceTransformer
import logging
import os

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class SearchResults:
    """Container for search results with metadata"""
    documents: List[str]
    metadata: List[Dict[str, Any]]
    distances: List[float]
    error: Optional[str] = None
    
    @classmethod
    def from_chroma(cls, chroma_results: Dict) -> 'SearchResults':
        """Create SearchResults from ChromaDB query results"""
        return cls(
            documents=chroma_results['documents'][0] if chroma_results['documents'] else [],
            metadata=chroma_results['metadatas'][0] if chroma_results['metadatas'] else [],
            distances=chroma_results['distances'][0] if chroma_results['distances'] else []
        )
    
    @classmethod
    def empty(cls, error_msg: str) -> 'SearchResults':
        """Create empty results with error message"""
        return cls(documents=[], metadata=[], distances=[], error=error_msg)
    
    def is_empty(self) -> bool:
        """Check if results are empty"""
        return len(self.documents) == 0

class VectorStore:
    """Vector storage using ChromaDB for course content and metadata with enhanced error handling"""

    def __init__(self, chroma_path: str, embedding_model: str, max_results: int = 5):
        logger.info(f"🔧 Initializing VectorStore with max_results={max_results}, path={chroma_path}")

        # Validate parameters
        if max_results <= 0:
            raise ValueError(f"max_results must be > 0, got {max_results}. This would cause search failures!")

        if not chroma_path:
            raise ValueError("chroma_path cannot be empty")

        if not embedding_model:
            raise ValueError("embedding_model cannot be empty")

        self.max_results = max_results
        self.chroma_path = chroma_path
        self.embedding_model = embedding_model

        try:
            # Initialize ChromaDB client
            logger.info(f"📁 Setting up ChromaDB at: {chroma_path}")
            self.client = chromadb.PersistentClient(
                path=chroma_path,
                settings=Settings(anonymized_telemetry=False)
            )

            # Set up sentence transformer embedding function
            logger.info(f"🧠 Loading embedding model: {embedding_model}")
            self.embedding_function = chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            )

            # Create collections for different types of data
            logger.info("📚 Creating/loading collections...")
            self.course_catalog = self._create_collection("course_catalog")  # Course titles/instructors
            self.course_content = self._create_collection("course_content")  # Actual course material

            logger.info("✅ VectorStore initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize VectorStore: {e}")
            logger.error(f"   - chroma_path: {chroma_path}")
            logger.error(f"   - embedding_model: {embedding_model}")
            logger.error(f"   - max_results: {max_results}")
            raise RuntimeError(f"VectorStore initialization failed: {e}") from e
    
    def _create_collection(self, name: str):
        """Create or get a ChromaDB collection"""
        return self.client.get_or_create_collection(
            name=name,
            embedding_function=self.embedding_function
        )
    
    def search(self,
               query: str,
               course_name: Optional[str] = None,
               lesson_number: Optional[int] = None,
               limit: Optional[int] = None) -> SearchResults:
        """
        Main search interface that handles course resolution and content search.

        Args:
            query: What to search for in course content
            course_name: Optional course name/title to filter by
            lesson_number: Optional lesson number to filter by
            limit: Maximum results to return

        Returns:
            SearchResults object with documents and metadata
        """
        # Validate input
        if not query or not query.strip():
            logger.warning("⚠️  Empty query provided to search")
            return SearchResults.empty("Query cannot be empty")

        search_limit = limit if limit is not None else self.max_results

        if search_limit <= 0:
            logger.error(f"❌ Invalid search limit: {search_limit}. Must be > 0")
            return SearchResults.empty(f"Invalid search limit: {search_limit}")

        logger.info(f"🔍 Searching: '{query[:50]}{'...' if len(query) > 50 else ''}' (limit={search_limit})")
        if course_name:
            logger.info(f"   📚 Course filter: '{course_name}'")
        if lesson_number:
            logger.info(f"   📖 Lesson filter: {lesson_number}")

        # Step 1: Resolve course name if provided
        course_title = None
        if course_name:
            logger.info(f"🔎 Resolving course name: '{course_name}'")
            course_title = self._resolve_course_name(course_name)
            if not course_title:
                logger.warning(f"⚠️  No course found matching '{course_name}'")
                return SearchResults.empty(f"No course found matching '{course_name}'")
            logger.info(f"✅ Resolved to: '{course_title}'")

        # Step 2: Build filter for content search
        filter_dict = self._build_filter(course_title, lesson_number)
        if filter_dict:
            logger.info(f"🔧 Using filter: {filter_dict}")

        # Step 3: Search course content
        try:
            logger.debug(f"🔍 Executing vector search with n_results={search_limit}")
            results = self.course_content.query(
                query_texts=[query],
                n_results=search_limit,
                where=filter_dict
            )

            search_results = SearchResults.from_chroma(results)

            if search_results.is_empty():
                logger.info("📭 No results found for query")
            else:
                logger.info(f"✅ Found {len(search_results.documents)} results")
                logger.debug(f"   📄 Result summary: {[doc[:100] + '...' if len(doc) > 100 else doc for doc in search_results.documents[:2]]}")

            return search_results

        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            logger.error(f"   Query: '{query}'")
            logger.error(f"   Course: {course_name}")
            logger.error(f"   Lesson: {lesson_number}")
            logger.error(f"   Limit: {search_limit}")
            return SearchResults.empty(f"Search error: {str(e)}")
    
    def _resolve_course_name(self, course_name: str) -> Optional[str]:
        """Use vector search to find best matching course by name"""
        try:
            results = self.course_catalog.query(
                query_texts=[course_name],
                n_results=1
            )
            
            if results['documents'][0] and results['metadatas'][0]:
                # Return the title (which is now the ID)
                return results['metadatas'][0][0]['title']
        except Exception as e:
            print(f"Error resolving course name: {e}")
        
        return None
    
    def _build_filter(self, course_title: Optional[str], lesson_number: Optional[int]) -> Optional[Dict]:
        """Build ChromaDB filter from search parameters"""
        if not course_title and lesson_number is None:
            return None
            
        # Handle different filter combinations
        if course_title and lesson_number is not None:
            return {"$and": [
                {"course_title": course_title},
                {"lesson_number": lesson_number}
            ]}
        
        if course_title:
            return {"course_title": course_title}
            
        return {"lesson_number": lesson_number}
    
    def add_course_metadata(self, course: Course):
        """Add course information to the catalog for semantic search"""
        import json

        course_text = course.title
        
        # Build lessons metadata and serialize as JSON string
        lessons_metadata = []
        for lesson in course.lessons:
            lessons_metadata.append({
                "lesson_number": lesson.lesson_number,
                "lesson_title": lesson.title,
                "lesson_link": lesson.lesson_link
            })
        
        self.course_catalog.add(
            documents=[course_text],
            metadatas=[{
                "title": course.title,
                "instructor": course.instructor,
                "course_link": course.course_link,
                "lessons_json": json.dumps(lessons_metadata),  # Serialize as JSON string
                "lesson_count": len(course.lessons)
            }],
            ids=[course.title]
        )
    
    def add_course_content(self, chunks: List[CourseChunk]):
        """Add course content chunks to the vector store"""
        if not chunks:
            return
        
        documents = [chunk.content for chunk in chunks]
        metadatas = [{
            "course_title": chunk.course_title,
            "lesson_number": chunk.lesson_number,
            "chunk_index": chunk.chunk_index
        } for chunk in chunks]
        # Use title with chunk index for unique IDs
        ids = [f"{chunk.course_title.replace(' ', '_')}_{chunk.chunk_index}" for chunk in chunks]
        
        self.course_content.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def clear_all_data(self):
        """Clear all data from both collections"""
        try:
            self.client.delete_collection("course_catalog")
            self.client.delete_collection("course_content")
            # Recreate collections
            self.course_catalog = self._create_collection("course_catalog")
            self.course_content = self._create_collection("course_content")
        except Exception as e:
            print(f"Error clearing data: {e}")
    
    def get_existing_course_titles(self) -> List[str]:
        """Get all existing course titles from the vector store"""
        try:
            # Get all documents from the catalog
            results = self.course_catalog.get()
            if results and 'ids' in results:
                return results['ids']
            return []
        except Exception as e:
            print(f"Error getting existing course titles: {e}")
            return []
    
    def get_course_count(self) -> int:
        """Get the total number of courses in the vector store"""
        try:
            results = self.course_catalog.get()
            if results and 'ids' in results:
                return len(results['ids'])
            return 0
        except Exception as e:
            print(f"Error getting course count: {e}")
            return 0
    
    def get_lesson_link(self, course_title: str, lesson_number: int) -> Optional[str]:
        """Get the lesson link for a specific course and lesson number"""
        import json
        try:
            # Query the course catalog for the specific course
            results = self.course_catalog.get(
                ids=[course_title]
            )

            if not results or not results.get('metadatas'):
                return None

            metadata = results['metadatas'][0]
            lessons_json = metadata.get('lessons_json')

            if not lessons_json:
                return None

            # Parse the lessons metadata
            lessons = json.loads(lessons_json)

            # Find the lesson with matching number
            for lesson in lessons:
                if lesson.get('lesson_number') == lesson_number:
                    return lesson.get('lesson_link')

            return None
        except Exception as e:
            print(f"Error getting lesson link for {course_title} lesson {lesson_number}: {e}")
            return None

    def get_all_courses_metadata(self) -> List[Dict[str, Any]]:
        """Get metadata for all courses in the vector store"""
        import json
        try:
            results = self.course_catalog.get()
            if results and 'metadatas' in results:
                # Parse lessons JSON for each course
                parsed_metadata = []
                for metadata in results['metadatas']:
                    course_meta = metadata.copy()
                    if 'lessons_json' in course_meta:
                        course_meta['lessons'] = json.loads(course_meta['lessons_json'])
                        del course_meta['lessons_json']  # Remove the JSON string version
                    parsed_metadata.append(course_meta)
                return parsed_metadata
            return []
        except Exception as e:
            print(f"Error getting courses metadata: {e}")
            return []

    def get_course_link(self, course_title: str) -> Optional[str]:
        """Get course link for a given course title"""
        try:
            # Get course by ID (title is the ID)
            results = self.course_catalog.get(ids=[course_title])
            if results and 'metadatas' in results and results['metadatas']:
                metadata = results['metadatas'][0]
                return metadata.get('course_link')
            return None
        except Exception as e:
            print(f"Error getting course link: {e}")
            return None
    
    def get_lesson_link(self, course_title: str, lesson_number: int) -> Optional[str]:
        """Get lesson link for a given course title and lesson number"""
        import json
        try:
            # Get course by ID (title is the ID)
            results = self.course_catalog.get(ids=[course_title])
            if results and 'metadatas' in results and results['metadatas']:
                metadata = results['metadatas'][0]
                lessons_json = metadata.get('lessons_json')
                if lessons_json:
                    lessons = json.loads(lessons_json)
                    # Find the lesson with matching number
                    for lesson in lessons:
                        if lesson.get('lesson_number') == lesson_number:
                            return lesson.get('lesson_link')
            return None
        except Exception as e:
            print(f"Error getting lesson link: {e}")
    