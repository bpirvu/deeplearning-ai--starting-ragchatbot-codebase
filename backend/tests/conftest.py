import pytest
from types import SimpleNamespace
import tempfile
import os
from typing import List, Dict, Any
import sys

# Add backend directory to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import Course, Lesson, CourseChunk
from vector_store import SearchResults
from config import Config


@pytest.fixture
def sample_lessons():
    """Sample lesson data for testing"""
    return [
        Lesson(lesson_number=1, title="Introduction to AI", lesson_link="https://example.com/lesson1"),
        Lesson(lesson_number=2, title="Machine Learning Basics", lesson_link="https://example.com/lesson2"),
        Lesson(lesson_number=3, title="Deep Learning", lesson_link=None)
    ]


@pytest.fixture
def sample_course(sample_lessons):
    """Sample course data for testing"""
    return Course(
        title="Introduction to Machine Learning",
        course_link="https://example.com/course",
        instructor="Dr. Jane Smith",
        lessons=sample_lessons
    )


@pytest.fixture
def sample_course_chunks(sample_course):
    """Sample course chunks for testing"""
    return [
        CourseChunk(
            content="Course Introduction to Machine Learning Lesson 1 content: This is an introduction to artificial intelligence and machine learning concepts.",
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=1
        ),
        CourseChunk(
            content="Course Introduction to Machine Learning Lesson 2 content: Machine learning algorithms can be supervised, unsupervised, or reinforcement learning.",
            course_title=sample_course.title,
            lesson_number=2,
            chunk_index=2
        ),
        CourseChunk(
            content="Deep learning is a subset of machine learning using neural networks with multiple layers.",
            course_title=sample_course.title,
            lesson_number=3,
            chunk_index=3
        )
    ]


@pytest.fixture
def sample_search_results():
    """Sample search results for testing"""
    return SearchResults(
        documents=[
            "This is an introduction to artificial intelligence and machine learning concepts.",
            "Machine learning algorithms can be supervised, unsupervised, or reinforcement learning."
        ],
        metadata=[
            {"course_title": "Introduction to Machine Learning", "lesson_number": 1, "chunk_index": 0},
            {"course_title": "Introduction to Machine Learning", "lesson_number": 2, "chunk_index": 2}
        ],
        distances=[0.1, 0.3]
    )


@pytest.fixture
def empty_search_results():
    """Empty search results for testing"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[]
    )


@pytest.fixture
def error_search_results():
    """Error search results for testing"""
    return SearchResults.empty("No course found matching 'NonExistent Course'")


@pytest.fixture
def mock_vector_store(mocker, sample_search_results, sample_course):
    """Mock VectorStore for testing"""
    mock_store = mocker.Mock()
    mock_store.search.return_value = sample_search_results
    mock_store.get_lesson_link.return_value = "https://example.com/lesson1"
    mock_store.get_all_courses_metadata.return_value = [
        {
            "title": sample_course.title,
            "instructor": sample_course.instructor,
            "course_link": sample_course.course_link,
            "lessons": [
                {
                    "lesson_number": lesson.lesson_number,
                    "lesson_title": lesson.title,
                    "lesson_link": lesson.lesson_link
                } for lesson in sample_course.lessons
            ]
        }
    ]
    mock_store._resolve_course_name.return_value = sample_course.title
    return mock_store


@pytest.fixture
def mock_anthropic_client(mocker):
    """Mock Anthropic client for testing"""
    mock_client = mocker.Mock()

    # Create mock response for direct text response
    mock_text_response = SimpleNamespace(
        content=[SimpleNamespace(text="This is a test response from Claude")],
        stop_reason="end_turn"
    )

    # Create mock response for tool use
    mock_tool_content = SimpleNamespace(
        type="tool_use",
        name="search_course_content",
        input={"query": "test query"},
        id="tool_123"
    )

    mock_tool_response = SimpleNamespace(
        content=[mock_tool_content],
        stop_reason="tool_use"
    )

    # Create final response after tool use
    mock_final_response = SimpleNamespace(
        content=[SimpleNamespace(text="Based on the search results, here is the answer...")],
        stop_reason="end_turn"
    )

    mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]

    return mock_client


@pytest.fixture
def test_config():
    """Test configuration with fixed values"""
    config = Config()
    # Override problematic values for testing
    config.MAX_RESULTS = 5  # Fixed value instead of 0
    config.ANTHROPIC_API_KEY = "test_api_key"
    config.CHROMA_PATH = ":memory:"  # Use in-memory for tests
    return config


@pytest.fixture
def broken_config():
    """Configuration with the actual broken values from production"""
    config = Config()
    # Use actual broken values to test the issue
    config.MAX_RESULTS = 0  # This is the bug!
    config.ANTHROPIC_API_KEY = "test_api_key"
    config.CHROMA_PATH = ":memory:"
    return config


@pytest.fixture
def temp_course_document():
    """Create a temporary course document for testing"""
    content = """Course Title: Test Course
Course Link: https://example.com/test-course
Course Instructor: Test Instructor

Lesson 1: Introduction
Lesson Link: https://example.com/lesson1
This is the introduction lesson content. It covers basic concepts and provides an overview of what students will learn.

Lesson 2: Advanced Topics
This is the advanced topics lesson. It covers more complex concepts and builds on the introduction.
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def mock_tool_manager(mocker):
    """Mock ToolManager for testing"""
    mock_manager = mocker.Mock()
    mock_manager.get_tool_definitions.return_value = [
        {
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"}
                },
                "required": ["query"]
            }
        }
    ]
    mock_manager.execute_tool.return_value = "Test search results from tool manager"
    mock_manager.get_last_sources.return_value = ["Test Course - Lesson 1"]
    mock_manager.get_last_source_links.return_value = ["https://example.com/lesson1"]
    mock_manager.reset_sources.return_value = None
    return mock_manager


@pytest.fixture
def mock_session_manager(mocker):
    """Mock SessionManager for testing"""
    mock_manager = mocker.Mock()
    mock_manager.create_session.return_value = "test_session_123"
    mock_manager.get_conversation_history.return_value = "Previous conversation: User asked about AI"
    mock_manager.add_exchange.return_value = None
    return mock_manager


# Mock ChromaDB for testing
@pytest.fixture
def mock_chromadb(mocker):
    """Mock ChromaDB client and collections"""
    mock_client_class = mocker.patch('chromadb.PersistentClient')
    mock_embedding_class = mocker.patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')

    mock_client = mocker.Mock()
    mock_collection = mocker.Mock()
    mock_embedding_fn = mocker.Mock()

    # Configure embedding function mock
    mock_embedding_class.return_value = mock_embedding_fn

    # Configure collection mock with SimpleNamespace-like responses
    mock_collection.query.return_value = {
        'documents': [["Test document content", "Another test document"]],
        'metadatas': [[
            {"course_title": "Test Course", "lesson_number": 1, "chunk_index": 0},
            {"course_title": "Test Course", "lesson_number": 2, "chunk_index": 1}
        ]],
        'distances': [[0.1, 0.3]]
    }
    mock_collection.add.return_value = None
    mock_collection.get.return_value = {
        'ids': ['Test Course'],
        'metadatas': [{
            'title': 'Test Course',
            'instructor': 'Test Instructor',
            'course_link': 'https://example.com/course',
            'lessons_json': '[]',
            'lesson_count': 0
        }]
    }

    # Configure client mock
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_client.delete_collection.return_value = None
    mock_client_class.return_value = mock_client

    yield mock_client, mock_collection