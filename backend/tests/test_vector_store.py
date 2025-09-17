import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk


class TestSearchResults:
    """Test suite for SearchResults class"""

    def test_from_chroma_with_results(self):
        """Test creating SearchResults from ChromaDB results"""
        # Arrange
        chroma_results = {
            'documents': [["Document 1", "Document 2"]],
            'metadatas': [[{"course_title": "Test Course", "lesson_number": 1}, {"course_title": "Test Course", "lesson_number": 2}]],
            'distances': [[0.1, 0.3]]
        }

        # Act
        results = SearchResults.from_chroma(chroma_results)

        # Assert
        assert len(results.documents) == 2
        assert results.documents[0] == "Document 1"
        assert results.documents[1] == "Document 2"
        assert len(results.metadata) == 2
        assert results.metadata[0]["course_title"] == "Test Course"
        assert len(results.distances) == 2
        assert results.error is None

    def test_from_chroma_empty_results(self):
        """Test creating SearchResults from empty ChromaDB results"""
        # Arrange
        chroma_results = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }

        # Act
        results = SearchResults.from_chroma(chroma_results)

        # Assert
        assert len(results.documents) == 0
        assert len(results.metadata) == 0
        assert len(results.distances) == 0
        assert results.error is None
        assert results.is_empty()

    def test_empty_with_error(self):
        """Test creating empty results with error message"""
        # Act
        results = SearchResults.empty("Test error message")

        # Assert
        assert results.is_empty()
        assert results.error == "Test error message"
        assert len(results.documents) == 0

    def test_is_empty(self):
        """Test is_empty method"""
        # Test with empty results
        empty_results = SearchResults(documents=[], metadata=[], distances=[])
        assert empty_results.is_empty()

        # Test with non-empty results
        non_empty_results = SearchResults(
            documents=["doc"],
            metadata=[{"key": "value"}],
            distances=[0.1]
        )
        assert not non_empty_results.is_empty()


class TestVectorStore:
    """Test suite for VectorStore class"""

    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_initialization(self, mock_embedding_fn, mock_client_class):
        """Test VectorStore initialization"""
        # Arrange
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client

        # Act
        store = VectorStore("/test/path", "test-model", max_results=5)

        # Assert
        assert store.max_results == 5
        mock_client_class.assert_called_once()
        assert mock_client.get_or_create_collection.call_count == 2  # Two collections created

    def test_search_basic_query(self, mock_chromadb):
        """Test basic search functionality"""
        # Arrange
        mock_client, mock_collection = mock_chromadb
        store = VectorStore("/test/path", "test-model", max_results=5)

        # Act
        results = store.search("machine learning basics")

        # Assert
        mock_collection.query.assert_called_once_with(
            query_texts=["machine learning basics"],
            n_results=5,
            where=None
        )
        assert not results.is_empty()
        assert len(results.documents) == 2

    def test_search_with_course_filter(self, mock_chromadb):
        """Test search with course name filter"""
        # Arrange
        mock_client, mock_collection = mock_chromadb
        store = VectorStore("/test/path", "test-model", max_results=5)

        # Mock course resolution
        store.course_catalog = Mock()
        store.course_catalog.query.return_value = {
            'documents': [["Test Course"]],
            'metadatas': [[{"title": "Test Course"}]]
        }

        # Act
        results = store.search("algorithms", course_name="Test Course")

        # Assert
        # Should call _resolve_course_name and then search with filter
        mock_collection.query.assert_called_once_with(
            query_texts=["algorithms"],
            n_results=5,
            where={"course_title": "Test Course"}
        )

    def test_search_with_lesson_filter(self, mock_chromadb):
        """Test search with lesson number filter"""
        # Arrange
        mock_client, mock_collection = mock_chromadb
        store = VectorStore("/test/path", "test-model", max_results=5)

        # Act
        results = store.search("concepts", lesson_number=1)

        # Assert
        mock_collection.query.assert_called_once_with(
            query_texts=["concepts"],
            n_results=5,
            where={"lesson_number": 1}
        )

    def test_search_with_both_filters(self, mock_chromadb):
        """Test search with both course and lesson filters"""
        # Arrange
        mock_client, mock_collection = mock_chromadb
        store = VectorStore("/test/path", "test-model", max_results=5)

        # Mock course resolution
        store.course_catalog = Mock()
        store.course_catalog.query.return_value = {
            'documents': [["Test Course"]],
            'metadatas': [[{"title": "Test Course"}]]
        }

        # Act
        results = store.search("introduction", course_name="Test Course", lesson_number=1)

        # Assert
        mock_collection.query.assert_called_once_with(
            query_texts=["introduction"],
            n_results=5,
            where={"$and": [
                {"course_title": "Test Course"},
                {"lesson_number": 1}
            ]}
        )

    def test_search_with_custom_limit(self, mock_chromadb):
        """Test search with custom limit parameter"""
        # Arrange
        mock_client, mock_collection = mock_chromadb
        store = VectorStore("/test/path", "test-model", max_results=5)

        # Act
        results = store.search("test query", limit=3)

        # Assert
        mock_collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=3,  # Should use custom limit
            where=None
        )

    def test_search_with_zero_max_results(self, mock_chromadb):
        """Test search with MAX_RESULTS=0 (should fail validation)"""
        # Arrange
        mock_client, mock_collection = mock_chromadb

        # Act & Assert - VectorStore should prevent max_results=0
        with pytest.raises(ValueError, match="max_results must be > 0, got 0"):
            VectorStore("/test/path", "test-model", max_results=0)

    def test_search_course_not_found(self, mock_chromadb):
        """Test search when course name cannot be resolved"""
        # Arrange
        mock_client, mock_collection = mock_chromadb
        store = VectorStore("/test/path", "test-model", max_results=5)

        # Mock course resolution to return None
        store.course_catalog = Mock()
        store.course_catalog.query.return_value = {
            'documents': [[]],
            'metadatas': [[]]
        }

        # Act
        results = store.search("test query", course_name="Nonexistent Course")

        # Assert
        assert results.error == "No course found matching 'Nonexistent Course'"
        assert results.is_empty()

    def test_search_exception_handling(self, mock_chromadb):
        """Test search exception handling"""
        # Arrange
        mock_client, mock_collection = mock_chromadb
        mock_collection.query.side_effect = Exception("Database connection error")
        store = VectorStore("/test/path", "test-model", max_results=5)

        # Act
        results = store.search("test query")

        # Assert
        assert results.error == "Search error: Database connection error"
        assert results.is_empty()

    def test_resolve_course_name_success(self, mock_chromadb):
        """Test successful course name resolution"""
        # Arrange
        mock_client, mock_collection = mock_chromadb
        store = VectorStore("/test/path", "test-model", max_results=5)

        # Mock catalog query
        store.course_catalog = Mock()
        store.course_catalog.query.return_value = {
            'documents': [["Introduction to Machine Learning"]],
            'metadatas': [[{"title": "Introduction to Machine Learning"}]]
        }

        # Act
        resolved_name = store._resolve_course_name("Machine Learning")

        # Assert
        assert resolved_name == "Introduction to Machine Learning"
        store.course_catalog.query.assert_called_once_with(
            query_texts=["Machine Learning"],
            n_results=1
        )

    def test_resolve_course_name_not_found(self, mock_chromadb):
        """Test course name resolution when no match found"""
        # Arrange
        mock_client, mock_collection = mock_chromadb
        store = VectorStore("/test/path", "test-model", max_results=5)

        # Mock catalog query with empty results
        store.course_catalog = Mock()
        store.course_catalog.query.return_value = {
            'documents': [[]],
            'metadatas': [[]]
        }

        # Act
        resolved_name = store._resolve_course_name("Nonexistent Course")

        # Assert
        assert resolved_name is None

    def test_build_filter_no_filters(self, mock_chromadb):
        """Test filter building with no parameters"""
        # Arrange
        mock_client, mock_collection = mock_chromadb
        store = VectorStore("/test/path", "test-model", max_results=5)

        # Act
        filter_dict = store._build_filter(None, None)

        # Assert
        assert filter_dict is None

    def test_build_filter_course_only(self, mock_chromadb):
        """Test filter building with course only"""
        # Arrange
        mock_client, mock_collection = mock_chromadb
        store = VectorStore("/test/path", "test-model", max_results=5)

        # Act
        filter_dict = store._build_filter("Test Course", None)

        # Assert
        assert filter_dict == {"course_title": "Test Course"}

    def test_build_filter_lesson_only(self, mock_chromadb):
        """Test filter building with lesson only"""
        # Arrange
        mock_client, mock_collection = mock_chromadb
        store = VectorStore("/test/path", "test-model", max_results=5)

        # Act
        filter_dict = store._build_filter(None, 2)

        # Assert
        assert filter_dict == {"lesson_number": 2}

    def test_build_filter_both_parameters(self, mock_chromadb):
        """Test filter building with both parameters"""
        # Arrange
        mock_client, mock_collection = mock_chromadb
        store = VectorStore("/test/path", "test-model", max_results=5)

        # Act
        filter_dict = store._build_filter("Test Course", 2)

        # Assert
        expected = {
            "$and": [
                {"course_title": "Test Course"},
                {"lesson_number": 2}
            ]
        }
        assert filter_dict == expected

    def test_add_course_metadata(self, mock_chromadb, sample_course):
        """Test adding course metadata"""
        # Arrange
        mock_client, mock_collection = mock_chromadb
        store = VectorStore("/test/path", "test-model", max_results=5)

        # Act
        store.add_course_metadata(sample_course)

        # Assert
        store.course_catalog.add.assert_called_once()
        call_args = store.course_catalog.add.call_args

        assert call_args[1]['documents'] == [sample_course.title]
        assert call_args[1]['ids'] == [sample_course.title]
        assert call_args[1]['metadatas'][0]['title'] == sample_course.title
        assert call_args[1]['metadatas'][0]['instructor'] == sample_course.instructor

    def test_add_course_content(self, mock_chromadb, sample_course_chunks):
        """Test adding course content chunks"""
        # Arrange
        mock_client, mock_collection = mock_chromadb
        store = VectorStore("/test/path", "test-model", max_results=5)

        # Act
        store.add_course_content(sample_course_chunks)

        # Assert
        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args

        assert len(call_args[1]['documents']) == len(sample_course_chunks)
        assert len(call_args[1]['metadatas']) == len(sample_course_chunks)
        assert len(call_args[1]['ids']) == len(sample_course_chunks)

        # Check first chunk metadata
        first_metadata = call_args[1]['metadatas'][0]
        assert first_metadata['course_title'] == sample_course_chunks[0].course_title
        assert first_metadata['lesson_number'] == sample_course_chunks[0].lesson_number
        assert first_metadata['chunk_index'] == sample_course_chunks[0].chunk_index

    def test_add_course_content_empty(self, mock_chromadb):
        """Test adding empty course content"""
        # Arrange
        mock_client, mock_collection = mock_chromadb
        store = VectorStore("/test/path", "test-model", max_results=5)

        # Act
        store.add_course_content([])

        # Assert
        mock_collection.add.assert_not_called()

    def test_clear_all_data(self, mock_chromadb):
        """Test clearing all data"""
        # Arrange
        mock_client, mock_collection = mock_chromadb
        store = VectorStore("/test/path", "test-model", max_results=5)

        # Act
        store.clear_all_data()

        # Assert
        assert mock_client.delete_collection.call_count == 2
        mock_client.delete_collection.assert_any_call("course_catalog")
        mock_client.delete_collection.assert_any_call("course_content")

    def test_get_existing_course_titles(self, mock_chromadb):
        """Test getting existing course titles"""
        # Arrange
        mock_client, mock_collection = mock_chromadb
        store = VectorStore("/test/path", "test-model", max_results=5)

        # Mock catalog get
        store.course_catalog = Mock()
        store.course_catalog.get.return_value = {
            'ids': ['Course 1', 'Course 2', 'Course 3']
        }

        # Act
        titles = store.get_existing_course_titles()

        # Assert
        assert titles == ['Course 1', 'Course 2', 'Course 3']

    def test_get_course_count(self, mock_chromadb):
        """Test getting course count"""
        # Arrange
        mock_client, mock_collection = mock_chromadb
        store = VectorStore("/test/path", "test-model", max_results=5)

        # Mock catalog get
        store.course_catalog = Mock()
        store.course_catalog.get.return_value = {
            'ids': ['Course 1', 'Course 2']
        }

        # Act
        count = store.get_course_count()

        # Assert
        assert count == 2

    def test_get_lesson_link(self, mock_chromadb):
        """Test getting lesson link"""
        # Arrange
        mock_client, mock_collection = mock_chromadb
        store = VectorStore("/test/path", "test-model", max_results=5)

        # Mock catalog get with lessons_json
        lessons_json = '[{"lesson_number": 1, "lesson_title": "Intro", "lesson_link": "https://example.com/lesson1"}]'
        store.course_catalog = Mock()
        store.course_catalog.get.return_value = {
            'metadatas': [{'lessons_json': lessons_json}]
        }

        # Act
        link = store.get_lesson_link("Test Course", 1)

        # Assert
        assert link == "https://example.com/lesson1"

    def test_get_lesson_link_not_found(self, mock_chromadb):
        """Test getting lesson link when lesson not found"""
        # Arrange
        mock_client, mock_collection = mock_chromadb
        store = VectorStore("/test/path", "test-model", max_results=5)

        # Mock catalog get with empty results
        store.course_catalog = Mock()
        store.course_catalog.get.return_value = {
            'metadatas': []
        }

        # Act
        link = store.get_lesson_link("Test Course", 1)

        # Assert
        assert link is None


class TestVectorStoreBugReproduction:
    """Specific tests to verify the MAX_RESULTS=0 bug has been fixed"""

    def test_bug_reproduction_zero_max_results(self, mock_chromadb):
        """Test that VectorStore properly validates against MAX_RESULTS=0 bug"""
        # Arrange - simulate the broken config that would have caused the bug
        mock_client, mock_collection = mock_chromadb

        # Act & Assert - VectorStore should now prevent the bug by validation
        with pytest.raises(ValueError, match="max_results must be > 0, got 0"):
            VectorStore("/test/path", "test-model", max_results=0)  # Should fail validation!

    def test_bug_fix_verification(self, mock_chromadb):
        """Test that verifies the bug is fixed with proper MAX_RESULTS"""
        # Arrange - simulate the fixed config
        mock_client, mock_collection = mock_chromadb
        store = VectorStore("/test/path", "test-model", max_results=5)  # FIXED!

        # Act
        results = store.search("machine learning basics")

        # Assert - this should work correctly
        mock_collection.query.assert_called_once_with(
            query_texts=["machine learning basics"],
            n_results=5,  # Now this will return results!
            where=None
        )
        assert not results.is_empty()  # Results returned because n_results > 0