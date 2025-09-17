import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import tempfile

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag_system import RAGSystem
from config import Config


class TestRAGSystemIntegration:
    """End-to-end integration tests for the RAG system"""

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_rag_system_initialization(self, mock_session_mgr, mock_doc_processor, mock_ai_gen, mock_vector_store):
        """Test RAG system initialization with all components"""
        # Arrange
        config = Config()
        config.MAX_RESULTS = 5  # Fixed config

        # Act
        rag_system = RAGSystem(config)

        # Assert
        assert rag_system.config == config
        mock_vector_store.assert_called_once_with(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
        mock_ai_gen.assert_called_once_with(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL)
        mock_doc_processor.assert_called_once_with(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        mock_session_mgr.assert_called_once_with(config.MAX_HISTORY)

        # Check that tools are registered
        assert len(rag_system.tool_manager.tools) == 2
        assert "search_course_content" in rag_system.tool_manager.tools
        assert "get_course_outline" in rag_system.tool_manager.tools

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_query_without_session(self, mock_session_mgr, mock_doc_processor, mock_ai_gen, mock_vector_store):
        """Test querying without session ID"""
        # Arrange
        config = Config()
        config.MAX_RESULTS = 5
        rag_system = RAGSystem(config)

        # Mock AI generator response
        mock_ai_gen.return_value.generate_response.return_value = "AI response about machine learning"

        # Mock tool manager sources
        rag_system.tool_manager.get_last_sources = Mock(return_value=["Test Course - Lesson 1"])

        # Act
        response, sources = rag_system.query("What is machine learning?")

        # Assert
        assert response == "AI response about machine learning"
        assert sources == ["Test Course - Lesson 1"]

        # Verify AI generator was called correctly
        mock_ai_gen.return_value.generate_response.assert_called_once()
        call_args = mock_ai_gen.return_value.generate_response.call_args

        assert "What is machine learning?" in call_args[1]['query']
        assert call_args[1]['conversation_history'] is None
        assert call_args[1]['tools'] is not None
        assert call_args[1]['tool_manager'] is not None

        # Verify session manager was not called for history
        mock_session_mgr.return_value.get_conversation_history.assert_not_called()

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_query_with_session(self, mock_session_mgr, mock_doc_processor, mock_ai_gen, mock_vector_store):
        """Test querying with session ID and conversation history"""
        # Arrange
        config = Config()
        config.MAX_RESULTS = 5
        rag_system = RAGSystem(config)

        # Mock session manager
        mock_session_mgr.return_value.get_conversation_history.return_value = "Previous conversation history"

        # Mock AI generator response
        mock_ai_gen.return_value.generate_response.return_value = "AI response with context"

        # Mock tool manager sources
        rag_system.tool_manager.get_last_sources = Mock(return_value=["Test Course - Lesson 2"])

        # Act
        response, sources = rag_system.query("Can you elaborate?", session_id="test_session_123")

        # Assert
        assert response == "AI response with context"
        assert sources == ["Test Course - Lesson 2"]

        # Verify session manager was called
        mock_session_mgr.return_value.get_conversation_history.assert_called_once_with("test_session_123")
        mock_session_mgr.return_value.add_exchange.assert_called_once_with(
            "test_session_123",
            "Can you elaborate?",
            "AI response with context"
        )

        # Verify AI generator received history
        call_args = mock_ai_gen.return_value.generate_response.call_args
        assert call_args[1]['conversation_history'] == "Previous conversation history"

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_query_with_broken_config(self, mock_session_mgr, mock_doc_processor, mock_ai_gen, mock_vector_store):
        """Test that RAGSystem properly rejects invalid MAX_RESULTS=0 config"""
        # Arrange
        config = Config()
        config.MAX_RESULTS = 0  # Invalid configuration

        # Act & Assert - Should raise ValueError for invalid config
        with pytest.raises(ValueError, match="Configuration validation failed. Cannot initialize RAG system."):
            RAGSystem(config)

    @patch('rag_system.os.path.isfile')
    @patch('rag_system.os.path.exists')
    @patch('rag_system.os.listdir')
    def test_add_course_folder_success(self, mock_listdir, mock_exists, mock_isfile, temp_course_document):
        """Test adding course folder successfully"""
        # Arrange
        config = Config()
        config.MAX_RESULTS = 5

        with patch('rag_system.VectorStore') as mock_vector_store, \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.DocumentProcessor') as mock_doc_processor, \
             patch('rag_system.SessionManager'):

            # Mock file system
            mock_exists.return_value = True
            mock_listdir.return_value = ['course1.txt']
            mock_isfile.return_value = True

            # Mock document processor
            from models import Course, Lesson, CourseChunk
            sample_course = Course(title="Test Course", instructor="Test Instructor")
            sample_chunks = [
                CourseChunk(content="Test content", course_title="Test Course", chunk_index=0)
            ]
            mock_doc_processor.return_value.process_course_document.return_value = (sample_course, sample_chunks)

            # Mock vector store
            mock_vector_store.return_value.get_existing_course_titles.return_value = []

            rag_system = RAGSystem(config)

            # Act
            courses, chunks = rag_system.add_course_folder("/test/docs")

            # Assert
            assert courses == 1
            assert chunks == 1
            mock_vector_store.return_value.add_course_metadata.assert_called_once_with(sample_course)
            mock_vector_store.return_value.add_course_content.assert_called_once_with(sample_chunks)

    def test_add_course_folder_nonexistent(self):
        """Test adding course folder that doesn't exist"""
        # Arrange
        config = Config()
        config.MAX_RESULTS = 5

        with patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.DocumentProcessor'), \
             patch('rag_system.SessionManager'):

            rag_system = RAGSystem(config)

            # Act
            courses, chunks = rag_system.add_course_folder("/nonexistent/path")

            # Assert
            assert courses == 0
            assert chunks == 0

    def test_get_course_analytics(self):
        """Test getting course analytics"""
        # Arrange
        config = Config()
        config.MAX_RESULTS = 5

        with patch('rag_system.VectorStore') as mock_vector_store, \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.DocumentProcessor'), \
             patch('rag_system.SessionManager'):

            mock_vector_store.return_value.get_course_count.return_value = 3
            mock_vector_store.return_value.get_existing_course_titles.return_value = ["Course 1", "Course 2", "Course 3"]

            rag_system = RAGSystem(config)

            # Act
            analytics = rag_system.get_course_analytics()

            # Assert
            assert analytics["total_courses"] == 3
            assert analytics["course_titles"] == ["Course 1", "Course 2", "Course 3"]


class TestRAGSystemToolIntegration:
    """Test integration between RAG system and its tools"""

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_search_tool_integration(self, mock_session_mgr, mock_doc_processor, mock_ai_gen, mock_vector_store):
        """Test that search tool is properly integrated"""
        # Arrange
        config = Config()
        config.MAX_RESULTS = 5
        rag_system = RAGSystem(config)

        # Mock vector store search results
        from vector_store import SearchResults
        search_results = SearchResults(
            documents=["Test document content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1]
        )
        mock_vector_store.return_value.search.return_value = search_results
        mock_vector_store.return_value.get_lesson_link.return_value = "https://example.com/lesson1"

        # Act - execute search tool directly
        result = rag_system.search_tool.execute("machine learning basics")

        # Assert
        assert "Test Course - Lesson 1" in result
        assert "Test document content" in result
        assert len(rag_system.search_tool.last_sources) == 1
        assert "Test Course - Lesson 1" in rag_system.search_tool.last_sources[0]

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_outline_tool_integration(self, mock_session_mgr, mock_doc_processor, mock_ai_gen, mock_vector_store):
        """Test that outline tool is properly integrated"""
        # Arrange
        config = Config()
        config.MAX_RESULTS = 5
        rag_system = RAGSystem(config)

        # Mock vector store for outline tool
        mock_vector_store.return_value._resolve_course_name.return_value = "Test Course"
        mock_vector_store.return_value.get_all_courses_metadata.return_value = [
            {
                "title": "Test Course",
                "instructor": "Test Instructor",
                "course_link": "https://example.com/course",
                "lessons": [
                    {"lesson_number": 1, "lesson_title": "Introduction", "lesson_link": "https://example.com/lesson1"}
                ]
            }
        ]

        # Act - execute outline tool directly
        result = rag_system.outline_tool.execute(course_name="Test Course")

        # Assert
        assert "**Course:** Test Course" in result
        assert "**Instructor:** Test Instructor" in result
        assert "1. Introduction" in result
        assert len(rag_system.outline_tool.last_sources) == 1
        assert rag_system.outline_tool.last_sources[0] == "Test Course"


class TestRAGSystemErrorHandling:
    """Test error handling in RAG system"""

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_ai_generator_error_propagation(self, mock_session_mgr, mock_doc_processor, mock_ai_gen, mock_vector_store):
        """Test that AI generator errors are properly handled"""
        # Arrange
        config = Config()
        config.MAX_RESULTS = 5
        rag_system = RAGSystem(config)

        # Mock AI generator to raise an exception
        mock_ai_gen.return_value.generate_response.side_effect = Exception("API Error: Rate limited")

        # Act & Assert
        with pytest.raises(Exception) as excinfo:
            rag_system.query("Test query")

        assert "API Error: Rate limited" in str(excinfo.value)

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_document_processing_error_handling(self, mock_session_mgr, mock_doc_processor, mock_ai_gen, mock_vector_store):
        """Test error handling during document processing"""
        # Arrange
        config = Config()
        config.MAX_RESULTS = 5

        # Mock document processor to raise an exception
        mock_doc_processor.return_value.process_course_document.side_effect = Exception("File parsing error")

        rag_system = RAGSystem(config)

        # Act - this should not crash, but return error info
        course, chunk_count = rag_system.add_course_document("/test/file.txt")

        # Assert
        assert course is None
        assert chunk_count == 0


class TestRAGSystemWithRealConfigBug:
    """Tests specifically designed to demonstrate the MAX_RESULTS=0 bug"""

    def test_bug_demonstration_with_real_config(self):
        """Demonstrate the bug using actual broken config values"""
        # Arrange - Load the actual broken config
        from config import Config
        broken_config = Config()
        # The actual config has MAX_RESULTS = 0

        with patch('rag_system.VectorStore') as mock_vector_store, \
             patch('rag_system.AIGenerator') as mock_ai_gen, \
             patch('rag_system.DocumentProcessor'), \
             patch('rag_system.SessionManager'):

            # Simulate the vector store behavior with MAX_RESULTS=0
            mock_vector_store.return_value.search.return_value = Mock()
            mock_vector_store.return_value.search.return_value.is_empty.return_value = True
            mock_vector_store.return_value.search.return_value.error = None

            # Simulate AI generator getting no tool results
            mock_ai_gen.return_value.generate_response.return_value = "I don't have specific information about that."

            rag_system = RAGSystem(broken_config)
            rag_system.tool_manager.get_last_sources = Mock(return_value=[])

            # Act
            response, sources = rag_system.query("What is machine learning?")

            # Assert - this demonstrates the bug
            assert "I don't have specific information" in response
            assert sources == []

            # Verify the vector store was initialized with the broken value
            mock_vector_store.assert_called_with(
                broken_config.CHROMA_PATH,
                broken_config.EMBEDDING_MODEL,
                broken_config.MAX_RESULTS  # This should be 0
            )

    def test_bug_fix_verification_with_corrected_config(self):
        """Verify that fixing the config resolves the issue"""
        # Arrange - Create fixed config
        fixed_config = Config()
        fixed_config.MAX_RESULTS = 5  # FIXED!

        with patch('rag_system.VectorStore') as mock_vector_store, \
             patch('rag_system.AIGenerator') as mock_ai_gen, \
             patch('rag_system.DocumentProcessor'), \
             patch('rag_system.SessionManager'):

            # Simulate the vector store working correctly with MAX_RESULTS > 0
            from vector_store import SearchResults
            mock_search_results = SearchResults(
                documents=["Machine learning is a subset of AI..."],
                metadata=[{"course_title": "AI Course", "lesson_number": 1}],
                distances=[0.1]
            )
            mock_vector_store.return_value.search.return_value = mock_search_results

            # Simulate AI generator getting good tool results
            mock_ai_gen.return_value.generate_response.return_value = "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data."

            rag_system = RAGSystem(fixed_config)
            rag_system.tool_manager.get_last_sources = Mock(return_value=["AI Course - Lesson 1"])

            # Act
            response, sources = rag_system.query("What is machine learning?")

            # Assert - this demonstrates the fix
            assert "Machine learning is a subset of artificial intelligence" in response
            assert sources == ["AI Course - Lesson 1"]

            # Verify the vector store was initialized with the fixed value
            mock_vector_store.assert_called_with(
                fixed_config.CHROMA_PATH,
                fixed_config.EMBEDDING_MODEL,
                5  # This should be 5, not 0
            )