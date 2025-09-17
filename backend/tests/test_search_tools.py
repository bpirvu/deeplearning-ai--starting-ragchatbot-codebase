import pytest
from types import SimpleNamespace
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test suite for CourseSearchTool"""

    def test_basic_search_success(self, mock_vector_store, sample_search_results):
        """Test successful basic search without filters"""
        # Arrange
        tool = CourseSearchTool(mock_vector_store)

        # Act
        result = tool.execute(query="machine learning basics")

        # Assert
        assert result is not None
        assert "Introduction to Machine Learning - Lesson 1" in result
        assert "Introduction to Machine Learning - Lesson 2" in result
        assert "This is an introduction to artificial intelligence" in result
        assert "Machine learning algorithms can be supervised" in result

        # Verify vector store was called correctly
        mock_vector_store.search.assert_called_once_with(
            query="machine learning basics",
            course_name=None,
            lesson_number=None
        )

    def test_search_with_course_filter(self, mock_vector_store, sample_search_results):
        """Test search with course name filter"""
        # Arrange
        tool = CourseSearchTool(mock_vector_store)

        # Act
        result = tool.execute(query="algorithms", course_name="Machine Learning")

        # Assert
        assert result is not None
        mock_vector_store.search.assert_called_once_with(
            query="algorithms",
            course_name="Machine Learning",
            lesson_number=None
        )

    def test_search_with_lesson_filter(self, mock_vector_store, sample_search_results):
        """Test search with lesson number filter"""
        # Arrange
        tool = CourseSearchTool(mock_vector_store)

        # Act
        result = tool.execute(query="concepts", lesson_number=1)

        # Assert
        assert result is not None
        mock_vector_store.search.assert_called_once_with(
            query="concepts",
            course_name=None,
            lesson_number=1
        )

    def test_search_with_both_filters(self, mock_vector_store, sample_search_results):
        """Test search with both course name and lesson number filters"""
        # Arrange
        tool = CourseSearchTool(mock_vector_store)

        # Act
        result = tool.execute(
            query="introduction",
            course_name="Machine Learning",
            lesson_number=1
        )

        # Assert
        assert result is not None
        mock_vector_store.search.assert_called_once_with(
            query="introduction",
            course_name="Machine Learning",
            lesson_number=1
        )

    def test_search_empty_results(self, mock_vector_store, empty_search_results):
        """Test search when no results are found"""
        # Arrange
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)

        # Act
        result = tool.execute(query="nonexistent topic")

        # Assert
        assert result == "No relevant content found."

    def test_search_empty_results_with_course_filter(self, mock_vector_store, empty_search_results):
        """Test search with no results and course filter"""
        # Arrange
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)

        # Act
        result = tool.execute(query="nonexistent", course_name="Some Course")

        # Assert
        assert result == "No relevant content found in course 'Some Course'."

    def test_search_empty_results_with_lesson_filter(self, mock_vector_store, empty_search_results):
        """Test search with no results and lesson filter"""
        # Arrange
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)

        # Act
        result = tool.execute(query="nonexistent", lesson_number=5)

        # Assert
        assert result == "No relevant content found in lesson 5."

    def test_search_empty_results_with_both_filters(self, mock_vector_store, empty_search_results):
        """Test search with no results and both filters"""
        # Arrange
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)

        # Act
        result = tool.execute(
            query="nonexistent",
            course_name="Some Course",
            lesson_number=3
        )

        # Assert
        assert result == "No relevant content found in course 'Some Course' in lesson 3."

    def test_search_error_handling(self, mock_vector_store, error_search_results):
        """Test search when vector store returns an error"""
        # Arrange
        mock_vector_store.search.return_value = error_search_results
        tool = CourseSearchTool(mock_vector_store)

        # Act
        result = tool.execute(query="test query")

        # Assert
        assert result == "No course found matching 'NonExistent Course'"

    def test_source_tracking(self, mock_vector_store, sample_search_results):
        """Test that sources are properly tracked"""
        # Arrange
        tool = CourseSearchTool(mock_vector_store)

        # Act
        result = tool.execute(query="machine learning")

        # Assert
        assert len(tool.last_sources) == 2
        assert tool.last_sources[0] == "Introduction to Machine Learning - Lesson 1"
        assert tool.last_sources[1] == "Introduction to Machine Learning - Lesson 2"

    def test_source_links_tracking(self, mock_vector_store, sample_search_results):
        """Test that source links are properly tracked"""
        # Arrange
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        tool = CourseSearchTool(mock_vector_store)

        # Act
        result = tool.execute(query="machine learning")

        # Assert
        assert len(tool.last_source_links) == 2
        # Both should call get_lesson_link for their respective lessons
        assert mock_vector_store.get_lesson_link.call_count == 2

    def test_get_tool_definition(self, mock_vector_store):
        """Test that tool definition is correctly structured"""
        # Arrange
        tool = CourseSearchTool(mock_vector_store)

        # Act
        definition = tool.get_tool_definition()

        # Assert
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert definition["input_schema"]["type"] == "object"
        assert "query" in definition["input_schema"]["properties"]
        assert "course_name" in definition["input_schema"]["properties"]
        assert "lesson_number" in definition["input_schema"]["properties"]
        assert definition["input_schema"]["required"] == ["query"]


class TestCourseOutlineTool:
    """Test suite for CourseOutlineTool"""

    def test_get_course_outline_success(self, mock_vector_store, sample_course):
        """Test successful course outline retrieval"""
        # Arrange
        tool = CourseOutlineTool(mock_vector_store)

        # Act
        result = tool.execute(course_name="Machine Learning")

        # Assert
        assert "**Course:** Introduction to Machine Learning" in result
        assert "**Instructor:** Dr. Jane Smith" in result
        assert "**Course Link:** https://example.com/course" in result
        assert "**Lessons (3 total):**" in result
        assert "1. Introduction to AI" in result
        assert "2. Machine Learning Basics" in result
        assert "3. Deep Learning" in result

    def test_get_course_outline_no_course_name(self, mock_vector_store):
        """Test course outline when course_name is missing"""
        # Arrange
        tool = CourseOutlineTool(mock_vector_store)

        # Act
        result = tool.execute()

        # Assert
        assert result == "Error: course_name parameter is required"

    def test_get_course_outline_course_not_found(self, mock_vector_store):
        """Test course outline when course is not found"""
        # Arrange
        mock_vector_store._resolve_course_name.return_value = None
        tool = CourseOutlineTool(mock_vector_store)

        # Act
        result = tool.execute(course_name="Nonexistent Course")

        # Assert
        assert result == "No course found matching 'Nonexistent Course'"

    def test_get_course_outline_metadata_not_found(self, mock_vector_store):
        """Test course outline when metadata is not found"""
        # Arrange
        mock_vector_store._resolve_course_name.return_value = "Some Course"
        mock_vector_store.get_all_courses_metadata.return_value = []
        tool = CourseOutlineTool(mock_vector_store)

        # Act
        result = tool.execute(course_name="Some Course")

        # Assert
        assert result == "Course metadata not found for 'Some Course'"

    def test_get_tool_definition(self, mock_vector_store):
        """Test that outline tool definition is correctly structured"""
        # Arrange
        tool = CourseOutlineTool(mock_vector_store)

        # Act
        definition = tool.get_tool_definition()

        # Assert
        assert definition["name"] == "get_course_outline"
        assert "description" in definition
        assert definition["input_schema"]["type"] == "object"
        assert "course_name" in definition["input_schema"]["properties"]
        assert definition["input_schema"]["required"] == ["course_name"]


class TestToolManager:
    """Test suite for ToolManager"""

    def test_register_tool(self, mock_vector_store):
        """Test tool registration"""
        # Arrange
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)

        # Act
        manager.register_tool(search_tool)

        # Assert
        assert "search_course_content" in manager.tools
        assert manager.tools["search_course_content"] == search_tool

    def test_get_tool_definitions(self, mock_vector_store):
        """Test getting all tool definitions"""
        # Arrange
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        outline_tool = CourseOutlineTool(mock_vector_store)
        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)

        # Act
        definitions = manager.get_tool_definitions()

        # Assert
        assert len(definitions) == 2
        tool_names = [defn["name"] for defn in definitions]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names

    def test_execute_tool_success(self, mock_vector_store, sample_search_results):
        """Test successful tool execution"""
        # Arrange
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)

        # Act
        result = manager.execute_tool("search_course_content", query="test query")

        # Assert
        assert result is not None
        assert "Introduction to Machine Learning" in result

    def test_execute_tool_not_found(self, mock_vector_store):
        """Test execution of non-existent tool"""
        # Arrange
        manager = ToolManager()

        # Act
        result = manager.execute_tool("nonexistent_tool", query="test")

        # Assert
        assert result == "Tool 'nonexistent_tool' not found"

    def test_get_last_sources(self, mock_vector_store, sample_search_results):
        """Test getting last sources from tools"""
        # Arrange
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)

        # Execute a search to populate sources
        manager.execute_tool("search_course_content", query="test query")

        # Act
        sources = manager.get_last_sources()

        # Assert
        assert len(sources) > 0
        assert "Introduction to Machine Learning" in sources[0]

    def test_get_last_source_links(self, mock_vector_store, sample_search_results):
        """Test getting last source links from tools"""
        # Arrange
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)

        # Execute a search to populate source links
        manager.execute_tool("search_course_content", query="test query")

        # Act
        source_links = manager.get_last_source_links()

        # Assert
        assert len(source_links) > 0

    def test_reset_sources(self, mock_vector_store, sample_search_results):
        """Test resetting sources from all tools"""
        # Arrange
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)

        # Execute a search to populate sources
        manager.execute_tool("search_course_content", query="test query")
        assert len(manager.get_last_sources()) > 0

        # Act
        manager.reset_sources()

        # Assert
        assert len(manager.get_last_sources()) == 0
        assert len(manager.get_last_source_links()) == 0