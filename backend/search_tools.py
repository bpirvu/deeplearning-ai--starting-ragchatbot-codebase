from typing import Dict, Any, Optional, Protocol
from abc import ABC, abstractmethod
from vector_store import VectorStore, SearchResults


class Tool(ABC):
    """Abstract base class for all tools"""
    
    @abstractmethod
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters"""
        pass


class CourseSearchTool(Tool):
    """Tool for searching course content with semantic course name matching"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last search
        self.last_source_links = []  # Track source links from last search
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "search_course_content",
            "description": "Search course materials with smart course name matching and lesson filtering",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string", 
                        "description": "What to search for in the course content"
                    },
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')"
                    },
                    "lesson_number": {
                        "type": "integer",
                        "description": "Specific lesson number to search within (e.g. 1, 2, 3)"
                    }
                },
                "required": ["query"]
            }
        }
    
    def execute(self, query: str, course_name: Optional[str] = None, lesson_number: Optional[int] = None) -> str:
        """
        Execute the search tool with given parameters.
        
        Args:
            query: What to search for
            course_name: Optional course filter
            lesson_number: Optional lesson filter
            
        Returns:
            Formatted search results or error message
        """
        
        # Use the vector store's unified search interface
        results = self.store.search(
            query=query,
            course_name=course_name,
            lesson_number=lesson_number
        )
        
        # Handle errors
        if results.error:
            return results.error
        
        # Handle empty results
        if results.is_empty():
            filter_info = ""
            if course_name:
                filter_info += f" in course '{course_name}'"
            if lesson_number:
                filter_info += f" in lesson {lesson_number}"
            return f"No relevant content found{filter_info}."
        
        # Format and return results
        return self._format_results(results)
    
    def _format_results(self, results: SearchResults) -> str:
        """Format search results with course and lesson context"""
        formatted = []
        sources = []  # Track sources for the UI
        source_links = []  # Track source links for the UI

        for doc, meta in zip(results.documents, results.metadata):
            course_title = meta.get('course_title', 'unknown')
            lesson_num = meta.get('lesson_number')

            # Build context header
            header = f"[{course_title}"
            if lesson_num is not None:
                header += f" - Lesson {lesson_num}"
            header += "]"

            # Track source for the UI
            source = course_title
            if lesson_num is not None:
                source += f" - Lesson {lesson_num}"
            sources.append(source)

            # Look up lesson link if lesson number exists
            lesson_link = None
            if lesson_num is not None:
                lesson_link = self.store.get_lesson_link(course_title, lesson_num)
            source_links.append(lesson_link)

            formatted.append(f"{header}\n{doc}")

        # Store sources and links for retrieval
        self.last_sources = sources
        self.last_source_links = source_links

        return "\n\n".join(formatted)


class CourseOutlineTool(Tool):
    """Tool for retrieving course outline information including lessons"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last search
        self.last_source_links = []  # Track source links from last search

    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "get_course_outline",
            "description": "Get complete course outline with title, link, and lesson list for a specific course",
            "input_schema": {
                "type": "object",
                "properties": {
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')"
                    }
                },
                "required": ["course_name"]
            }
        }

    def execute(self, **kwargs) -> str:
        """
        Execute the outline tool to get course structure.

        Args:
            **kwargs: Should contain 'course_name' - Course name to get outline for

        Returns:
            Formatted course outline or error message
        """

        course_name = kwargs.get('course_name')
        if not course_name:
            return "Error: course_name parameter is required"

        # Resolve course name using vector store's semantic matching
        resolved_course_title = self.store._resolve_course_name(course_name)

        if not resolved_course_title:
            return f"No course found matching '{course_name}'"

        # Get all courses metadata and find the matching course
        all_courses = self.store.get_all_courses_metadata()

        course_data = None
        for course in all_courses:
            if course.get('title') == resolved_course_title:
                course_data = course
                break

        if not course_data:
            return f"Course metadata not found for '{resolved_course_title}'"

        # Format the course outline
        return self._format_outline(course_data)

    def _format_outline(self, course_data: Dict[str, Any]) -> str:
        """Format course outline with title, link, and lessons"""
        formatted = []

        # Course title
        course_title = course_data.get('title', 'Unknown Course')
        formatted.append(f"**Course:** {course_title}")

        # Course instructor
        instructor = course_data.get('instructor')
        if instructor:
            formatted.append(f"**Instructor:** {instructor}")

        # Course link
        course_link = course_data.get('course_link')
        if course_link:
            formatted.append(f"**Course Link:** {course_link}")

        # Lessons
        lessons = course_data.get('lessons', [])
        if lessons:
            formatted.append(f"\n**Lessons ({len(lessons)} total):**")
            for lesson in sorted(lessons, key=lambda x: x.get('lesson_number', 0)):
                lesson_num = lesson.get('lesson_number')
                lesson_title = lesson.get('lesson_title', 'Untitled')
                lesson_link = lesson.get('lesson_link')

                lesson_info = f"{lesson_num}. {lesson_title}"
                if lesson_link:
                    lesson_info += f" - [Link]({lesson_link})"
                formatted.append(lesson_info)
        else:
            formatted.append("\n**Lessons:** No lessons found")

        # Track sources for the UI
        self.last_sources = [course_title]
        self.last_source_links = [course_link] if course_link else [None]

        return "\n".join(formatted)


class ToolManager:
    """Manages available tools for the AI"""
    
    def __init__(self):
        self.tools = {}
    
    def register_tool(self, tool: Tool):
        """Register any tool that implements the Tool interface"""
        tool_def = tool.get_tool_definition()
        tool_name = tool_def.get("name")
        if not tool_name:
            raise ValueError("Tool must have a 'name' in its definition")
        self.tools[tool_name] = tool

    
    def get_tool_definitions(self) -> list:
        """Get all tool definitions for Anthropic tool calling"""
        return [tool.get_tool_definition() for tool in self.tools.values()]
    
    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with given parameters"""
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found"
        
        return self.tools[tool_name].execute(**kwargs)
    
    def get_last_sources(self) -> list:
        """Get sources from the last search operation"""
        # Check all tools for last_sources attribute
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources') and tool.last_sources:
                return tool.last_sources
        return []

    def get_last_source_links(self) -> list:
        """Get source links from the last search operation"""
        # Check all tools for last_source_links attribute
        for tool in self.tools.values():
            if hasattr(tool, 'last_source_links') and tool.last_source_links:
                return tool.last_source_links
        return []

    def reset_sources(self):
        """Reset sources from all tools that track sources"""
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources'):
                tool.last_sources = []
            if hasattr(tool, 'last_source_links'):
                tool.last_source_links = []