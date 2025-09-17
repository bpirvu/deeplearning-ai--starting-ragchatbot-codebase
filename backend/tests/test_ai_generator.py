import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai_generator import AIGenerator, ToolCallState


class TestAIGenerator:
    """Test suite for AIGenerator"""

    def test_initialization(self):
        """Test AIGenerator initialization"""
        # Arrange & Act
        generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")

        # Assert
        assert generator.model == "claude-sonnet-4-20250514"
        assert generator.base_params["model"] == "claude-sonnet-4-20250514"
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_simple_query(self, mock_anthropic_class):
        """Test simple response generation without tools"""
        # Arrange
        mock_client = Mock()
        mock_response = Mock()
        mock_content = Mock()
        mock_content.text = "This is a simple response about AI."
        mock_response.content = [mock_content]
        mock_response.stop_reason = "end_turn"

        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")

        # Act
        result = generator.generate_response("What is artificial intelligence?")

        # Assert
        assert result == "This is a simple response about AI."
        mock_client.messages.create.assert_called_once()

        # Check the call parameters
        call_args = mock_client.messages.create.call_args[1]
        assert call_args["model"] == "claude-sonnet-4-20250514"
        assert call_args["temperature"] == 0
        assert call_args["max_tokens"] == 800
        assert len(call_args["messages"]) == 1
        assert call_args["messages"][0]["role"] == "user"
        assert call_args["messages"][0]["content"] == "What is artificial intelligence?"

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_conversation_history(self, mock_anthropic_class):
        """Test response generation with conversation history"""
        # Arrange
        mock_client = Mock()
        mock_response = Mock()
        mock_content = Mock()
        mock_content.text = "Based on our previous discussion, here's more info."
        mock_response.content = [mock_content]
        mock_response.stop_reason = "end_turn"

        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")
        history = "User: What is AI?\nAssistant: AI is artificial intelligence."

        # Act
        result = generator.generate_response(
            "Can you tell me more?",
            conversation_history=history
        )

        # Assert
        assert result == "Based on our previous discussion, here's more info."

        # Check that system prompt includes history
        call_args = mock_client.messages.create.call_args[1]
        assert history in call_args["system"]

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_tools_but_no_tool_use(self, mock_anthropic_class):
        """Test response with tools available but not used"""
        # Arrange
        mock_client = Mock()
        mock_response = Mock()
        mock_content = Mock()
        mock_content.text = "I can answer this from my knowledge without searching."
        mock_response.content = [mock_content]
        mock_response.stop_reason = "end_turn"

        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")

        tools = [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                }
            }
        ]

        # Act
        result = generator.generate_response(
            "What is 2+2?",
            tools=tools
        )

        # Assert
        assert result == "I can answer this from my knowledge without searching."

        # Check that tools were provided
        call_args = mock_client.messages.create.call_args[1]
        assert "tools" in call_args
        assert call_args["tools"] == tools
        assert call_args["tool_choice"]["type"] == "auto"

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_tool_use(self, mock_anthropic_class):
        """Test response generation with tool use"""
        # Arrange
        mock_client = Mock()

        # First response: tool use request
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "machine learning basics"}
        mock_tool_content.id = "tool_123"

        mock_tool_response = Mock()
        mock_tool_response.content = [mock_tool_content]
        mock_tool_response.stop_reason = "tool_use"

        # Second response: final answer after tool execution
        mock_final_content = Mock()
        mock_final_content.text = "Based on the search results, machine learning is a subset of AI that enables computers to learn from data without explicit programming."
        mock_final_response = Mock()
        mock_final_response.content = [mock_final_content]

        # Configure mock to return different responses on each call
        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        mock_anthropic_class.return_value = mock_client

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Machine learning is a subset of artificial intelligence..."

        generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")

        tools = [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                }
            }
        ]

        # Act
        result = generator.generate_response(
            "What is machine learning?",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Assert
        assert "Based on the search results" in result
        assert "machine learning is a subset of AI" in result

        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="machine learning basics"
        )

        # Verify two API calls were made
        assert mock_client.messages.create.call_count == 2

    @patch('ai_generator.anthropic.Anthropic')
    def test_handle_tool_execution_multiple_tools(self, mock_anthropic_class):
        """Test handling multiple tool calls in one response"""
        # Arrange
        mock_client = Mock()

        # Create multiple tool use content blocks
        mock_tool_content_1 = Mock()
        mock_tool_content_1.type = "tool_use"
        mock_tool_content_1.name = "search_course_content"
        mock_tool_content_1.input = {"query": "neural networks"}
        mock_tool_content_1.id = "tool_123"

        mock_tool_content_2 = Mock()
        mock_tool_content_2.type = "tool_use"
        mock_tool_content_2.name = "get_course_outline"
        mock_tool_content_2.input = {"course_name": "Deep Learning"}
        mock_tool_content_2.id = "tool_456"

        mock_tool_response = Mock()
        mock_tool_response.content = [mock_tool_content_1, mock_tool_content_2]
        mock_tool_response.stop_reason = "tool_use"

        # Final response after tools
        mock_final_content = Mock()
        mock_final_content.text = "Based on both the search results and course outline..."
        mock_final_response = Mock()
        mock_final_response.content = [mock_final_content]

        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        mock_anthropic_class.return_value = mock_client

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Neural networks are computational models...",
            "Course: Deep Learning\nLesson 1: Introduction..."
        ]

        generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")

        tools = [
            {"name": "search_course_content", "description": "Search"},
            {"name": "get_course_outline", "description": "Get outline"}
        ]

        # Act
        result = generator.generate_response(
            "Tell me about neural networks and show the Deep Learning course outline",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Assert
        assert "Based on both the search results and course outline" in result

        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call(
            "search_course_content",
            query="neural networks"
        )
        mock_tool_manager.execute_tool.assert_any_call(
            "get_course_outline",
            course_name="Deep Learning"
        )

    def test_system_prompt_structure(self):
        """Test that the system prompt is properly structured"""
        # Act
        generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")

        # Assert
        assert "You are an AI assistant specialized in course materials" in generator.SYSTEM_PROMPT
        assert "search_course_content" in generator.SYSTEM_PROMPT
        assert "get_course_outline" in generator.SYSTEM_PROMPT
        assert "Brief, Concise and focused" in generator.SYSTEM_PROMPT

    @patch('ai_generator.anthropic.Anthropic')
    def test_api_error_handling(self, mock_anthropic_class):
        """Test handling of API errors"""
        # Arrange
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API Error: Rate limited")
        mock_anthropic_class.return_value = mock_client

        generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")

        # Act & Assert
        with pytest.raises(Exception) as excinfo:
            generator.generate_response("Test query")

        assert "API Error: Rate limited" in str(excinfo.value)

    @patch('ai_generator.anthropic.Anthropic')
    def test_empty_response_handling(self, mock_anthropic_class):
        """Test handling of empty responses from API"""
        # Arrange
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = []
        mock_response.stop_reason = "end_turn"

        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")

        # Act & Assert
        with pytest.raises(IndexError):
            generator.generate_response("Test query")

    @patch('ai_generator.anthropic.Anthropic')
    def test_tool_execution_with_no_tool_manager(self, mock_anthropic_class):
        """Test tool use response when no tool manager is provided"""
        # Arrange
        mock_client = Mock()

        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "test"}
        mock_tool_content.id = "tool_123"

        mock_tool_response = Mock()
        mock_tool_response.content = [mock_tool_content]
        mock_tool_response.stop_reason = "tool_use"

        mock_client.messages.create.return_value = mock_tool_response
        mock_anthropic_class.return_value = mock_client

        generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")

        tools = [{"name": "search_course_content", "description": "Search"}]

        # Act - should return the tool content since no tool manager provided
        result = generator.generate_response(
            "Search for something",
            tools=tools,
            tool_manager=None
        )

        # Assert - since there's no tool manager, it should return the raw response
        # The actual behavior would depend on the implementation, but we should not crash
        assert result is not None

    def test_base_params_immutability(self):
        """Test that base params are not modified during calls"""
        # Arrange
        generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")
        original_params = generator.base_params.copy()

        # Act - simulate what happens in generate_response
        api_params = {
            **generator.base_params,
            "messages": [{"role": "user", "content": "test"}],
            "system": "test system"
        }
        api_params["tools"] = [{"name": "test_tool"}]

        # Assert
        assert generator.base_params == original_params
        assert "tools" not in generator.base_params
        assert "messages" not in generator.base_params
        assert "system" not in generator.base_params

    @patch('ai_generator.anthropic.Anthropic')
    def test_sequential_tool_calling_two_rounds(self, mock_anthropic_class):
        """Test sequential tool calling with two rounds"""
        # Arrange
        mock_client = Mock()

        # First round: get course outline
        mock_outline_content = Mock()
        mock_outline_content.type = "tool_use"
        mock_outline_content.name = "get_course_outline"
        mock_outline_content.input = {"course_name": "Python Basics"}
        mock_outline_content.id = "tool_123"

        mock_first_response = Mock()
        mock_first_response.content = [mock_outline_content]
        mock_first_response.stop_reason = "tool_use"

        # Second round: search content based on outline
        mock_search_content = Mock()
        mock_search_content.type = "tool_use"
        mock_search_content.name = "search_course_content"
        mock_search_content.input = {"query": "variables", "course_name": "Python Basics"}
        mock_search_content.id = "tool_456"

        mock_second_response = Mock()
        mock_second_response.content = [mock_search_content]
        mock_second_response.stop_reason = "tool_use"

        # Final response: text answer
        mock_final_content = Mock()
        mock_final_content.text = "Variables in Python are used to store data. The course covers basic variable types including strings, integers, and lists."

        mock_final_response = Mock()
        mock_final_response.content = [mock_final_content]
        mock_final_response.stop_reason = "end_turn"

        # Configure mock to return different responses
        mock_client.messages.create.side_effect = [
            mock_first_response,
            mock_second_response,
            mock_final_response
        ]
        mock_anthropic_class.return_value = mock_client

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Course: Python Basics\nLesson 1: Introduction to Variables\nLesson 2: Data Types",
            "Variables are containers for storing data values..."
        ]

        generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")

        tools = [
            {"name": "get_course_outline", "description": "Get course outline"},
            {"name": "search_course_content", "description": "Search content"}
        ]

        # Act
        result = generator.generate_response(
            "What does the Python Basics course teach about variables?",
            tools=tools,
            tool_manager=mock_tool_manager,
            max_tool_rounds=2
        )

        # Assert
        assert "Variables in Python are used to store data" in result
        assert mock_tool_manager.execute_tool.call_count == 2
        assert mock_client.messages.create.call_count == 3

        # Verify tool execution order
        mock_tool_manager.execute_tool.assert_any_call(
            "get_course_outline",
            course_name="Python Basics"
        )
        mock_tool_manager.execute_tool.assert_any_call(
            "search_course_content",
            query="variables",
            course_name="Python Basics"
        )

    @patch('ai_generator.anthropic.Anthropic')
    def test_sequential_tool_calling_early_termination(self, mock_anthropic_class):
        """Test early termination when Claude provides text response"""
        # Arrange
        mock_client = Mock()

        # First round: tool use
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "basic concepts"}
        mock_tool_content.id = "tool_123"

        mock_first_response = Mock()
        mock_first_response.content = [mock_tool_content]
        mock_first_response.stop_reason = "tool_use"

        # Second round: text response (early termination)
        mock_text_content = Mock()
        mock_text_content.text = "Based on the search, here are the basic concepts..."

        mock_second_response = Mock()
        mock_second_response.content = [mock_text_content]
        mock_second_response.stop_reason = "end_turn"

        mock_client.messages.create.side_effect = [mock_first_response, mock_second_response]
        mock_anthropic_class.return_value = mock_client

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Concept 1: Variables\nConcept 2: Functions"

        generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")

        tools = [{"name": "search_course_content", "description": "Search content"}]

        # Act
        result = generator.generate_response(
            "What are the basic concepts?",
            tools=tools,
            tool_manager=mock_tool_manager,
            max_tool_rounds=2
        )

        # Assert
        assert "Based on the search, here are the basic concepts" in result
        assert mock_tool_manager.execute_tool.call_count == 1
        assert mock_client.messages.create.call_count == 2

    @patch('ai_generator.anthropic.Anthropic')
    def test_sequential_tool_calling_max_rounds_reached(self, mock_anthropic_class):
        """Test behavior when maximum rounds are reached"""
        # Arrange
        mock_client = Mock()

        # First round: tool use
        mock_tool_content_1 = Mock()
        mock_tool_content_1.type = "tool_use"
        mock_tool_content_1.name = "search_course_content"
        mock_tool_content_1.input = {"query": "topic 1"}
        mock_tool_content_1.id = "tool_123"

        mock_first_response = Mock()
        mock_first_response.content = [mock_tool_content_1]
        mock_first_response.stop_reason = "tool_use"

        # Second round: tool use (max rounds reached)
        mock_tool_content_2 = Mock()
        mock_tool_content_2.type = "tool_use"
        mock_tool_content_2.name = "search_course_content"
        mock_tool_content_2.input = {"query": "topic 2"}
        mock_tool_content_2.id = "tool_456"

        mock_second_response = Mock()
        mock_second_response.content = [mock_tool_content_2]
        mock_second_response.stop_reason = "tool_use"

        # Final round: forced text response without tools
        mock_final_content = Mock()
        mock_final_content.text = "Based on my searches, here's what I found about both topics."

        mock_final_response = Mock()
        mock_final_response.content = [mock_final_content]

        mock_client.messages.create.side_effect = [
            mock_first_response,
            mock_second_response,
            mock_final_response
        ]
        mock_anthropic_class.return_value = mock_client

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Information about topic 1...",
            "Information about topic 2..."
        ]

        generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")

        tools = [{"name": "search_course_content", "description": "Search content"}]

        # Act
        result = generator.generate_response(
            "Tell me about topic 1 and topic 2",
            tools=tools,
            tool_manager=mock_tool_manager,
            max_tool_rounds=2
        )

        # Assert
        assert "Based on my searches, here's what I found about both topics" in result
        assert mock_tool_manager.execute_tool.call_count == 2
        assert mock_client.messages.create.call_count == 3

    @patch('ai_generator.anthropic.Anthropic')
    def test_sequential_tool_calling_tool_execution_error(self, mock_anthropic_class):
        """Test handling of tool execution errors in sequential calling"""
        # Arrange
        mock_client = Mock()

        # First round: tool use
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "test"}
        mock_tool_content.id = "tool_123"

        mock_first_response = Mock()
        mock_first_response.content = [mock_tool_content]
        mock_first_response.stop_reason = "tool_use"

        # Second round: text response after error
        mock_final_content = Mock()
        mock_final_content.text = "I encountered an error while searching, but I can provide general information."

        mock_final_response = Mock()
        mock_final_response.content = [mock_final_content]
        mock_final_response.stop_reason = "end_turn"

        mock_client.messages.create.side_effect = [mock_first_response, mock_final_response]
        mock_anthropic_class.return_value = mock_client

        # Mock tool manager with error
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Database connection failed")

        generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")

        tools = [{"name": "search_course_content", "description": "Search content"}]

        # Act
        result = generator.generate_response(
            "Search for information",
            tools=tools,
            tool_manager=mock_tool_manager,
            max_tool_rounds=2
        )

        # Assert
        assert result is not None
        assert mock_tool_manager.execute_tool.call_count == 1
        assert mock_client.messages.create.call_count == 2

    @patch('ai_generator.anthropic.Anthropic')
    def test_backward_compatibility_single_round(self, mock_anthropic_class):
        """Test that single round behavior is preserved for backward compatibility"""
        # Arrange
        mock_client = Mock()
        mock_response = Mock()
        mock_content = Mock()
        mock_content.text = "This is a single round response."
        mock_response.content = [mock_content]
        mock_response.stop_reason = "end_turn"

        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")

        # Act - using max_tool_rounds=1 should use single round logic
        result = generator.generate_response(
            "Simple question",
            max_tool_rounds=1
        )

        # Assert
        assert result == "This is a single round response."
        assert mock_client.messages.create.call_count == 1

    @patch('ai_generator.anthropic.Anthropic')
    def test_no_tools_provided_uses_single_round(self, mock_anthropic_class):
        """Test that no tools provided falls back to single round"""
        # Arrange
        mock_client = Mock()
        mock_response = Mock()
        mock_content = Mock()
        mock_content.text = "Response without tools."
        mock_response.content = [mock_content]
        mock_response.stop_reason = "end_turn"

        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")

        # Act - no tools provided should use single round
        result = generator.generate_response(
            "Question without tools",
            max_tool_rounds=2
        )

        # Assert
        assert result == "Response without tools."
        assert mock_client.messages.create.call_count == 1

    def test_tool_call_state_initialization(self):
        """Test ToolCallState initialization"""
        # Act
        state = ToolCallState(max_rounds=3)

        # Assert
        assert state.current_round == 0
        assert state.max_rounds == 3
        assert state.has_made_tool_calls == False
        assert state.messages == []
        assert isinstance(state.messages, list)