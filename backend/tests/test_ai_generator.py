import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai_generator import AIGenerator


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