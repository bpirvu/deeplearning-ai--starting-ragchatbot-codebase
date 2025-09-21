import anthropic
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class ToolCallState:
    """Tracks state during sequential tool calling rounds"""
    current_round: int = 0
    max_rounds: int = 2
    has_made_tool_calls: bool = False
    messages: List[Dict[str, Any]] = field(default_factory=list)


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive tools for course information.

Available Tools:
1. **search_course_content**: For questions about specific course content or detailed educational materials
2. **get_course_outline**: For questions about course structure, lessons, or course information

Tool Usage Guidelines:
- **Course outline queries**: Use get_course_outline for questions about course structure, lessons, what's covered in a course, course overview, or lesson lists
- **Course content queries**: Use search_course_content for specific material within courses
- **Sequential tool usage**: You can make multiple tool calls in sequence to better answer complex questions:
  - Example: First get course outline to understand structure, then search specific content
  - Example: Search one course for a topic, then search another course for comparison
  - Maximum 2 tool calling rounds per conversation
- Synthesize tool results into accurate, fact-based responses
- If tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course outline questions**: Use get_course_outline tool first, then provide complete course information including title, course link, and lesson details (number and title for each lesson)
- **Course content questions**: Use search_course_content tool first, then answer
- **Complex queries**: Break down into multiple tool calls if needed to provide comprehensive answers
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results" or "according to the tool"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None,
                         max_tool_rounds: int = 2) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_tool_rounds: Maximum number of sequential tool calling rounds

        Returns:
            Generated response as string
        """

        # Use sequential tool calling if tools and tool_manager provided
        if tools and tool_manager and max_tool_rounds > 1:
            return self._generate_with_sequential_tools(
                query, conversation_history, tools, tool_manager, max_tool_rounds
            )

        # Fall back to original single-round behavior for backward compatibility
        return self._generate_single_round(query, conversation_history, tools, tool_manager)
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle execution of tool calls and get follow-up response.
        
        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()
        
        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})
        
        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name, 
                    **content_block.input
                )
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result
                })
        
        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        
        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"]
        }
        
        # Get final response
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text

    def _generate_with_sequential_tools(self, query: str,
                                      conversation_history: Optional[str] = None,
                                      tools: Optional[List] = None,
                                      tool_manager=None,
                                      max_tool_rounds: int = 2) -> str:
        """
        Generate AI response with sequential tool calling support.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_tool_rounds: Maximum number of sequential tool calling rounds

        Returns:
            Generated response as string
        """

        # Initialize state
        state = ToolCallState(max_rounds=max_tool_rounds)
        state.messages = [{"role": "user", "content": query}]

        # Build system content
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )
        
        print("System content:", system_content)

        # Multi-round loop
        while state.current_round < state.max_rounds:
            response = self._execute_round(state, system_content, tools, tool_manager)
            
            print("-" * 80)
            print("Round:", state.current_round)
            print("Response type:", response.content[0].type)
            if response.content[0].type == "tool_use":
                print("Tool use detected:", response.content[0].name, response.content[0].input)
            elif response.content[0].type == "text":
                print("Text response:", response.content[0].text)
            print("-" * 80)

            # Check termination conditions
            if self._should_terminate(response, state):
                return self._extract_final_response(response)

            # Execute tools and prepare for next round
            self._process_tool_calls(response, state, tool_manager)
            state.current_round += 1

        # Final round without tools
        return self._execute_final_round(state, system_content)

    def _generate_single_round(self, query: str,
                             conversation_history: Optional[str] = None,
                             tools: Optional[List] = None,
                             tool_manager=None) -> str:
        """
        Original single-round logic for backward compatibility.
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Get response from Claude
        response = self.client.messages.create(**api_params)

        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager)

        # Return direct response
        return response.content[0].text

    def _execute_round(self, state: ToolCallState, system_content: str,
                      tools: Optional[List], tool_manager) -> Any:
        """Execute a single round of Claude API call"""

        # Prepare API parameters
        api_params = {
            **self.base_params,
            "messages": state.messages.copy(),
            "system": system_content
        }

        # Add tools if available and we haven't exceeded tool rounds
        if tools and state.current_round < state.max_rounds:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Make API call
        response = self.client.messages.create(**api_params)
        return response

    def _should_terminate(self, response, state: ToolCallState) -> bool:
        """Determine if we should stop the multi-round process"""

        # Terminate if no tool use detected
        if response.stop_reason != "tool_use":
            return True

        # Continue if we have tool calls and haven't exceeded rounds
        return False

    def _process_tool_calls(self, response, state: ToolCallState, tool_manager):
        """Process tool calls and update conversation state for next round"""

        if response.stop_reason != "tool_use":
            return

        # Add Claude's response with tool calls to conversation
        state.messages.append({"role": "assistant", "content": response.content})

        # Execute all tool calls and collect results
        tool_results = []
        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    result = tool_manager.execute_tool(
                        content_block.name,
                        **content_block.input
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": result
                    })
                    state.has_made_tool_calls = True

                except Exception as e:
                    # Handle tool execution errors gracefully
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": f"Tool execution failed: {e}"
                    })

        # Add tool results as single message
        if tool_results:
            state.messages.append({"role": "user", "content": tool_results})

    def _execute_final_round(self, state: ToolCallState, system_content: str) -> str:
        """Execute final round without tools to get text response"""

        # Final API call without tools
        final_params = {
            **self.base_params,
            "messages": state.messages,
            "system": system_content
        }

        # Get final response
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text

    def _extract_final_response(self, response) -> str:
        """Extract text response from API response"""
        return response.content[0].text