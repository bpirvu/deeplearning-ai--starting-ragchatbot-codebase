# Changelog
All notable changes to this project will be documented in this file.

## [Unreleased]
### Added
- GitHub Actions workflow for Claude Code integration to respond to @claude mentions in issues and PR comments
- GitHub Actions workflow for automated PR reviews with Claude Code including tracking comments
- Sequential tool calling support for AI generator allowing Claude to make up to 2 tool calls in separate API rounds
- ToolCallState class for tracking state during multi-round tool calling sessions
- Enhanced system prompt with guidance for sequential tool usage patterns
- Support for complex queries requiring multiple searches, comparisons, and multi-part questions
- Comprehensive test suite covering sequential tool calling scenarios including:
  - Two-round tool calling flows
  - Early termination when Claude provides text response
  - Maximum rounds reached handling
  - Tool execution error recovery
  - Backward compatibility verification
- Smart termination logic based on Claude's response patterns
- Graceful error handling for tool execution failures during sequential rounds
- Dark/light theme toggle with sticky bottom positioning in sidebar
- Professional light theme with WCAG-compliant color palette
- Theme persistence using localStorage with system preference detection
- Smooth CSS transitions for seamless theme switching
- Accessibility features including dynamic ARIA labels and keyboard navigation
- Icon-based theme toggle button following existing design patterns

### Changed
- Updated commit-all command to clarify using Claude Code command syntax (/log-changes)
- Refactored `generate_response` method in AIGenerator to support sequential tool calling while maintaining backward compatibility
- Updated system prompt to include examples and guidance for multi-step reasoning patterns
- Enhanced conversation context preservation between tool calling rounds
- Refactored entire test suite to use pytest-mock and SimpleNamespace for improved readability and maintainability
- Frontend sidebar layout to use flexbox for better content organization and sticky positioning
- CSS architecture to support theme switching with custom properties and smooth transitions

### Fixed
- Issue where Claude couldn't make additional tool calls after seeing results from previous tools
- Limited ability to handle complex queries requiring information from different courses/lessons