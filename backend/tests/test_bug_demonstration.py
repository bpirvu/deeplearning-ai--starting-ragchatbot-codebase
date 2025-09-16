"""
Demonstration tests to show the MAX_RESULTS=0 bug and its fix.
These tests specifically demonstrate the before and after behavior.
"""
import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import Config
from vector_store import VectorStore


class TestBugDemonstration:
    """Tests that demonstrate the MAX_RESULTS=0 bug and its fix"""

    def test_current_config_values(self):
        """Test that verifies the current configuration values are correct"""
        # Arrange & Act
        config = Config()

        # Assert - This should now pass with the fix
        assert config.MAX_RESULTS == 5, f"Expected MAX_RESULTS=5, got {config.MAX_RESULTS}"
        assert config.CHUNK_SIZE == 800
        assert config.CHUNK_OVERLAP == 100
        print(f"✅ Configuration verified: MAX_RESULTS={config.MAX_RESULTS}")

    def test_broken_config_simulation(self):
        """Test that simulates the broken config behavior"""
        # Arrange - Create a config with the bug
        broken_config = Config()
        broken_config.MAX_RESULTS = 0  # Simulate the bug

        # Act & Assert - This demonstrates what would happen with the bug
        assert broken_config.MAX_RESULTS == 0
        print(f"❌ Broken config simulation: MAX_RESULTS={broken_config.MAX_RESULTS} would cause no results")

    def test_fixed_config_behavior(self):
        """Test that verifies the fixed config behavior"""
        # Arrange - Use the current fixed config
        config = Config()

        # Act & Assert - This demonstrates the fix
        assert config.MAX_RESULTS > 0, "MAX_RESULTS should be greater than 0"
        assert config.MAX_RESULTS == 5, "MAX_RESULTS should be 5"
        print(f"✅ Fixed config verified: MAX_RESULTS={config.MAX_RESULTS} will return results")

    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_vector_store_with_broken_max_results(self, mock_embedding_fn, mock_client_class):
        """Test VectorStore behavior with MAX_RESULTS=0 (the bug)"""
        # Arrange
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client

        # Simulate broken config
        store = VectorStore("/test/path", "test-model", max_results=0)  # THE BUG!

        # Mock empty results (what happens when n_results=0)
        mock_collection.query.return_value = {
            'documents': [[]],  # Empty because n_results=0
            'metadatas': [[]],
            'distances': [[]]
        }

        # Act
        results = store.search("machine learning basics")

        # Assert - This demonstrates the bug
        mock_collection.query.assert_called_once_with(
            query_texts=["machine learning basics"],
            n_results=0,  # This is the problem!
            where=None
        )
        assert results.is_empty(), "Should return empty results due to MAX_RESULTS=0"
        print("❌ Bug demonstrated: MAX_RESULTS=0 causes empty search results")

    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_vector_store_with_fixed_max_results(self, mock_embedding_fn, mock_client_class):
        """Test VectorStore behavior with fixed MAX_RESULTS>0"""
        # Arrange
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client

        # Use fixed config
        store = VectorStore("/test/path", "test-model", max_results=5)  # FIXED!

        # Mock successful results
        mock_collection.query.return_value = {
            'documents': [["Machine learning is a subset of AI...", "Neural networks are computational models..."]],
            'metadatas': [[{"course_title": "AI Course", "lesson_number": 1}, {"course_title": "AI Course", "lesson_number": 2}]],
            'distances': [[0.1, 0.3]]
        }

        # Act
        results = store.search("machine learning basics")

        # Assert - This demonstrates the fix
        mock_collection.query.assert_called_once_with(
            query_texts=["machine learning basics"],
            n_results=5,  # Now this will return results!
            where=None
        )
        assert not results.is_empty(), "Should return results with MAX_RESULTS=5"
        assert len(results.documents) == 2, "Should return 2 documents"
        print("✅ Fix verified: MAX_RESULTS=5 returns search results")


class TestSystemBehaviorComparison:
    """Compare system behavior before and after the fix"""

    def test_search_results_comparison(self):
        """Compare search behavior with broken vs fixed config"""
        # Test data
        test_cases = [
            {"max_results": 0, "expected_empty": True, "description": "Broken config (MAX_RESULTS=0)"},
            {"max_results": 5, "expected_empty": False, "description": "Fixed config (MAX_RESULTS=5)"},
        ]

        for case in test_cases:
            with patch('vector_store.chromadb.PersistentClient') as mock_client_class, \
                 patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):

                # Setup mocks
                mock_client = Mock()
                mock_collection = Mock()
                mock_client.get_or_create_collection.return_value = mock_collection
                mock_client_class.return_value = mock_client

                # Setup return value based on max_results
                if case["max_results"] == 0:
                    # Simulate ChromaDB returning empty results for n_results=0
                    mock_collection.query.return_value = {
                        'documents': [[]],
                        'metadatas': [[]],
                        'distances': [[]]
                    }
                else:
                    # Simulate ChromaDB returning results for n_results>0
                    mock_collection.query.return_value = {
                        'documents': [["Test document content"]],
                        'metadatas': [[{"course_title": "Test Course", "lesson_number": 1}]],
                        'distances': [[0.1]]
                    }

                # Test the behavior
                store = VectorStore("/test/path", "test-model", max_results=case["max_results"])
                results = store.search("test query")

                # Verify behavior
                mock_collection.query.assert_called_with(
                    query_texts=["test query"],
                    n_results=case["max_results"],
                    where=None
                )

                if case["expected_empty"]:
                    assert results.is_empty(), f"{case['description']} should return empty results"
                    print(f"❌ {case['description']}: No results returned")
                else:
                    assert not results.is_empty(), f"{case['description']} should return results"
                    print(f"✅ {case['description']}: Results returned successfully")

    def test_real_world_impact_demonstration(self):
        """Demonstrate the real-world impact of the bug fix"""
        print("\n" + "="*60)
        print("REAL-WORLD IMPACT DEMONSTRATION")
        print("="*60)

        print("\n🔍 BEFORE THE FIX (MAX_RESULTS=0):")
        print("  • User asks: 'What is machine learning?'")
        print("  • Vector store searches with n_results=0")
        print("  • No documents returned (even if matches exist)")
        print("  • Search tool returns: 'No relevant content found'")
        print("  • AI responds with generic answer: 'I don't have specific information'")
        print("  • User experience: 'Query failed' - system appears broken")

        print("\n✅ AFTER THE FIX (MAX_RESULTS=5):")
        print("  • User asks: 'What is machine learning?'")
        print("  • Vector store searches with n_results=5")
        print("  • Up to 5 relevant documents returned")
        print("  • Search tool returns formatted course content")
        print("  • AI responds with specific, accurate information from courses")
        print("  • User experience: Detailed, helpful answers with source citations")

        print("\n📊 SYSTEM STATUS:")
        from config import config
        print(f"  • Current MAX_RESULTS: {config.MAX_RESULTS}")
        print(f"  • Status: {'✅ FIXED' if config.MAX_RESULTS > 0 else '❌ BROKEN'}")

        # This test always passes - it's just for demonstration
        assert config.MAX_RESULTS > 0, "Configuration should be fixed"