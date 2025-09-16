import os
import logging
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration settings for the RAG system with validation"""
    # Anthropic API settings
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"

    # Embedding model settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Document processing settings
    CHUNK_SIZE: int = 800       # Size of text chunks for vector storage
    CHUNK_OVERLAP: int = 100     # Characters to overlap between chunks
    MAX_RESULTS: int = 5         # Maximum search results to return
    MAX_HISTORY: int = 2         # Number of conversation messages to remember

    # Database paths
    CHROMA_PATH: str = "./chroma_db"  # ChromaDB storage location

    def validate(self) -> List[str]:
        """
        Validate configuration settings and return list of issues found.

        Returns:
            List of validation error messages (empty if all valid)
        """
        issues = []

        # Critical validations
        if not self.ANTHROPIC_API_KEY:
            issues.append("ANTHROPIC_API_KEY is not set - API calls will fail")
        elif not self.ANTHROPIC_API_KEY.startswith(('sk-ant-', 'sk-')):
            issues.append("ANTHROPIC_API_KEY format appears invalid")

        # Configuration value validations
        if self.MAX_RESULTS <= 0:
            issues.append(f"MAX_RESULTS ({self.MAX_RESULTS}) must be > 0 - this will cause search failures!")

        if self.CHUNK_SIZE <= 0:
            issues.append(f"CHUNK_SIZE ({self.CHUNK_SIZE}) must be > 0")

        if self.CHUNK_OVERLAP < 0:
            issues.append(f"CHUNK_OVERLAP ({self.CHUNK_OVERLAP}) must be >= 0")

        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            issues.append(f"CHUNK_OVERLAP ({self.CHUNK_OVERLAP}) should be < CHUNK_SIZE ({self.CHUNK_SIZE})")

        if self.MAX_HISTORY < 0:
            issues.append(f"MAX_HISTORY ({self.MAX_HISTORY}) must be >= 0")

        # Warnings for suboptimal values
        if self.MAX_RESULTS > 20:
            issues.append(f"MAX_RESULTS ({self.MAX_RESULTS}) is very high - may impact performance")

        if self.CHUNK_SIZE > 2000:
            issues.append(f"CHUNK_SIZE ({self.CHUNK_SIZE}) is very large - may impact relevance")

        return issues

    def validate_and_log(self) -> bool:
        """
        Validate configuration and log issues.

        Returns:
            True if configuration is valid, False if critical issues found
        """
        issues = self.validate()

        if not issues:
            logger.info("✅ Configuration validation passed")
            return True

        critical_issues = []
        warnings = []

        for issue in issues:
            if any(word in issue.lower() for word in ['must be', 'will fail', 'cause search failures']):
                critical_issues.append(issue)
            else:
                warnings.append(issue)

        # Log warnings
        for warning in warnings:
            logger.warning(f"⚠️  Config Warning: {warning}")

        # Log critical issues
        for critical in critical_issues:
            logger.error(f"❌ Config Error: {critical}")

        if critical_issues:
            logger.error(f"❌ Configuration has {len(critical_issues)} critical issue(s) that will cause system failures!")
            return False
        else:
            logger.info(f"✅ Configuration validation passed with {len(warnings)} warning(s)")
            return True

    def get_summary(self) -> str:
        """Get a summary of current configuration for debugging"""
        return f"""
Configuration Summary:
• MAX_RESULTS: {self.MAX_RESULTS} (search result limit)
• CHUNK_SIZE: {self.CHUNK_SIZE} chars
• CHUNK_OVERLAP: {self.CHUNK_OVERLAP} chars
• MAX_HISTORY: {self.MAX_HISTORY} messages
• ANTHROPIC_MODEL: {self.ANTHROPIC_MODEL}
• EMBEDDING_MODEL: {self.EMBEDDING_MODEL}
• CHROMA_PATH: {self.CHROMA_PATH}
• API_KEY: {'✅ Set' if self.ANTHROPIC_API_KEY else '❌ Missing'}
"""

# Create and validate config on import
config = Config()

# Validate configuration on startup
if __name__ != "__main__":  # Don't validate during direct script execution
    try:
        config.validate_and_log()
    except Exception as e:
        logger.error(f"❌ Configuration validation failed with exception: {e}")
        logger.info("📋 Current configuration:" + config.get_summary())


