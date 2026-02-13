import os
from dotenv import load_dotenv


class EmbeddingGenerator:
    """Factory class that creates embedding objects based on the provider."""

    SUPPORTED_PROVIDERS = ["ollama", "openai"]

    def __init__(self, provider, model, **kwargs):
        """
        Initialize and create the embedding object.

        Args:
            provider: The embedding provider - "ollama" or "openai"
            model: The embedding model name. Defaults vary by provider:
                   - ollama: "nomic-embed-text"
                   - openai: "text-embedding-3-small"
            **kwargs: Additional provider-specific arguments
                      - ollama: base_url (default: "http://localhost:11434")
                      - openai: api_key (reads from env if not provided)
        """
        self.provider = provider.lower()
        if self.provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported provider: '{provider}'. "
                f"Supported providers: {self.SUPPORTED_PROVIDERS}"
            )

        self.model = model
        self.kwargs = kwargs
        self.embeddings = self._create_embeddings()

    def _create_embeddings(self):
        if self.provider == "ollama":
            return self._create_ollama_embeddings()
        elif self.provider == "openai":
            return self._create_openai_embeddings()

    def _create_ollama_embeddings(self):
        from langchain_ollama import OllamaEmbeddings

        model = self.model or "nomic-embed-text"
        base_url = self.kwargs.get("base_url", "http://localhost:11434")

        return OllamaEmbeddings(model=model, base_url=base_url)

    def _create_openai_embeddings(self):
        from langchain_openai import OpenAIEmbeddings

        load_dotenv()
        api_key = self.kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY in .env or pass api_key parameter."
            )

        model = self.model or "text-embedding-3-small"

        return OpenAIEmbeddings(model=model, api_key=api_key)

    def get_embeddings(self):
        """Return the created embedding object."""
        return self.embeddings
