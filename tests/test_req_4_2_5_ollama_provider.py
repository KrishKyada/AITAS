"""
Test REQ-4.2.5: System shall support local models via the OLLAMA provider setting.
"""
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from config.config import Config, Provider, ModelConfig
from client.llm_client import LLMClient


class TestOllamaProvider(unittest.TestCase):
    """Test cases for Ollama provider functionality."""

    def test_config_provider_enum_has_ollama(self):
        """Test that Provider enum includes OLLAMA option."""
        self.assertTrue(hasattr(Provider, 'OLLAMA'))
        self.assertEqual(Provider.OLLAMA, "ollama")

    def test_config_defaults_to_api_provider(self):
        """Test that default provider is API."""
        config = Config()
        self.assertEqual(config.provider, Provider.API)

    def test_config_can_be_set_to_ollama(self):
        """Test that provider can be set to OLLAMA."""
        config = Config()
        config.provider = Provider.OLLAMA
        self.assertEqual(config.provider, Provider.OLLAMA)

    def test_ollama_api_key_returns_placeholder(self):
        """Test that Ollama provider returns placeholder API key."""
        config = Config(provider=Provider.OLLAMA)
        api_key = config.api_key
        self.assertIsNotNone(api_key)

    def test_ollama_base_url_configured(self):
        """Test that Ollama base URL is configurable."""
        config = Config(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11435")
        self.assertEqual(config.ollama_base_url, "http://localhost:11435")

    def test_ollama_default_base_url(self):
        """Test that Ollama has default base URL."""
        config = Config(provider=Provider.OLLAMA)
        self.assertEqual(config.ollama_base_url, "http://localhost:11434")

    def test_ollama_base_url_in_config_properties(self):
        """Test that api_key property returns correct Ollama endpoint."""
        config = Config(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
        base_url = config.base_url
        self.assertIn("localhost:11434", base_url)
        self.assertIn("/v1", base_url)

    def test_llm_client_uses_ollama_endpoint(self):
        """Test that LLMClient correctly uses Ollama endpoint."""
        config = Config(provider=Provider.OLLAMA)
        client = LLMClient(config)
        
        openai_client = client.get_client()
        self.assertIsNotNone(openai_client)
        self.assertIn("11434", config.base_url)

    def test_ollama_provider_validation(self):
        """Test that config validates Ollama provider correctly."""
        config = Config(provider=Provider.OLLAMA)
        
        errors = config.validate()
        key_errors = [e for e in errors if "API key" in e]
        self.assertEqual(len(key_errors), 0)

    def test_provider_from_cli_string(self):
        """Test that provider can be created from string."""
        provider_str = "ollama"
        provider = Provider(provider_str)
        self.assertEqual(provider, Provider.OLLAMA)

    def test_model_config_with_ollama(self):
        """Test that model config works with Ollama provider."""
        config = Config(
            provider=Provider.OLLAMA,
            model=ModelConfig(name="mistral", temperature=0.7),
        )
        self.assertEqual(config.model.name, "mistral")
        self.assertEqual(config.provider, Provider.OLLAMA)

    def test_multiple_ollama_instances(self):
        """Test creating multiple Ollama configurations."""
        config1 = Config(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
        config2 = Config(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11435")
        
        self.assertNotEqual(config1.ollama_base_url, config2.ollama_base_url)
        self.assertEqual(config1.provider, config2.provider)

    def test_ollama_and_api_providers_independent(self):
        """Test that API and Ollama providers can coexist in different configs."""
        api_config = Config(provider=Provider.API)
        ollama_config = Config(provider=Provider.OLLAMA)
        
        self.assertNotEqual(api_config.provider, ollama_config.provider)
        self.assertEqual(api_config.provider, Provider.API)
        self.assertEqual(ollama_config.provider, Provider.OLLAMA)


if __name__ == "__main__":
    unittest.main()
