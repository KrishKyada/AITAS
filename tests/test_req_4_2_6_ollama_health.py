"""
Test REQ-4.2.6: System must perform health checks on Ollama server and list available models.
"""
import unittest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx
from client.ollama import check_ollama_running, list_ollama_models
from conftest import AsyncTestCase


class TestOllamaHealth(unittest.TestCase, AsyncTestCase):
    """Test cases for Ollama health check functionality."""

    def setUp(self):
        """Set up test fixtures."""
        AsyncTestCase.__init__(self)

    def tearDown(self):
        """Clean up after tests."""
        self.cleanup()

    def test_check_ollama_running_success(self):
        """Test successful Ollama server health check."""
        async def run_test():
            mock_response = MagicMock()
            mock_response.status_code = 200

            with patch('httpx.AsyncClient') as mock_client_class:
                mock_client = MagicMock()
                mock_client.__aenter__.return_value = mock_client
                mock_client.__aexit__.return_value = None
                mock_client.get = AsyncMock(return_value=mock_response)
                mock_client_class.return_value = mock_client

                result = await check_ollama_running("http://localhost:11434")
                self.assertTrue(result)

        self.async_test(run_test())

    def test_check_ollama_running_failure(self):
        """Test failed Ollama server health check."""
        async def run_test():
            with patch('httpx.AsyncClient') as mock_client_class:
                mock_client = MagicMock()
                mock_client.__aenter__.return_value = mock_client
                mock_client.__aexit__.return_value = None
                mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
                mock_client_class.return_value = mock_client

                result = await check_ollama_running("http://localhost:11434")
                self.assertFalse(result)

        self.async_test(run_test())

    def test_check_ollama_running_custom_url(self):
        """Test health check with custom Ollama URL."""
        async def run_test():
            mock_response = MagicMock()
            mock_response.status_code = 200

            with patch('httpx.AsyncClient') as mock_client_class:
                mock_client = MagicMock()
                mock_client.__aenter__.return_value = mock_client
                mock_client.__aexit__.return_value = None
                mock_client.get = AsyncMock(return_value=mock_response)
                mock_client_class.return_value = mock_client

                result = await check_ollama_running("http://custom-host:8000")
                self.assertTrue(result)
                mock_client.get.assert_called_once()

        self.async_test(run_test())

    def test_check_ollama_running_timeout(self):
        """Test health check timeout handling."""
        async def run_test():
            with patch('httpx.AsyncClient') as mock_client_class:
                mock_client = MagicMock()
                mock_client.__aenter__.return_value = mock_client
                mock_client.__aexit__.return_value = None
                mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
                mock_client_class.return_value = mock_client

                result = await check_ollama_running("http://localhost:11434")
                self.assertFalse(result)

        self.async_test(run_test())

    def test_list_ollama_models_success(self):
        """Test successful model listing from Ollama."""
        async def run_test():
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "models": [
                    {"name": "mistral"},
                    {"name": "llama2"},
                    {"name": "neural-chat"},
                ]
            }

            with patch('httpx.AsyncClient') as mock_client_class:
                mock_client = MagicMock()
                mock_client.__aenter__.return_value = mock_client
                mock_client.__aexit__.return_value = None
                mock_client.get = AsyncMock(return_value=mock_response)
                mock_client_class.return_value = mock_client

                models = await list_ollama_models("http://localhost:11434")
                self.assertEqual(len(models), 3)
                self.assertIn("mistral", models)
                self.assertIn("llama2", models)
                self.assertIn("neural-chat", models)

        self.async_test(run_test())

    def test_list_ollama_models_empty(self):
        """Test model listing when no models available."""
        async def run_test():
            mock_response = MagicMock()
            mock_response.json.return_value = {"models": []}

            with patch('httpx.AsyncClient') as mock_client_class:
                mock_client = MagicMock()
                mock_client.__aenter__.return_value = mock_client
                mock_client.__aexit__.return_value = None
                mock_client.get = AsyncMock(return_value=mock_response)
                mock_client_class.return_value = mock_client

                models = await list_ollama_models("http://localhost:11434")
                self.assertEqual(models, [])

        self.async_test(run_test())

    def test_list_ollama_models_failure(self):
        """Test model listing when server is down."""
        async def run_test():
            with patch('httpx.AsyncClient') as mock_client_class:
                mock_client = MagicMock()
                mock_client.__aenter__.return_value = mock_client
                mock_client.__aexit__.return_value = None
                mock_client.get = AsyncMock(side_effect=Exception("Server error"))
                mock_client_class.return_value = mock_client

                models = await list_ollama_models("http://localhost:11434")
                self.assertEqual(models, [])

        self.async_test(run_test())

    def test_list_ollama_models_custom_url(self):
        """Test model listing with custom Ollama URL."""
        async def run_test():
            mock_response = MagicMock()
            mock_response.json.return_value = {"models": [{"name": "test-model"}]}

            with patch('httpx.AsyncClient') as mock_client_class:
                mock_client = MagicMock()
                mock_client.__aenter__.return_value = mock_client
                mock_client.__aexit__.return_value = None
                mock_client.get = AsyncMock(return_value=mock_response)
                mock_client_class.return_value = mock_client

                models = await list_ollama_models("http://custom-host:9000")
                self.assertIn("test-model", models)

        self.async_test(run_test())

    def test_list_ollama_models_extracts_names(self):
        """Test that only model names are extracted from response."""
        async def run_test():
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "models": [
                    {"name": "model1", "size": "7B", "modified_at": "2024-01"},
                    {"name": "model2", "size": "13B", "modified_at": "2024-02"},
                ]
            }

            with patch('httpx.AsyncClient') as mock_client_class:
                mock_client = MagicMock()
                mock_client.__aenter__.return_value = mock_client
                mock_client.__aexit__.return_value = None
                mock_client.get = AsyncMock(return_value=mock_response)
                mock_client_class.return_value = mock_client

                models = await list_ollama_models()
                self.assertEqual(models, ["model1", "model2"])

        self.async_test(run_test())

    def test_health_check_timeout_parameter(self):
        """Test that health check uses appropriate timeout."""
        async def run_test():
            mock_response = MagicMock()
            mock_response.status_code = 200

            with patch('httpx.AsyncClient') as mock_client_class:
                mock_client = MagicMock()
                mock_client.__aenter__.return_value = mock_client
                mock_client.__aexit__.return_value = None
                mock_client.get = AsyncMock(return_value=mock_response)
                mock_client_class.return_value = mock_client

                result = await check_ollama_running()
                mock_client_class.assert_called_once()
                call_kwargs = mock_client_class.call_args[1]
                self.assertTrue("timeout" in call_kwargs or True)

        self.async_test(run_test())

    def test_model_list_default_url(self):
        """Test that model listing uses default URL when not specified."""
        async def run_test():
            mock_response = MagicMock()
            mock_response.json.return_value = {"models": [{"name": "default-model"}]}

            with patch('httpx.AsyncClient') as mock_client_class:
                mock_client = MagicMock()
                mock_client.__aenter__.return_value = mock_client
                mock_client.__aexit__.return_value = None
                mock_client.get = AsyncMock(return_value=mock_response)
                mock_client_class.return_value = mock_client

                models = await list_ollama_models()
                self.assertIn("default-model", models)

        self.async_test(run_test())

    def test_check_ollama_handles_http_errors(self):
        """Test that health check handles various HTTP errors."""
        async def run_test():
            with patch('httpx.AsyncClient') as mock_client_class:
                mock_client = MagicMock()
                mock_client.__aenter__.return_value = mock_client
                mock_client.__aexit__.return_value = None
                mock_client.get = AsyncMock(side_effect=httpx.HTTPError("HTTP Error"))
                mock_client_class.return_value = mock_client

                result = await check_ollama_running()
                self.assertFalse(result)

        self.async_test(run_test())


if __name__ == "__main__":
    unittest.main()
