import httpx


async def list_ollama_models(base_url: str = "http://localhost:11434") -> list[str]:
    """Fetch available model names from a running Ollama instance."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{base_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = data.get("models", [])
            return [m["name"] for m in models]
            
    except Exception:
        return []


async def check_ollama_running(base_url: str = "http://localhost:11434") -> bool:
    """Return True if the Ollama server is reachable."""
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            resp = await client.get(base_url)
            return resp.status_code == 200
    except Exception:
        return False