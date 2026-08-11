import threading
from core.clients.olllama_client import Qwen3Client
from core.clients.openrouter_client import OpenRouterModelClient
from core.clients.azure_client import AzureOpenAIClient
from core.config.config import MODEL_PROFILES

_thread_local = threading.local()


def get_model_profile(model_name: str) -> dict:
    return MODEL_PROFILES.get(
        model_name,
        {"backend": "ollama", "model_id": model_name},
    )


def get_client(model_name: str):
    """
    Returns a thread-local client instance for the requested model.
    Instantiates OpenRouterModelClient, AzureOpenAIClient, or Qwen3Client based on backend.
    """
    if not hasattr(_thread_local, "client"):
        profile = get_model_profile(model_name)

        if profile["backend"] == "openrouter":
            _thread_local.client = OpenRouterModelClient(
                model_id=profile["model_id"]
            )
        elif profile["backend"] == "azure":
            _thread_local.client = AzureOpenAIClient(
                model=model_name,
                resource_name=profile.get("resource_name"),
                deployment_name=profile.get("deployment_name"),
                api_key=profile.get("api_key"),
            )
        else:
            _thread_local.client = Qwen3Client(
                model=profile["model_id"]
            )

    return _thread_local.client