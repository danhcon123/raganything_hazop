"""
Ollama LLM function that works for both LightRAG and RAG-Anything.

lightrag.llm.ollama.ollama_model_complete reads the model name from
kwargs["hashing_kv"]. LightRAG always passes hashing_kv, but RAG-Anything's
modal processors (table/equation descriptions) call the LLM without it,
which fails with KeyError: 'hashing_kv' and silently falls back to the raw table.

make_ollama_llm_func() binds model, host and timeout up front instead.
"""

from lightrag.llm.ollama import _ollama_model_if_cache


def make_ollama_llm_func(model: str, host: str = "http://localhost:11434", timeout: int = 600):
    async def ollama_llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        kwargs.pop("hashing_kv", None)
        if kwargs.pop("keyword_extraction", None):
            kwargs["format"] = "json"
        kwargs.setdefault("host", host)
        kwargs.setdefault("timeout", timeout)
        return await _ollama_model_if_cache(
            model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            **kwargs,
        )

    return ollama_llm_func
