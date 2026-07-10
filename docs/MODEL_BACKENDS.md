# Model Backends

The code supports local HuggingFace/Transformers models, vLLM serving, and OpenAI-compatible APIs.

## Local or vLLM Backend

Set the local model path in the YAML configuration:

```yaml
llm:
  backend: vllm
  default_model: /path/to/models/Qwen3-14B
```

## OpenAI-Compatible Backend

Set:

```yaml
runtime:
  use_vllm: false
  llm_backend: openai_compat

llm:
  backend: openai_compat
  server_base_url: https://api.openai.com/v1
  server_api_key: ${OPENAI_API_KEY}
  server_model_name: your-model-name
```

Then export your key:

```bash
export OPENAI_API_KEY=your_api_key
```
