# Performance Orientated MLX Fork

On a Mac mini M4 (32 GB RAM), this fork has been observed to sustain around **12 tokens/second** for Mistral Small 3.2 24B-class workloads as both of these can now be enabled:

1. **Speculative Decoding** via `draft_model_id_or_path` + `num_draft_tokens`
2. **Prompt Caching** via the built-in KV cache path


## MLX-Textgen Fork Delta

This fork is intentionally minimal and tracks the upstream project:
- Upstream: https://github.com/nath1295/MLX-Textgen

## What changed in this fork

The main functional delta is support for draft-model speculative decoding via two new model config fields:

- `draft_model_id_or_path`
- `num_draft_tokens`

These fields can be set per model in `model_config.yaml`.

```yaml
model_configs:
- model_id_or_path: /path/to/main_model
  tokenizer_repo_or_path: null
  model_kwargs: null
  tokenizer_kwargs: null
  draft_model_id_or_path: /path/to/draft_model
  num_draft_tokens: 4
  model_name: null
  enable_cache: true
  preprocess_batch_size: 512
  extra_stop_words: null
  reasoning_parser: null
  default_template: null
```

If `draft_model_id_or_path` is omitted or `null`, decoding runs as before.

