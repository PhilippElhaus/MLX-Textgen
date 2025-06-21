def main():
    import argparse
    from typing import Optional, List, Union
    from .utils import PACKAGE_NAME, set_cache_dir, get_package_cache_dir
    from .server import serve_api

    parser = argparse.ArgumentParser(prog=PACKAGE_NAME, description=f'Welcome to {PACKAGE_NAME} CLI')
    subparsers = parser.add_subparsers(dest='command')

    # Subcommand for setting default config
    def set_config_cli() -> None:
        current = get_package_cache_dir()
        new = input(f'Cache directory for {PACKAGE_NAME} [{current}]: ')
        if new.strip():
            set_cache_dir(new.strip())


    parser_set_config = subparsers.add_parser('cachedir', help='Set cache default directory.')
    parser_set_config.set_defaults(func=set_config_cli)

    # Subcommand for serving OpenAI API endpoint
    def config_and_serve(
        config_file: Optional[str] = None,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        revision: Optional[str] = None,
        model_name: Optional[str] = None,
        host: str = '127.0.0.1',
        port: int = 5001,
        api_key: Optional[str] = None,
        min_tokens: int = 20,
        max_reprocess_tokens: int = 250,
        replace_threshold: float = 0.95,
        max_capacity: int = 50,
        use_reasoning_content: bool = False
        ):
        from .log_utils import get_logger
        from .engine import ModelConfig
        import yaml

        if model_path:
            mconfig = dict(
                model_id_or_path=model_path,
                tokenizer_repo_or_path=tokenizer_path,
                model_kwargs=dict(revision=revision),
                tokenizer_kwargs=dict(revision=revision),
                model_name=model_name
            )
            config = dict(
                model_configs = [mconfig],
                host=host,
                port=port,
                api_keys=[api_key] if api_key else None,
                min_tokens=min_tokens,
                max_reprocess_tokens=max_reprocess_tokens,
                replace_threshold=replace_threshold,
                max_capacity=max_capacity,
                use_reasoning_content=use_reasoning_content
            )
        elif config_file:
            with open(config_file, 'r') as f:
                config = yaml.load(f, yaml.SafeLoader)

        else:
            raise ValueError('Must provide one of "config_file" and "model_path".')
        

        config['model_configs'] = [ModelConfig.model_validate(mc) for mc in config['model_configs']]
        config['logger'] = get_logger("MLX Textgen")
        serve_api(**config)

    parser_serve = subparsers.add_parser('serve', help='Start the MLX Textgen OpenAI-cmopatible API server.')
    parser_serve.add_argument('-m', '--model-path', type=str, 
        default=None, help='Path to the model or the HuggingFace repository name if only one model should be served.')
    parser_serve.add_argument('--tokenizer-path', type=str, 
        default=None, help='Path to the tokenizer or the HuggingFace repository name if only one model should be served. If None is given, it will be the model_path. Defaults to None.')
    parser_serve.add_argument('--revision', type=str, 
        default=None, help='Revision of the repository if an HF repository is given. Defaults to None.')
    parser_serve.add_argument('--model-name', type=str,
        default=None, help='Model name appears in the API endpoint. If None is given, it will be created automatically with the model path. Defaults to None.')
    parser_serve.add_argument('-cf', '--config-file', type=str, 
        default=None, 
        help='Path of the config file that store the configs of all models wanted to be served. If this is passed, all other arguments will be ignored.')
    parser_serve.add_argument('--api-key', type=str, default=None, help='API key to access the endpoints. Defaults to None.')
    parser_serve.add_argument('-p', '--port', type=int, 
                        default=5001, help='Port to server the API endpoints.')
    parser_serve.add_argument('--host', type=str, 
                        default='127.0.0.1', help='Host to bind the server to. Defaults to "127.0.0.1".')
    parser_serve.add_argument('--min-tokens', type=int, default=20, help='Minimum number of tokens in the cache to be considered for saving.')
    parser_serve.add_argument('--max-reprocess-tokens', type=int, default=250, 
                        help='Maximum number of tokens to be dicarded if a cache is regarded as worth saving, but another similar cache exists.')
    parser_serve.add_argument('--replace-threshold', type=float, default=0.95,
                        help='Percentage threshold to consider two cache similar in terms of token prefix. Affected by "max_reprocess_tokens" for longer prompts.')
    parser_serve.add_argument('--max-capacity', type=int, default=50, help='Maximum number of cache per model to save. Older ones will be discarded.')
    parser_serve.add_argument('--use-reasoning-content', type=bool, default=False, help='Whether to put thoughts of reasoning models in reasoning_content instead of content in /v1/chat/completions endpoint.')
    parser_serve.set_defaults(func=config_and_serve)

    # Subcommand for creating config file
    def create_config(num_models: int):
        import yaml
        from .engine import ModelConfig

        mconfig = [ModelConfig(model_id_or_path=f'/path/to/model_{i}').model_dump()for i in range(num_models)]

        config = dict(
            model_configs=mconfig,
            host='127.0.0.1',
            port=5001,
            api_keys=None,
            min_tokens=20,
            max_reprocess_tokens=250,
            replace_threshold=0.95,
            max_capacity=50,
            use_reasoning_content=False
        )
        with open('model_config.yaml', 'w') as f:
            yaml.dump(config, f, sort_keys=False)

    parser_cf = subparsers.add_parser('createconfig', help='Creating config file.')
    parser_cf.add_argument('-n', '--num-models', type=int, default=1, help='Number of model examples in the config file.')
    parser_cf.set_defaults(func=create_config)

    args = parser.parse_args()
    if args.command:
        args_kwargs = vars(args)
        args.func(**{k: v for k, v in args_kwargs.items() if k not in ['command', 'func']})
    else:
        parser.print_help()

if __name__ == '__main__':
    main()