import uvicorn
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse
from .engine import ModelEngine, TextCompletionInput, ChatCompletionInput
from .model_utils import PACKAGE_NAME, ModelConfig
import asyncio
import argparse
import logging, json, warnings, yaml, time, uuid, os
from typing import Union, Dict, Any, List, Tuple, Callable, Optional, AsyncGenerator

warnings.filterwarnings('ignore')
logging.basicConfig(format='[(%(levelname)s) %(asctime)s]: %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# Add debug logger
debug_logger = logging.getLogger('openai_stream')
debug_logger.setLevel(logging.DEBUG)
debug_logger.propagate = False  # Prevent logs from going to console
handler = logging.FileHandler('openai_stream.log')
handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
debug_logger.addHandler(handler)

# Configure
def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f'{PACKAGE_NAME}.server',
        description='Run an OpenAI-compatible LLM server.',
    )
    parser.add_argument('-m', '--model-path', type=str,
        default=None, help='Path to the model or the HuggingFace repository name if only one model should be served.')
    parser.add_argument('--tokenizer-path', type=str,
        default=None, help='Path to the tokenizer or the HuggingFace repository name if only one model should be served. If None is given, it will be the model_path. Defaults to None.')
    parser.add_argument('--adapter-path', type=str,
        default=None, help='Path to the adapter for the model. Defaults to None.')
    parser.add_argument('--revision', type=str,
        default=None, help='Rivision of the repository if an HF repository is given. Defaults to None.')
    parser.add_argument('-q', '--quantize', type=str,
        default='fp16', help='Model qunatization, options are "fp16", "q8", "q4", "q2". Defaults to "fp16", meaning no quantization.')
    parser.add_argument('--model-name', type=str,
        default=None, help='Model name appears in the API endpoint. If None is given, it will be created automatically with the model path. Defaults to None.')
    parser.add_argument('-cf', '--config-file', type=str,
        default=None,
        help='Path of the config file that store the configs of all models wanted to be served. If this is passed, "model-path", "quantize", and "model-name" will be ignored.')
    parser.add_argument('--prefill-step-size', type=int,
        default=512, help='Batch size for model prompt processing. Defaults to 512.')
    parser.add_argument('-mk', '--max-keep', type=int,
        default=50, help='Maximum number of cache history for each model to keep. Defaults to 50.')
    parser.add_argument('--token-threshold', type=int,
        default=20,
        help='Minimum number of tokens in the prompt plus generated text to trigger prompt caching. Shorter prompts do not require caching to speed up generation. Defaults to 20.')
    parser.add_argument('--api-key', type=str, default=None, help='API key to access the endpoints. Defaults to None.')
    parser.add_argument('-p', '--port', type=int,
                        default=5001, help='Port to server the API endpoints.')
    parser.add_argument('--host', type=str,
                        default='127.0.0.1', help='Host to bind the server to. Defaults to "127.0.0.1".')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging to file')

    return parser

def parse_args() -> Tuple[ModelEngine, str, int, List[str]]:
    engine_args = get_arg_parser().parse_args()
    host = engine_args.host
    port = engine_args.port
    api_keys = [engine_args.api_key] if engine_args.api_key is not None else []
    if engine_args.config_file is not None:
        with open(engine_args.config_file, 'r') as f:
            model_args = yaml.safe_load(f)
        if isinstance(model_args, list):
            model_args = [ModelConfig(**args) for args in model_args]
        else:
            model_args = [ModelConfig(**model_args)]
    elif engine_args.model_path is not None:
        model_args = [
            ModelConfig(
                model_id_or_path=engine_args.model_path,
                tokenizer_id_or_path=engine_args.tokenizer_path,
                adapter_path=engine_args.adapter_path,
                quant=engine_args.quantize,
                revision=engine_args.revision,
                model_name=engine_args.model_name
            )
        ]
    else:
        raise ValueError('Either model_path or config_file has to be provide.')
    _engine = ModelEngine(models=model_args, prefill_step_size=engine_args.prefill_step_size,
        token_threshold=engine_args.token_threshold, max_keep=engine_args.max_keep, logger=logger)
    return _engine, host, port, api_keys

def convert_arguments(new: str, old: str, args: Dict[str, Any], transform_fn: Optional[Callable] = None) -> Dict[str, Any]:
    value = args.pop(new, None)
    if value is not None:
        args[old] = value if transform_fn is None else transform_fn(value)
    return args

def cleanup_logs(log_file='openai_stream.log', max_size_mb=10):
    """Cleanup log files if they get too large."""
    try:
        if os.path.exists(log_file) and os.path.getsize(log_file) > max_size_mb * 1024 * 1024:
            # Rename existing file with timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            backup = f"{log_file}.{timestamp}"
            os.rename(log_file, backup)

            # Keep max 5 backups
            backups = [f for f in os.listdir('.') if f.startswith(f"{log_file}.")]
            if len(backups) > 5:
                backups.sort()
                for old_backup in backups[:-5]:
                    os.remove(old_backup)
    except Exception as e:
        logger.error(f"Error cleaning up logs: {e}")

engine, host, port, api_keys = parse_args()

app = FastAPI()
semaphore = asyncio.Semaphore(1)

class OpenAIStreamResponse(StreamingResponse):
    """A custom streaming response that ensures OpenAI API compatibility."""

    def __init__(self, content, *args, **kwargs):
        headers = kwargs.pop("headers", {})
        headers.update({
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Important for Nginx
        })
        super().__init__(content, headers=headers, *args, **kwargs)
async def async_generate_stream(args: Dict[str, Any]):
    """Generate text completion stream with OpenAI API compatibility."""
    args['completion_type'] = 'text_completion'
    completion_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    model = args['model']

    # Start the generation in a separate thread
    generator = await asyncio.to_thread(engine.generate, **args)

    # Process the generator to ensure OpenAI API compatibility
    try:
        for tokens in generator:
            if isinstance(tokens, list):
                for token in tokens:
                    # Check if the token is valid
                    if not token or 'choices' not in token or not token['choices']:
                        continue

                    # Create a new token with valid OpenAI-API structure
                    new_token = {
                        "id": completion_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": token['choices'][0]['index'] if 'index' in token['choices'][0] else 0,
                                "text": token['choices'][0]['text'] if 'text' in token['choices'][0] else "",
                                "finish_reason": token['choices'][0].get('finish_reason', None)
                            }
                        ]
                    }

                    # Send new token to client
                    token_str = f"data: {json.dumps(new_token)}\n\n"
                    debug_logger.debug(f"Sending: {token_str.strip()}")
                    yield token_str
            else:
                if not tokens or 'choices' not in tokens or not tokens['choices']:
                    continue

                new_token = {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": tokens['choices'][0]['index'] if 'index' in tokens['choices'][0] else 0,
                            "text": tokens['choices'][0]['text'] if 'text' in tokens['choices'][0] else "",
                            "finish_reason": tokens['choices'][0].get('finish_reason', None)
                        }
                    ]
                }
                token_str = f"data: {json.dumps(new_token)}\n\n"
                debug_logger.debug(f"Sending: {token_str.strip()}")
                yield token_str

        # End the stream with [DONE]
        done_str = "data: [DONE]\n\n"
        debug_logger.debug(f"Sending DONE: {done_str.strip()}")
        yield done_str

    except Exception as e:
        logger.error(f"Stream error: {e}")
        error_str = f"data: {{\"error\": \"{str(e)}\"}}\n\n"
        yield error_str
        yield "data: [DONE]\n\n"

async def async_generate(args: Dict[str, Any]):
    """Non-streaming text completion generation."""
    args['completion_type'] = 'text_completion'

    # Use asyncio.to_thread to run engine.generate in a separate thread
    result = await asyncio.to_thread(engine.generate, **args)

    # Ensure consistent ID and created timestamp
    completion_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    # Normalize the output for consistency
    if 'id' in result:
        result['id'] = completion_id
    if 'created' in result:
        result['created'] = created

    return result

async def async_chat_generate_stream(args: Dict[str, Any]):
    """Generate chat completion stream with OpenAI API compatibility."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    model = args['model']

    # 1. First, send the role message - ALLEEN role in delta!
    first_message = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "system_fingerprint": None,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant"},  # Alleen role, geen content of tool_calls
                "finish_reason": None,
                "logprobs": None
            }
        ]
    }
    debug_logger.debug(f"Sending role message: {json.dumps(first_message)}")
    yield f"data: {json.dumps(first_message)}\n\n"

    # 2. Start the generation in a separate thread
    generator = await asyncio.to_thread(engine.chat_generate, **args)

    # 3. Process the generator to ensure OpenAI API compatibility
    try:
        for tokens in generator:
            if isinstance(tokens, list):
                for token in tokens:
                    if not token or 'choices' not in token or not token['choices']:
                        continue

                    new_token = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "system_fingerprint": None,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {}, # Vullen we hieronder in
                                "finish_reason": None,
                                "logprobs": None
                            }
                        ]
                    }

                    # Retrieve delta content from original token
                    if ('delta' in token['choices'][0]):
                        # IMPORTA: Only content in delta, no role or empty tool_calls!
                        if 'content' in token['choices'][0]['delta'] and token['choices'][0]['delta']['content']:
                            new_token['choices'][0]['delta'] = {"content": token['choices'][0]['delta']['content']}
                        # When tool calls is present, only tool calls in delta
                        elif 'tool_calls' in token['choices'][0]['delta'] and token['choices'][0]['delta']['tool_calls']:
                            new_token['choices'][0]['delta'] = {"tool_calls": token['choices'][0]['delta']['tool_calls']}
                        else:
                            # Skip empty deltas
                            continue

                    # Finish reason copy when present
                    if 'finish_reason' in token['choices'][0] and token['choices'][0]['finish_reason']:
                        new_token['choices'][0]['finish_reason'] = token['choices'][0]['finish_reason']

                    # Stuur nieuwe token
                    token_str = f"data: {json.dumps(new_token)}\n\n"
                    debug_logger.debug(f"Sending: {token_str.strip()}")
                    yield token_str
            else:
                if not tokens or 'choices' not in tokens or not tokens['choices']:
                    continue

                new_token = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "system_fingerprint": None,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {}, # Vullen we hieronder in
                            "finish_reason": None,
                            "logprobs": None
                        }
                    ]
                }

                if ('delta' in tokens['choices'][0]):
                    if 'content' in tokens['choices'][0]['delta'] and tokens['choices'][0]['delta']['content']:
                        new_token['choices'][0]['delta'] = {"content": tokens['choices'][0]['delta']['content']}
                    elif 'tool_calls' in tokens['choices'][0]['delta'] and tokens['choices'][0]['delta']['tool_calls']:
                        new_token['choices'][0]['delta'] = {"tool_calls": tokens['choices'][0]['delta']['tool_calls']}
                    else:
                        continue

                if 'finish_reason' in tokens['choices'][0] and tokens['choices'][0]['finish_reason']:
                    new_token['choices'][0]['finish_reason'] = tokens['choices'][0]['finish_reason']

                token_str = f"data: {json.dumps(new_token)}\n\n"
                debug_logger.debug(f"Sending: {token_str.strip()}")
                yield token_str

        # 4. Send a final message with empty delta and finish_reason=stop
        final_message = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "system_fingerprint": None,
            "choices": [
                {
                    "index": 0,
                    "delta": {},  # Empty delta object to indicate the end of the stream
                    "finish_reason": "stop",
                    "logprobs": None
                }
            ]
        }
        final_str = f"data: {json.dumps(final_message)}\n\n"
        debug_logger.debug(f"Sending final message: {final_str.strip()}")
        yield final_str

        # 5. End the stream with [DONE]
        done_str = "data: [DONE]\n\n"
        debug_logger.debug(f"Sending DONE: {done_str.strip()}")
        yield done_str

    except Exception as e:
        logger.error(f"Stream error: {e}")
        error_str = f"data: {{\"error\": \"{str(e)}\"}}\n\n"
        yield error_str
        yield "data: [DONE]\n\n"

async def async_chat_generate(args: Dict[str, Any]):
    """Non-streaming chat completion generation."""
    result = await asyncio.to_thread(engine.chat_generate, **args)

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if 'id' in result:
        result['id'] = completion_id
    if 'created' in result:
        result['created'] = created
    if 'system_fingerprint' not in result:
        result['system_fingerprint'] = None

    if 'choices' in result:
        for choice in result['choices']:
            if 'logprobs' not in choice:
                choice['logprobs'] = None

    return result

@app.post('/v1/completions',  response_model=None)
async def completions(request: Request) -> Union[OpenAIStreamResponse, JSONResponse]:
    content = await request.json()
    api_key = request.headers.get('authorization', 'Bearer ').removeprefix('Bearer ')
    if ((api_key not in api_keys) and (len(api_keys) != 0)):
        return JSONResponse(jsonable_encoder(dict(error='Invalid API key.')), status_code=404)
    model = content.get('model')
    if model not in engine.models.keys():
        return JSONResponse(jsonable_encoder(dict(error=f'Model "{model}" does not exist.')), status_code=404)
    stream = content.get('stream', False)
    content = convert_arguments('max_completion_tokens', 'max_tokens', content)
    content = convert_arguments('frequency_penalty', 'repetition_penalty', content)
    content = convert_arguments('response_format', 'guided_json', content, transform_fn=lambda x: x.get('json_schema', dict()).get('schema'))
    if isinstance(content.get('stop', None), str):
        content['stop'] = [content['stop']]
    args_model = ChatCompletionInput(**content)
    logger.info(args_model)
    args = args_model.model_dump()
    async with semaphore:
        if stream:
            return OpenAIStreamResponse(async_chat_generate_stream(args))
        else:
            result = await async_chat_generate(args)
            return JSONResponse(jsonable_encoder(result))

@app.get('/v1/models')
async def get_models(request: Request) -> JSONResponse:
    api_key = request.headers.get('authorization', 'Bearer ').removeprefix('Bearer ')
    if ((api_key not in api_keys) and (len(api_keys) != 0)):
        return JSONResponse(jsonable_encoder(dict(error='Invalid API key.')), status_code=404)
    return JSONResponse(content=jsonable_encoder(dict(object='list', data=engine.model_info)))

@app.get('/v1/models/{model_id}')
async def get_model(request: Request, model_id: str) -> JSONResponse:
    api_key = request.headers.get('authorization', 'Bearer ').removeprefix('Bearer ')
    if ((api_key not in api_keys) and (len(api_keys) != 0)):
        return JSONResponse(jsonable_encoder(dict(error='Invalid API key.')), status_code=404)
    model_dict = {info['id']: info for info in engine.model_info}
    if model_id not in model_dict.keys():
        return JSONResponse(jsonable_encoder(dict(error='Invalid model ID.')), status_code=404)
    return JSONResponse(content=jsonable_encoder(model_dict[model_id]))

def main():
    """Main entry point for the server."""
    # Cleanup logs if needed
    cleanup_logs()
    uvicorn.run(app, port=port, host=host)

if __name__ == '__main__':
    main()