from typing import List, Any, Optional, Union, Dict, TYPE_CHECKING
if TYPE_CHECKING:
    from .engine import ModelConfig
    from logging import Logger


def serve_api(
        model_configs: List["ModelConfig"],
        logger: "Logger",
        host: str = '127.0.0.1',
        port: int = 5001,
        api_keys: Optional[Union[str, List[str]]] = None,
        min_tokens: int = 20,
        max_reprocess_tokens: int = 250,
        replace_threshold: float = 0.95,
        max_capacity: int = 50,
        use_reasoning_content: bool = False
    ):
    from .engine import InferenceEngine
    from .utils import PACKAGE_NAME
    from . import __version__
    from fastapi import FastAPI, HTTPException, status, Request
    from fastapi.encoders import jsonable_encoder
    from fastapi.responses import JSONResponse, StreamingResponse
    from copy import deepcopy
    import json
    import asyncio
    import uvicorn

    engine = InferenceEngine(
        model_configs=model_configs, 
        min_tokens=min_tokens,
        max_reprocess_tokens=max_reprocess_tokens,
        replace_threshold=replace_threshold,
        max_capacity=max_capacity,
        use_reasoning_content=use_reasoning_content,
        logger=logger
    )
    api_keys = api_keys if api_keys else []
    if isinstance(api_keys, str):
        api_keys = [api_keys]

    app = FastAPI()
    semaphore = asyncio.Semaphore(1)

    def _validate_api_key(request: Request):
        api_key = request.headers.get('authorization', 'Bearer ').removeprefix('Bearer ')
        if api_keys and (api_key not in api_keys):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key."
            )
        
    def log_request(content: Dict[str, Any]):
        alt = deepcopy(content)
        if 'prompt' in alt:
            if len(alt['prompt']) > 200:
                alt['prompt'] = alt['prompt'][:100] + '...' + alt['prompt'][-100:]
        
        if 'messages' in alt:
            for msgs in alt['messages']:
                if not isinstance(msgs, list):
                    if isinstance(msgs.get('content'), list):
                        for c in msgs['content']:
                            img = c.get('image_url')
                            if img:
                                if isinstance(img, dict):
                                    img['url'] = img['url'] if not img['url'].startswith('data:') else 'base64image_string'
                            text = c.get('text')
                            if text and (len(text) > 200):
                                c['text'] = text[:99] + '...' + text[-99:]
                    elif isinstance(msgs.get('content'), str):
                        text = msgs['content']
                        if len(text) > 200:
                            msgs['content'] = text[:99] + '...' + text[-99:]
                else:
                    for msg in msgs:
                        if isinstance(msg.get('content'), list):
                            for c in msg['content']:
                                img = c.get('image_url')
                                if img:
                                    if isinstance(img, dict):
                                        img['url'] = img['url'] if not img['url'].startswith('data:') else 'base64image_string'
                                text = c.get('text')
                                if text and (len(text) > 200):
                                    c['text'] = text[:99] + '...' + text[-99:]
                        elif isinstance(msg.get('content'), str):
                            text = msg['content']
                            if len(text) > 200:
                                msg['content'] = text[:99] + '...' + text[-99:]

        
        logger.info(json.dumps(alt, indent=2))

    @app.get('/v1/models')
    async def get_models(request: Request) -> JSONResponse:
        _validate_api_key(request)
        return JSONResponse(content=jsonable_encoder(dict(object='list', data=engine.model_info))) 

    @app.get('/v1/models/{model_id}')
    async def get_model(request: Request, model_id: str) -> JSONResponse:
        _validate_api_key(request)
        model_dict = {info['id']: info for info in engine.model_info}
        if model_id not in model_dict:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f'Model "{model_id}" does not exist.'
            )
        return JSONResponse(content=jsonable_encoder(model_dict[model_id]))
    
    @app.post('/v1/completions',  response_model=None)
    async def completions(request: Request) -> Union[StreamingResponse, JSONResponse]:
        content = await request.json()
        log_request(content)
        _validate_api_key(request)
        model = content.get('model')
        if model not in engine.model_dict.keys():
            return JSONResponse(jsonable_encoder(dict(error=f'Model "{model}" does not exist.')), status_code=404)
        
        if isinstance(content.get('stop', None), str):
            content['stop'] = [content['stop']]

        stream = content.get('stream', False)

        async with semaphore:
            if stream:
                async def gen():
                    try:
                        generator = await asyncio.to_thread(engine.generate, **content)
                        for chunk in generator:
                            yield f'data: {chunk.model_dump_json()}\n\n'
                        yield 'data: [DONE]'
                    except Exception as e:
                        logger.error(str(e)[:500])
                        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)[:500])
                return StreamingResponse(gen(), media_type="text/event-stream")
            
            else:
                try:
                    output = await asyncio.to_thread(engine.generate, **content)
                    return JSONResponse(jsonable_encoder(output.model_dump()))
                except Exception as e:
                    logger.error(str(e)[:500])
                    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)[:500])
                
    @app.post('/v1/chat/completions',  response_model=None)
    async def completions(request: Request) -> Union[StreamingResponse, JSONResponse]:
        content = await request.json()
        log_request(content)
        _validate_api_key(request)
        model = content.get('model')
        if model not in engine.model_dict.keys():
            return JSONResponse(jsonable_encoder(dict(error=f'Model "{model}" does not exist.')), status_code=404)
        
        if isinstance(content.get('stop', None), str):
            content['stop'] = [content['stop']]

        if content.get('max_tokens', None) and (not content.get('max_completion_tokens', None)):
            content['max_completion_tokens'] = content['max_tokens']

        stream = content.get('stream', False)

        async with semaphore:
            if stream:
                async def gen():
                    try:
                        generator = await asyncio.to_thread(engine.chat_generate, **content)
                        for chunk in generator:
                            yield f'data: {chunk.model_dump_json()}\n\n'
                        yield 'data: [DONE]'
                    except Exception as e:
                        logger.error(str(e)[:500])
                        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)[:500])
                return StreamingResponse(gen(), media_type="text/event-stream")
            
            else:
                try:
                    output = await asyncio.to_thread(engine.chat_generate, **content)
                    return JSONResponse(jsonable_encoder(output.model_dump()))
                except Exception as e:
                    logger.error(e)
                    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
                
    print(f'{PACKAGE_NAME} OpenAI-compatible LLM API server version: {__version__}')
                
    uvicorn.run(app, port=port, host=host)



        

    