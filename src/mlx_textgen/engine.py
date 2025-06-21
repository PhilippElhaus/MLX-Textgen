from typing import List, Dict, Optional, Any, Literal, Union, Iterator, TYPE_CHECKING
from .chat_utils import DEFAULT_TEMPLATES
from pydantic import BaseModel
if TYPE_CHECKING:
    from logging import Logger
    from .model_utils import LLMModel
    from .generation_utils import TextCompletionOutput, ChatCompletionOutput

def print_images_list(images: List):
    new = []
    for i in images:
        if isinstance(i, list):
            new.append(print_images_list(i))
        elif isinstance(i, str):
            new.append(i[:50])
        else:
            new.append(i)
    return new

class ModelConfig(BaseModel):
    model_id_or_path: str
    tokenizer_repo_or_path: Optional[str] = None
    model_kwargs: Optional[Dict[str, Any]] = None
    tokenizer_kwargs: Optional[Dict[str, Any]] = None
    model_name: Optional[str] = None
    enable_cache: bool = True
    preprocess_batch_size: int = 512
    extra_stop_words: Optional[Union[str, List[str]]] = None
    reasoning_parser: Optional[Literal['deepseek_r1']] = None
    default_template: Optional[DEFAULT_TEMPLATES] = None

def get_model_name(model_id_or_path: str, model_name: Optional[str] = None) -> str:
        if model_name:
             return model_name
        name = model_id_or_path.split('/')[-1].split('\\')[-1]
        name = name.lower().replace('_', '-')
        if 'bit' in name.split('-')[-1]:
            name = '-'.join(name.split('-')[:-1])
        return name

class InferenceEngine:

    def __init__(self,
            model_configs: List[ModelConfig],
            min_tokens: int = 20,
            max_reprocess_tokens: int = 250,
            replace_threshold: float = 0.95,
            max_capacity: int = 50,
            use_reasoning_content: bool = False,
            logger: Optional["Logger"] = None
        ):
        self._logger = logger
        self._use_reasoning_content = use_reasoning_content
        from .model_utils import make_model_exist
        import time
        start = time.perf_counter()
        self._model_dict = dict()
        for mc in model_configs:
            model_name = get_model_name(mc.model_id_or_path, mc.model_name)
            if model_name not in self._model_dict:
                self._model_dict[model_name] = mc
                kwargs = mc.model_kwargs if mc.model_kwargs else dict()
                make_model_exist(model_id_or_path=mc.model_id_or_path, **kwargs)
            else:
                 msg = f'More than one model is named as "{model_name}". Please set the "model_name" argument differently for these models.'
                 self.log(msg, 'error')
                 ValueError(msg)
        self._model = None
        self._cache_manage_config = dict(
             min_tokens=min_tokens,
             max_reprocess_tokens=max_reprocess_tokens,
             replace_threshold=replace_threshold,
             max_capacity=max_capacity
        )
        end = time.perf_counter()
        self.log(f'All models prepared locally. Time taken: {end - start:.3f}s.')

    def log(self, msg: str, level: Literal["error", "warning", "info", "debug"] = "info") -> None:
        """Logs a message to the logger at the specified level.

        Args:
            msg (str): The message to log.
            level (Literal["error", "warning", "info", "debug"], optional): The logging level. Defaults to "info".
        """
        levels = dict(
            error=40,
            warning=30,
            info=20,
            debug=10
        )
        if self._logger:
            self._logger.log(level=levels.get(level), msg=msg)

    @property
    def model(self) -> Optional["LLMModel"]:
        return self._model
    
    @property
    def model_dict(self) -> Dict[str, ModelConfig]:
        return self._model_dict
    
    @property
    def model_info(self) -> List[Dict[str, Optional[Union[str, int, List, Dict[str, Any]]]]]:
        if not hasattr(self, '_model_info'):
            self._model_info = [self._get_model_info(k, v) for k, v in self.model_dict.items()]
        return self._model_info
    
    def _get_model_info(self, key: str, config: ModelConfig) -> Dict[str, Optional[Union[str, int, List, Dict[str, Any]]]]:
        from datetime import datetime as dt
        config = dict(
            id=key, 
            object='model', 
            created=int(dt.now().timestamp()), 
            owned_by=None, 
            permission=[], 
            root='root',
            info=dict(
                tokenizer_id=config.tokenizer_repo_or_path if config.tokenizer_repo_or_path else config.model_id_or_path,
                tokenizer_kwargs=config.tokenizer_kwargs
            )
        )
        return config

    def load_model(self, model_name: str) -> None:
        import time

        start = time.perf_counter()
        if model_name not in self._model_dict:
            error = f'Model "{model_name}" does not exist.'
            self.log(error, 'error')
            raise ValueError(error)
        
        if self.model and (self.model.model_name == model_name):
            return
        
        elif self.model:
            self.model.unload()
            del self._model
            self._model = None

        from .model_utils import LLMModel
        mc = self.model_dict[model_name]
        self._model = LLMModel(
            model_id_or_path=mc.model_id_or_path,
            tokenizer_repo_or_path=mc.tokenizer_repo_or_path,
            model_kwargs=mc.model_kwargs,
            tokenizer_kwargs=mc.tokenizer_kwargs,
            model_name=mc.model_name,
            logger=self._logger,
            enable_cache=mc.enable_cache,
            cache_manage_config=self._cache_manage_config,
            preprocess_batch_size=mc.preprocess_batch_size,
            extra_stop_words=mc.extra_stop_words,
            reasoning_parser=mc.reasoning_parser,
            default_template=mc.default_template
        )
        end = time.perf_counter()
        self.log(f'Model "{model_name}" loaded. Time taken: {end - start:.3f}s.')

    def generate(self,
            model: str,
            prompt: Union[str, List[str]],
            images: Optional[List[Optional[List[str]]]] = None,
            logit_bias: Optional[Dict[str, int]] = None,
            logprobs: Optional[int] = None,
            stream: bool = False,
            n: int = 1,
            max_tokens: int = 4096,
            stop: Optional[List[str]] = None,
            seed: Optional[int] = None,
            presence_penalty: float = 0.0,
            frequency_penalty: float = 0.0,
            repetition_penalty: float = 1.0,
            penalty_context_size: Optional[int] = 1000,
            temperature: float = 0.0,
            top_k: Optional[int] = None,
            top_p: float = 1.0,
            min_p: float = 0.0,
            min_tokens_to_keep: int = 1,
            guided_json: Optional[Dict[str, Any]] = None,
            guided_choice: Optional[List[str]] = None,
            guided_regex: Optional[str] = None,
            guided_grammar: Optional[str] = None,
            whitespace_pattern: Optional[str] = None,
            **kwargs
        ) -> Union["TextCompletionOutput", Iterator["TextCompletionOutput"]]:
        from .sampling_utils import SamplingParams
        from datetime import datetime as dt
        import uuid
        from .generation_utils import TextCompletionOutput, to_completion_logprobs
        self.load_model(model)
        params = SamplingParams(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            stop=stop if stop else [],
            max_completion_tokens=max_tokens,
            max_reasoning_tokens=0,
            min_tokens_to_keep=min_tokens_to_keep,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            penalty_context_size=penalty_context_size,
            logit_bias={int(k): v for k, v in logit_bias} if logit_bias else None,
            seed=seed,
            guided_json=guided_json,
            guided_choice=guided_choice,
            guided_regex=guided_regex,
            guided_grammar=guided_grammar,
            whitespace_pattern=whitespace_pattern,
            logprobs=True if logprobs else False,
            top_logprobs=logprobs if logprobs else 4
        )
        cpl_id = 'cmpl-' + uuid.uuid4().hex
        created = int(dt.now().timestamp())
        if stream:
            def gen_output():
                output = self.model.stream(prompts=prompt, sampling_params=params, images=images, n=n, is_thinking=False)
                status = [None] if isinstance(prompt, str) else [None] * len(prompt)
                prompt_tokens = None
                completion_tokens = [0] if isinstance(prompt, str) else [0] * len(prompt)
                for gos in output:
                    choices = [
                        dict(
                            index=go.index,
                            text=go.token,
                            finish_reason=go.finish_reason,
                            logprobs=to_completion_logprobs([go.logprobs]) if params.logprobs and (not s) else None
                        )
                    for s, go in zip(status, gos)
                    ]
                    if prompt_tokens is None:
                        prompt_tokens = sum([go.input_tokens for go in gos])
                    completion_tokens = [go.output_tokens if not s else c for c, s, go in zip(completion_tokens, status, gos)]
                    status = [go.finish_reason if not s else s for s, go in zip(status, gos)]
                    cmpl = dict(
                        id=cpl_id,
                        created=created,
                        model=model,
                        choices=choices
                    )
                    yield TextCompletionOutput.model_validate(cmpl)
                
                # Usage information
                completion_tokens = sum(completion_tokens)
                cmpl = dict(
                    id=cpl_id,
                    created=created,
                    model=model,
                    choices=[dict(index=go.index, text='') for go in gos],
                    usage=dict(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=prompt_tokens + completion_tokens)
                )
                yield TextCompletionOutput.model_validate(cmpl)
            return gen_output()
        
        else:
            output = self.model.generate(prompts=prompt, sampling_params=params, images=images, n=n, is_thinking=False)
            prompt_tokens = sum(output['input_tokens'])
            completion_tokens = sum(output['output_tokens'])
            usage = dict(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
            logprobs = output['logprobs'] if output['logprobs'] else range(len(output['indices']))
            cmpl = dict(
                id=cpl_id,
                created=created,
                model=model,
                choices=[dict(
                    index=i,
                    text=t,
                    logprobs=to_completion_logprobs(l) if params.logprobs else None,
                    finish_reason=fr
                ) for i, t, l, fr in zip(
                    output['indices'], output['texts'], logprobs, output['finish_reasons']
                )],
                usage=usage
            )
            return TextCompletionOutput.model_validate(cmpl)

    def chat_generate(self,
            model: str,
            messages: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
            logit_bias: Optional[Dict[str, int]] = None,
            logprobs: bool = False,
            top_logprobs: int = 4,
            stream: bool = False,
            n: int = 1,
            max_completion_tokens: int = 4096,
            reasoning_effort: Optional[Literal['low', 'medium', 'high']] = None,
            max_reasoning_tokens: Optional[int] = None,
            stop: Optional[List[str]] = None,
            response_format: Optional[Dict[str, Any]] = None,
            tools: Optional[List[Dict[str, Any]]] = None,
            tool_choice: Union[Literal['auto', 'required', 'none'], Dict[str, Any]] = 'auto',
            seed: Optional[int] = None,
            presence_penalty: float = 0.0,
            frequency_penalty: float = 0.0,
            repetition_penalty: float = 1.0,
            penalty_context_size: Optional[int] = 1000,
            temperature: float = 0.0,
            top_k: Optional[int] = None,
            top_p: float = 1.0,
            min_p: float = 0.0,
            min_tokens_to_keep: int = 1,
            guided_json: Optional[Dict[str, Any]] = None,
            guided_choice: Optional[List[str]] = None,
            guided_regex: Optional[str] = None,
            guided_grammar: Optional[str] = None,
            whitespace_pattern: Optional[str] = None,
            use_reasoning_content: Optional[bool] = None,
            **kwargs
        ) -> Union["ChatCompletionOutput", Iterator["ChatCompletionOutput"]]:
        from .chat_utils import OpenAIToolSchema, build_function_call_schema, convert_tool_to_json_schema, ToolChoiceSchema, ResponseFormat
        from datetime import datetime as dt
        import uuid
        import json
        from .sampling_utils import SamplingParams
        from .generation_utils import ChatCompletionOutput
        if (not tools) and (tool_choice == 'auto'):
            _tool_choice = 'none'
            _tools = None
        else:
            _tool_choice = tool_choice
            _tools = tools if (_tool_choice in ('auto', 'required')) or (isinstance(_tool_choice, dict)) else None
            if isinstance(_tool_choice, dict):
                ToolChoiceSchema.model_validate(_tool_choice)

        if _tools:
            [OpenAIToolSchema.model_validate(t) for t in _tools]
        elif (_tool_choice == 'required') or isinstance(_tool_choice, dict):
            error = 'Required function calling, but no tools are provided.'
            self.log(error, 'error')
            raise ValueError(error)

        if guided_json and response_format:
            error = 'Either use "response_format" or "guided_json", but not both.'
            self.log(error, 'error')
            raise ValueError(error)
        
        _guided_json = ResponseFormat.model_validate(response_format).json_schema if response_format else guided_json

        if ((_tool_choice == 'required') or isinstance(_tool_choice, dict)) and _guided_json:
            error = 'Cannot use json schema alongside with tool calling.'
            self.log(error, 'error')
            raise ValueError(error)
        
        if _tool_choice == 'required':
            _guided_json = build_function_call_schema(_tools)
        elif isinstance(_tool_choice, dict):
            tool_name = _tool_choice['function']['name']
            tool_schemas = [t for t in _tools if t['function']['name'] == tool_name]
            if len(tool_schemas) == 0:
                error = f'Provided tool choice "{tool_name}" not in given list of tools.'
                self.log(error, 'error')
                raise ValueError(error)
            _guided_json = build_function_call_schema(tool_schemas)

        self.load_model(model)

        if self.model.chat_template.reasoning_start:
            rmap = dict(low=512, medium=2048, high=4096)
            if max_reasoning_tokens is not None:
                _max_reasoning_tokens = max_reasoning_tokens
            elif reasoning_effort:
                _max_reasoning_tokens = rmap(reasoning_effort)
            else:
                _max_reasoning_tokens = rmap['medium']

        elif max_reasoning_tokens or reasoning_effort:
            _max_reasoning_tokens = 0
            self.log(f'Model "{self.model.model_name}" is not configured to support reasoning. Setting max_reasoning_tokens to 0.')

        else:
            _max_reasoning_tokens = 0

        use_reasoning_content = self._use_reasoning_content if use_reasoning_content is None else use_reasoning_content

        if _max_reasoning_tokens:
            tparams = SamplingParams(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                stop=[self.model.chat_template.reasoning_end.rstrip()],
                max_completion_tokens=max_completion_tokens,
                max_reasoning_tokens=_max_reasoning_tokens,
                min_tokens_to_keep=min_tokens_to_keep,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty,
                penalty_context_size=penalty_context_size,
                logit_bias={int(k): v for k, v in logit_bias} if logit_bias else None,
                seed=seed,
                guided_json=None,
                guided_choice=None,
                guided_regex=None,
                guided_grammar=None,
                whitespace_pattern=whitespace_pattern,
                logprobs=logprobs,
                top_logprobs=top_logprobs
            )
        else:
            tparams = None

        stop_str = stop if stop else []
        if _tool_choice == 'auto' and (self.model.chat_template.tool_start not in stop_str):
            stop_str.append(self.model.chat_template.tool_start)
        params = SamplingParams(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            stop=stop_str,
            max_completion_tokens=max_completion_tokens,
            max_reasoning_tokens=_max_reasoning_tokens,
            min_tokens_to_keep=min_tokens_to_keep,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            penalty_context_size=penalty_context_size,
            logit_bias={int(k): v for k, v in logit_bias} if logit_bias else None,
            seed=seed,
            guided_json=_guided_json,
            guided_choice=guided_choice,
            guided_regex=guided_regex,
            guided_grammar=guided_grammar,
            whitespace_pattern=whitespace_pattern,
            logprobs=logprobs,
            top_logprobs=top_logprobs
        )

        cpl_id = 'chatcmpl-' + uuid.uuid4().hex
        created = int(dt.now().timestamp())

        if len(messages) == 0:
            error = 'No message sequences provided.'
            self.log(error, 'error')
            raise ValueError(error)
        
        if self.model.is_vision and _max_reasoning_tokens and (n > 1):
            error = 'Vsion model cannot have `n` > 2 when reasoning is enabled.'
            self.log(error, 'error')
            raise ValueError(error)
        
        
        msgs = [messages] if isinstance(messages[0], dict) else msgs
        for m in msgs:
            if len(m) == 0:
                error = 'Cannot have empty messages sequences.'
                self.log(error, 'error')
                raise ValueError(error)
                        
        if tparams:
            pi_pairs = [
                self.model.chat_template.apply_chat_template(m, tools=_tools, reasoning=True)
                for m in msgs
            ]
        else:
            pi_pairs = [
                self.model.chat_template.apply_chat_template(m, tools=_tools, tool_choice=_tool_choice, reasoning=False)
                for m in msgs
            ]

        prompts = [pp[0] for pp in pi_pairs]
        images = [pp[1] for pp in pi_pairs]

        if stream:
            def gen_tokens():
                nonlocal prompts, images, tparams, params, n, cpl_id, created
                object='chat.completion.chunk'
                indices = list(range(len(prompts) * n))
                prompt_tokens = None
                reasoning_tokens = [0] * len(indices)
                completion_tokens = [0] * len(indices)
                cmpls = [dict(
                    id=cpl_id,
                    object=object,
                    created=created,
                    model=model,
                    choices = [dict(index=i, delta=dict(role='assistant', content=''))]
                ) for i in indices]
                for c in cmpls:
                    yield ChatCompletionOutput.model_validate(c)
                if tparams:
                    gen_outputs = [''] * len(indices)
                    status = [None] * len(indices)
                    toutput = self.model.stream(prompts=prompts, sampling_params=tparams, images=images, n=n, is_thinking=True)
                    prefs = [self.model.chat_template.reasoning_start if not use_reasoning_content else ''] * len(indices)
                    tend = self.model.chat_template.reasoning_end if self.model.chat_template.reasoning_end else ''
                    for gos in toutput:
                        gen_outputs = [t + go.token if not s else t for s, t, go in zip(status, gen_outputs, gos)]
                        cmpls = [dict(
                            id=cpl_id,
                            object=object,
                            created=created,
                            model=model,
                            choices = [dict(
                                index=go.index, 
                                delta=dict(
                                    content='',
                                    reasoning_content=p + go.token if not s else None
                                    ) if use_reasoning_content else dict(
                                        content=(p + go.token + tend if go.finish_reason else p + go.token) if not s else None),
                                finish_reason=None,
                                logprobs=dict(content=[go.logprobs]) if tparams.logprobs and (not s) else None
                            )]
                        ) for s, go, p in zip(status, gos, prefs)]
                        prefs = ['' if go.token is not None else p for p, go in zip(prefs, gos)]
                        if prompt_tokens is None:
                            prompt_tokens = [go.input_tokens for go in gos]
                        reasoning_tokens = [go.output_tokens if s else rt for s, go, rt in zip(status, gos, reasoning_tokens)]
                        status = [go.finish_reason if not s else s for s, go in zip(status, gos)]

                        for cmpl in cmpls:
                            if cmpl['choices'][0]['delta']['reasoning_content' if use_reasoning_content else 'content'] is not None:
                                yield ChatCompletionOutput.model_validate(cmpl)

                    new_prompts = []
                    new_images = []
                    for prompt in prompts:
                        new_prompts.extend([prompt] * n)
                    for img in images:
                        new_images.extend([img] * n)
                    new_prompts = [p + gt + self.model.chat_template.reasoning_end for p, gt in zip(new_prompts, gen_outputs)]
                    prompts = new_prompts
                    images = new_images

                    if (_tool_choice == 'required') or (isinstance(_tool_choice, dict)):
                        prompts = [p + self.model.chat_template.tool_start for p in prompts]
                        is_tool_call = True
                    else:
                        is_tool_call = False

                    output = self.model.stream(prompts=prompts, sampling_params=params, images=images, n=1, is_thinking=False)

                else:
                    is_tool_call = (_tool_choice == 'required') or (isinstance(_tool_choice, dict))
                    output = self.model.stream(prompts=prompts, sampling_params=params, images=images, n=n, is_thinking=False)

                # Definite tool calls
                if is_tool_call:
                    tool_call_strs = [''] * len(indices)
                    call_ids = ['call_' + uuid.uuid4().hex[:8] for i in range(len(indices))]
                    status = [None] * len(indices)
                    logprob_list = [[]] * len(indices)
                    for gos in output:
                        tool_call_strs = [tcs + go.token for tcs, go in zip(tool_call_strs, gos)]
                        if prompt_tokens is None:
                            prompt_tokens = [go.input_tokens for go in gos]
                        completion_tokens = [go.output_tokens if not s else c for s, go, c in zip(status, gos, completion_tokens)]
                        if params.logprobs:
                            logprob_list = [lp + [go.logprobs] if not s else lp for s, go, lp in zip(status, gos, logprob_list)]
                        status = [s if s else go.finish_reason for s, go in zip(status, gos)]

                    tool_call_list = [json.loads(tcs) for tcs in tool_call_strs]
                    cmpls = [dict(
                        id=cpl_id,
                        object=object,
                        created=created,
                        model=model,
                        choices=[
                            dict(
                                index=go.index,
                                logprobs=dict(content=lp) if params.logprobs else None,
                                delta=dict(
                                    tool_calls=[dict(index=0, id=cid, function=dict(name=tc['name'], arguments=json.dumps(tc['arguments'])))]
                                )
                            )
                        ]
                    ) for go, cid, tc, lp in zip(gos, call_ids, tool_call_list, logprob_list)]
                    for cmpl in cmpls:
                        yield ChatCompletionOutput.model_validate(cmpl)

                    cmpls = [dict(
                        id=cpl_id,
                        object=object,
                        created=created,
                        model=model,
                        choices=[
                            dict(index=go.index, delta=dict(), finish_reason='tool_calls')
                        ],
                        usage=dict(prompt_tokens=pt, completion_tokens=ct + rt, total_tokens=pt + ct + rt, completion_tokens_details=dict(reasoning_tokens=rt))
                    ) for go, pt, rt, ct in zip(gos, prompt_tokens, reasoning_tokens, completion_tokens)]
                    for cmpl in cmpls:
                        yield ChatCompletionOutput.model_validate(cmpl)

                # Any other form of generation except 'auto'
                elif _tool_choice == 'none':
                    status = [None] * len(indices)
                    for gos in output:
                        cmpls = [dict(
                            id=cpl_id,
                            object=object,
                            created=created,
                            model=model,
                            choices=[
                                dict(
                                    index=go.index,
                                    logprobs=dict(content=[go.logprobs]) if params.logprobs else None,
                                    delta=dict(
                                        content=go.token if go.token and (not s) else None
                                    )
                                )
                            ]
                        ) for go, s in zip(gos, status)]
                        if prompt_tokens is None:
                            prompt_tokens = [go.input_tokens for go in gos]
                        completion_tokens = [go.output_tokens if not s else c for s, go, c in zip(status, gos, completion_tokens)]
                        status = [s if s else go.finish_reason for s, go in zip(status, gos)]
                        for cmpl in cmpls:
                            if cmpl['choices'][0]['delta']['content'] is not None:
                                yield ChatCompletionOutput.model_validate(cmpl)

                        # Yield ended sequences as well
                        cmpls = [dict(
                            id=cpl_id,
                            object=object,
                            created=created,
                            model=model,
                            choices=[
                                dict(
                                    index=go.index,
                                    delta=dict(),
                                    finish_reason=go.finish_reason
                                )
                            ],
                            usage=dict(prompt_tokens=pt, completion_tokens=ct + rt, total_tokens=pt + ct + rt, completion_tokens_details=dict(reasoning_tokens=rt))
                        ) for go, ct, rt, pt in zip(gos, completion_tokens, reasoning_tokens, prompt_tokens) if go.finish_reason]
                        for cmpl in cmpls:
                            yield ChatCompletionOutput.model_validate(cmpl)

                # Deal with tool_choice="auto" case
                else:
                    status = [None] * len(indices)
                    if len(prompts) != len(indices):
                        new_prompts = []
                        new_images = []
                        for p, img in zip(prompts, images):
                            new_prompts.extend([p] * n)
                            new_images.extend([img] * n)
                        prompts = new_prompts
                        images = new_images

                    for gos in output:
                        cmpls = [dict(
                            id=cpl_id,
                            object=object,
                            created=created,
                            model=model,
                            choices=[
                                dict(
                                    index=go.index,
                                    logprobs=dict(content=[go.logprobs]) if params.logprobs else None,
                                    delta=dict(
                                        content=go.token if go.token and (not s) else None
                                    )
                                )
                            ]
                        ) for go, s in zip(gos, status)]
                        if prompt_tokens is None:
                            prompt_tokens = [go.input_tokens for go in gos]
                        completion_tokens = [go.output_tokens if not s else c for s, go, c in zip(status, gos, completion_tokens)]
                        prompts = [p + go.token if not s else p for s, go, p in zip(status, gos, prompts)]
                        status = [s if s else (go.finish_reason if go.stop_str != self.model.chat_template.tool_start else 'tool_call_start') for s, go in zip(status, gos)]
                        for cmpl in cmpls:
                            if cmpl['choices'][0]['delta']['content'] is not None:
                                yield ChatCompletionOutput.model_validate(cmpl)

                        # Yield ended sequences as well
                        cmpls = [dict(
                            id=cpl_id,
                            object=object,
                            created=created,
                            model=model,
                            choices=[
                                dict(
                                    index=go.index,
                                    delta=dict(),
                                    finish_reason=go.finish_reason
                                )
                            ],
                            usage=dict(prompt_tokens=pt, completion_tokens=ct + rt, total_tokens=pt + ct + rt, completion_tokens_details=dict(reasoning_tokens=rt))
                        ) for go, ct, rt, pt in zip(gos, completion_tokens, reasoning_tokens, prompt_tokens) if (go.finish_reason and (go.stop_str != self.model.chat_template.tool_start))]
                        for cmpl in cmpls:
                            yield ChatCompletionOutput.model_validate(cmpl)

                        # Dealing with tool calls for those with tool_call_start
                        if len([s for s in status if status == 'tool_call_start']) > 0:
                            # Set guided decoding to tool call schema
                            params.guided_choice = None
                            params.guided_regex = None
                            params.guided_grammar = None
                            params.guided_json = build_function_call_schema(_tools)
                            
                            prompts = [p + self.model.chat_template.tool_start for p, s in zip(prompts, status) if s == 'tool_call_start']
                            images = [img for img, s in zip(images, status) if s == 'tool_call_start']
                            indices = [i for i, s in zip(indices, status) if s == 'tool_call_start']
                            prompt_tokens = [pt for pt, s in zip(prompt_tokens, status) if s == 'tool_call_start']
                            completion_tokens = [ct for ct, s in zip(completion_tokens, status) if s == 'tool_call_start']
                            reasoning_tokens = [rt for rt, s in zip(reasoning_tokens, status) if s == 'tool_call_start']
                            tool_call_strs = [''] * len(indices)
                            call_ids = ['call_' + uuid.uuid4().hex[:8] for i in range(len(indices))]
                            logprob_list = [[]] * len(indices)
                            status = [None] * len(indices)
                            output = self.model.stream(prompts, params, images, n=1, is_thinking=False)

                            for gos in output:
                                tool_call_strs = [tcs + go.token for tcs, go in zip(tool_call_strs, gos)]
                                completion_tokens = [go.output_tokens if not s else c for s, go, c in zip(status, gos, completion_tokens)]
                                if params.logprobs:
                                    logprob_list = [lp + [go.logprobs] if not s else lp for s, go, lp in zip(status, gos, logprob_list)]
                                status = [s if s else go.finish_reason for s, go in zip(status, gos)]

                            tool_call_list = [json.loads(tcs) for tcs in tool_call_strs]
                            cmpls = [dict(
                                id=cpl_id,
                                object=object,
                                created=created,
                                model=model,
                                choices=[
                                    dict(
                                        index=i,
                                        logprobs=dict(content=lp) if params.logprobs else None,
                                        delta=dict(
                                            tool_calls=[dict(index=0, id=cid, function=dict(name=tc['name'], arguments=json.dumps(tc['arguments'])))]
                                        )
                                    )
                                ]
                            ) for cid, tc, lp, i in zip(call_ids, tool_call_list, logprob_list, indices)]
                            for cmpl in cmpls:
                                yield ChatCompletionOutput.model_validate(cmpl)

                            cmpls = [dict(
                                id=cpl_id,
                                object=object,
                                created=created,
                                model=model,
                                choices=[
                                    dict(index=i, delta=dict(), finish_reason='tool_calls')
                                ],
                                usage=dict(prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct + rt)
                            ) for i, pt, rt, ct in zip(indices, prompt_tokens, reasoning_tokens, completion_tokens)]
                            for cmpl in cmpls:
                                yield ChatCompletionOutput.model_validate(cmpl)

            return gen_tokens()

        else:
            object = 'chat.completion'
            indices = list(range(len(prompts) * n))
            cmpls = dict(
                id=cpl_id,
                object=object,
                created=created,
                model=model,
                choices = [
                    dict(
                        index=i,
                        message=dict(
                            role='assistant',
                            content='',
                            reasoning_content='',
                            tool_calls=[]
                        ),
                        finish_reason=None,
                        logprobs=dict(content=[]) if params.logprobs else None
                    )
                for i in indices],
                usage = dict(
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    completion_tokens_details=dict(
                        reasoning_tokens=0
                    )
                )
            )
            if tparams:
                output = self.model.generate(prompts=prompts, images=images, sampling_params=tparams, n=n, is_thinking=True)
                texts = output['texts']
                input_tokens = output['input_tokens']
                output_tokens = output['output_tokens']
                logprobs = output['logprobs']
                for i, t in zip(indices, texts):
                    if use_reasoning_content:
                        cmpls['choices'][i]['message']['reasoning_content'] = t
                    else:
                        cmpls['choices'][i]['message']['content'] = self.model.chat_template.reasoning_start + t + self.model.chat_template.reasoning_end
                if params.logprobs:
                    for i, lp in zip(indices, logprobs):
                        cmpls['choices'][i]['logprobs']['content'].extend(lp)                   

                if cmpls['usage']['prompt_tokens'] == 0:
                    cmpls['usage']['prompt_tokens']  = sum(input_tokens)

                cmpls['usage']['completion_tokens'] += sum(output_tokens)
                cmpls['usage']['completion_tokens_details']['reasoning_tokens'] = sum(output_tokens)
                
                if n > 1:
                    new_prompts = []
                    new_images = []
                    for p, img in zip(prompts, images):
                        new_prompts.extend([p] * n)
                        new_images.extend([img] * n)
                
                    prompts = new_prompts
                    images = new_images
                
                prompts = [p + t + self.model.chat_template.reasoning_end for p, t in zip(prompts, texts)]

                if (_tool_choice == 'required') or (isinstance(_tool_choice, dict)):
                    prompts = [p + self.model.chat_template.tool_start for p in prompts]

                output = self.model.generate(prompts=prompts, images=images, sampling_params=params, n=1, is_thinking=False)

            else:
                output = self.model.generate(prompts=prompts, images=images, sampling_params=params, n=n, is_thinking=False)

            if (_tool_choice == 'required') or (isinstance(_tool_choice, dict)):
                texts = output['texts']
                input_tokens = output['input_tokens']
                output_tokens = output['output_tokens']
                logprobs = output['logprobs']

                tool_call_dicts = [json.loads(t) for t in texts]
                cids = ['call_' + uuid.uuid4().hex[:8] for i in range(len(indices))]
                tool_calls = [dict(
                    index=0,
                    id=cid,
                    type='function',
                    function=dict(name=tc['name'], arguments=json.dumps(tc['arguments']))
                ) for tc, cid in zip(tool_call_dicts, cids)]

                if cmpls['usage']['prompt_tokens'] == 0:
                    cmpls['usage']['prompt_tokens'] = sum(input_tokens)

                if params.logprobs:
                    for i, lp in zip(indices, logprobs):
                        cmpls['choices'][i]['logprobs']['content'].extend(lp)    

                cmpls['usage']['completion_tokens'] += sum(output_tokens)

                cmpls['usage']['total_tokens'] = cmpls['usage']['completion_tokens'] + cmpls['usage']['prompt_tokens']

                for i, tc in zip(indices, tool_calls):
                    cmpls['choices'][i]['message']['tool_calls'].append(tc)
                    cmpls['choices'][i]['finish_reason'] = 'tool_calls'

            else:
                texts = output['texts']
                input_tokens = output['input_tokens']
                output_tokens = output['output_tokens']
                logprobs = output['logprobs']
                finish_reasons = output['finish_reasons']
                stop_strs = output['stop_strs']
                is_auto = _tool_choice == 'auto'

                finish_reasons = ['tool_calls' if ((ss == self.model.chat_template.tool_start) and is_auto) else fr for fr, ss in zip(finish_reasons, stop_strs)]

                if cmpls['usage']['prompt_tokens'] == 0:
                    cmpls['usage']['prompt_tokens'] = sum(input_tokens)

                if params.logprobs:
                    for i, lp in zip(indices, logprobs):
                        cmpls['choices'][i]['logprobs']['content'].extend(lp)    

                cmpls['usage']['completion_tokens'] += sum(output_tokens)

                cmpls['usage']['total_tokens'] = cmpls['usage']['completion_tokens'] + cmpls['usage']['prompt_tokens']

                for i, t, fr in zip(indices, texts, finish_reasons):
                    cmpls['choices'][i]['message']['content'] += t
                    cmpls['choices'][i]['finish_reason'] = fr

                if len([fr for fr in finish_reasons if fr == 'tool_calls']) > 0:
                    if (n > 1) and (len(prompts) != len(indices)):
                        new_prompts = []
                        new_images = []
                        for p, img in zip(prompts, images):
                            new_prompts.extend([p] * n)
                            new_images.extend([img] * n)
                    
                        prompts = new_prompts
                        images = new_images

                    indices = [i for i, fr in zip(indices, finish_reasons) if fr == 'tool_calls']
                    prompts = [p + t + self.model.chat_template.tool_start  for p, fr, t in zip(prompts, finish_reasons, texts) if fr == 'tool_calls']
                    images = [img for img, fr in zip(images, finish_reasons) if fr == 'tool_calls']

                    params.guided_choice = None
                    params.guided_regex = None
                    params.guided_grammar = None
                    params.guided_json = build_function_call_schema(_tools)
                    
                    output = self.model.generate(prompts, params, images, n=1, is_thinking=False)

                    texts = output['texts']
                    input_tokens = output['input_tokens']
                    output_tokens = output['output_tokens']
                    logprobs = output['logprobs']

                    if params.logprobs:
                        for i, lp in zip(indices, logprobs):
                            cmpls['choices'][i]['logprobs']['content'].extend(lp)    

                    cmpls['usage']['completion_tokens'] += sum(output_tokens)

                    cmpls['usage']['total_tokens'] += sum(output_tokens)

                    for i, t, fr in zip(indices, texts, finish_reasons):
                        tc = json.loads(t)
                        cmpls['choices'][i]['message']['tool_calls'].append(dict(
                            index=0,
                            id='call_' + uuid.uuid4().hex[:8],
                            type='function',
                            function=dict(name=tc['name'], arguments=json.dumps(tc['arguments']))
                        ))

            indices = list(range(len(cmpls['choices'])))
            for i in indices:
                if not cmpls['choices'][i]['message']['reasoning_content']:
                    cmpls['choices'][i]['message']['reasoning_content'] = None
                if not cmpls['choices'][i]['message']['content']:
                    cmpls['choices'][i]['message']['content'] = None

            return ChatCompletionOutput.model_validate(cmpls)

        




                
                
                        



                    
                    
                    


                

                

                        



                        


        
        




        

        


        
