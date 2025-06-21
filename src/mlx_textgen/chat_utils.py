from typing import Literal, List, Dict, Any, Optional, Union, Tuple, TYPE_CHECKING
from pydantic import BaseModel, model_validator
if TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer

class TextContent(BaseModel):
    type: Literal['text']
    text: str

class ImageURL(BaseModel):
    url: str

class ImageContent(BaseModel):
    type: Literal['image_url', 'input_image']
    image_url: Union[ImageURL, str]

class Parameters(BaseModel):
    type: Literal['object']
    properties: Dict[str, Any]

class FunctionSchema(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Parameters

class OpenAIToolSchema(BaseModel):
    type: Literal['function']
    function: FunctionSchema

class ToolChoiceFunction(BaseModel):
    name: str

class ToolChoiceSchema(BaseModel):
    type: Literal['function']
    function: ToolChoiceFunction

class ResponseFormat(BaseModel):
    type: Literal['json_schema']
    json_schema: Dict[str, Any]

class Function(BaseModel):
    name: str
    arguments: str

class ToolCall(BaseModel):
    id: str
    type: Literal['function']
    function: Function

class ChatMessage(BaseModel):
    role: Literal['system', 'developer', 'user', 'assistant', 'tool']
    content: Optional[Union[str, List[Union[TextContent, ImageContent]]]] = None
    tool_calls: Optional[List[ToolCall]] = None

    @property
    def tool_call_list(self):
        if self.tool_calls:
            import json
            tool_call_list = []
            for t in self.tool_calls:
                t_dict = t.model_dump()
                arg_str = t_dict['function']['arguments']
                if arg_str.strip():
                    t_dict['function']['arguments'] = json.loads(arg_str.strip())
                else:
                    t_dict['function']['arguments'] = {}
                tool_call_list.append(t_dict)
            return tool_call_list

    def model_post_init(self, __context):
        self.role = 'system' if self.role == 'developer' else self.role
        if isinstance(self.content, list):
            if all(c.type == 'text' for c in self.content):
                self.content = '\n'.join([c.text for c in self.content])

        if self.tool_calls:
            import json
            tool_call_list = []
            for t in self.tool_calls:
                t_dict = t.model_dump()
                arg_str = t_dict['function']['arguments']
                if arg_str.strip():
                    t_dict['function']['arguments'] = json.loads(arg_str.strip())
                else:
                    t_dict['function']['arguments'] = {}
                tool_call_list.append(t_dict)
            self.tool_calls = tool_call_list
        else:
            self.tool_calls = None

    @model_validator(mode='after')
    def content_validation(self):
        if self.role == 'assistant':
            if (self.content is not None) and (not isinstance(self.content, str)) and (any(x.type != 'text' for x in self.content)):
                raise ValueError('"assistant" content must be string.')
        
            elif (self.content is None) and (not self.tool_calls):
                raise ValueError('No "tool_calls" or "content" for the assistant message.')
            
            elif(self.content is not None) and (self.tool_calls is not None):
                raise ValueError('Cannot have both tool_calls and content.')
        
        elif self.role != 'user':
            if self.content is None:
                raise ValueError(f'Content cannot be None for role "{self.role}".')
            elif (isinstance(self.content, list)) and (any(c.type != 'text' for c in self.content)):
                raise ValueError(f'Content can only be string for role "{self.role}".')
                        
        return self

# Model to format mapping
model_to_format = {
    # Models using message_list_with_image format
    "idefics2": "message_list_with_image",
    "idefics3": "message_list_with_image_first",
    "aya_vision": "message_list_with_image",
    "mistral3": "message_list_with_image_first",
    "qwen2_vl": "message_list_with_image",
    "qwen2_5_vl": "message_list_with_image_first",
    "kimi_vl": "message_list_with_image",
    "llama4": "message_list_with_image",
    "smolvlm": "message_list_with_image_first",
    "llava": "message_list_with_image",
    "llava_next": "message_list_with_image",
    "mllama": "message_list_with_image",
    # Models using message_list_with_content_image format
    "internvl_chat": "message_list_with_image_type",
    "pixtral": "message_list_with_image_type",
    # Models using <start_of_image>
    "gemma3": "message_with_start_image_token",
    # Models using <image>\n
    "llava-qwen2": "message_with_image_token_new_line",
    "bunny-llama": "message_with_image_token_new_line",
    "deepseek_vl_v2": "message_with_image_token_new_line",
    # Models using message_with_image_token format
    "multi_modality": "message_with_image_token",
    # Models using <|image_i|>
    "phi3_v": "message_with_numbered_image_tokens",
    # Models using prompt_with_image_token format
    "paligemma": "prompt_with_image_token",
    # Models using prompt_only format
    "florence2": "prompt_only",
    "molmo": "prompt_only",
}

def convert_tool_to_json_schema(tool: Dict[str, Any]) -> Dict[str, Any]:
    params_json = tool.get('function', dict()).get('parameters')
    if params_json is not None:
        if 'title' not in params_json.keys():
            params_json['title'] = tool['function']['name'].title() + 'Args'
        return params_json
    else:
        return tool
    
def build_function_call_schema(tools: list[Dict[str, Any]]) -> Dict[str, Any]:
    defs = dict()
    ref_list = []
    for tool in tools:
        tool_name = tool['function']['name']
        tool_title = tool_name.title()
        defs[tool_title] = dict(
            properties=dict(
                name=dict(const=tool_name, title='Name', type='string'),
                arguments={'$ref': f'#/$defs/{tool_title}Args'}
            ),
            required=['name', 'arguments'],
            title=tool_title,
            type='object'
        )
        defs[tool_title + 'Args'] = convert_tool_to_json_schema(tool)
        ref_list.append({'$ref': f'#/$defs/{tool_title}'})

    schema = {'$defs': defs, 'anyOf': ref_list, 'title': 'FunctionCall'} if len(ref_list) > 1 else {'$defs': defs, '$ref': list(ref_list[0].values())[0], 'title': 'FunctionCall'}
    return schema

def get_message_json(model_type: str, message: ChatMessage, image_count: int = 0) -> Dict[str, Optional[Union[str, List[Dict[str, str]]]]]:
    format_type = model_to_format.get(model_type.lower())
    if not format_type:
        raise ValueError(f'Cannot find format type for model type "model_type".')
    
    if message.role != 'user':
        return message.model_dump()
    
    if format_type in ('message_list_with_image', "message_list_with_image_first"):
        if isinstance(message.content, str):
            content = [{"type": "text", "text": message.content}]
            return dict(role=message.role, content=content)
        elif isinstance(message.content, list):
            content = [{"type": "text", "text": c.text} if c.type == 'text' else {"type": "image"} for c in message.content]
            return dict(role=message.role, content=content)
        else:
            return message.model_dump()
        
    elif format_type == 'message_list_with_image_type':
        if isinstance(message.content, str):
            content = [{"type": "text", "content": message.content}]
            return dict(role=message.role, content=content)
        elif isinstance(message.content, list):
            content = [{"type": "text", "content": c.text} if c.type == 'text' else {"type": "image"} for c in message.content]
            return dict(role=message.role, content=content)
        else:
            return message.model_dump()
        
    elif format_type in ("message_with_start_image_token", "message_with_image_token_new_line", "message_with_image_token"):
        if format_type == 'message_with_start_image_token':
            image_token = '<start_of_image>'
        elif format_type == 'message_with_image_token_new_line':
            image_token = '<image>\n'
        else:
            image_token = '<image>'

        if isinstance(message.content, str):
            return dict(role=message.role, content=message.content)
        elif isinstance(message.content, list):
            content = ''
            for c in message.content:
                if c.type == 'text':
                    content += c.text
                else:
                    content += image_token
            return dict(role=message.role, content=content)
        else:
            return message.model_dump()
        
    elif format_type == 'message_with_numbered_image_tokens':
        current_image_count = image_count + 1
        if isinstance(message.content, str):
            return dict(role=message.role, content=message.content)
        elif isinstance(message.content, list):
            content = ''
            for c in message.content:
                if c.type == 'text':
                    content += c.text
                else:
                    content += f'<|image_{current_image_count}|> '
                    current_image_count += 1
            return dict(role=message.role, content=content)
        else:
            return message.model_dump()
        
    else:
        raise ValueError('Model chat template type not supported.')
    
def convert_vision_message_list(messages: List[ChatMessage], model_type: str) -> List[Dict[str, Optional[Union[str, List[Dict[str, str]]]]]]:
    image_count = 0
    img_count_list = []
    for m in messages:
        img_count_list.append(image_count)
        if isinstance(m.content, list):
            image_count += sum([1 if c.type == 'imput_image' else 0 for c in m.content])

    msg_dicts = [get_message_json(model_type, m, c) for m, c in zip(messages, img_count_list)]
    return msg_dicts


DEFAULT_TEMPLATES = Literal['chatml', 'llama3', 'gemma', 'deepseek', 'openchat', 'phi']

class ChatTemplate:

    def __init__(self, 
                 tokenizer: "PreTrainedTokenizer",
                 model_type: str,
                 default_template: Optional[DEFAULT_TEMPLATES] = None, 
                 is_vision: bool = False,
                 reasoning_parser: Optional[Literal['deepseek_r1']] = None
                 ):
        from .chat_presets import PRESETS, PRESETS_EOT
        self._tokenizer = tokenizer
        self._model_type = model_type
        self._is_vision = is_vision
        self._reasoning_parser = reasoning_parser
        self._ban_none_content = False
        if default_template:
            if default_template in PRESETS.keys():
                self.tokenizer.chat_template = PRESETS[default_template]
                self.eot = PRESETS_EOT[default_template]
            else:
                raise ValueError(f'Default chat template "{default_template}" does not exist.')
        elif self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = PRESETS['chatml']
            self.eot = PRESETS_EOT['chatml']
        else:
            self.eot = self.tokenizer.eos_token

    @property
    def tokenizer(self) -> "PreTrainedTokenizer":
        return self._tokenizer

    @property
    def is_vision(self) -> bool:
        return self._is_vision
    
    @property
    def reasoning_start(self) -> Optional[str]:
        if self._reasoning_parser == 'deepseek_r1':
            return '<think>\n'
        
    @property
    def reasoning_end(self) -> Optional[str]:
        if self._reasoning_parser == 'deepseek_r1':
            return '\n</think>\n\n'
    
    @property
    def support_system(self) -> bool:
        if not hasattr(self, '_support_system'):
            try:
                system = 'Test system message, see if exist.'
                messages = [dict(role='system', content=system), dict(role='user', content='Hi there')]
                messages = [ChatMessage.model_validate(m).model_dump() for m in messages]
                prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                self._support_system = system in prompt
            except:
                self._support_system = False
        return self._support_system
    
    @property
    def allow_multiple_assistant(self) -> bool:
        if not hasattr(self, '_allow_multiple_assistant'):
            try:
                from .chat_presets import MSG_SINGLE_ASSISTANT
                messages = MSG_SINGLE_ASSISTANT * 2
                messages = [ChatMessage.model_validate(m).model_dump() for m in messages]
                self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, continue_final_message=True)
                self._allow_multiple_assistant = True
            except:
                self._allow_multiple_assistant = False
        return self._allow_multiple_assistant
    
    @property
    def support_tool_call(self) -> bool:
        if not hasattr(self, '_support_tool_call'):
            try:
                from .chat_presets import MSG_WITH_TOOL, TOOL_LIST
                messages = MSG_WITH_TOOL[:1]
                messages = [ChatMessage.model_validate(m).model_dump() for m in messages]
                p_with_tool = self.tokenizer.apply_chat_template(messages, tools=TOOL_LIST, add_generation_prompt=True, tokenize=False)
                p_wo_tool = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                self._support_tool_call = p_with_tool != p_wo_tool
            except:
                self._support_tool_call = False
        return self._support_tool_call
    
    @property
    def tool_start(self) -> str:
        if not hasattr(self, '_tool_start'):
            if self.support_tool_call:
                from .chat_presets import MSG_WITH_TOOL, TOOL_LIST
                messages = [ChatMessage.model_validate(m).model_dump() for m in MSG_WITH_TOOL]
                try:
                    p_with_tool = self.tokenizer.apply_chat_template(messages, tools=TOOL_LIST, add_generation_prompt=True, tokenize=False)
                except:
                    messages = [self._content_to_str_with_tool_call(m) for m in messages]
                    p_with_tool = self.tokenizer.apply_chat_template(messages, tools=TOOL_LIST, add_generation_prompt=True, tokenize=False)
                    self._ban_none_content = True
                p_wo_tool = self.tokenizer.apply_chat_template(messages[:1], tools=TOOL_LIST, add_generation_prompt=True, tokenize=False)
                diff_str = p_with_tool.removeprefix(p_wo_tool)
                tool_first_index = diff_str.find('{')
                self._tool_start = diff_str[:tool_first_index]
                if self.reasoning_start:
                    rchunk = self.reasoning_start + self.reasoning_end
                    if rchunk in self._tool_start:
                        self._tool_start = self._tool_start.split(rchunk)[-1]
                    
            else:
                self._tool_start = '<tool_call>\n'
        return self._tool_start
    
    def _content_to_str_with_tool_call(self, message: Dict[str, Any]) -> Dict[str, Any]:
        if (message['role'] == 'assistant') and ('tool_calls' in message):
            if message.get('content', None) is None:
                message['content'] = ''
        return message
    
    def _validate_msg_seq(self, messages: List[Union[Dict[str, Any], ChatMessage]]) -> List[ChatMessage]:
        if len(messages) == 0:
            raise ValueError('Cannot have an empty message list.')
        msgs = [ChatMessage.model_validate(m) if not isinstance(m, ChatMessage) else m for m in messages]

        system_count = sum([1 if m.role == 'system' else 0 for m in msgs])
        if system_count > 1:
            raise ValueError('Cannot have more than one system messages.')
        elif (system_count == 1) and (msgs[0].role != 'system'):
            raise ValueError('If system messages is provided, it must be the first message.')
        elif (len(msgs) < 2 and (system_count == 1)):
            raise ValueError('Cannot have a messages list with only a system message.')
        
        if system_count and (not self.support_system):
            system = '<system>\n' + msgs[0].content.strip() + '\n</system>'
            if msgs[1].role != 'user':
                msgs = [ChatMessage(role='user', content=system)] + msgs[:1]
            elif isinstance(msgs[1].content, list):
                msgs[1].content = [TextContent(type='text', text=system + '\n\n')] + msgs[1].content
            else:
                msgs[1].content = system + '\n\n' + msgs[1].content

        msgs_wo_sys = msgs[1:] if system_count else msgs
        if (not self.allow_multiple_assistant) and (msgs_wo_sys[0].role != 'user'):
            msgs_wo_sys = [ChatMessage(role='user', content='')] + msgs_wo_sys

        last_role = None
        for i, m in enumerate(msgs_wo_sys):
            if not self.support_tool_call and m.role == 'tool':
                m.role = 'user'
                if isinstance(m.content, list):
                    m.content = [TextContent(type='text', text='<tool_response>\n')] + m.content + [TextContent(type='text', text='\n</tool_response>')]
                elif isinstance(m.content, str):
                    m.content = '<tool_response>\n' + m.content + '\n</tool_response>'

            if (not self.is_vision) and (isinstance(m.content, list)):
                raise ValueError('Text only template only support string contents.')

            if (not self.allow_multiple_assistant) and (m.role == last_role):
                raise ValueError('Current chat template only support user/assistant/user/assistant message sequences.')
            last_role = m.role

        msgs = [msgs[0]] + msgs_wo_sys if system_count else msgs_wo_sys
        return msgs
    
    def apply_chat_template(self,
            messages: List[Union[Dict[str, Any], ChatMessage]],
            tools: Optional[List[Dict[str, Any]]] = None,
            tool_choice: Union[Literal['none', 'auto', 'required'], Dict[str, Union[str, Dict[str, str]]]] = 'auto',
            reasoning: bool = False,
            add_generation_prompt: bool = True
        ) -> Tuple[str, Optional[List[str]]]:
        import json
        from .chat_presets import DEFAULT_TOOL_SYSTEM
        msgs = self._validate_msg_seq(messages)
        images = []
        for m in msgs:
            if isinstance(m.content, list):
                for c in m.content:
                    if isinstance(c, ImageContent):
                        if isinstance(c.image_url, ImageURL):
                            image_str = c.image_url.url
                        else:
                            image_str = c.image_url
                        images.append(image_str)
        images = images if images else None


        msgs = convert_vision_message_list(msgs, self._model_type) if self.is_vision else [m.model_dump() for m in msgs]
        tools = tools if tool_choice != 'none' else None
        if tools:
            [OpenAIToolSchema.model_validate(t) for t in tools]
        continue_final_message = False
        if msgs[-1]['role'] =='assistant':
            continue_final_message = True
            add_generation_prompt = False
        if tools and (not self.support_tool_call):
            tool_json = '\n'.join([json.dumps(tool) for tool in tools])
            if msgs[0]['role'] == 'system':
                content = msgs[0].get('content', '') + DEFAULT_TOOL_SYSTEM.replace('$$tool_list$$', tool_json)
                msgs[0]['content'] = content.strip()
            else:
                msgs = [dict(role='system', content=DEFAULT_TOOL_SYSTEM.replace('$$tool_list$$', tool_json).strip())] + msgs

        last_role = msgs[-1]['role']       

        if self._ban_none_content:
            msgs = [self._content_to_str_with_tool_call(m) for m in msgs] 
            
        prompt = self.tokenizer.apply_chat_template(conversation=msgs, 
            tools=tools if self.support_tool_call else None, tokenize=False, 
            add_generation_prompt=add_generation_prompt, 
            continue_final_message=continue_final_message)
        
        if last_role == 'user':
            if reasoning and self.reasoning_start:
                if not prompt.rstrip().endswith(self.reasoning_start.rstrip()):
                    prompt += self.reasoning_start
            elif self.reasoning_start:
                if not prompt.rstrip().endswith(self.reasoning_start.rstrip()):
                    prompt += self.reasoning_start
                prompt += self.reasoning_end

        if (tool_choice == 'required') or isinstance(tool_choice, dict):
            prompt += self.tool_start

        return prompt, images

        



        
    






    