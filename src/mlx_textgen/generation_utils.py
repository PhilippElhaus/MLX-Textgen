from typing import List, Optional, Literal, Any, Dict, Union
from dataclasses import dataclass
from pydantic import BaseModel, Field
from mlx.core import array

def string_partial_pause(text: str, stop: List[str], stop_len: List[int]) -> Optional[str]:
    """Checks if the end of a string partially matches any of the stop strings.

    This function iterates through the `stop` list and checks if the end of the
    input `text` matches the beginning of any of the stop strings up to a
    certain length specified by `stop_len`. It returns the matched portion of
    the stop string if a match is found, otherwise it returns None.

    Args:
        text (str): The input string to check.
        stop (List[str]): A list of stop strings to compare against.
        stop_len (List[int]): A list of maximum lengths to consider for each
            stop string.  Should be the same length as `stop`.
            
    Returns:
        Optional[str: The matching portion of a stop string if found,
            otherwise None.
    """
    for s, l in zip(stop, stop_len):
        clen = min(len(text), l)
        for i in range(clen):
            seg = clen - i
            if text[-seg:] == s[:seg]:
                return s[:seg]

def get_stop(text: str, stop: List[str]) -> Optional[str]:
    """Checks if any of the stop strings are present in the text.

    This function iterates through the `stop` list and checks if any of the
    stop strings are present as substrings within the input `text`.
    It returns the first stop string found in the text, or None if none are found.

    Args:
        text (str): The input string to check.
        stop (List[str]): A list of stop strings to search for.

    Returns:
        Optional[str]: The first stop string found in the text, or None.
    """
    for s in stop:
        if s in text:
            return s

@dataclass
class NewStringToken:
    new_token: str
    stop_str: Optional[str]

@dataclass
class GenerationOutput:
    index: int
    token: str
    token_id: Optional[int]
    stop_str: Optional[str]
    logprobs: Optional[Dict[str, Any]]
    input_tokens: int
    output_tokens: int
    finish_reason: Optional[Literal['stop', 'length', 'tool_calls']]

class StringStop:
    """String stop checker to prevent partial stop sequences.
    """
    def __init__(self, num_prompt: int, stop: List[str]) -> None:
        self.num_prompt = num_prompt
        self.buffer = [''] * num_prompt
        self.to_yield = [''] * num_prompt
        stop_pair = [(s, len(s)) for s in set(stop)]
        stop_pair.sort(key=lambda x: x[1], reverse=True)
        self.stop = [s[0] for s in stop_pair]
        self.stop_len = [s[1] for s in stop_pair]
        self.is_stop = [False] * num_prompt

    def get_finalised_token_strings(self, tokens: List[str]) -> List[NewStringToken]:
        """Processes a list of newly generated tokens, checking for stop sequences and preparing tokens to yield.

        This method updates the internal buffer with the new tokens, checks for complete stop sequences,
        and identifies partial stop sequences. It then determines the tokens to yield and updates the
        buffer accordingly.  The `is_stop` flag prevents further processing once a stop sequence is found
        for a specific sequence.

        Args:
            tokens (List[str]): A list of newly generated tokens, one for each prompt.  The length of this
                list must match the number of prompts specified during initialization.

        Raises:
            ValueError: If the number of provided tokens does not match the expected number of prompts.

        Returns:
            List[NewStringToken]: A list of `NewStringToken` objects, one for each prompt. Each object
                contains the token to yield and the stop string that was found (if any).
        """
        if len(tokens) != self.num_prompt:
            raise ValueError('Number of provided tokens not ')
        
        self.buffer = [o + n if not s else o for o, n, s in zip(self.buffer, tokens, self.is_stop)]
        stop = [get_stop(t, self.stop) if not si else None for t, si in zip(self.buffer, self.is_stop)]
        self.is_stop = [(si or bool(s)) for si, s in zip(self.is_stop, stop)]
        to_yield = [b.split(s)[0] if s else b for b, s in zip(self.buffer, stop)]
        temp_stop = [string_partial_pause(b, self.stop, self.stop_len) if not si else None for b, si in zip(to_yield, self.is_stop)]
        to_yield = [b.removesuffix(ts) if ts else b for b, ts in zip(to_yield, temp_stop)]
        self.buffer = [b.removeprefix(ty) if not si else '' for b, ty, si in zip(self.buffer, to_yield, self.is_stop)]
        return [NewStringToken(new_token=ty, stop_str=s) for ty, s in zip(to_yield, stop)]
    
    def get_remains(self) -> List[NewStringToken]:
        return [NewStringToken(new_token=b, stop_str=None) for b in self.buffer]
    
def to_completion_logprobs(logprobs: List[Dict[str, Any]]):
    tokens = [lp['token'] for lp in logprobs]
    token_logprobs = [lp['logprob'] for lp in logprobs]
    top_logprobs = [{l['token']: l['logprob'] for l in lp['top_logprobs']} 
                    for lp in logprobs
                    ]
    return dict(tokens=tokens, token_logprobs=token_logprobs, top_logprobs=top_logprobs)
    
class TopLogprob(BaseModel):
    token: str
    logprob: float

class Logprob(BaseModel):
    token: str
    logprob: float
    top_logprobs: List[TopLogprob]

class CompletionLogprobs(BaseModel):
    tokens: List[str]
    token_logprobs: List[float]
    top_logprobs: List[Dict[str, float]]

class LogprobsObject(BaseModel):
    content: List[Logprob]

class CompletionUsageDetails(BaseModel):
    reasoning_tokens: int

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    completion_tokens_details: Optional[CompletionUsageDetails] = None

class TextCompletionChoice(BaseModel):
    index: int
    finish_reason: Optional[Literal['stop', 'length']] = None
    text: str
    logprobs: Optional[CompletionLogprobs] = None

class TextCompletionOutput(BaseModel):
    id: str = Field(pattern='^cmpl-[a-z0-9]{32}$')
    object: Literal['text_completion'] = 'text_completion'
    created: int
    model: str
    choices: List[TextCompletionChoice]
    usage: Optional[Usage] = None

class FunctionInput(BaseModel):
    name: str
    arguments: Optional[str]

class ToolCall(BaseModel):
    index: int
    id: str = Field(pattern='^call_[a-z0-9]{8}$')
    function: FunctionInput
    type: Literal['function'] = 'function'

class ChatCompletionDelta(BaseModel):
    role: Optional[Literal['assistant']] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)

class ChatCompletionStreamChoice(BaseModel):
    index: int
    finish_reason: Optional[Literal['stop', 'length', 'tool_calls']] = None
    logprobs: Optional[LogprobsObject] = None
    delta: ChatCompletionDelta

class ChatMessage(BaseModel):
    role: Literal['assistant'] = 'assistant'
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal['stop', 'length', 'tool_calls']] = None
    logprobs: Optional[LogprobsObject] = None

class ChatCompletionOutput(BaseModel):
    id: str = Field(pattern='^chatcmpl-[a-z0-9]{32}$')
    object: Literal['chat.completion', 'chat.completion.chunk']
    created: int
    model: str
    choices: List[Union[ChatCompletionChoice, ChatCompletionStreamChoice]]
    usage: Optional[Usage] = None







