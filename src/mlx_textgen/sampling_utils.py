from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
from pydantic import BaseModel, Field, model_validator
import mlx.core as mx

if TYPE_CHECKING:
    from mlx.core import array
    try:
        from outlines.processors.base_logits_processor import OutlinesLogitsProcessor
        from outlines.processors.structured import JSONLogitsProcessor, RegexLogitsProcessor, CFGLogitsProcessor
        from outlines.models.transformers import TransformerTokenizer
    except:
        pass


def apply_repetition_penalty(logits: "array", token_ids: "array", penalty: float = 1.0, context_size: Optional[int] = None, row_range: Optional["array"] = None) -> "array":
    """Applies a repetition penalty to the logits based on previously seen tokens.

    This penalty discourages the model from repeating tokens that have
    already appeared in the sequence.

    Args:
        logits (array): The logits to adjust.
        token_ids (array): The previously generated token ids.
        penalty (float, optional): The penalty to apply. Should be >= 1.0.
            Values > 1 penalize repetition, values < 1 encourage it. Default: 1.0.
        context_size (Optional[int], optional): The context size to consider for the repetition penalty.
            If None, the entire history is used. Default: None.
        row_range (Optional[array], optional): An optional array of indices to use for selecting logits. Defaults to None.

    Returns:
        array: The adjusted logits.
    """
    if penalty == 1:
        return logits
    
    tokens = token_ids if context_size is None else token_ids[:, -context_size:]
    if tokens.shape[1] > 0:
        rows = mx.arange(tokens.shape[0])[:, mx.newaxis] if row_range is None else row_range
        selected_logits = logits[rows, tokens]
        selected_logits = mx.where(
            selected_logits < 0, selected_logits * penalty, selected_logits / penalty
        )
        logits[rows, tokens] = selected_logits
    return logits

def apply_presence_penalty(logits: "array", token_ids: "array", penalty: float = 0.0, context_size: Optional[int] = None, row_range: Optional["array"] = None) -> "array":
    """Applies a presence penalty to the logits based on previously seen tokens.

    This penalty adds a fixed bias to the logits of tokens that have
    already appeared in the sequence.  This encourages the model to
    generate novel tokens.
    
    Args:
        logits (array): The logits to adjust.
        token_ids (array): The previously generated token ids.
        penalty (float, optional): The penalty to apply. Should be >= 0.
            Values > 0 penalize presence, values < 0 encourage it. Default: 0.0.
        context_size (Optional[int], optional): The context size to consider for the presence penalty.
            If None, the entire history is used. Default: None.
        row_range (Optional[array], optional): An optional array of indices to use for selecting logits. Defaults to None.

    Returns:
        array: The adjusted logits.
    """
    if penalty == 0:
        return logits
    
    tokens = token_ids if context_size is None else token_ids[:, -context_size:]
    if tokens.shape[1] > 0:
        rows = mx.arange(tokens.shape[0])[:, mx.newaxis] if row_range is None else row_range
        logits[rows, tokens] -= penalty
    return logits


def apply_frequency_penalty(logits: "array", token_ids: "array", penalty: float = 0.0, context_size: Optional[int] = None, vocab_ids: Optional["array"] = None) -> "array":
    """Applies a frequency penalty to the logits based on the frequency of previously seen tokens.

    Args:
        logits (array): The logits to adjust.
        token_ids (array): The previously generated token ids.
        penalty (float, optional): The penalty to apply. Should be greater than or equal to 0.
            Values > 0 discourage frequent tokens, while values < 0 encourage them. Default: 0.0.
        context_size (Optional[int, optional): The context size to consider for the frequency penalty.
            If None, the entire history is used. Default: None.
        vocab_ids (Optional[array], optional): An optional array of vocab ids to use for calculating the frequency. Defaults to None.

    Returns:
        array: The adjusted logits.
    """
    if penalty == 0:
        return logits

    tokens = token_ids if context_size is None else token_ids[:, -context_size:]

    if tokens.size == 0:
        return logits

    tids = mx.arange(logits.shape[1], dtype=tokens.dtype) if vocab_ids is None else vocab_ids
    frequency_factor = (tokens[..., mx.newaxis] == tids).sum(axis=1) ** 0.5

    logits -= frequency_factor * penalty

    return logits

def create_logit_bias_args(logit_bias: Dict[int, float]) -> Dict[str, "array"]:
    """Creates arguments for applying a logit bias.

    Args:
        logit_bias (Dict[int, float]): A dictionary mapping token ids to bias values.

    Returns:
        Dict[str, "array"]: A dictionary containing the logit keys and biases as mlx arrays.
    """
    args = dict(
        logit_key=mx.array(list(logit_bias.keys())),
        logit_bias=mx.array(list(logit_bias.values()))
    )
    return args


def apply_logit_bias(logits: "array", logit_keys: "array", logit_bias: "array") -> "array":
    """Applies a logit bias to the logits.

    Args:
        logits (array): The logits to adjust.
        logit_keys (array): The token ids to apply the bias to.
        logit_bias (array): The bias values to apply to the logits.

    Returns:
        array: The adjusted logits.
    """
    if logit_bias is None:
        return logits
    
    logits[:, logit_keys] += logit_bias
    return logits
    
def apply_temperature(logits: "array", temperature: float = 0.0, row_range: Optional["array"] = None) -> "array":
    """Applies temperature scaling to the logits.

    If temperature is 0, performs greedy sampling by setting the probability
    of the most likely token to 1 and all others to 0. Otherwise, divides
    the logits by the temperature.

    Args:
        logits (array): The logits to adjust.
        temperature (float, optional): The temperature to apply. If 0, performs greedy sampling.
            Defaults to 0.0.
        row_range (Optional["array"], optional): An optional array of indices to use for selecting logits.
            Defaults to None.
    Returns:
        array: The adjusted logits.
    """
    if temperature != 0:
        return logits / temperature
    
    else:
        from mlx.core import arange, inf
        indices = logits.argmax(axis=1).reshape(-1, 1)
        if row_range is None:
            rows = arange(logits.shape[0]).reshape(-1, 1)
        else:
            rows = row_range
        logits[:, :] = -inf
        logits[rows, indices] = 1
        return logits
    
def apply_top_k(logits: "array", top_k: Optional[int] = None, row_range: Optional["array"] = None) -> "array":
    """Applies top-k filtering to the logits.

    This keeps only the top k tokens with the highest probabilities and
    sets the probabilities of the remaining tokens to negative infinity,
    effectively removing them from consideration.

    Args:
        logits (array): The logits to adjust.
        top_k (Optional[int, optional): The number of top tokens to keep.
            If None, no filtering is applied. Defaults to None.
        row_range (Optional["array"], optional): An optional array of indices to use for selecting logits.
            Defaults to None.

    Returns:
        array: The adjusted logits.
    """
    if (top_k is None) or (logits.shape[1] < top_k):
        return logits
    
    rows = mx.arange(logits.shape[0]).reshape(-1, 1) if row_range is None else row_range
    token_sorted = mx.argsort(-logits)
    logits[rows, token_sorted[:, top_k:]] = -mx.inf
    return logits

def apply_top_p(logits: "array", top_p: float = 1.0, is_prob: bool = False, row_range: Optional["array"] = None) -> "array":
    """Applies top-p filtering to the logits.

    This keeps only the tokens with a cumulative probability above a
    certain threshold (top_p) and sets the probabilities of the remaining
    tokens to zero, effectively removing them from consideration.

    Args:
        logits (array): The logits to adjust.
        top_p (float, optional): The cumulative probability threshold.
            Should be between 0 and 1. If 1, no filtering is applied.
            Defaults to 1.0.
        is_prob (bool, optional): Whether the input is probabilities or logits.
            If False, softmax is applied to the logits before filtering.
            Defaults to False.
        row_range (Optional["array"], optional): An optional array of indices to use for selecting logits.
            Defaults to None.

    Returns:
        array: The adjusted logits or probabilities.
    """
    if top_p == 1:
        return logits if is_prob else mx.softmax(logits, axis=-1)
    
    rows = mx.arange(logits.shape[0]).reshape(-1, 1) if row_range is None else row_range
    probs = mx.softmax(logits, axis=-1) if not is_prob else logits
    token_sorted = mx.argsort(probs)
    sorted_probs = probs[rows, token_sorted]
    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
    top_probs = mx.where(
        cumulative_probs > 1 - top_p,
        sorted_probs,
        0,
    )
    probs[rows, token_sorted] = top_probs
    return probs

def apply_min_p(logits: "array", min_p: float = 0.0, is_prob: bool = False, min_tokens_to_keep: int = 1, row_range: Optional["array"] = None) -> "array":
    """Applies min-p filtering to the logits.

    This keeps only the tokens with a probability above a
    certain threshold (min_p * max(probs)) and sets the probabilities
    of the remaining tokens to zero, effectively removing them from
    consideration.  A minimum number of tokens are always kept to
    avoid completely removing all candidates.

    Args:
        logits (array): The logits to adjust.
        min_p (float, optional): The minimum probability threshold,
            as a fraction of the most likely probability. Should be
            between 0 and 1. If 0, no filtering is applied.
            Defaults to 0.0.
        is_prob (bool, optional): Whether the input is probabilities or
            logits.  If False, softmax is applied to the logits before
            filtering. Defaults to False.
        min_tokens_to_keep (int, optional): The minimum number of tokens
            to keep, regardless of their probability. Defaults to 1.
        row_range (Optional["array"], optional): An optional array of
            indices to use for selecting logits. Defaults to None.

    Returns:
        array: The adjusted logits or probabilities.
    """
    if min_p == 0:
        return logits if is_prob else mx.softmax(logits, axis=-1)
    
    rows = mx.arange(logits.shape[0]).reshape(-1, 1) if row_range is None else row_range
    probs = mx.softmax(logits, axis=-1) if not is_prob else logits
    token_sorted = mx.argsort(-probs)
    sorted_probs = probs[rows, token_sorted]
    top_probs = probs.max(axis=-1).reshape(-1, 1)
    scaled_min_p = min_p * top_probs
    tokens_to_remove = sorted_probs < scaled_min_p
    tokens_to_remove[..., :min_tokens_to_keep] = False
    selected_probs = mx.where(tokens_to_remove, 0, sorted_probs)
    probs[rows, token_sorted] = selected_probs
    return probs

def create_json_logit_processor(json_schema: Dict[str, Any], tokenizer: "TransformerTokenizer", whitespace_pattern: Optional[str] = None) -> "JSONLogitsProcessor":
    """Creates a JSON logits processor for constrained generation.

    The JSON logits processor ensures that the generated text conforms to the
    provided JSON schema.

    Args:
        json_schema (Dict[str, Any]): The JSON schema to constrain the generation.
        tokenizer (TransformerTokenizer): The tokenizer used by the model.
        whitespace_pattern (Optional[str], optional): A regex pattern defining what constitutes a whitespace.
            Defaults to None which uses the tokenizer's default whitespace pattern.

    Returns:
        JSONLogitsProcessor: A JSON logits processor instance.
    """
    from outlines.processors.structured import JSONLogitsProcessor
    return JSONLogitsProcessor(schema=json_schema, tokenizer=tokenizer, whitespace_pattern=whitespace_pattern)

def create_regex_logit_processor(regex_pattern: str, tokenizer: "TransformerTokenizer") -> "RegexLogitsProcessor":
    """Creates a regex logits processor for constrained generation.

    The regex logits processor ensures that the generated text conforms to the
    provided regex pattern.

    Args:
        regex_pattern (str): The regex pattern to constrain the generation.
        tokenizer (TransformerTokenizer): The tokenizer used by the model.

    Returns:
        RegexLogitsProcessor: A regex logits processor instance.
    """
    from outlines.processors.structured import RegexLogitsProcessor
    return RegexLogitsProcessor(regex_string=regex_pattern, tokenizer=tokenizer)

def create_choice_logit_processor(choices: List[str], tokenizer: "TransformerTokenizer") -> "RegexLogitsProcessor":
    """Creates a choice logits processor for constrained generation.

    The choice logits processor ensures that the generated text is one of the
    provided choices.

    Args:
        choices (List[str): A list of strings representing the possible choices.
        tokenizer (TransformerTokenizer): The tokenizer used by the model.

    Returns:
        RegexLogitsProcessor: A regex logits processor instance.
    """
    regex_pattern = r"(" + r"|".join(list(set(choices))) + r")"
    return create_regex_logit_processor(regex_pattern=regex_pattern, tokenizer=tokenizer)

def create_cfg_logit_processor(cfg_str: str, tokenizer: "TransformerTokenizer") -> "CFGLogitsProcessor":
    """Creates a CFG logits processor for constrained generation.

    The CFG logits processor ensures that the generated text conforms to the
    provided Context-Free Grammar (CFG).

    Args:
        cfg_str (str): The CFG string to constrain the generation.
        tokenizer (TransformerTokenizer): The tokenizer used by the model.

    Returns:
        CFGLogitsProcessor: A CFG logits processor instance.
    """
    from outlines.processors.structured import CFGLogitsProcessor
    return CFGLogitsProcessor(cfg_str=cfg_str, tokenizer=tokenizer)

class SamplingParams(BaseModel):
    temperature: float = Field(ge = 0.0, default= 0.0)
    top_k: Optional[int] = Field(gt = 0, default = None)
    top_p: float = Field(gt = 0.0, le = 1.0, default = 1.0)
    min_p: float = Field(ge = 0.0, le = 1.0, default = 0.0)
    stop: List[str] = Field(default_factory=list)
    max_completion_tokens: int = Field(gt = 0, default=4096)
    max_reasoning_tokens: int = Field(ge = 0, default=0)
    min_tokens_to_keep: int = Field(gt = 0, default = 1)
    frequency_penalty: float = Field(ge = -2.0, le = 2.0, default = 0.0)
    presence_penalty: float = Field(ge = -2.0, le = 2.0, default = 0.0)
    repetition_penalty: float = Field(gt = 0.0, default = 1.0)
    penalty_context_size: Optional[int] = Field(gt = 0, default = 1000)
    logit_bias: Optional[Dict[int, float]] = Field(default = None)
    seed: Optional[int] = Field(default = None)
    guided_json: Optional[Dict[str, Any]] = None
    guided_choice: Optional[List[str]] = None
    guided_regex: Optional[str] = None
    guided_grammar: Optional[str] = None
    whitespace_pattern: Optional[str] = None
    logprobs: bool = False
    top_logprobs: int = Field(gt=0, default=4)

    @model_validator(mode='after')
    def validate_single_penalty_type(self) -> "SamplingParams":
        """
        Ensures that only one of frequency_penalty, presence_penalty,
        or repetition_penalty is active (not at its default/neutral value).
        """
        active_penalties = 0

        if self.frequency_penalty != 0.0:
            active_penalties += 1
        if self.presence_penalty != 0.0:
            active_penalties += 1
        if self.repetition_penalty != 1.0: # Default is 1.0 (no penalty)
            active_penalties += 1
        
        if active_penalties > 1:
            raise ValueError(
                "Only one of 'frequency_penalty', 'presence_penalty', or 'repetition_penalty' "
                "can be active (i.e., set to a non-default value) at a time."
            )
        return self
    
    @model_validator(mode='after')
    def validate_single_guided_decoding_processor(self) -> "SamplingParams":
        active_processor = 0

        for p in [self.guided_json, self.guided_choice, self.guided_regex, self.guided_grammar]:
            if p is not None:
                active_processor += 1
        
        if active_processor > 1:
            raise ValueError(
                'Only one of "guided_json", "guided_choice", "guided_regex", "guided_grammar" can be used at a time.'
            )
        
        if active_processor > 0:
            from importlib.util import find_spec
            if find_spec("outlines") is None:
                raise ValueError('Guided decoding is not supported as Outlines is not installed. Please install Outlines with `pip install outlines` to enable guided decoding.')

        return self

class Sampler:
    """Implements the sampling procedure for the model.
    """
    def __init__(self, params: SamplingParams, tokenizer: Optional["TransformerTokenizer"] = None) -> None:
        """Initializes the Sampler with the given parameters and tokenizer.

        Args:
            params (SamplingParams): The sampling parameters to use.
            tokenizer (Optional["TransformerTokenizer"], optional): The tokenizer used by the model. Defaults to None.
        """
        self.params = params
        self.tokenizer = tokenizer

        if self.params.logit_bias:
            self.logit_bias_args = create_logit_bias_args(self.params.logit_bias)

        if self.params.seed is not None:
            mx.random.seed(self.params.seed)

        if self.params.guided_json:
            self.structured_processor = create_json_logit_processor(self.params.guided_json, tokenizer=tokenizer, whitespace_pattern=self.params.whitespace_pattern)
        elif self.params.guided_choice:
            self.structured_processor = create_choice_logit_processor(self.params.guided_choice, tokenizer=tokenizer)
        elif self.params.guided_regex:
            self.structured_processor = create_regex_logit_processor(self.params.guided_regex, tokenizer=tokenizer)
        elif self.params.guided_grammar:
            self.structured_processor = create_cfg_logit_processor(self.params.guided_grammar, tokenizer=tokenizer)
        else:
            self.structured_processor = None

    def sample(self, logits: "array", token_ids: "array", start_index: int) -> Tuple["array", "array"]:
        """Samples the next token based on the logits.

        Applies various sampling techniques such as guided decoding, logit bias,
        frequency/presence/repetition penalties, temperature scaling, top-k,
        top-p, and min-p filtering.

        Args:
            logits (array): The logits to sample from.
            token_ids (array): The previously generated token ids.
            start_index (int): The starting index of the current generation step.

        Returns:
            Tuple[array, array]: A tuple containing the new tokens and the log probabilities.
        """
        if not hasattr(self, 'vocab_ids'):
            self.vocab_ids = mx.arange(logits.shape[1], dtype=token_ids.dtype)

        if not hasattr(self, 'row_range'):
            self.row_range = mx.arange(logits.shape[0]).reshape(-1, 1)

        if self.row_range.shape[0] != logits.shape[0]:
            self.row_range = mx.arange(logits.shape[0]).reshape(-1, 1)

        if self.structured_processor:
            logits = self.structured_processor(input_ids=token_ids[:, start_index:], logits=logits)

        if self.params.logit_bias:
            logits = apply_logit_bias(logits, **self.logit_bias_args)

        if self.params.frequency_penalty != 0:
            logits = apply_frequency_penalty(
                logits=logits, 
                token_ids=token_ids, 
                penalty=self.params.frequency_penalty, 
                context_size=self.params.penalty_context_size, 
                vocab_ids=self.vocab_ids
            )

        elif self.params.presence_penalty != 0:
            logits = apply_presence_penalty(
                logits=logits, 
                token_ids=token_ids, 
                penalty=self.params.presence_penalty, 
                context_size=self.params.penalty_context_size, 
                row_range=self.row_range
            )

        elif self.params.repetition_penalty != 1:
            logits = apply_repetition_penalty(
                logits=logits, 
                token_ids=token_ids, 
                penalty=self.params.repetition_penalty, 
                context_size=self.params.penalty_context_size, 
                row_range=self.row_range
            )

        logits = apply_temperature(logits=logits, temperature=self.params.temperature, row_range=self.row_range)
        logits = apply_top_k(logits=logits, top_k=self.params.top_k, row_range=self.row_range)

        logprobs = logits - mx.logsumexp(logits, axis=-1).reshape(-1, 1)

        probs = apply_min_p(logits=logits, min_p=self.params.min_p, is_prob=False, min_tokens_to_keep=self.params.min_tokens_to_keep, row_range=self.row_range)
        probs = apply_top_p(logits=probs, is_prob=True, top_p=self.params.top_p, row_range=self.row_range)

        new_tokens = mx.random.categorical(mx.log(probs), axis=-1).reshape(-1, 1)

        return new_tokens, logprobs
        


