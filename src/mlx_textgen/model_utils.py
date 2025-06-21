from typing import TYPE_CHECKING, Dict, Any, Optional, Literal, Type, List, Union, Tuple, Iterator
from dataclasses import dataclass
from pydantic import BaseModel
from mlx.core import new_stream, default_device
import contextlib
if TYPE_CHECKING:
    from logging import Logger
    from mlx.nn import Module
    from .tokenizer_utils import BaseDetokenizer
    from transformers.processing_utils import ProcessorMixin
    from transformers.tokenization_utils import PreTrainedTokenizer
    from transformers.image_processing_utils import BaseImageProcessor
    from mlx_lm.models.cache import KVCache, RotatingKVCache
    from mlx.core import array, Stream
    from PIL.Image import Image
    from .cache_utils import CacheManager
    from .sampling_utils import SamplingParams
    from .generation_utils import GenerationOutput
    from .chat_utils import DEFAULT_TEMPLATES, ChatTemplate

GENERATION_STREAM = new_stream(default_device())

class Images(BaseModel):
    images: List[Optional[List[str]]]

@dataclass
class InferencePlan:
    token_ids: "array"
    cache: List[Union["KVCache", "RotatingKVCache"]]
    cache_offset: int
    token_offset: int
    offsets: "array"
    prompt_batches: List[Tuple[Tuple[int, int], Optional[Tuple[int, int, int]]]]
    attention_mask: "array"
    b64_images: Optional[List[Optional[List[str]]]] = None
    pixel_values: Optional["array"] = None
    kwargs_list: Optional[List[Optional[Dict[str, Any]]]] = None
    
@contextlib.contextmanager
def wired_limit(model: "Module", streams: Optional[List["Stream"]] = None):
    """
    A context manager to temporarily change the wired limit.

    Note, the wired limit should not be changed during an async eval.  If an
    async eval could be running pass in the streams to synchronize with prior
    to exiting the context manager.
    """
    from mlx.utils import tree_reduce
    import mlx.core as mx
    model_bytes = tree_reduce(
        lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc, model, 0
    )
    max_rec_size = mx.metal.device_info()["max_recommended_working_set_size"]
    if model_bytes > 0.9 * max_rec_size:
        model_mb = model_bytes // 2**20
        max_rec_mb = max_rec_size // 2**20
        print(
            f"[WARNING] Generating with a model that requires {model_mb} MB "
            f"which is close to the maximum recommended size of {max_rec_mb} "
            "MB. This can be slow. See the documentation for possible work-arounds: "
            "https://github.com/ml-explore/mlx-lm/tree/main#large-models"
        )
    old_limit = mx.set_wired_limit(max_rec_size)
    try:
        yield None
    finally:
        if streams is not None:
            for s in streams:
                mx.synchronize(s)
        else:
            mx.synchronize()
        mx.set_wired_limit(old_limit)

def image_to_b64_string(image: str, cache_dir: str) -> str:
    from base64 import b64encode
    from requests import get
    from pathlib import Path

    if image.startswith('https://') or image.startswith('http://'):
        import os
        import json
        info_dir = os.path.join(cache_dir, 'web_image_info.json')
        if os.path.exists(info_dir):
            with open(info_dir, 'r') as f:
                info = json.load(f)
        else:
            info = dict()
        try:
            if image in info:
                with open(os.path.join(cache_dir, info[image]), 'r') as f:
                    b64_str = f.read()
            else:
                res = get(image)
                res.raise_for_status()
                b64_str = b64encode(res.content).decode('utf-8')
                indices = [int(n.removesuffix('.json').removeprefix('img_'))for n in info.values()]
                new = max(indices) + 1 if indices else 0
                new = f'img_{new}.json'
                info[image] = new
                with open(info_dir, 'w') as f:
                    json.dump(info, f, indent=2)
                with open(os.path.join(cache_dir, new), 'w') as f:
                    f.write(b64_str)

        except Exception as e:
            raise ValueError(f'Failed to load image from URL: "{image}" with error {e}')
        
    elif (len(image) < 10000) and (Path(image).is_file()):
        with open(image, 'rb') as f:
            b64_str = b64encode(f.read()).decode('utf-8')

    elif image.startswith('data:'):
        b64_str = image.removeprefix('data:').split(';base64,')[-1]

    else:
        b64_str = image

    return b64_str

def b64_to_img_object(b64_image: str) -> "Image":
    from base64 import b64decode
    from io import BytesIO
    from PIL.Image import open
    try:
        image_bytes = b64decode(b64_image)
        return open(BytesIO(image_bytes))
    except Exception as e:
        raise ValueError(f'Failed to decode base64 string: "{b64_image}" with error {e}')

def convert_images_to_base64(images: List[Optional[List[str]]]) -> List[Optional[List[str]]]:
    # Validation
    Images(images=images)

    import os
    from .utils import get_package_cache_dir

    image_cache_dir = os.path.join(get_package_cache_dir(), 'web_image_cache')
    os.makedirs(image_cache_dir, exist_ok=True)

    converted = []
    for imgs in images:
        if not imgs:
            converted.append(None)
        else:
            converted.append([image_to_b64_string(img, image_cache_dir) for img in imgs])
    return converted
    
def convert_images_to_img_objects(b64_images: List[Optional[List[str]]]) -> List[Optional[List["Image"]]]:
    converted = []
    for imgs in b64_images:
        if not imgs:
            converted.append(None)
        else:
            converted.append([b64_to_img_object(img) for img in imgs])
    return converted

def find_conseq_indices_pairs(input_list: List[int], batch_size: int) -> List[Tuple[Tuple[int, int], bool]]:
    if not input_list:
        return []

    output = []
    i = 0
    n = len(input_list)

    while i < n:
        start_index = i
        current_value = input_list[i]

        if current_value == 0:
            j = i
            while j < n and input_list[j] == 0:
                j += 1
                if (j - i) == batch_size:
                    output.append(((i, j), False))
                    i = j
                    start_index = i
            if i < j:
                output.append(((start_index, j), False))
                i = j
        else:
            j = i
            while j < n and input_list[j] != 0:
                j += 1
            output.append(((i, j), True))
            i = j

    return output

def make_model_exist(model_id_or_path: str, **kwargs):
    import os
    from pathlib import Path

    is_dir = os.path.exists(model_id_or_path)

    if is_dir:
        model_path = Path(model_id_or_path)
    
    else:
        from mlx_lm.utils import get_model_path
        model_path = get_model_path(model_id_or_path, revision=kwargs.pop('revision', None))

    return model_path

class LLMModel:

    def __init__(
            self, 
            model_id_or_path: str,
            tokenizer_repo_or_path: Optional[str] = None,
            model_kwargs: Optional[Dict[str, Any]] = None,
            tokenizer_kwargs: Optional[Dict[str, Any]] = None,
            model_name: Optional[str] = None,
            logger: Optional["Logger"] = None,
            verbose: bool = True,
            enable_cache: bool = True,
            cache_manage_config: Optional[Dict[str, Any]] = None,
            preprocess_batch_size: int = 512,
            extra_stop_words: Optional[Union[str, List[str]]] = None,
            reasoning_parser: Optional[Literal['deepseek_r1']] = None,
            default_template: Optional["DEFAULT_TEMPLATES"] = None,
            **kwargs
        ) -> None:
        from .cache_utils import CacheManager
        from .chat_utils import ChatTemplate
        self._logger = logger
        self._verbose = verbose
        self._model_id_or_path = model_id_or_path
        self._extra_stop_words = [] if extra_stop_words is None else extra_stop_words
        if isinstance(self._extra_stop_words, str):
            self._extra_stop_words = [self._extra_stop_words]
        self._tokenizer_repo_or_path = tokenizer_repo_or_path if tokenizer_repo_or_path else model_id_or_path
        model_kwargs = model_kwargs if model_kwargs else {}
        tokenizer_kwargs = tokenizer_kwargs if tokenizer_kwargs else {}
        self._model_name = model_name

        self._load_model_and_config(model_id_or_path=model_id_or_path, **model_kwargs)
        self._load_processor(tokenizer_repo_or_path=self._tokenizer_repo_or_path, **tokenizer_kwargs)

        self._enable_cache = enable_cache
        self._preprocess_batch_size = preprocess_batch_size
        cache_manage_config = cache_manage_config if cache_manage_config else {}
        self._cache_manager = CacheManager(
            model_name=self.model_real_name, 
            is_vision=self.is_vision,
            image_token_id=self.image_token_id,
            extra_image_tokens=self.extra_image_tokens,
            logger=self._logger,
            **cache_manage_config
        )
        self._chat_template = ChatTemplate(self.tokenizer, model_type=self.config['model_type'], default_template=default_template, is_vision=self.is_vision, reasoning_parser=reasoning_parser)
        
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
        elif self._verbose:
            print(f'{level.upper()}: {msg.strip()}')

    @property
    def original_config(self) -> Dict[str, Any]:
        """Retrieves the original configuration of the pre-trained model.

        This method uses the `transformers` library to load the configuration
        directly from the `model_id_or_path` specified during initialization.
        The configuration is then returned as a dictionary.

        Returns:
            Dict[str, Any]: A dictionary containing the original configuration parameters of the model.
        """
        if not hasattr(self, '_original_config'):
            from transformers import AutoConfig
            self._original_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=self._model_id_or_path).to_dict()
        return self._original_config
    
    @property
    def is_vision(self) -> bool:
        """Checks if the model is a vision model based on its configuration.

        This method inspects the model's original configuration to determine
        if it contains a 'vision_config' key. The presence of this key
        indicates that the model is designed for vision-related tasks.

        Returns:
            bool: True if the model is a vision model, False otherwise.
        """
        return 'vision_config' in self.original_config
    
    @property
    def model(self) -> "Module":
        """The loaded mlx.nn.Module.

        Returns:
            Module: The loaded model.
        """
        return self._model
    
    @property
    def config(self) -> Dict[str, Any]:
        """Retrieves the configuration used to load the mlx model.

        Returns:
            Dict[str, Any]: The configuration dictionary.
        """
        return self._config
    
    @property
    def vision_config(self) -> Dict[str, Any]:
        """Retrieves the vision configuration of the model.

        This property accesses the 'vision_config' key within the model's
        overall configuration. If the key is present, its value (a dictionary
        containing vision-specific configuration parameters) is returned.
        If the 'vision_config' key is not found, an empty dictionary is returned,
        indicating that the model does not have a separate vision configuration.

        Returns:
            Dict[str, Any]: The vision configuration dictionary, or an empty dictionary if not found.
        """
        return self._config.get('vision_config', {})
    
    @property
    def image_token_id(self) -> Optional[int]:
        """Retrieves the token ID for the image placeholder.

        This property attempts to retrieve the ID of the token used to
        represent an image within the model's vocabulary. This is
        typically used in multimodal models that process both text and images.
        The token ID is read from the model's configuration under the
        'image_token_index' key.
        
        Returns:
            Optional[int]: The token ID of the image placeholder, or None if not found in the configuration.
        """
        if not hasattr(self, '_image_token_id'):
            self._image_token_id = self.config.get('image_token_index', self.config.get('image_token_id'))
        return self._image_token_id
    
    @property
    def boi_token_id(self) -> Optional[int]:
        """Retrieves the token ID representing the beginning of an image.

        This property accesses the model's configuration to find the ID
        of the token that signifies the start of an image sequence.  This
        is used in multimodal models to delineate image input.

        Returns:
            Optional[int]: The token ID for the beginning-of-image token, or None if not found in the configuration.
        """
        return self.config.get('boi_token_index')
    
    @property
    def eoi_token_id(self) -> Optional[int]:
        """Retrieves the token ID representing the end of an image.

        This property accesses the model's configuration to find the ID
        of the token that signifies the end of an image sequence. This
        is used in multimodal models to delineate image input.

        Returns:
            Optional[int]: The token ID for the end-of-image token, or None if not found in the configuration.
        """
        return self.config.get('eoi_token_index')
    
    @property
    def extra_image_tokens(self) -> List[int]:
        if not hasattr(self, '_extra_image_tokens'):
            self._extra_image_tokens = []
            if self.config['model_type'] == 'mistral3':
                img_break_tokens = self.tokenizer('[IMG_BREAK]', add_special_tokens=False)['input_ids']
                if len(img_break_tokens) == 1:
                    self._extra_image_tokens.append(img_break_tokens[0])
                else:
                    self.log(msg='Cannot add image break token for "mistral3" model.', level='warning')
        return self._extra_image_tokens
    
    @property
    def processor(self) -> Optional["ProcessorMixin"]:
        """Retrieves the processor associated with the model.

        For vision models, this will be an instance of a class like
        `transformers.AutoProcessor`.  For language models, this will
        be `None`.

        Returns:
            Optional[ProcessorMixin]: The loaded processor, or None if not applicable.
        """
        return self._processor
    
    @property
    def image_processor(self) -> Optional["BaseImageProcessor"]:
        """Retrieves the image processor associated with the model's processor.

        This property is relevant for vision models that use a processor
        containing an image processor component. If the model's processor
        has an 'image_processor' attribute, it is returned. Otherwise,
        this property returns None.

        Returns:
            Optional[BaseImageProcessor: The image processor, or None if not available.
        """
        if self.processor:
            if hasattr(self.processor, 'image_processor'):
                return self.processor.image_processor
    
    @property
    def tokenizer(self) -> "PreTrainedTokenizer":
        """Retrieves the tokenizer associated with the model.

        For vision models, the tokenizer is part of the processor.
        For language models, this is a standard tokenizer.

        Returns:
            PreTrainedTokenizer: The loaded tokenizer.
        """
        return self._tokenizer
    
    @property
    def model_name(self) -> str:
        """Returns the name of the model.

        If a model name was provided during initialization, that name is returned.
        Otherwise, the model name is extracted from the model ID or path.

        Returns:
            str: The name of the model.
        """
        if not self._model_name:
            name = self._model_id_or_path.split('/')[-1].split('\\')[-1]
            name = name.lower().replace('_', '-')
            if 'bit' in name.split('-')[-1]:
                name = '-'.join(name.split('-')[:-1])
            self._model_name = name
        return self._model_name
    
    @property
    def model_real_name(self) -> str:
        if not hasattr(self, '_model_real_name'):
            name = self._model_id_or_path.split('/')[-1].split('\\')[-1]
            name = name.lower().replace('_', '-')
            if 'bit' in name.split('-')[-1]:
                name = '-'.join(name.split('-')[:-1])
            if self.quantization:
                name += f'-q{self.quantization}'
            self._model_real_name = name
        return self._model_real_name
    
    @property
    def model_local_path(self) -> str:
        """Returns the local path to the model.

        This is the path on the file system where the model
        is stored.  It will be a directory.

        Returns:
            str: The local path to the model.
        """
        return self._model_local_path
    
    @property
    def detokenizer_class(self) -> Type["BaseDetokenizer"]:
        """Returns the detokenizer class to use for this model.

        The detokenizer class is determined during initialization based
        on the tokenizer configuration.  It will be a subclass of
        BaseDetokenizer.

        Returns:
            Type[BaseDetokenizer]: The detokenizer class.
        """
        return self._detokenizer_class
    
    @property
    def quantization(self) -> Optional[int]:
        """Returns the quantization level of the model, if any.

        This property attempts to read the quantization level (number of bits)
        from the model's configuration. It looks for a 'quantization' key
        in the configuration dictionary, and within that, a 'bits' key.
        If both keys are present and the 'bits' value is an integer, that
        value is returned. If either key is missing or the 'bits' value
        is not an integer, None is returned.

        Returns:
            Optional[int]: The number of bits used for quantization, or None if the model is not quantized or the quantization level is not specified in the config.
        """
        if not hasattr(self, '_quantization'):
            self._quantization = self.config.get('quantization', dict()).get('bits')
        return self._quantization
    
    @property
    def enable_cache(self) -> bool:
        """Indicates whether caching is enabled for this model.

        When enabled, the key-value cache (KVCache) is used to store
        intermediate results during inference, which can significantly
        improve performance for autoregressive generation tasks.

        Returns:
            bool: True if caching is enabled, False otherwise.
        """
        return self._enable_cache

    @property
    def preprocess_batch_size(self) -> int:
        """Returns the batch size used for preprocessing prompts.

        This value controls how many tokens are processed together
        during prompt processing.

        Returns:
            int: The batch size used for preprocessing.
        """
        return self._preprocess_batch_size

    @property
    def cache_manager(self) -> "CacheManager":
        """Retrieves the CacheManager instance associated with this model.

        The CacheManager is responsible for managing the key-value cache used
        during inference. It handles tasks such as preallocation, resizing,
        and clearing the cache.

        Returns:
            CacheManager: The CacheManager instance.
        """
        return self._cache_manager

    @property
    def chat_template(self) -> "ChatTemplate":
        return self._chat_template

    def _load_model_and_config(self, model_id_or_path: str, **kwargs) -> None:
        """Loads the model and its configuration.

        This method handles loading the model and its configuration,
        whether from a local directory or by downloading it from a
        remote repository.  It also handles the distinction between
        vision models (using `mlx_vlm`) and language models (using `mlx_lm`).

        Args:
            model_id_or_path (str): The identifier or path to the model.
            **kwargs: Additional keyword arguments passed to the model loading functions.
        """
        import os
        import time
        from pathlib import Path

        start = time.perf_counter()

        from mlx.core import clear_cache

        is_dir = os.path.exists(model_id_or_path)

        if is_dir:
            model_path = Path(model_id_or_path)
        
        else:
            from mlx_lm.utils import get_model_path
            model_path = get_model_path(model_id_or_path, revision=kwargs.pop('revision', None))

        self._model_local_path = str(model_path)

        if self.is_vision:
            from mlx_vlm.utils import load_model, load_config
            self._model = load_model(model_path=model_path, **kwargs)
            self._config = load_config(model_path=model_path, **kwargs)
        
        else:
            from mlx_lm.utils import load_model
            self._model, self._config = load_model(model_path=model_path, model_config=kwargs if kwargs else {})

        clear_cache()
        end = time.perf_counter()
        self.log(f'Model and config for "{self.model_name}" loaded. Time taken: {end - start:.3f}s.')

    def unload(self) -> None:
        import mlx.core as mx
        import gc

        del self._model, self._tokenizer, self._processor, self._chat_template, self._config, self._cache_manager
        mx.clear_cache()
        gc.collect()

    def _get_detokenizer_class(self, tokenizer_repo_or_path: str) -> None:
        """Determines the appropriate detokenizer class based on the tokenizer configuration.

        This method examines the `tokenizer.json` file in the specified
        `tokenizer_repo_or_path` to identify the type of decoder being used.
        Based on the decoder type, it selects the corresponding detokenizer
        class (NaiveDetokenizer, SPMDetokenizer, or a variant of SPMDetokenizer).

        Args:
            tokenizer_repo_or_path (str): The repository or path containing the tokenizer configuration files.
        """
        import os
        from huggingface_hub import hf_hub_download
        from .tokenizer_utils import NaiveDetokenizer, SPMDetokenizer, _is_spm_decoder, _is_spm_decoder_no_space
        from functools import partial
        import json

        detokenizer_class = NaiveDetokenizer

        tokenizer_file = os.path.join(tokenizer_repo_or_path, "tokenizer.json")
        if not os.path.exists(tokenizer_file):
            tokenizer_file = hf_hub_download(repo_id=tokenizer_repo_or_path, filename='tokenizer.json')
        with open(tokenizer_file, "r") as fid:
            tokenizer_content = json.load(fid)
        if "decoder" in tokenizer_content:
            if _is_spm_decoder(tokenizer_content["decoder"]):
                self.log(f'Model "{self.model_name}" is using SPM decoder.')
                detokenizer_class = SPMDetokenizer
            elif _is_spm_decoder_no_space(tokenizer_content["decoder"]):
                self.log(f'Model "{self.model_name}" is using SPM decoder with trim_space=False.')
                detokenizer_class = partial(SPMDetokenizer, trim_space=False)
            else:
                self.log(f'Model "{self.model_name}" is using Naive decoder.')

        self._detokenizer_class = detokenizer_class

    def _load_processor(self, tokenizer_repo_or_path: str, **kwargs) -> None:
        """Loads the tokenizer and processor (if applicable) from a specified repository or path.

        This method uses the `transformers` library to load either a tokenizer or a processor,
        depending on whether the model is a vision model or a language model. For vision models,
        an AutoProcessor is loaded, which includes both a tokenizer and image processing tools.
        For language models, an AutoTokenizer is loaded. The appropriate detokenizer class is
        then determined based on the tokenizer's configuration.

        Args:
            tokenizer_repo_or_path (str): The repository or path from which to load the tokenizer.
            **kwargs: Additional keyword arguments to pass to the `from_pretrained` method of the tokenizer or processor.
        """
        import time
        
        start = time.perf_counter()
        if self.is_vision:
            from transformers import AutoProcessor
            self._processor: "ProcessorMixin" = AutoProcessor.from_pretrained(tokenizer_repo_or_path, **kwargs)
            self._tokenizer = self._processor.tokenizer
            if self._tokenizer.chat_template is None:
                self._tokenizer.chat_template = self._processor.chat_template
            elif (self._processor.chat_template) and (self._processor.chat_template != self._tokenizer.chat_template):
                self._tokenizer.chat_template = self._processor.chat_template

        else:
            from transformers import AutoTokenizer
            self._processor = None
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo_or_path, **kwargs)
            if not self._tokenizer.pad_token:
                self._tokenizer.pad_token = self._tokenizer.eos_token

        self._get_detokenizer_class(tokenizer_repo_or_path)
        end = time.perf_counter()
        additional_info = ' and Preprocessor' if self.is_vision else ''
        self.log(f'Tokenizer{additional_info} loaded for model "{self.model_name}". Time taken: {end - start:.3f}s.')

    def get_empty_cache(self) -> List[Union["KVCache", "RotatingKVCache"]]:
        """Returns an empty key-value cache (KVCache) suitable for use with this model.

        The type of cache returned depends on the model architecture.
        For example, some models use a rotating KVCache.  The cache
        is created using the `make_prompt_cache` function from `mlx_lm.models.cache`.

        Returns:
            List[Union["KVCache", "RotatingKVCache"]]: An empty key-value cache.
        """
        from mlx_lm.models.cache import make_prompt_cache

        model = self.model.language_model if self.is_vision else self.model
        return make_prompt_cache(model)

    def prepare_inputs(self,
            prompts: Union[str, List[str]],
            images: Optional[List[Optional[List[str]]]] = None
        ) -> Dict[str, "array"]:
        """Prepares the inputs for the model.

        This method tokenizes the input prompts and processes any provided images
        into a format suitable for the model. It handles both vision and
        language models, using the appropriate tokenizer and processor.
        For vision models, it leverages the `mlx_vlm.utils.prepare_inputs` function
        to handle image and text integration.  For language models, it uses the
        tokenizer directly. It also handles batch processing of prompts for
        language models.

        Args:
            prompts (Union[str, List[str]]): The input text prompt(s). Can be a single string or a list of strings.
            images (Optional[Union[str, List[str], optional): The input image(s). Can be a single image path or a list of image paths. Defaults to None.

        Returns:
            Dict[str, "array"]: A dictionary containing the prepared inputs, including token IDs and attention masks.

        Raises:
            ValueError: If a vision model receives more than one prompt, or if a language model receives image inputs.
        """
        import time
        start = time.perf_counter()

        def clean_prompt(prompt: str, bos_token: str):
            new = prompt
            if bos_token:
                while new.startswith(bos_token):
                    new = new.removeprefix(bos_token)
            return new

        prompts = [prompts] if isinstance(prompts, str) else prompts
        prompts = [clean_prompt(prompt, self.tokenizer.bos_token) for prompt in prompts]
        images = images if images else [None] * len(prompts)
        images = [img if img else None for img in images]

        if self.is_vision:
            if len(prompts) > 1:
                error = f'Vision model does not support batch processing currently. Unable to process {len(prompts)} with "{self.model_name}".'
                self.log(error, level='error')
                raise ValueError(error)
            if images[0]:
                from mlx_vlm.utils import prepare_inputs
                b64_images = convert_images_to_base64([images[0]])
                bio_images = convert_images_to_img_objects(b64_images)
                inputs = prepare_inputs(
                    processor=self.processor,
                    images=bio_images[0],
                    prompts=prompts,
                    image_token_index=self.image_token_id
                )
                inputs['b64_images'] = b64_images
            else:
                inputs = self.tokenizer(prompts, padding=True, padding_side='left', return_tensors='mlx', return_attention_mask=True)

        else:
            if any(x is not None for x in images):
                error = f'Model "{self.model_name}" does not support image inputs.'
                self.log(error, level='error')
                raise ValueError(error)
            inputs = self.tokenizer(prompts, padding=True, padding_side='left', return_tensors='mlx', return_attention_mask=True)

        inputs['offsets'] = (inputs['attention_mask'] == 0).sum(axis=1)

        end = time.perf_counter()
        self.log(f'Prepared inputs for model "{self.model_name}". Time taken: {end - start:.3f}s.')
        return inputs
    
    def _create_batches(self, token_ids: "array", pixel_values: Optional["array"] = None) -> List[Tuple[Tuple[int, int], Optional[Tuple[int, int, int]]]]:
        num_tokens = token_ids.shape[1]
        num_images = pixel_values.shape[0] if (pixel_values is not None) and self.is_vision else 0

        def get_batch_tuples(num_tokens: int, batch_size: int) -> List[Tuple[int, int]]:
            num_batches = (num_tokens // batch_size) + 1 if num_tokens % batch_size else (num_tokens // batch_size)
            batches = [
                (
                    batch_size * i,
                    min(batch_size * (i + 1), num_tokens)
                )
                for i in range(num_batches)
            ]
            return batches

        if not num_images:
            batches = get_batch_tuples(num_tokens, self.preprocess_batch_size)
            return [(b, None) for b in batches]
        
        else:
            # Only work with batch size=1
            from .cache_utils import split_list_by_image_token
            text_seqs, img_seqs = split_list_by_image_token(token_ids[0].tolist(), [self.image_token_id] + self.extra_image_tokens)
            text_batches = [get_batch_tuples(len(ts), self.preprocess_batch_size) for ts in text_seqs]
            text_sizes = [[(tb[1] - tb[0], False) for tb in tbs] for tbs in text_batches]
            image_sizes = [(len(i), True) for i in img_seqs]
            chunk_sizes = text_sizes[0]
            for i in range(len(image_sizes)):
                chunk_sizes.append(image_sizes[i])
                chunk_sizes.extend(text_sizes[i + 1])
            
            final_batches = []
            current_start = 0
            current_end = 0
            current_image_start = 0
            current_image_end = 0
            current_size = 0

            for size, is_image in chunk_sizes:
                current_size += size
                current_end += size
                if is_image:
                    current_image_end += 1

                if (current_size >= self.preprocess_batch_size) or is_image:
                    t_batch = (current_start, current_end)
                    i_batch = None if current_image_end == current_image_start else (current_image_start, current_image_end, size)
                    final_batches.append((t_batch, i_batch))

                    current_size = 0
                    current_start = current_end 
                    current_image_start = current_image_end

            if current_size:
                t_batch = (current_start, current_end)
                i_batch = None if current_image_end == current_image_start else (current_image_start, current_image_end, size)
                final_batches.append((t_batch, i_batch))

            return final_batches

    def create_inference_plan(self,
            prompts: Union[str, List[str]],
            images: Optional[List[Optional[List[str]]]] = None
        ) -> InferencePlan:
        """Create the plan for batch processsing and inference.

        Args:
            prompts (Union[str, List[str]]): The prompts for generation.
            images (Optional[List[Optional[List[str]]]], optional): The images for the prompt if the model is a vision model. Defaults to None.

        Returns:
            InferencePlan: An inference plan object for prompt processing.
        """
        inputs = self.prepare_inputs(prompts=prompts, images=images)
        import time

        start = time.perf_counter()

        from .cache_utils import split_list_by_image_token
        from mlx.core import array

        b64_images = inputs.get('b64_images')
        b64_images = b64_images if b64_images else [None] * inputs['input_ids'].shape[0]

        if self.enable_cache:
            cache, token_offset = self.cache_manager.get_cache(
                create_cache_fn=self.get_empty_cache,
                token_ids=inputs['input_ids'],
                offsets=inputs['offsets'],
                images=b64_images
            )
        else:
            cache = self.get_empty_cache()
            token_offset = 0

        token_ids = inputs['input_ids']
        tokens_to_process = token_ids[:, token_offset:]
        processed_tokens = token_ids[:, :token_offset]
        num_processed_images = [
            len(split_list_by_image_token(processed_tokens[i].tolist(), image_token_id=self.image_token_id)[1])
            for i in range(token_ids.shape[0])
            ] if self.is_vision and (processed_tokens.shape[1] != 0) else [0] * token_ids.shape[0]
        pixel_values = inputs.get('pixel_values')
        pixel_values = pixel_values[num_processed_images[0]:, :] if pixel_values is not None else None
        batches = self._create_batches(token_ids=tokens_to_process, pixel_values=pixel_values)
        final_batches = [
            (
                (ts + cache[0].offset, te + cache[0].offset),
                img if img is None else (img[0] + num_processed_images[0], img[1] + num_processed_images[0], img[2])
            )
            for (ts, te), img in batches
        ]

        kwargs_list = [dict()] * len(final_batches)

        kwargs_keys = [k for k in inputs.keys() if k not in ('input_ids', 'attention_mask', 'pixel_values', 'offsets', 'b64_images')]
        
        for kw in kwargs_keys:
            val = inputs[kw]
            # print(kw, val)
            # print(inputs.get('pixel_values').shape)
            if isinstance(val, array) and (inputs.get('pixel_values') is not None) and (inputs.get('pixel_values').shape[0] == val.shape[0]):
                for i in range(len(kwargs_list)):
                    if final_batches[i][1]:
                        kwargs_list[i][kw] = val[final_batches[i][1][0]:final_batches[i][1][1]]

        output = InferencePlan(
            token_ids=token_ids,
            cache=cache,
            cache_offset=cache[0].offset,
            token_offset=token_offset,
            offsets=inputs['offsets'],
            prompt_batches=final_batches,
            attention_mask=inputs['attention_mask'],
            b64_images=inputs.get('b64_images'),
            pixel_values=pixel_values,
            kwargs_list=kwargs_list
        )
        end = time.perf_counter()
        self.log(f'Inference plan created. Time taken: {end - start:.3f}s.')
        return output
    
    def _inference(self, 
            token_ids: "array",
            cache: List[Union["KVCache", "RotatingKVCache"]],
            mask: Optional["array"],
            pixel_values: Optional['array'] = None,
            **kwargs
            ) -> "array":
        """The core inference from the mlx model.

        Args:
            token_ids (array): Token IDs to process.
            cache (List[Union[&quot;KVCache&quot;, &quot;RotatingKVCache&quot;]]): List of cache object.
            mask (Optional[&quot;array&quot;]): Attention mask for vision model.
            pixel_values (Optional[&#39;array&#39;], optional): Image pixel values for vision model. Defaults to None.

        Returns:
            array: The output logits.
        """
        from mlx.core import stream
        with stream(GENERATION_STREAM):
            if self.is_vision:
                return self.model(token_ids, cache=cache, mask=mask, pixel_values=pixel_values, **kwargs).logits[:, -1, :]
            
            else:
                return self.model(token_ids, cache=cache)[:, -1, :]
        
    def process_prompt(self, inference_plan: InferencePlan) -> Tuple["array", List[List[int]]]:
        import time
        start = time.perf_counter()

        from mlx.core import clear_cache, eval

        logits = None
        token_ids = inference_plan.token_ids
        num_prompts = token_ids.shape[0]
        num_tokens = num_prompts * token_ids.shape[1]
        processed_tokens = inference_plan.token_offset * num_prompts
        unprocessed_tokens = num_tokens - processed_tokens
        processed = 0
        kwargs_dicts = inference_plan.kwargs_list if inference_plan.kwargs_list else [dict()] * len(inference_plan.prompt_batches)
        image_diffs = []

        with wired_limit(self.model, streams=[GENERATION_STREAM]):
            for ((ts, te), img), kw in zip(inference_plan.prompt_batches, kwargs_dicts):
                bstart = time.perf_counter()
                batch_size = (te - ts) * num_prompts
                processed += batch_size
                pixel_values = inference_plan.pixel_values[img[0]:img[1]] if img else None
                image_size = img[2] if img else None
                logits = self._inference(token_ids=token_ids[:, ts:te], cache=inference_plan.cache, mask=inference_plan.attention_mask[:, :te], pixel_values=pixel_values, **kw)
                eval([c.state for c in inference_plan.cache])
                if image_size:
                    diff_till_now = sum(image_diffs) if image_diffs else 0
                    img_diff = token_ids[:, :te].shape[1] - inference_plan.cache[0].offset - diff_till_now
                    image_diffs.append(img_diff)
                clear_cache()
                bend = time.perf_counter() - bstart
                tps = batch_size / bend
                image_msg = f'Image tokens length: {image_size}. '
                self.log(f'Total of {processed}/{unprocessed_tokens} processed tokens. {image_msg}Batch size: {batch_size} tokens. Time taken for current batch: {bend:.3f}s. {tps:.3f} t/s.')

        end = time.perf_counter() - start
        tps = num_tokens / end
        self.log(f'Processed {num_tokens} tokens in total. Total time taken: {end:.3f}s. {tps:.3f} t/s.')
        return logits, [image_diffs] if image_diffs else [[]] * inference_plan.token_ids.shape[0]
    
    def _get_logprobs(self, logprobs: "array", current_ids: "array", current_tokens: List[str], top_logprobs: int, row_range: "array"):
        from mlx.core import argsort, inf, where
        lp = where(logprobs == inf, 1e4, logprobs)
        lp = where(lp == -inf, -1e4, lp)
        current_logprobs = lp[row_range, current_ids].reshape(-1).tolist()
        top_ids = argsort(-logprobs, axis=-1)[:, :top_logprobs]
        top_probs = lp[row_range, top_ids].tolist()
        top_tokens = [self.tokenizer.batch_decode(t) for t in top_ids[:, :, None].tolist()]
        output = [
            dict(
                token=ct,
                logprob=cl,
                top_logprobs=[
                    dict(
                        token=t,
                        logprob=l
                    ) for t, l in zip(tt, tl)
                ]
            )
            for ct, cl, tt, tl in zip(current_tokens, current_logprobs, top_tokens, top_probs)
        ]
        return output

    def _stream(self, inference_plan: InferencePlan, logits: "array", image_diffs: List[List[int]], sampling_params: "SamplingParams", is_thinking: bool = False) -> Iterator[List["GenerationOutput"]]:
        from time import perf_counter

        start = perf_counter()

        from .sampling_utils import Sampler
        from .generation_utils import StringStop, GenerationOutput
        import mlx.core as mx
        try:
            from outlines.models.transformers import TransformerTokenizer
        except:
            TransformerTokenizer = None

        max_tokens = sampling_params.max_reasoning_tokens if is_thinking else sampling_params.max_completion_tokens
        stop = sampling_params.stop if sampling_params.stop else []
        if self.tokenizer.eos_token and (self.tokenizer.eos_token not in stop):
            stop.append(self.tokenizer.eos_token)
        for s in self._extra_stop_words:
            if s not in stop:
                stop.append(s)
        ttokenizer = None if TransformerTokenizer is None else TransformerTokenizer(tokenizer=self.tokenizer)
        token_ids = inference_plan.token_ids
        cache = inference_plan.cache
        mask = inference_plan.attention_mask
        start_index = token_ids.shape[1]

        def gen_tokens():
            nonlocal token_ids, cache, mask, start_index, inference_plan, logits, start
            offsets = inference_plan.offsets
            input_tokens = ((offsets * -1) + token_ids.shape[1]).tolist()
            sampler = Sampler(params=sampling_params, tokenizer=ttokenizer)
            detokenizer = self.detokenizer_class(tokenizer=self.tokenizer)
            stopper = StringStop(num_prompt=token_ids.shape[0], stop=stop)

            new_tokens, logprobs = sampler.sample(logits=logits, token_ids=token_ids, start_index=start_index)
            new_token_list = new_tokens.tolist()
            detokenizer.add_tokens(new_token_list)
            token_ids = mx.concat([token_ids, new_tokens], axis=1)
            new_str_tokens_obj = stopper.get_finalised_token_strings(tokens=detokenizer.last_segments)
            new_str_tokens = [o.new_token for o in new_str_tokens_obj]
            stop_strings = [o.stop_str for o in new_str_tokens_obj]
            mask = mx.concat([mask, mx.ones(shape=(mask.shape[0], 1), dtype=mask.dtype)], axis=1)
            output_tokens = 1
            is_stop = [bool(st) for st in stop_strings]

            if sampling_params.logprobs:
                logprob_list = self._get_logprobs(logprobs, new_tokens, current_tokens=new_str_tokens, top_logprobs=sampling_params.top_logprobs, row_range=sampler.row_range)

            gos = [
                GenerationOutput(
                    index=i,
                    token=t,
                    token_id=tid[0],
                    stop_str=s,
                    logprobs=logprob_list[i] if sampling_params.logprobs else None,
                    input_tokens=it,
                    output_tokens=output_tokens,
                    finish_reason='stop' if s else ('length' if max_tokens == 1 else None)
                )
                    for i, (t, tid, it, s)
                    in enumerate(zip(new_str_tokens, new_token_list, input_tokens, stop_strings))
            ]
            yield gos

            for i in range(max_tokens - 1):
                try:
                    if all(is_stop):
                        break
                    logits = self._inference(new_tokens, cache=cache, mask=mask, pixel_values=None)

                    new_tokens, logprobs = sampler.sample(logits=logits, token_ids=token_ids, start_index=start_index)
                    new_token_list = new_tokens.tolist()
                    detokenizer.add_tokens(new_token_list)
                    token_ids = mx.concat([token_ids, new_tokens], axis=1)
                    new_str_tokens_obj = stopper.get_finalised_token_strings(tokens=detokenizer.last_segments)
                    new_str_tokens = [o.new_token for o in new_str_tokens_obj]
                    stop_strings = [o.stop_str for o in new_str_tokens_obj]
                    mask = mx.concat([mask, mx.ones(shape=(mask.shape[0], 1), dtype=mask.dtype)], axis=1)
                    output_tokens += 1

                    if i == (max_tokens - 2):
                        detokenizer.finalize()
                        final_str_tokens_obj = stopper.get_finalised_token_strings(tokens=detokenizer.last_segments)
                        final_str_tokens = [o.new_token for o in final_str_tokens_obj]
                        final_stop_strings = [o.stop_str for o in final_str_tokens_obj]
                        new_str_tokens = [n + f + sf.new_token for n, f, sf in zip(new_str_tokens, final_str_tokens, stopper.get_remains())]
                        stop_strings = [s if s else f for s, f in zip(stop_strings, final_stop_strings)]

                    is_stop = [(si or bool(st)) for si, st in zip(is_stop, stop_strings)]
                        
                    if sampling_params.logprobs:
                        logprob_list = self._get_logprobs(logprobs, new_tokens, current_tokens=new_str_tokens, top_logprobs=sampling_params.top_logprobs, row_range=sampler.row_range)

                    gos = [
                        GenerationOutput(
                            index=j,
                            token=t,
                            token_id=tid[0],
                            stop_str=s,
                            logprobs=logprob_list[j] if sampling_params.logprobs else None,
                            input_tokens=it,
                            output_tokens=output_tokens,
                            finish_reason='stop' if s else (None if (i != (max_tokens - 2)) or si else 'length')
                        )
                            for j, (t, tid, it, s, si)
                            in enumerate(zip(new_str_tokens, new_token_list, input_tokens, stop_strings, is_stop))
                    ]
                    yield gos
                except GeneratorExit:
                    self.log(f'Generation stopped prematurely.')
                    del inference_plan, cache, logits
                    mx.clear_cache()
                    break
                except Exception as e:
                    raise

            mx.clear_cache()
            gen_end = perf_counter()
            token_generated = output_tokens * mask.shape[0]
            tps = token_generated / (gen_end - start)
            self.log(f'{token_generated} tokens generated. Time taken: {gen_end - start:.3f}s. {tps:.3f} t/s.')
            if self.enable_cache:
                try:
                    self.cache_manager.save_cache(cache, 
                        token_ids=token_ids[:, :-1], 
                        offsets=offsets, 
                        create_cache_fn=self.get_empty_cache, 
                        images=inference_plan.b64_images,
                        image_diffs=image_diffs)
                except:
                    pass
            try:
                del inference_plan, cache, logits
            except:
                pass
            mx.clear_cache()

        return gen_tokens()
    
    def stream(
            self,
            prompts: Union[str, List[str]],
            sampling_params: "SamplingParams",
            images: Optional[List[Optional[List[str]]]] = None,
            n: int = 1,
            is_thinking: bool = False
        ) -> Iterator[List["GenerationOutput"]]:
        import mlx.core as mx
        imgs = [None] if images is None else images 
        inference_plan = self.create_inference_plan(prompts=prompts, images=images if any(img is not None for img in imgs) else None)
        logits, image_diffs = self.process_prompt(inference_plan)
        
        # Make duplicates of everything for n > 1
        if n > 1:
            inference_plan.attention_mask = mx.repeat(inference_plan.attention_mask, repeats=n, axis=0)
            inference_plan.token_ids = mx.repeat(inference_plan.token_ids, repeats=n, axis=0)
            for c in inference_plan.cache:
                c.keys = mx.repeat(c.keys, repeats=n, axis=0)
                c.values = mx.repeat(c.values, repeats=n, axis=0)
            inference_plan.offsets = mx.repeat(inference_plan.offsets, repeats=n, axis=0)
            logits = mx.repeat(logits, repeats=n, axis=0)
            if inference_plan.b64_images is not None:
                new_images = []
                new_image_diffs = []
                for i, il in enumerate(inference_plan.b64_images):
                    new_image_diffs.extend([image_diffs[i]] * n)
                    new_images.extend([il] * n)
                inference_plan.b64_images = il
                image_diffs = new_image_diffs
        
        return self._stream(inference_plan, logits=logits, image_diffs=image_diffs, sampling_params=sampling_params, is_thinking=is_thinking)
    
    def generate(
            self,
            prompts: Union[str, List[str]],
            sampling_params: "SamplingParams",
            images: Optional[List[Optional[List[str]]]] = None,
            n: int = 1,
            is_thinking: bool = False
        ) -> Dict[str, Optional[Union[str, List[Optional[Union[str, int, List[Union[int, Dict[str, Any]]]]]]]]]:
        num_prompts = 1 if isinstance(prompts, str) else len(prompts)
        num_outputs = num_prompts * n
        output = dict(
            indices = list(range(num_outputs)),
            texts = [''] * num_outputs,
            token_ids = [[]] * num_outputs,
            stop_strs = [None] * num_outputs,
            logprobs=None if not sampling_params.logprobs else ([[]] * num_outputs),
            input_tokens = None,
            output_tokens = [0] * num_outputs,
            finish_reasons = [None] * num_outputs
        )
        for gos in self.stream(prompts, sampling_params, images, n, is_thinking):
            output['texts'] = [t + go.token if not fr else t for t, fr, go in zip(output['texts'], output['finish_reasons'], gos)]
            output['token_ids'] = [t + [go.token_id] if not fr else t for t, fr, go in zip(output['token_ids'], output['finish_reasons'], gos)]
            output['stop_strs'] = [go.stop_str if not fr else s for s, fr, go in zip(output['stop_strs'], output['finish_reasons'], gos)]
            if sampling_params.logprobs:
                output['logprobs'] = [lp + [go.logprobs] if not fr else lp for lp, fr, go in zip(output['logprobs'], output['finish_reasons'], gos)]
            output['input_tokens'] = [go.input_tokens for go in gos] if output['input_tokens'] is None else output['input_tokens']
            output['output_tokens'] = [go.output_tokens if not fr else ot for ot, fr, go in zip(output['output_tokens'], output['finish_reasons'], gos)]
            output['finish_reasons'] = [go.finish_reason if not fr else fr for fr, go in zip(output['finish_reasons'], gos)]
        return output
            

        









        
        
    



