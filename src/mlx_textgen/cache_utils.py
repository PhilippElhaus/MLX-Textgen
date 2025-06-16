from typing import List, Optional, Dict, Union, Tuple, Literal, Callable, TYPE_CHECKING
from pydantic import BaseModel, Field
if TYPE_CHECKING:
    from logging import Logger
    from mlx_lm.models.cache import KVCache, RotatingKVCache
    from mlx.core import array


def create_cache_dict(
        cache: List[Union["KVCache", "RotatingKVCache"]]
    ) -> Optional[Dict[str, "array"]]:
    """
    Converts a list of KVCache or RotatingKVCache objects into a dictionary of arrays suitable for saving.

    The dictionary keys are generated automatically based on the internal structure of the KVCache/RotatingKVCache objects.
    Only the cached portion of the keys and values are extracted when the cache offset is not zero.

    Args:
        cache (List[Union["KVCache", "RotatingKVCache"]]): A list of KVCache or RotatingKVCache objects.

    Returns:
        Optional[Dict[str, "array"]]: A dictionary containing the keys and values arrays from the cache, or None if the cache offset is 0.
    """
    from mlx.utils import tree_flatten
    offset = cache[0].offset
    if offset != 0:
        cache_dict = dict(tree_flatten([(c.keys[..., :c.offset, :], c.values[..., :c.offset, :]) for c in cache]))
        return cache_dict

def save_cache(
        cache: List[Union["KVCache", "RotatingKVCache"]], 
        file: str,
        model_name: str,
        metadata: Optional[Dict[str, str]] = None, 
        logger: Optional["Logger"] = None
    ) -> None:
    """
    Saves the provided KVCache or RotatingKVCache to a file in safetensors format.

    Args:
        cache (List[Union["KVCache", "RotatingKVCache"]]): A list of KVCache or RotatingKVCache objects to be saved.
        file (str): The file path where the cache will be saved (e.g., "path/to/cache.safetensors").
        model_name (str): The name of the model associated with this cache. This will be stored in the metadata.
        metadata (Optional[Dict[str, str]], optional): Optional metadata to be saved along with the cache. Defaults to None.
        logger (Optional["Logger"], optional): An optional logger object for logging progress and errors. Defaults to None.
    """
    from time import perf_counter
    from mlx.core import save_safetensors, clear_cache

    if logger:
        start = perf_counter()

    cache_dict = create_cache_dict(cache=cache)

    if cache_dict:
        metadata = metadata if metadata else {}
        metadata['model_name'] = model_name
        save_safetensors(file=file, arrays=cache_dict, metadata=metadata)

    del cache_dict
    clear_cache()

    if logger:
        end = perf_counter()
        logger.info(f'Save cache for model "{model_name}" to "{file}"; Time taken: {end - start:.3f}s.')

def load_cache(
        file: str, 
        cache: List[Union["KVCache", "RotatingKVCache"]], 
        logger: Optional["Logger"] = None
    ) -> Tuple[List[Union["KVCache", "RotatingKVCache"]], Dict[str, str]]:
    """
    Loads a KVCache or RotatingKVCache from a safetensors file.

    The function reads the safetensors file, extracts the cached keys and values, and updates the provided KVCache objects.
    It also returns any metadata stored within the file.

    Args:
        file (str): The path to the safetensors file containing the cache.
        cache (List[Union["KVCache", "RotatingKVCache"): A list of KVCache or RotatingKVCache objects to be updated with the loaded cache.
        logger (Optional["Logger"], optional): An optional logger object for logging progress and errors. Defaults to None.

    Returns:
        Tuple[List[Union["KVCache", "RotatingKVCache"]], Dict[str, str]]: A tuple containing the updated KVCache objects and a dictionary of metadata loaded from the file.
    """
    from time import perf_counter
    from mlx.core import load, eval, clear_cache
    from mlx.utils import tree_unflatten

    if logger:
        start = perf_counter()
    
    cache_dict, metadata = load(file, return_metadata=True)
    cache_list = tree_unflatten(list(cache_dict.items()))

    if len(cache_list) != len(cache):
        error = f'Cache file length and cache length mismatch.'

        if logger:
            logger.error(error)

        raise ValueError(error)
    
    elif not all(c.offset == 0 for c in cache):
        error = f'Provided list of caches are not empty.'

        if logger:
            logger.error(error)

        raise ValueError(error)   

    for i, (key, value) in enumerate(cache_list):
        cache[i].update_and_fetch(key, value)
        eval(cache[i].state)

    eval([c.state for c in cache])
    del cache_dict
    clear_cache()

    if logger:
        end = perf_counter()
        logger.info(f'Loaded cache from "{file}"; Time taken: {end - start:.3f}s.')
    
    return cache, metadata

def split_list_by_sublist(data_list: List[int], split_sublist: List[int]) -> List[List[int]]:
    """Splits a list into sublists, using a specified sublist as the delimiter.

    This function divides the input `data_list` into multiple sublists based on occurrences
    of the `split_sublist`. Each sublist contains the elements between the `split_sublist` occurrences.

    Args:
        data_list (List[int]): The list of integers to be split.
        split_sublist (List[int]): The sublist of integers used as the delimiter for splitting the `data_list`.

    Returns:
        List[List[int]]: A list of lists, where each inner list represents a sublist of the
                         original `data_list` delimited by the `split_sublist`.
    """
    result = []
    current_sublist = []
    i = 0
    n = len(data_list)
    m = len(split_sublist)

    while i < n:
        if i + m <= n and data_list[i:i + m] == split_sublist:
            result.append(current_sublist)
            current_sublist = []
            i += m  # Skip the splitting sublist
        else:
            current_sublist.append(data_list[i])
            i += 1

    result.append(current_sublist)

    return result

class CacheInfo(BaseModel):
    cache_id: str = Field(pattern='cache_\d+')
    token_ids: List[int]
    images: Optional[List[str]] = None
    last_modified: float

    @property
    def length(self) -> int:
        """Returns the length of the token_ids list.

        This is a cached property, so the length is only computed once.

        Returns:
            int: The number of tokens in the `token_ids` list.
        """
        if not hasattr(self, '_length'):
            self._length = len(self.token_ids)
        return self._length

class SeqIndex(BaseModel):
    token_ids: List[int]
    images: Optional[List[str]] = None
    index: int

    @property
    def length(self) -> int:
        """Returns the length of the token_ids list.

        This is a cached property, so the length is only computed once.

        Returns:
            int: The number of tokens in the `token_ids` list.
        """
        if not hasattr(self, '_length'):
            self._length = len(self.token_ids)
        return self._length

class CacheManager:
    """The CacheManager is responsible for storing and retrieving cached prompt completions to accelerate generation.
        It supports both text-only models and vision models (models that incorporate images into the prompt).

        The cache is stored in a directory named after the model under the mlx prompt cache directory
        (see `utils.get_prompt_cache_dir`).
        """
    def __init__(self, 
            model_name: str,
            is_vision: bool = False,
            image_token_id: Optional[int] = None, 
            num_tokens_per_image: Optional[int] = None,
            min_tokens: int = 20,
            max_reprocess_tokens: int = 250,
            replace_threshold: float = 0.95,
            max_capacity: int = 50,
            logger: Optional["Logger"] = None
            ) -> None:
        """Initializes the CacheManager with the specified parameters.

        The CacheManager is responsible for storing and retrieving cached prompt completions to accelerate generation.
        It supports both text-only models and vision models (models that incorporate images into the prompt).
        
        The cache is stored in a directory named after the model under the mlx prompt cache directory
        (see `utils.get_prompt_cache_dir`).

        Args:
            model_name (str): A unique identifier for the model using this cache manager.
                Used to create a dedicated cache directory within the mlx prompt cache directory.
            is_vision (bool, optional): Whether the model is a vision model (accepts images as input).
                Defaults to False.  If True, `image_token_id` and `num_tokens_per_image` must be set.
            image_token_id (Optional[int], optional): The token ID that represents an image in the model's
                vocabulary. Required if `is_vision` is True. Defaults to None.
            num_tokens_per_image (Optional[int], optional): The number of tokens used to represent each image.
                Required if `is_vision` is True. Defaults to None.
            min_tokens (int, optional): The minimum number of tokens a prompt must have to be considered
                for caching. Prompts shorter than this length will not be cached. Defaults to 20.
            max_reprocess_tokens (int, optional): When comparing sequences to determine if one sequence should be replaced by
                another cached sequence, the sequences need to share at least `replace_threshold` * `min(len(seq1), len(seq2))` tokens. This threshold is raised
                if the minimum length is low to `max(replace_threshold, (min_len - max_reprocess_tokens) / min_len)`. Defaults to 250.
            replace_threshold (float, optional): A similarity threshold (between 0 and 1) used to determine
                whether to replace an existing cache entry with a new one. A higher value means the new prompt must be
                more similar to the existing one to trigger replacement. Defaults to 0.95.
            max_capacity (int, optional): The maximum number of prompts to store in the cache. When the cache is full,
                the least recently used (LRU) entries are evicted to make space for new ones. Defaults to 50.
            logger (Optional["Logger"], optional): An optional logger object for logging cache operations.
                Defaults to None.

        Raises:
            ValueError: If `num_tokens_per_image` is None when `is_vision` is True.
            ValueError: If `num_tokens_per_image` is not a positive integer when `is_vision` is True.
            ValueError: If `image_token_id` is None when `is_vision` is True.
            ValueError: If `min_tokens` is less than 0.
            ValueError: If `max_capacity` is less than 1.
            ValueError: If `max_reprocess_tokens` is less than `min_tokens`.
            ValueError: If `replace_threshold` is not between 0 and 1.
        """
        from .utils import get_prompt_cache_dir
        import os

        self._logger = logger
        self.cache_dir = os.path.join(get_prompt_cache_dir(), model_name)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_info_dir = os.path.join(self.cache_dir, 'cached_prompts.json')

        if is_vision and (num_tokens_per_image is None):
            error = '"num_tokens_per_images" cannot be None for vision model.'
            self.log(error, level='error')
            raise ValueError(error)

        if is_vision and (num_tokens_per_image < 1):
            error = '"num_tokens_per_images" needs to be a positive integer for vision model.'
            self.log(error, level='error')
            raise ValueError(error)
        
        if is_vision and (image_token_id is None):
            error = '"image_token_id" cannot be None for vision model.'
            self.log(error, level='error')
            raise ValueError(error)
        
        if min_tokens < 0:
            error = '"min_tokens" needs to be an integer larger than or equal to 0.'
            self.log(error, level='error')
            raise ValueError(error)
        
        if max_capacity < 1:
            error = '"max_capacity" needs to be larger than or equal to 1.'
            self.log(error, level='error')
            raise ValueError(error)
        
        if max_reprocess_tokens < min_tokens:
            error = '"max_reprocess_tokens" needs to be larger than or equal to "min_tokens".'
            self.log(error, level='error')
            raise ValueError(error)
        
        if (replace_threshold <= 0) or (replace_threshold > 1):
            error = '"replace_threshold" needs to be between 0 and 1.'
            self.log(error, level='error')
            raise ValueError(error)
        
        self.model_name = model_name
        self.is_vision = is_vision
        self.image_token_id = image_token_id
        self.num_tokens_per_image = num_tokens_per_image
        self.min_tokens = min_tokens
        self.max_reprocess_tokens = max_reprocess_tokens
        self.replace_threshold = replace_threshold
        self.max_capacity = max_capacity

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
    def cache_info(self) -> List[CacheInfo]:
        """
        Retrieves a list of CacheInfo objects representing the cached prompts.

        The list is sorted by the last modified timestamp in descending order,
        so the most recently used caches appear first.

        Returns:
            List[CacheInfo]: A list of CacheInfo objects, sorted by last_modified (most recent first).
        """
        import os
        import json
        if not hasattr(self, '_cache_info'):
            if os.path.exists(self.cache_info_dir):
                with open(self.cache_info_dir, 'r') as f:
                    cache_info = json.load(f)
            else:
                cache_info = {}
            
            self._cache_info = [CacheInfo(cache_id=k, **v) for k, v in cache_info.items()]
        
        self._cache_info.sort(key=lambda x: x.last_modified, reverse=True)
        return self._cache_info
    
    def get_new_cache_id(self, num: int = 1) -> List[str]:
        """
        Generates a list of new, unique cache IDs.

        This method finds the next available cache IDs, ensuring they don't conflict with existing IDs.
        If there are gaps in the existing IDs, it will fill those gaps first. Otherwise, it will increment from the maximum existing ID.

        Args:
            num (int): The number of new cache IDs to generate. Must be at least 1.

        Returns:
            List[str]: A list of new, unique cache IDs.
        """
        if num < 1:
            error = f'Trying to get {num} cache IDs. Must at least get 1 new cache ID.'
            self.log(error, level='error')
            raise ValueError(error)

        existing = [int(cf.cache_id.removeprefix('cache_')) for cf in self.cache_info]
        if len(existing) == 0:
            return [f'cache_{i}' for i in range(num)]
        
        max_id = max(existing)
        count = 0
        new = []
        for i in range(max_id + 1):
            if i not in existing:
                new.append(f'cache_{i}')
                count += 1
                if count == num:
                    break
        
        remain = num - count
        if remain > 0:
            for i in range(remain):
                new.append(f'cache_{max_id + 1 + i}')
        return new
    
    def save_cache_info(self) -> None:
        """
        Saves the current cache information to a JSON file.

        This method serializes the `cache_info` list into a dictionary and saves it as a JSON file.
        The JSON file is used to persist the cache metadata across sessions.
        """
        import json
        cach_info_dict = {
            cf.cache_id: (dict(token_ids=cf.token_ids, images=cf.images, last_modified=cf.last_modified) if self.is_vision else dict(token_ids=cf.token_ids, last_modified=cf.last_modified)) for cf in self.cache_info
            }
        with open(self.cache_info_dir, 'w') as f:
            json.dump(cach_info_dict, f, indent=4)

    def drop_cache_by_id(self, cache_ids: Union[str, List[str]]) -> None:
        """
        Removes the specified cache(s) from the cache manager.

        This method removes cache information from the `cache_info` list, saves the updated cache information,
        and deletes the corresponding cache files from disk.

        Args:
            cache_ids (Union[str, List[str]]): A single cache ID or a list of cache IDs to remove.
        """
        import time
        start = time.perf_counter()

        import os

        existing = [cf.cache_id for cf in self.cache_info]
        to_drop = [cache_ids] if isinstance(cache_ids, str) else cache_ids
        to_drop = [td for td in to_drop if td in existing]

        self._cache_info = [cf for cf in self._cache_info if cf.cache_id not in to_drop]
        self.save_cache_info()

        for cid in to_drop:
            os.remove(os.path.join(self.cache_dir, f'{cid}.safetensors'))

        to_drop_list = [f'"{td}"' for td in to_drop]
        to_drop_str = ', '.join(to_drop_list)
        
        end = time.perf_counter()
        self.log(f'Cache dropped for model "{self.model_name}": {to_drop_str}. Time taken: {end - start:.3f}s.')

    def split_cache(self, 
            cache: List[Union["KVCache", "RotatingKVCache"]], 
            create_cache_fn: Callable[[], List[Union["KVCache", "RotatingKVCache"]]],
            offsets: "array"
        ) -> List[List[Union["KVCache", "RotatingKVCache"]]]:
        """
        Splits the provided KV cache into multiple smaller caches based on sequence offsets.

        This function divides an existing KV cache into a list of new KV caches, one for each sequence
        in the batch. It uses provided offsets to determine where each sequence's cached data begins
        within the original cache.  A `create_cache_fn` is used to generate new, empty caches with
        the same structure as the input `cache`.

        If the original `cache` is empty (indicated by a zero offset), the function returns an empty list.

        Args:
            cache (List[Union["KVCache", "RotatingKVCache"]]): The KV cache to split.  This cache contains cached
                data for a batch of sequences.
            create_cache_fn (Callable[[], List[Union["KVCache", "RotatingKVCache"]]]): A function that creates a new,
                empty KV cache with the same structure (number of layers, dimensions) as the input `cache`. This is
                used to initialize the individual caches for each sequence.
            offsets ("array"): An array of integer offsets.  Each offset indicates the starting index of the cached
                data for a specific sequence within the original `cache`. The length of this array should match the
                batch size (number of sequences) represented by the cache.

        Returns:
            List[List[Union["KVCache", "RotatingKVCache"]]]: A list of new KV caches. Each element in the outer list
            represents a single sequence from the batch, and contains a list of KVCache/RotatingKVCache objects (one
            for each layer in the model).  Returns an empty list if the input `cache` is empty.

        Raises:
            ValueError: If the batch size of the input `cache` does not match the length of the `offsets` array.
        """
        from mlx.core import eval, clear_cache

        if cache[0].offset == 0:
            return []
        
        bsize = cache[0].keys.shape[0]
        if bsize != offsets.shape[0]:
            error = 'Number of token sequences and number of offsets mismatched.'
            self.log(error, level='error')
            raise ValueError(error)
        
        new_cache = [create_cache_fn() for i in range(bsize)]
        for i, nc in enumerate(new_cache):
            for j, l in enumerate(nc):
                c = cache[j]
                l.update_and_fetch(c.keys[i:(i+1), :, offsets[i].tolist():c.offset, :], c.values[i:(i+1), :, offsets[i].tolist():c.offset, :])
                eval(c.state)
        del cache
        clear_cache()
        return new_cache

    def search_cache_non_vision(self, token_ids: List[int], cache_info: Optional[List[CacheInfo]] = None) -> Optional[Tuple[str, int]]:
        """Searches the cache for a matching prefix of token IDs (non-vision model).

        This function searches the existing cached prompts for the longest matching prefix
        with the input `token_ids`.  The function prioritizes longer shared prefixes.
        If multiple caches have the same shared prefix length, the cache with the shortest
        total length is selected.

        Args:
            token_ids (List[int]): The list of token IDs to search for in the cache.
            cache_info (Optional[List[CacheInfo]]): Optional list of cache infos. If None, use the internal `cache_info`.

        Returns:
            Optional[Tuple[str, int]]: A tuple containing the cache ID of the best match and the
            length of the shared token ID prefix, or None if no suitable match is found.
        """
        cache_info = self.cache_info if cache_info is None else cache_info
        if len(cache_info) == 0:
            return
        
        from itertools import takewhile
        
        selected = None
        current_shared = 0
        slen = max([cf.length for cf in cache_info])

        for cf in cache_info:
            shared = sum([1 for _ in takewhile(lambda x: x[0] == x[1], zip(token_ids, cf.token_ids))])
            if shared > current_shared:
                selected = cf.cache_id
                current_shared = shared
                slen = cf.length

            elif (shared == current_shared) and (cf.length < slen) and (current_shared != 0):
                selected = cf.cache_id
                current_shared = shared
                slen = cf.length

        if selected and (current_shared == len(token_ids)): # Need to have at least one token for the model to process before generation.
            current_shared -= 1

        if selected and (current_shared > 0):
            return selected, current_shared
        
        else:
            return
        
    def search_cache_vision(self, token_ids: List[int], images: Optional[List[str]] = None) -> Optional[Tuple[str, int]]:
        """Searches the cache for a matching prefix of token IDs and images (vision model).

        This function attempts to find the best matching cache entry for a given input consisting of token IDs and,
        for vision models, a list of image identifiers.  It prioritizes longer shared prefixes of token IDs and
        matching image identifiers when image tokens are present in the shared prefix. If multiple cache entries have
        the same shared prefix length, the cache entry with the shortest total length is selected.

        When image tokens are present in the shared prefix of tokens, the function compares the provided image
        identifiers with the image identifiers stored in the cache entry.  A cache entry is considered a better match
        if it has a longer shared prefix of image identifiers.

        If no images are provided or if no cached prompts include images, the function falls back to searching based
        solely on the token IDs.

        Args:
            token_ids (List[int]): The list of token IDs to search for in the cache.
            images (Optional[List[str]], optional): The list of image identifiers (e.g., file paths or unique IDs)
                corresponding to any image tokens present in the `token_ids`. Defaults to None.

        Returns:
            Optional[Tuple[str, int]]: A tuple containing the cache ID of the best match and the length of the
                shared token ID prefix. Returns None if no suitable match is found. The length returned represents
                the number of matching tokens, including any image tokens that have corresponding image identifier
                matches.
        """
        if (not images) or (len(self.cache_info) == 0):
            return self.search_cache_non_vision(token_ids=token_ids)
        


        cache_info = self.cache_info
        from itertools import takewhile        
        img_seq = [self.image_token_id] * self.num_tokens_per_image

        selected = None
        current_shared = 0
        slen = max([cf.length for cf in cache_info])


        for cf in cache_info:
            shared = sum([1 for _ in takewhile(lambda x: x[0] == x[1], zip(token_ids, cf.token_ids))])
            if shared > current_shared:
                num_images_in_shared = len([t for t in token_ids[:shared] if t == self.image_token_id]) // self.num_tokens_per_image
                if num_images_in_shared == 0:
                    selected = cf.cache_id
                    current_shared = shared
                    slen = cf.length

                else: # Compare the images
                    oimages = images[:num_images_in_shared]
                    cimages = cf.images[:num_images_in_shared]
                    shared_images = sum([1 for _ in takewhile(lambda x: x[0] == x[1], zip(oimages, cimages))])
                    if len(oimages) == shared_images:
                        selected = cf.cache_id
                        current_shared = shared
                        slen = cf.length
                    else:
                        text_seqs = split_list_by_sublist(token_ids[:shared], img_seq)
                        shared_seq = text_seqs[0]
                        for i in range(shared_images):
                            shared_seq += img_seq + text_seqs[i + 1]
                        shared = len(shared_seq)
                        if shared > current_shared:
                            selected = cf.cache_id
                            current_shared = shared
                            slen = cf.length

            elif (shared == current_shared) and (cf.length < slen) and (current_shared != 0):
                num_images_in_shared = len([t for t in token_ids[:shared] if t == self.image_token_id]) // self.num_tokens_per_image
                if num_images_in_shared == 0:
                    selected = cf.cache_id
                    current_shared = shared
                    slen = cf.length

                else: # Compare the images
                    oimages = images[:num_images_in_shared]
                    cimages = cf.images[:num_images_in_shared]
                    shared_images = sum([1 for _ in takewhile(lambda x: x[0] == x[1], zip(oimages, cimages))])
                    if len(oimages) == shared_images:
                        selected = cf.cache_id
                        current_shared = shared
                        
                        slen = cf.length
        # Need to have at least one token for the model to process before generation.
        if selected and (current_shared == len(token_ids)):
            if token_ids[current_shared - 1] != self.image_token_id:
                current_shared -= 1
            else: # Need to remove the entire image
                current_shared -= self.num_tokens_per_image

        if selected and (current_shared > 0):
            return selected, current_shared
        
        else:
            return

    def search_cache(self, token_ids: List[int], images: Optional[List[str]] = None) -> Optional[Tuple[str, int]]:
        """Searches the cache for a matching prompt.

        This method intelligently searches the cache, using either `search_cache_vision` for
        vision-enabled models (models that accept images) or `search_cache_non_vision` for text-only
        models. The appropriate search function is called based on the `is_vision` attribute
        of the `CacheManager`.

        Args:
            token_ids (List[int]): The sequence of token IDs representing the prompt.
            images (Optional[List[str]], optional):  A list of image identifiers (e.g., file paths)
                associated with the prompt. Required only for vision models. Defaults to None.

        Returns:
            Optional[Tuple[str, int]]: A tuple containing the cache ID of the best matching
                prompt and the length of the shared token prefix (the number of matching tokens).
                Returns None if no suitable match is found in the cache.
        """
        if self.is_vision:
            return self.search_cache_vision(token_ids=token_ids, images=images)
        else:
            return self.search_cache_non_vision(token_ids=token_ids)

    def get_cache(self, 
            create_cache_fn: Callable[[], List[Union["KVCache", "RotatingKVCache"]]],
            token_ids: "array",
            offsets: "array",
            images: Optional[List[Optional[List[str]]]] = None
        ) -> List[Union["KVCache", "RotatingKVCache"]]:
        """Retrieves and pre-fills the KV cache based on existing cached prompts, maximizing reuse for accelerated generation.

        This function searches the cache for the longest matching prefixes of the given token ID sequences
        (and associated images, if applicable). If matches are found, the corresponding KV cache states are
        loaded and used to pre-fill the provided `cache`. The function aims to leverage existing cached
        computations to avoid redundant processing.

        Args:
            create_cache_fn (Callable[[], List[Union["KVCache", "RotatingKVCache"]]]): A function that creates a new,
                empty KV cache with the same structure (layers, dimensions) as required by the model.  This
                function is used to initialize the cache if no suitable cached prompts are found, or to construct
                intermediate caches during the pre-filling process.
            token_ids ("array"): A 2D array of token IDs representing the input prompts. Each row corresponds to a
                separate prompt sequence.
            offsets ("array"): A 1D array of integer offsets. Each offset specifies the starting position of a prompt
                sequence within the `token_ids` array. This enables the function to handle batched inputs efficiently.
            images (Optional[List[Optional[List[str]]]], optional): A list of lists containing image identifiers for each
                prompt sequence. This argument is only relevant for vision models and should be set to None for
                text-only models. If a prompt contains images, the corresponding inner list should contain the
                identifiers (e.g., file paths) of those images. Defaults to None.

        Returns:
            List[Union["KVCache", "RotatingKVCache"]]: A list of KVCache or RotatingKVCache objects representing the
            pre-filled KV cache. If suitable cached prompts are found, the cache will be partially filled with
            the loaded states. If no matches are found, the function returns a newly created, empty cache.
        Raises:
            ValueError: If the number of token ID sequences in `token_ids` does not match the number of offsets
                provided in the `offsets` array. This indicates an inconsistency in the input data.
        """
        import time
        
        start = time.perf_counter()

        if token_ids.shape[0] != offsets.shape[0]:
            error = 'Number of token sequences and number of offsets mistmatch.'
            self.log(error, level='error')
            raise ValueError(error)
        
        if token_ids.shape[1] < self.min_tokens:
            cache = create_cache_fn()
            end = time.perf_counter()
            self.log(f'Existing cache not required as the prompts have fewer than {self.min_tokens} tokens. Time taken: {end - start:.3f}s.')
            return cache

        offset_list = offsets.tolist()
        token_seqs = [t[o:] for t, o in zip(token_ids.tolist(), offset_list)]
        search_results = [self.search_cache(tids, images[i]) for i, tids in enumerate(token_seqs)]
        coverage = [0 if sr is None else sr[1] + o for sr, o in zip(search_results, offset_list)]

        cache = create_cache_fn()
        min_coverage = min(coverage)

        if min_coverage == 0:
            end = time.perf_counter()
            self.log(f'No suitable cache found. Time taken: {end - start:.3f}s.')
            return cache
        
        import os
        from datetime import datetime
        from mlx.core import load, eval, clear_cache, zeros
        from mlx.utils import tree_unflatten
        
        search_results = [None if ((sr is None) or (o >= min_coverage)) else sr for sr, o in zip(search_results, offset_list)]
        cache_to_load = list(set([sr[0] for sr in search_results if sr]))
        cache_files = [os.path.join(self.cache_dir, f'{cid}.safetensors') for cid in cache_to_load]
        cache_dict = {}
        for cid, cf in zip(cache_to_load, cache_files):
            cd, metadata = load(cf, return_metadata=True)
            cache_dict[cid] = tree_unflatten(list(cd.items()))

        for i, c in enumerate(cache):
            shape = cache_dict[cid][i][0].shape
            kv_heads = shape[1]
            emb_size = shape[3]
            key_dtype = cache_dict[cid][i][0].dtype
            value_dtype = cache_dict[cid][i][1].dtype
            keys = zeros(shape=(token_ids.shape[0], kv_heads, min_coverage, emb_size), dtype=key_dtype)
            values = zeros(shape=(token_ids.shape[0], kv_heads, min_coverage, emb_size), dtype=value_dtype)

            for j, sr in enumerate(search_results):
                if sr is None:
                    continue
                else:
                    keys[j, :, offset_list[j]:, :] = cache_dict[sr[0]][i][0][0, :, :(min_coverage - offset_list[j]), :]
                    values[j, :, offset_list[j]:, :] = cache_dict[sr[0]][i][1][0, :, :(min_coverage - offset_list[j]), :]

            c.update_and_fetch(keys, values)
            eval(c.state)

        del cd, cache_dict
        clear_cache()

        ts = datetime.now().timestamp()
        for cid in cache_to_load:
            [sr for sr in self.cache_info if sr.cache_id == cid][0].last_modified = ts
        self.save_cache_info()
        
        cache_id_str = ', '.join([f'"{cid}"' for cid in cache_to_load])
        end = time.perf_counter()
        self.log(f'Reusing cache {cache_id_str}. {min_coverage} tokens for each prompt prefilled. Time taken: {end - start:.3f}s.')
        return cache
    
    def find_seq_to_keep_drop_update(self, token_ids: List[List[int]], images: Optional[List[Optional[List[str]]]] = None) -> Tuple[List[int], List[str], List[str]]:
        """Identifies sequences to keep, drop, or update in the cache.

        This function compares newly generated token ID sequences with existing cached sequences to determine
        whether each new sequence should be kept (added to the cache), dropped (ignored), or used to update
        an existing cache entry. The comparison is based on the length of shared token prefixes and, for vision
        models, the matching of image identifiers.
        Args:
            token_ids (List[List[int]]): A list of token ID sequences representing the newly generated prompts.
            images (Optional[List[Optional[List[str]]]], optional): A list of lists of image identifiers.
                Each inner list corresponds to the images associated with a token sequence (or None if there are no
                images for that sequence). This is only required for vision models. Defaults to None.

        Returns:
            Tuple[List[int], List[str], List[str]]: A tuple containing three lists:
                - keep (List[int]): A list of indices (corresponding to the input `token_ids` list) of the sequences
                  that should be added as new entries to the cache.
                - drop (List[str]): A list of cache IDs of existing cache entries that should be removed from the cache.
                - update (List[str]): A list of cache IDs of existing cache entries that should be updated with the
                  newly generated sequences.
        """
        import time
        start = time.perf_counter()

        from itertools import takewhile

        images = [None] * len(token_ids) if images == None else images
        if len(images) != len(token_ids):
            error = 'Number of image list and number of token sequences mismatch.'
            self.log(error, level='error')
            raise ValueError(error)
        seqs = [SeqIndex(token_ids=tids, images=images[i], index=i) for i, tids in enumerate(token_ids)]
        seqs = [s for s in seqs if s.length >= self.min_tokens]
        
        if len(seqs) == 0:
            end = time.perf_counter()
            self.log(f'Found no new cache to save. Time taken: {end - start:.3f}s.')
            return []

        # self comparing
        img_seq = [self.image_token_id] * self.num_tokens_per_image if self.is_vision else []
        to_process: List[SeqIndex] = [s for s in seqs]
        cont_seqs: List[SeqIndex] = []
        done_seqs = []
        while len(to_process) != 0:
            seq = to_process[0]
            comp = [s for s in to_process if s.index != seq.index]

            strong_seqs = []
            num_images = 0 if (not seq.images) or (not self.is_vision) else len(seq.images)

            for cseq in comp:
                num_share = sum([1 for _ in takewhile(lambda x: x[0] == x[1], zip(seq.token_ids, cseq.token_ids))])
                slen = seq.length
                clen = cseq.length
                if num_images:
                    num_images_in_share = len([t for t in seq.token_ids[:num_share] if t == self.image_token_id]) // self.num_tokens_per_image
                    if cseq.images:
                        num_share_images = sum([1 for _ in takewhile(lambda x: x[0] == x[1], zip(seq.images[:num_images_in_share], cseq.images[:num_images_in_share]))])
                    else:
                        num_share_images = 0
                    if num_share_images != num_images_in_share:
                        text_seqs = split_list_by_sublist(seq.token_ids[:num_share], img_seq)
                        shared_seq = text_seqs[0]
                        for i in range(num_share_images):
                            shared_seq += img_seq + text_seqs[i + 1]
                        num_share = len(shared_seq)
                min_len = min(slen, clen)
                threshold = max(self.replace_threshold, (min_len - self.max_reprocess_tokens) / min_len)
                if (clen > slen) and ((num_share / slen) > threshold):
                    strong_seqs.append(cseq)
                    break
                elif (slen >= clen) and ((num_share / clen) > threshold):
                    done_seqs.append(cseq)

            if not len(strong_seqs):
                cont_seqs.append(seq)

            done_seqs.append(seq)
            to_process = [s for s in to_process if s.index not in [cs.index for cs in done_seqs]]

        # Comparing to existing caches
        existing_cache = self.cache_info
        keep = []
        drop = []
        update = []
        if len(existing_cache) == 0:
            keep = [seq.index for seq in cont_seqs]
            end = time.perf_counter()
            self.log(f'Found {len(keep)} new caches to save. Time taken: {end - start:.3f}s.')
            return keep, drop, update
        
        for seq in cont_seqs:
            to_update = None
            to_update_len = 0
            num_images = 0 if (not seq.images) or (not self.is_vision) else len(seq.images)

            for cseq in existing_cache:
                num_share = sum([1 for _ in takewhile(lambda x: x[0] == x[1], zip(seq.token_ids, cseq.token_ids))])
                slen = seq.length
                clen = cseq.length
                if num_images:
                    num_images_in_share = len([t for t in seq.token_ids[:num_share] if t == self.image_token_id]) // self.num_tokens_per_image
                    if cseq.images:
                        num_share_images = sum([1 for _ in takewhile(lambda x: x[0] == x[1], zip(seq.images[:num_images_in_share], cseq.images[:num_images_in_share]))])
                    else:
                        num_share_images = 0
                    if num_share_images != num_images_in_share:
                        text_seqs = split_list_by_sublist(seq.token_ids[:num_share], img_seq)
                        shared_seq = text_seqs[0]
                        for i in range(num_share_images):
                            shared_seq += img_seq + text_seqs[i + 1]
                        num_share = len(shared_seq)
                min_len = min(slen, clen)
                threshold = max(self.replace_threshold, (min_len - self.max_reprocess_tokens) / min_len)
                if (clen >= slen) and ((num_share / slen) > threshold):
                    if cseq.length > to_update_len:
                        to_update = cseq.cache_id
                        to_update_len = cseq.length
                elif (slen > clen) and ((num_share / clen) > threshold):
                    drop.append(cseq.cache_id)

            if to_update:
                update.append(to_update)
            else:
                keep.append(seq.index)

        drop = list(set(drop))
        update = list(set(update))
        end = time.perf_counter()
        self.log(f'Found {len(keep)} new caches to save. Time taken: {end - start:.3f}s.')
        return keep, drop, update

    def save_cache(self,
            cache: List[Union["KVCache", "RotatingKVCache"]],
            token_ids: "array",
            offsets: "array",
            create_cache_fn: Callable[[], List[Union["KVCache", "RotatingKVCache"]]],
            images: Optional[List[Optional[List[str]]]] = None
        ) -> None:
        """Saves the provided KV cache to disk for later reuse.

        This method takes a KV cache, token IDs, and offsets, and saves the relevant portions of the
        cache to disk. It splits the input KV cache into smaller caches, one for each sequence in the
        batch, and then determines which of those sequences should be saved to the cache based on factors
        such as minimum token length and similarity to existing cached prompts.  The function then saves
        the selected caches to disk as safetensors files, along with metadata about the cached prompts.
        It also manages the cache's capacity, dropping least recently used (LRU) entries if necessary,
        and updating cache metadata.

        Args:
            cache (List[Union["KVCache", "RotatingKVCache"]]): The KV cache to save. This cache contains cached
                key/value states for a batch of sequences.
            token_ids ("array"): A 2D array of token IDs. Each row represents a sequence in the batch.  Used
                to identify which sequences should be cached and to create cache metadata.
            offsets ("array"): A 1D array of integer offsets. Each offset indicates the starting index of the
                corresponding sequence within the `token_ids` array. Used for batched inputs.
            create_cache_fn (Callable[[], List[Union["KVCache", "RotatingKVCache"]]]): A function that creates a new,
                empty KV cache with the same structure (number of layers, dimensions) as the input `cache`. This is
                used when splitting the original cache into smaller caches for individual sequences.
            images (Optional[List[Optional[List[str]]]], optional): A list of lists of image identifiers.
                Only required for vision models. If the model uses images, each inner list corresponds to the
                images associated with a token sequence (or None if there are no images for that sequence).
                Defaults to None.

        Raises:
            ValueError: If the number of tokens in the cache does not match the length of the corresponding
                token ID sequence.
            ValueError: If the number of caches does not match the number of token ID lists.
            ValueError: If the number of caches does not match the number of offsets.
        """
        import time
        start = time.perf_counter()

        from datetime import datetime
        from mlx.core import clear_cache
        import os

        if cache[0].state is None:
            end = time.perf_counter()
            self.log(f'No cache saved. Time taken: {end - start:.3f}s.')
            return

        B, kv_heads, num_tokens, embed_size = cache[0].keys[..., :cache[0].offset, :].shape

        if num_tokens != token_ids.shape[1]:
            error = 'Number of tokens and token ids mismatch while saving cache.'
            self.log(msg=error, level='error')
            raise ValueError(error)
        
        if B != token_ids.shape[0]:
            error = 'Number of cache and number of token id lists mismatch while saving cache.'
            self.log(msg=error, level='error')
            raise ValueError(error)
        
        if offsets.shape[0] != B:
            error = 'Number of cache and number of offsets mismatch while saving cache.'
            self.log(msg=error, level='error')
            raise ValueError(error)
        
        token_seqs = [t[o:] for t, o in zip(token_ids.tolist(), offsets.tolist())]
        caches = self.split_cache(cache=cache, create_cache_fn=create_cache_fn, offsets=offsets)
        seq_lens = [len(t) for t in token_seqs]
        token_seqs = [t for t, l in zip(token_seqs, seq_lens) if l >= self.min_tokens]
        caches = [c for c, l in zip(caches, seq_lens) if l >= self.min_tokens]

        if len(token_seqs) == 0:
            del caches
            clear_cache()
            end = time.perf_counter()
            self.log(f'No cache saved. Time taken: {end - start:.3f}s.')
            return

        keep, drop, update = self.find_seq_to_keep_drop_update(token_seqs, images=images)
        
        if update:
            ts = datetime.now().timestamp()
            for cid in update:
                [sr for sr in self.cache_info if sr.cache_id == cid][0].last_modified = ts
            self.save_cache_info()

        if keep:
            token_seqs = [ts for i, ts in enumerate(token_seqs) if i in keep]
            caches = [c for i, c in enumerate(caches) if i in keep]
            extra_to_drop = len(self.cache_info) + len(token_seqs) - len(drop) - self.max_capacity
            if extra_to_drop > 0:
                drop = drop + [cf.cache_id for cf in self.cache_info if cf.cache_id not in drop][-extra_to_drop:]
        
        if drop:
            self.drop_cache_by_id(drop)

        if keep:
            ts = datetime.now().timestamp()
            new_ids = self.get_new_cache_id(num=len(keep))
            new_cache_infos = [CacheInfo(cache_id=cid, token_ids=tids, images=images, last_modified=ts) for tids, cid in zip(token_seqs, new_ids)]
            new_files = [os.path.join(self.cache_dir, f'{cid}.safetensors') for cid in new_ids]
            for c, nf in zip(caches, new_files):
                save_cache(cache=c, file=nf, model_name=self.model_name, logger=self._logger)
            self._cache_info.extend(new_cache_infos)
            self.save_cache_info()

        del caches
        clear_cache()
        end = time.perf_counter()
        self.log(f'Save cache processed done. Total time taken: {end - start:.3f}s.')           

    def clear(self) -> None:
        cache_ids = [c.cache_id for c in self.cache_info]
        self.drop_cache_by_id(cache_ids=cache_ids)
