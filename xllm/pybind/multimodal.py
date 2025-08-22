from typing import Any, Dict, List, cast
from functools import lru_cache
from io import BytesIO
from PIL import Image
import torch

@lru_cache(maxsize=1)
def __cache_image_processor(
    processor_name: str,
    *args: Any,
    trust_remote_code: bool = False,
    **kwargs: Any,
):
    """Load an image processor for the given model name via HuggingFace."""
    # don't put this import at the top level
    # it will call torch.cuda.device_count()
    from transformers import AutoImageProcessor
    from transformers.image_processing_utils import BaseImageProcessor

    try:
        processor = AutoImageProcessor.from_pretrained(
            processor_name,
            *args,
            trust_remote_code=trust_remote_code,
            **kwargs)
    except ValueError as e:
        if not trust_remote_code:
            err_msg = (
                "Failed to load the image processor. If the image processor is "
                "a custom processor not yet available in the HuggingFace "
                "transformers library, consider setting "
                "`trust_remote_code=True` in LLM or using the "
                "`--trust-remote-code` flag in the CLI.")
            raise RuntimeError(err_msg) from e
        else:
            raise e

    return cast(BaseImageProcessor, processor)

def try_cat_feature(item):
    if isinstance(item, torch.Tensor):
        return item

    assert isinstance(item, list), f"expected list, got {type(item)}"
    lst = []

    for i in item:
        res = try_cat_feature(i)
        if isinstance(res, list):
            lst.extend(res)
        elif isinstance(res, torch.Tensor):
            lst.append(res)
        else:
            raise TypeError(f"expected list or torch.Tensor, got {type(res)}")

    if len(lst) == 1:
        return lst[0]
    if any(t.shape[1:] != lst[0].shape[1:] for t in lst):
        return lst

    return torch.cat(lst)

def preprocess(lst: List[str], model: str) -> Dict[str, Any]:
    
    images = [Image.open(BytesIO(item)) for item in lst]
    image_processor = __cache_image_processor(model, trust_remote_code=True)

    data = image_processor.preprocess(images, return_tensors="pt").data
    return { key: try_cat_feature(val)
            for key, val in data.items()
            }