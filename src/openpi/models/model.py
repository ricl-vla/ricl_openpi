import abc
from collections.abc import Sequence
import dataclasses
import enum
import logging
import pathlib
from typing import Generic, TypeVar

import augmax
from flax import nnx
from flax import struct
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from openpi.shared import image_tools
import openpi.shared.array_typing as at

logger = logging.getLogger("openpi")

ArrayT = TypeVar("ArrayT", at.Array, jax.ShapeDtypeStruct)


class ModelType(enum.Enum):
    """Supported model types."""

    PI0 = "pi0"
    PI0_FAST = "pi0_fast"


# The model always expects these images
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)


# This may need change if we release a small model.
IMAGE_RESOLUTION = (224, 224)


# Data format
#
# Data transforms produce the model input as a nested dictionary which is later converted
# into `Obesrvation` and `Actions` objects. See below.
#
# In the dictory form, this data should look like:
# {
#     # Observation data.
#     "image": {
#         "base_0_rgb": (float32|uint8)[*b, h, w, 3],  # RGB image in [-1, 1] or [0, 255]
#         ...  # Additional camera views
#     },
#     "image_mask": {
#         "base_0_rgb": bool[*b],  # True if image is valid
#         ...  # Masks for additional views
#     },
#     "state": float32[*b, s],  # Low-dimensional robot state
#     "tokenized_prompt": int32[*b, l],  # Optional, tokenized language prompt
#     "tokenized_prompt_mask": bool[*b, l],  # Optional, mask for tokenized prompt
#     "token_ar_mask": int32[*b, l],  # Optional, autoregressive mask for FAST model
#     "token_loss_mask": bool[*b, l],  # Optional, loss mask for FAST model
#
#      # Actions data.
#      "actions": float32[*b ah ad]
# }
# where:
#   *b = batch dimensions
#   h,w = image height/width
#   s = state dimension
#   l = sequence length
#
@at.typecheck
@struct.dataclass
class Observation(Generic[ArrayT]):
    """Holds observations, i.e., inputs to the model.

    See `Observation.from_dict` to see the expected dictionary form. This is the format
    that should be produced by the data transforms.
    """

    # Images, in [-1, 1] float32.
    images: dict[str, at.Float[ArrayT, "*b h w c"]]
    # Image masks, with same keys as images.
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]
    # Low-dimensional robot state.
    state: at.Float[ArrayT, "*b s"]

    # Tokenized prompt.
    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None
    # Tokenized prompt mask.
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None

    # pi0-fast model specific fields.

    # Token auto-regressive mask (for FAST autoregressive model).
    token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    # Token loss mask (for FAST autoregressive model).
    token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":
        """This method defines the mapping between unstructured data (i.e., nested dict) to the structured Observation format."""
        # Ensure that tokenized_prompt and tokenized_prompt_mask are provided together.
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")
        # If images are uint8, convert them to [-1, 1] float32.
        for key in data["image"]:
            if data["image"][key].dtype == np.uint8:
                data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        """Convert the Observation to a nested dict."""
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        return result


@at.typecheck
@struct.dataclass
class ObservationPrefixPostfix(Generic[ArrayT]):
    """Holds observations, i.e., inputs to the model.

    See `Observation.from_dict` to see the expected dictionary form. This is the format
    that should be produced by the data transforms.
    """

    # Images, in [-1, 1] float32.
    images: dict[str, at.Float[ArrayT, "*b h w c"]]
    # Image masks, with same keys as images.
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]
    # Low-dimensional robot state.
    state: at.Float[ArrayT, "*b s"]

    # Tokenized prompt.
    tokenized_prompt_prefix: at.Int[ArrayT, "*b l_prefix"] | None = None
    tokenized_prompt_postfix: at.Int[ArrayT, "*b l_postfix"] | None = None
    # Tokenized prompt mask.
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None

    # pi0-fast model specific fields.

    # Token auto-regressive mask (for FAST autoregressive model).
    token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    # Token loss mask (for FAST autoregressive model).
    token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":
        """This method defines the mapping between unstructured data (i.e., nested dict) to the structured Observation format."""
        # Ensure that tokenized_prompt_prefix, tokenized_prompt_postfix, and tokenized_prompt_mask are provided together.
        if ("tokenized_prompt_prefix" in data) != ("tokenized_prompt_postfix" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt_prefix, tokenized_prompt_postfix, and tokenized_prompt_mask must be provided together.")
        # If images are uint8, convert them to [-1, 1] float32.
        for key in data["image"]:
            if data["image"][key].dtype == np.uint8:
                data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt_prefix=data.get("tokenized_prompt_prefix"),
            tokenized_prompt_postfix=data.get("tokenized_prompt_postfix"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        """Convert the Observation to a nested dict."""
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        return result
    

@at.typecheck
@struct.dataclass
class RiclObservation(Generic[ArrayT]):
    """Holds observations, i.e., inputs to the model.

    See `Observation.from_dict` to see the expected dictionary form. This is the format
    that should be produced by the data transforms.

    TODO: Make below brute forced (quick and dirty) class actually elegant.
    """

    # Images, in [-1, 1] float32.
    query_images: dict[str, at.Float[ArrayT, "*b h w c"]] | None = None
    retrieved_0_images: dict[str, at.Float[ArrayT, "*b h w c"]] | None = None
    retrieved_1_images: dict[str, at.Float[ArrayT, "*b h w c"]] | None = None
    retrieved_2_images: dict[str, at.Float[ArrayT, "*b h w c"]] | None = None
    retrieved_3_images: dict[str, at.Float[ArrayT, "*b h w c"]] | None = None
    retrieved_4_images: dict[str, at.Float[ArrayT, "*b h w c"]] | None = None
    retrieved_5_images: dict[str, at.Float[ArrayT, "*b h w c"]] | None = None
    retrieved_6_images: dict[str, at.Float[ArrayT, "*b h w c"]] | None = None
    retrieved_7_images: dict[str, at.Float[ArrayT, "*b h w c"]] | None = None
    retrieved_8_images: dict[str, at.Float[ArrayT, "*b h w c"]] | None = None
    retrieved_9_images: dict[str, at.Float[ArrayT, "*b h w c"]] | None = None
    retrieved_10_images: dict[str, at.Float[ArrayT, "*b h w c"]] | None = None
    retrieved_11_images: dict[str, at.Float[ArrayT, "*b h w c"]] | None = None
    retrieved_12_images: dict[str, at.Float[ArrayT, "*b h w c"]] | None = None
    retrieved_13_images: dict[str, at.Float[ArrayT, "*b h w c"]] | None = None
    retrieved_14_images: dict[str, at.Float[ArrayT, "*b h w c"]] | None = None
    retrieved_15_images: dict[str, at.Float[ArrayT, "*b h w c"]] | None = None
    retrieved_16_images: dict[str, at.Float[ArrayT, "*b h w c"]] | None = None
    retrieved_17_images: dict[str, at.Float[ArrayT, "*b h w c"]] | None = None
    retrieved_18_images: dict[str, at.Float[ArrayT, "*b h w c"]] | None = None
    retrieved_19_images: dict[str, at.Float[ArrayT, "*b h w c"]] | None = None
    # Image masks, with same keys as images.
    query_image_masks: dict[str, at.Bool[ArrayT, "*b"]] | None = None
    retrieved_0_image_masks: dict[str, at.Bool[ArrayT, "*b"]] | None = None
    retrieved_1_image_masks: dict[str, at.Bool[ArrayT, "*b"]] | None = None
    retrieved_2_image_masks: dict[str, at.Bool[ArrayT, "*b"]] | None = None
    retrieved_3_image_masks: dict[str, at.Bool[ArrayT, "*b"]] | None = None
    retrieved_4_image_masks: dict[str, at.Bool[ArrayT, "*b"]] | None = None
    retrieved_5_image_masks: dict[str, at.Bool[ArrayT, "*b"]] | None = None
    retrieved_6_image_masks: dict[str, at.Bool[ArrayT, "*b"]] | None = None
    retrieved_7_image_masks: dict[str, at.Bool[ArrayT, "*b"]] | None = None
    retrieved_8_image_masks: dict[str, at.Bool[ArrayT, "*b"]] | None = None
    retrieved_9_image_masks: dict[str, at.Bool[ArrayT, "*b"]] | None = None
    retrieved_10_image_masks: dict[str, at.Bool[ArrayT, "*b"]] | None = None
    retrieved_11_image_masks: dict[str, at.Bool[ArrayT, "*b"]] | None = None
    retrieved_12_image_masks: dict[str, at.Bool[ArrayT, "*b"]] | None = None
    retrieved_13_image_masks: dict[str, at.Bool[ArrayT, "*b"]] | None = None
    retrieved_14_image_masks: dict[str, at.Bool[ArrayT, "*b"]] | None = None
    retrieved_15_image_masks: dict[str, at.Bool[ArrayT, "*b"]] | None = None
    retrieved_16_image_masks: dict[str, at.Bool[ArrayT, "*b"]] | None = None
    retrieved_17_image_masks: dict[str, at.Bool[ArrayT, "*b"]] | None = None
    retrieved_18_image_masks: dict[str, at.Bool[ArrayT, "*b"]] | None = None
    retrieved_19_image_masks: dict[str, at.Bool[ArrayT, "*b"]] | None = None
    # Distances between embeddings
    exp_lamda_distances: at.Float[ArrayT, "*b num_observations 1"] | None = None
    # Low-dimensional robot state.
    query_state: at.Float[ArrayT, "*b s"] | None = None
    retrieved_0_state: at.Float[ArrayT, "*b s"] | None = None
    retrieved_1_state: at.Float[ArrayT, "*b s"] | None = None
    retrieved_2_state: at.Float[ArrayT, "*b s"] | None = None
    retrieved_3_state: at.Float[ArrayT, "*b s"] | None = None
    retrieved_4_state: at.Float[ArrayT, "*b s"] | None = None
    retrieved_5_state: at.Float[ArrayT, "*b s"] | None = None
    retrieved_6_state: at.Float[ArrayT, "*b s"] | None = None
    retrieved_7_state: at.Float[ArrayT, "*b s"] | None = None
    retrieved_8_state: at.Float[ArrayT, "*b s"] | None = None
    retrieved_9_state: at.Float[ArrayT, "*b s"] | None = None
    retrieved_10_state: at.Float[ArrayT, "*b s"] | None = None
    retrieved_11_state: at.Float[ArrayT, "*b s"] | None = None
    retrieved_12_state: at.Float[ArrayT, "*b s"] | None = None
    retrieved_13_state: at.Float[ArrayT, "*b s"] | None = None
    retrieved_14_state: at.Float[ArrayT, "*b s"] | None = None
    retrieved_15_state: at.Float[ArrayT, "*b s"] | None = None
    retrieved_16_state: at.Float[ArrayT, "*b s"] | None = None
    retrieved_17_state: at.Float[ArrayT, "*b s"] | None = None
    retrieved_18_state: at.Float[ArrayT, "*b s"] | None = None
    retrieved_19_state: at.Float[ArrayT, "*b s"] | None = None

    # Tokenized prompt prefix
    query_tokenized_prompt_prefix: at.Int[ArrayT, "*b l_query_prefix"] | None = None
    retrieved_0_tokenized_prompt_prefix: at.Int[ArrayT, "*b l_prefix"] | None = None
    retrieved_1_tokenized_prompt_prefix: at.Int[ArrayT, "*b l_prefix"] | None = None
    retrieved_2_tokenized_prompt_prefix: at.Int[ArrayT, "*b l_prefix"] | None = None
    retrieved_3_tokenized_prompt_prefix: at.Int[ArrayT, "*b l_prefix"] | None = None
    retrieved_4_tokenized_prompt_prefix: at.Int[ArrayT, "*b l_prefix"] | None = None
    retrieved_5_tokenized_prompt_prefix: at.Int[ArrayT, "*b l_prefix"] | None = None
    retrieved_6_tokenized_prompt_prefix: at.Int[ArrayT, "*b l_prefix"] | None = None
    retrieved_7_tokenized_prompt_prefix: at.Int[ArrayT, "*b l_prefix"] | None = None
    retrieved_8_tokenized_prompt_prefix: at.Int[ArrayT, "*b l_prefix"] | None = None
    retrieved_9_tokenized_prompt_prefix: at.Int[ArrayT, "*b l_prefix"] | None = None
    retrieved_10_tokenized_prompt_prefix: at.Int[ArrayT, "*b l_prefix"] | None = None
    retrieved_11_tokenized_prompt_prefix: at.Int[ArrayT, "*b l_prefix"] | None = None
    retrieved_12_tokenized_prompt_prefix: at.Int[ArrayT, "*b l_prefix"] | None = None
    retrieved_13_tokenized_prompt_prefix: at.Int[ArrayT, "*b l_prefix"] | None = None
    retrieved_14_tokenized_prompt_prefix: at.Int[ArrayT, "*b l_prefix"] | None = None
    retrieved_15_tokenized_prompt_prefix: at.Int[ArrayT, "*b l_prefix"] | None = None
    retrieved_16_tokenized_prompt_prefix: at.Int[ArrayT, "*b l_prefix"] | None = None
    retrieved_17_tokenized_prompt_prefix: at.Int[ArrayT, "*b l_prefix"] | None = None
    retrieved_18_tokenized_prompt_prefix: at.Int[ArrayT, "*b l_prefix"] | None = None
    retrieved_19_tokenized_prompt_prefix: at.Int[ArrayT, "*b l_prefix"] | None = None
    # Tokenized prompt postfix
    query_tokenized_prompt_postfix: at.Int[ArrayT, "*b l_query_postfix"] | None = None
    retrieved_0_tokenized_prompt_postfix: at.Int[ArrayT, "*b l_postfix"] | None = None
    retrieved_1_tokenized_prompt_postfix: at.Int[ArrayT, "*b l_postfix"] | None = None
    retrieved_2_tokenized_prompt_postfix: at.Int[ArrayT, "*b l_postfix"] | None = None
    retrieved_3_tokenized_prompt_postfix: at.Int[ArrayT, "*b l_postfix"] | None = None
    retrieved_4_tokenized_prompt_postfix: at.Int[ArrayT, "*b l_postfix"] | None = None
    retrieved_5_tokenized_prompt_postfix: at.Int[ArrayT, "*b l_postfix"] | None = None
    retrieved_6_tokenized_prompt_postfix: at.Int[ArrayT, "*b l_postfix"] | None = None
    retrieved_7_tokenized_prompt_postfix: at.Int[ArrayT, "*b l_postfix"] | None = None
    retrieved_8_tokenized_prompt_postfix: at.Int[ArrayT, "*b l_postfix"] | None = None
    retrieved_9_tokenized_prompt_postfix: at.Int[ArrayT, "*b l_postfix"] | None = None
    retrieved_10_tokenized_prompt_postfix: at.Int[ArrayT, "*b l_postfix"] | None = None
    retrieved_11_tokenized_prompt_postfix: at.Int[ArrayT, "*b l_postfix"] | None = None
    retrieved_12_tokenized_prompt_postfix: at.Int[ArrayT, "*b l_postfix"] | None = None
    retrieved_13_tokenized_prompt_postfix: at.Int[ArrayT, "*b l_postfix"] | None = None
    retrieved_14_tokenized_prompt_postfix: at.Int[ArrayT, "*b l_postfix"] | None = None
    retrieved_15_tokenized_prompt_postfix: at.Int[ArrayT, "*b l_postfix"] | None = None
    retrieved_16_tokenized_prompt_postfix: at.Int[ArrayT, "*b l_postfix"] | None = None
    retrieved_17_tokenized_prompt_postfix: at.Int[ArrayT, "*b l_postfix"] | None = None
    retrieved_18_tokenized_prompt_postfix: at.Int[ArrayT, "*b l_postfix"] | None = None
    retrieved_19_tokenized_prompt_postfix: at.Int[ArrayT, "*b l_postfix"] | None = None
    # Tokenized prompt mask.
    query_tokenized_prompt_mask: at.Bool[ArrayT, "*b l_query"] | None = None
    retrieved_0_tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None
    retrieved_1_tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None
    retrieved_2_tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None
    retrieved_3_tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None
    retrieved_4_tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None
    retrieved_5_tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None
    retrieved_6_tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None
    retrieved_7_tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None
    retrieved_8_tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None
    retrieved_9_tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None
    retrieved_10_tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None
    retrieved_11_tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None
    retrieved_12_tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None
    retrieved_13_tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None
    retrieved_14_tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None
    retrieved_15_tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None
    retrieved_16_tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None
    retrieved_17_tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None
    retrieved_18_tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None
    retrieved_19_tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None

    # pi0-fast model specific fields.

    # Token auto-regressive mask (for FAST autoregressive model).
    query_token_ar_mask: at.Int[ArrayT, "*b l_query"] | None = None
    retrieved_0_token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    retrieved_1_token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    retrieved_2_token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    retrieved_3_token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    retrieved_4_token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    retrieved_5_token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    retrieved_6_token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    retrieved_7_token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    retrieved_8_token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    retrieved_9_token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    retrieved_10_token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    retrieved_11_token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    retrieved_12_token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    retrieved_13_token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    retrieved_14_token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    retrieved_15_token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    retrieved_16_token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    retrieved_17_token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    retrieved_18_token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    retrieved_19_token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    # Token loss mask (for FAST autoregressive model).
    query_token_loss_mask: at.Bool[ArrayT, "*b l_query"] | None = None
    retrieved_0_token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None
    retrieved_1_token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None
    retrieved_2_token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None
    retrieved_3_token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None
    retrieved_4_token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None
    retrieved_5_token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None
    retrieved_6_token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None
    retrieved_7_token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None
    retrieved_8_token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None
    retrieved_9_token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None
    retrieved_10_token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None
    retrieved_11_token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None
    retrieved_12_token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None
    retrieved_13_token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None
    retrieved_14_token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None
    retrieved_15_token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None
    retrieved_16_token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None
    retrieved_17_token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None
    retrieved_18_token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None
    retrieved_19_token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None
    
    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT], num_retrieved_observations: int) -> "RiclObservation[ArrayT]":
        """This method defines the mapping between unstructured data (i.e., nested dict) to the structured Observation format."""
        assert num_retrieved_observations <= 20, f"TODO: fix this brute force code"
        # Idenify all prefix
        all_prefix = [f"retrieved_{i}_" for i in range(num_retrieved_observations)] + ["query_"]
        # Ensure that tokenized_prompt_prefix, tokenized_prompt_postfix, and tokenized_prompt_mask are provided together.
        for prefix in all_prefix:
            if (f"{prefix}tokenized_prompt_prefix" in data) != (f"{prefix}tokenized_prompt_postfix" in data) != (f"{prefix}tokenized_prompt_mask" in data):
                raise ValueError(f"{prefix}tokenized_prompt_prefix, {prefix}tokenized_prompt_postfix, and {prefix}tokenized_prompt_mask must be provided together.")
        # If images are uint8, convert them to [-1, 1] float32.
        image_keys = list(data[f"query_image"].keys())
        for prefix in all_prefix:
            for key in image_keys:
                if data[f"{prefix}image"][key].dtype == np.uint8:
                    data[f"{prefix}image"][key] = data[f"{prefix}image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
        return cls(
            query_images=data.get("query_image"),
            retrieved_0_images=data.get("retrieved_0_image"), 
            retrieved_1_images=data.get("retrieved_1_image"), 
            retrieved_2_images=data.get("retrieved_2_image"), 
            retrieved_3_images=data.get("retrieved_3_image"), 
            retrieved_4_images=data.get("retrieved_4_image"), 
            retrieved_5_images=data.get("retrieved_5_image"), 
            retrieved_6_images=data.get("retrieved_6_image"), 
            retrieved_7_images=data.get("retrieved_7_image"), 
            retrieved_8_images=data.get("retrieved_8_image"), 
            retrieved_9_images=data.get("retrieved_9_image"), 
            retrieved_10_images=data.get("retrieved_10_image"), 
            retrieved_11_images=data.get("retrieved_11_image"), 
            retrieved_12_images=data.get("retrieved_12_image"), 
            retrieved_13_images=data.get("retrieved_13_image"), 
            retrieved_14_images=data.get("retrieved_14_image"), 
            retrieved_15_images=data.get("retrieved_15_image"), 
            retrieved_16_images=data.get("retrieved_16_image"), 
            retrieved_17_images=data.get("retrieved_17_image"), 
            retrieved_18_images=data.get("retrieved_18_image"), 
            retrieved_19_images=data.get("retrieved_19_image"),
            query_image_masks=data.get("query_image_mask"),
            retrieved_0_image_masks=data.get("retrieved_0_image_mask"), 
            retrieved_1_image_masks=data.get("retrieved_1_image_mask"), 
            retrieved_2_image_masks=data.get("retrieved_2_image_mask"), 
            retrieved_3_image_masks=data.get("retrieved_3_image_mask"), 
            retrieved_4_image_masks=data.get("retrieved_4_image_mask"), 
            retrieved_5_image_masks=data.get("retrieved_5_image_mask"), 
            retrieved_6_image_masks=data.get("retrieved_6_image_mask"), 
            retrieved_7_image_masks=data.get("retrieved_7_image_mask"), 
            retrieved_8_image_masks=data.get("retrieved_8_image_mask"), 
            retrieved_9_image_masks=data.get("retrieved_9_image_mask"), 
            retrieved_10_image_masks=data.get("retrieved_10_image_mask"), 
            retrieved_11_image_masks=data.get("retrieved_11_image_mask"), 
            retrieved_12_image_masks=data.get("retrieved_12_image_mask"), 
            retrieved_13_image_masks=data.get("retrieved_13_image_mask"), 
            retrieved_14_image_masks=data.get("retrieved_14_image_mask"), 
            retrieved_15_image_masks=data.get("retrieved_15_image_mask"), 
            retrieved_16_image_masks=data.get("retrieved_16_image_mask"), 
            retrieved_17_image_masks=data.get("retrieved_17_image_mask"), 
            retrieved_18_image_masks=data.get("retrieved_18_image_mask"), 
            retrieved_19_image_masks=data.get("retrieved_19_image_mask"),
            exp_lamda_distances=data.get("exp_lamda_distances"),
            query_state=data.get("query_state"),
            retrieved_0_state=data.get("retrieved_0_state"), 
            retrieved_1_state=data.get("retrieved_1_state"), 
            retrieved_2_state=data.get("retrieved_2_state"), 
            retrieved_3_state=data.get("retrieved_3_state"), 
            retrieved_4_state=data.get("retrieved_4_state"), 
            retrieved_5_state=data.get("retrieved_5_state"), 
            retrieved_6_state=data.get("retrieved_6_state"), 
            retrieved_7_state=data.get("retrieved_7_state"), 
            retrieved_8_state=data.get("retrieved_8_state"), 
            retrieved_9_state=data.get("retrieved_9_state"), 
            retrieved_10_state=data.get("retrieved_10_state"), 
            retrieved_11_state=data.get("retrieved_11_state"), 
            retrieved_12_state=data.get("retrieved_12_state"), 
            retrieved_13_state=data.get("retrieved_13_state"), 
            retrieved_14_state=data.get("retrieved_14_state"), 
            retrieved_15_state=data.get("retrieved_15_state"), 
            retrieved_16_state=data.get("retrieved_16_state"), 
            retrieved_17_state=data.get("retrieved_17_state"), 
            retrieved_18_state=data.get("retrieved_18_state"), 
            retrieved_19_state=data.get("retrieved_19_state"),
            query_tokenized_prompt_prefix=data.get("query_tokenized_prompt_prefix"),
            retrieved_0_tokenized_prompt_prefix=data.get("retrieved_0_tokenized_prompt_prefix"), 
            retrieved_1_tokenized_prompt_prefix=data.get("retrieved_1_tokenized_prompt_prefix"), 
            retrieved_2_tokenized_prompt_prefix=data.get("retrieved_2_tokenized_prompt_prefix"), 
            retrieved_3_tokenized_prompt_prefix=data.get("retrieved_3_tokenized_prompt_prefix"), 
            retrieved_4_tokenized_prompt_prefix=data.get("retrieved_4_tokenized_prompt_prefix"), 
            retrieved_5_tokenized_prompt_prefix=data.get("retrieved_5_tokenized_prompt_prefix"), 
            retrieved_6_tokenized_prompt_prefix=data.get("retrieved_6_tokenized_prompt_prefix"), 
            retrieved_7_tokenized_prompt_prefix=data.get("retrieved_7_tokenized_prompt_prefix"), 
            retrieved_8_tokenized_prompt_prefix=data.get("retrieved_8_tokenized_prompt_prefix"), 
            retrieved_9_tokenized_prompt_prefix=data.get("retrieved_9_tokenized_prompt_prefix"), 
            retrieved_10_tokenized_prompt_prefix=data.get("retrieved_10_tokenized_prompt_prefix"), 
            retrieved_11_tokenized_prompt_prefix=data.get("retrieved_11_tokenized_prompt_prefix"), 
            retrieved_12_tokenized_prompt_prefix=data.get("retrieved_12_tokenized_prompt_prefix"), 
            retrieved_13_tokenized_prompt_prefix=data.get("retrieved_13_tokenized_prompt_prefix"), 
            retrieved_14_tokenized_prompt_prefix=data.get("retrieved_14_tokenized_prompt_prefix"), 
            retrieved_15_tokenized_prompt_prefix=data.get("retrieved_15_tokenized_prompt_prefix"), 
            retrieved_16_tokenized_prompt_prefix=data.get("retrieved_16_tokenized_prompt_prefix"), 
            retrieved_17_tokenized_prompt_prefix=data.get("retrieved_17_tokenized_prompt_prefix"), 
            retrieved_18_tokenized_prompt_prefix=data.get("retrieved_18_tokenized_prompt_prefix"), 
            retrieved_19_tokenized_prompt_prefix=data.get("retrieved_19_tokenized_prompt_prefix"),
            query_tokenized_prompt_postfix=data.get("query_tokenized_prompt_postfix"),
            retrieved_0_tokenized_prompt_postfix=data.get("retrieved_0_tokenized_prompt_postfix"), 
            retrieved_1_tokenized_prompt_postfix=data.get("retrieved_1_tokenized_prompt_postfix"), 
            retrieved_2_tokenized_prompt_postfix=data.get("retrieved_2_tokenized_prompt_postfix"), 
            retrieved_3_tokenized_prompt_postfix=data.get("retrieved_3_tokenized_prompt_postfix"), 
            retrieved_4_tokenized_prompt_postfix=data.get("retrieved_4_tokenized_prompt_postfix"), 
            retrieved_5_tokenized_prompt_postfix=data.get("retrieved_5_tokenized_prompt_postfix"), 
            retrieved_6_tokenized_prompt_postfix=data.get("retrieved_6_tokenized_prompt_postfix"), 
            retrieved_7_tokenized_prompt_postfix=data.get("retrieved_7_tokenized_prompt_postfix"), 
            retrieved_8_tokenized_prompt_postfix=data.get("retrieved_8_tokenized_prompt_postfix"), 
            retrieved_9_tokenized_prompt_postfix=data.get("retrieved_9_tokenized_prompt_postfix"), 
            retrieved_10_tokenized_prompt_postfix=data.get("retrieved_10_tokenized_prompt_postfix"), 
            retrieved_11_tokenized_prompt_postfix=data.get("retrieved_11_tokenized_prompt_postfix"), 
            retrieved_12_tokenized_prompt_postfix=data.get("retrieved_12_tokenized_prompt_postfix"), 
            retrieved_13_tokenized_prompt_postfix=data.get("retrieved_13_tokenized_prompt_postfix"), 
            retrieved_14_tokenized_prompt_postfix=data.get("retrieved_14_tokenized_prompt_postfix"), 
            retrieved_15_tokenized_prompt_postfix=data.get("retrieved_15_tokenized_prompt_postfix"), 
            retrieved_16_tokenized_prompt_postfix=data.get("retrieved_16_tokenized_prompt_postfix"), 
            retrieved_17_tokenized_prompt_postfix=data.get("retrieved_17_tokenized_prompt_postfix"), 
            retrieved_18_tokenized_prompt_postfix=data.get("retrieved_18_tokenized_prompt_postfix"), 
            retrieved_19_tokenized_prompt_postfix=data.get("retrieved_19_tokenized_prompt_postfix"),
            query_tokenized_prompt_mask=data.get("query_tokenized_prompt_mask"),
            retrieved_0_tokenized_prompt_mask=data.get("retrieved_0_tokenized_prompt_mask"),
            retrieved_1_tokenized_prompt_mask=data.get("retrieved_1_tokenized_prompt_mask"),
            retrieved_2_tokenized_prompt_mask=data.get("retrieved_2_tokenized_prompt_mask"),
            retrieved_3_tokenized_prompt_mask=data.get("retrieved_3_tokenized_prompt_mask"),
            retrieved_4_tokenized_prompt_mask=data.get("retrieved_4_tokenized_prompt_mask"),
            retrieved_5_tokenized_prompt_mask=data.get("retrieved_5_tokenized_prompt_mask"),
            retrieved_6_tokenized_prompt_mask=data.get("retrieved_6_tokenized_prompt_mask"),
            retrieved_7_tokenized_prompt_mask=data.get("retrieved_7_tokenized_prompt_mask"),
            retrieved_8_tokenized_prompt_mask=data.get("retrieved_8_tokenized_prompt_mask"),
            retrieved_9_tokenized_prompt_mask=data.get("retrieved_9_tokenized_prompt_mask"),
            retrieved_10_tokenized_prompt_mask=data.get("retrieved_10_tokenized_prompt_mask"),
            retrieved_11_tokenized_prompt_mask=data.get("retrieved_11_tokenized_prompt_mask"),
            retrieved_12_tokenized_prompt_mask=data.get("retrieved_12_tokenized_prompt_mask"),
            retrieved_13_tokenized_prompt_mask=data.get("retrieved_13_tokenized_prompt_mask"),
            retrieved_14_tokenized_prompt_mask=data.get("retrieved_14_tokenized_prompt_mask"),
            retrieved_15_tokenized_prompt_mask=data.get("retrieved_15_tokenized_prompt_mask"),
            retrieved_16_tokenized_prompt_mask=data.get("retrieved_16_tokenized_prompt_mask"),
            retrieved_17_tokenized_prompt_mask=data.get("retrieved_17_tokenized_prompt_mask"),
            retrieved_18_tokenized_prompt_mask=data.get("retrieved_18_tokenized_prompt_mask"),
            retrieved_19_tokenized_prompt_mask=data.get("retrieved_19_tokenized_prompt_mask"),
            query_token_ar_mask=data.get("query_token_ar_mask"),
            retrieved_0_token_ar_mask=data.get("retrieved_0_token_ar_mask"),
            retrieved_1_token_ar_mask=data.get("retrieved_1_token_ar_mask"),
            retrieved_2_token_ar_mask=data.get("retrieved_2_token_ar_mask"),
            retrieved_3_token_ar_mask=data.get("retrieved_3_token_ar_mask"),
            retrieved_4_token_ar_mask=data.get("retrieved_4_token_ar_mask"),
            retrieved_5_token_ar_mask=data.get("retrieved_5_token_ar_mask"),
            retrieved_6_token_ar_mask=data.get("retrieved_6_token_ar_mask"),
            retrieved_7_token_ar_mask=data.get("retrieved_7_token_ar_mask"),
            retrieved_8_token_ar_mask=data.get("retrieved_8_token_ar_mask"),
            retrieved_9_token_ar_mask=data.get("retrieved_9_token_ar_mask"),
            retrieved_10_token_ar_mask=data.get("retrieved_10_token_ar_mask"),
            retrieved_11_token_ar_mask=data.get("retrieved_11_token_ar_mask"),
            retrieved_12_token_ar_mask=data.get("retrieved_12_token_ar_mask"),
            retrieved_13_token_ar_mask=data.get("retrieved_13_token_ar_mask"),
            retrieved_14_token_ar_mask=data.get("retrieved_14_token_ar_mask"),
            retrieved_15_token_ar_mask=data.get("retrieved_15_token_ar_mask"),
            retrieved_16_token_ar_mask=data.get("retrieved_16_token_ar_mask"),
            retrieved_17_token_ar_mask=data.get("retrieved_17_token_ar_mask"),
            retrieved_18_token_ar_mask=data.get("retrieved_18_token_ar_mask"),
            retrieved_19_token_ar_mask=data.get("retrieved_19_token_ar_mask"),
            query_token_loss_mask=data.get("query_token_loss_mask"),
            retrieved_0_token_loss_mask=data.get("retrieved_0_token_loss_mask"),
            retrieved_1_token_loss_mask=data.get("retrieved_1_token_loss_mask"),
            retrieved_2_token_loss_mask=data.get("retrieved_2_token_loss_mask"),
            retrieved_3_token_loss_mask=data.get("retrieved_3_token_loss_mask"),
            retrieved_4_token_loss_mask=data.get("retrieved_4_token_loss_mask"),
            retrieved_5_token_loss_mask=data.get("retrieved_5_token_loss_mask"),
            retrieved_6_token_loss_mask=data.get("retrieved_6_token_loss_mask"),
            retrieved_7_token_loss_mask=data.get("retrieved_7_token_loss_mask"),
            retrieved_8_token_loss_mask=data.get("retrieved_8_token_loss_mask"),
            retrieved_9_token_loss_mask=data.get("retrieved_9_token_loss_mask"),
            retrieved_10_token_loss_mask=data.get("retrieved_10_token_loss_mask"),
            retrieved_11_token_loss_mask=data.get("retrieved_11_token_loss_mask"),
            retrieved_12_token_loss_mask=data.get("retrieved_12_token_loss_mask"),
            retrieved_13_token_loss_mask=data.get("retrieved_13_token_loss_mask"),
            retrieved_14_token_loss_mask=data.get("retrieved_14_token_loss_mask"),
            retrieved_15_token_loss_mask=data.get("retrieved_15_token_loss_mask"),
            retrieved_16_token_loss_mask=data.get("retrieved_16_token_loss_mask"),
            retrieved_17_token_loss_mask=data.get("retrieved_17_token_loss_mask"),
            retrieved_18_token_loss_mask=data.get("retrieved_18_token_loss_mask"),
            retrieved_19_token_loss_mask=data.get("retrieved_19_token_loss_mask"),
        )

# Defines the format of the actions. This field is included as "actions" inside the dictionary
# produced by the data transforms.
Actions = at.Float[ArrayT, "*b ah ad"]


def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
) -> Observation:
    """Preprocess the observations by performing image augmentations (if train=True), resizing (if necessary), and
    filling in a default image mask (if necessary).
    """

    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key]
        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad(image, *image_resolution)

        if train:
            # Convert from [-1, 1] to [0, 1] for augmax.
            image = image / 2.0 + 0.5

            transforms = []
            if "wrist" not in key:
                height, width = image.shape[1:3]
                transforms += [
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                    augmax.Resize(width, height),
                    augmax.Rotate((-5, 5)),
                ]
            transforms += [
                augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
            ]
            sub_rngs = jax.random.split(rng, image.shape[0])
            image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)

            # Back to [-1, 1].
            image = image * 2.0 - 1.0

        out_images[key] = image

    # obtain mask
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)
        else:
            out_masks[key] = jnp.asarray(observation.image_masks[key])

    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )


def preprocess_observation_prefix_postfix(
    rng: at.KeyArrayLike | None,
    observation: ObservationPrefixPostfix,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
) -> ObservationPrefixPostfix:
    """Preprocess the observations by performing image augmentations (if train=True), resizing (if necessary), and
    filling in a default image mask (if necessary).
    """

    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key]
        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad(image, *image_resolution)

        if train:
            # Convert from [-1, 1] to [0, 1] for augmax.
            image = image / 2.0 + 0.5

            transforms = []
            if "wrist" not in key:
                height, width = image.shape[1:3]
                transforms += [
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                    augmax.Resize(width, height),
                    augmax.Rotate((-5, 5)),
                ]
            transforms += [
                augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
            ]
            sub_rngs = jax.random.split(rng, image.shape[0])
            image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)

            # Back to [-1, 1].
            image = image * 2.0 - 1.0

        out_images[key] = image

    # obtain mask
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)
        else:
            out_masks[key] = jnp.asarray(observation.image_masks[key])

    return ObservationPrefixPostfix(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt_prefix=observation.tokenized_prompt_prefix,
        tokenized_prompt_postfix=observation.tokenized_prompt_postfix,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )


def extract_observation_from_ricl_observation(
    ricl_observation: RiclObservation,
    prefix: str,
) -> ObservationPrefixPostfix:
    return ObservationPrefixPostfix(
        images=getattr(ricl_observation, f"{prefix}images"),
        image_masks=getattr(ricl_observation, f"{prefix}image_masks"),
        state=getattr(ricl_observation, f"{prefix}state"),
        tokenized_prompt_prefix=getattr(ricl_observation, f"{prefix}tokenized_prompt_prefix"),
        tokenized_prompt_postfix=getattr(ricl_observation, f"{prefix}tokenized_prompt_postfix"),
        tokenized_prompt_mask=getattr(ricl_observation, f"{prefix}tokenized_prompt_mask"),
        token_ar_mask=getattr(ricl_observation, f"{prefix}token_ar_mask"),
        token_loss_mask=getattr(ricl_observation, f"{prefix}token_loss_mask"),
    )


@dataclasses.dataclass(frozen=True)
class BaseModelConfig(abc.ABC):
    """Configuration shared by all models. Specific models should inherit from this class, and implement the `create`
    method to create the corresponding model.
    """

    # Action space dimension.
    action_dim: int
    # Action sequence length.
    action_horizon: int
    # Tokenized prompt maximum length.
    max_token_len: int

    @property
    @abc.abstractmethod
    def model_type(self) -> ModelType:
        """The model type."""

    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> "BaseModel":
        """Create a new model, initializing parameters."""

    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "BaseModel":
        """Create a model with the given parameters."""
        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        if remove_extra_params:
            params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), params)
        at.check_pytree_equality(expected=state.to_pure_dict(), got=params, check_shapes=True, check_dtypes=False)
        state.replace_by_pure_dict(params)
        return nnx.merge(graphdef, state)

    @abc.abstractmethod
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[Observation, Actions]:
        """Returns the input specification for the model. Values are jax.ShapeDtypeStruct."""

    def fake_obs(self, batch_size: int = 1) -> Observation:
        observation_spec, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)

    def fake_act(self, batch_size: int = 1) -> Actions:
        _, action_spec = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)


@dataclasses.dataclass
class BaseModel(nnx.Module, abc.ABC):
    """Base class for all model implementations. Specific models should inherit from this class. They should call
    super().__init__() to initialize the shared attributes (action_dim, action_horizon, and max_token_len).
    """

    action_dim: int
    action_horizon: int
    max_token_len: int

    @abc.abstractmethod
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        actions: Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]: ...

    @abc.abstractmethod
    def sample_actions(self, rng: at.KeyArrayLike, observation: Observation) -> Actions: ...


def restore_params(
    params_path: pathlib.Path | str,
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    dtype: jnp.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
) -> at.Params:
    """Restores unstructured params PyTree from a checkpoint.

    This works with checkpoints saved with `save_state` during openpi training (see `training/checkpoints.py`) as
    well as pre-trained checkpoints released for openpi.

    Args:
        params_path: The local path to the checkpoint directory.
        restore_type: The type to restore the params as. Can be set to `np.ndarray` to load the params as a numpy array.
        dtype: The dtype to restore all params as. If not provided, will use the original dtype from the checkpoint.
        sharding: The sharding to use for the params. If not provided, the params will be replicated across all devices.

    Returns:
        The restored params.
    """
    params_path = pathlib.Path(params_path).resolve()
    if not params_path.exists():
        raise FileNotFoundError(f"Model params not found at: {params_path}")

    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        item = {"params": metadata["params"]}

        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype), item
                ),
            ),
        )["params"]

    # If the params were saved with `save_state` during openpi training, every key path will end with "value", which is
    # added by `nnx.State`. We remove the "value" suffix here and always return what NNX calls a "pure dict".
    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)
