"""SimCodecMixin — save/load mixin for simulation codecs.

Drop-in replacement for CodecPytorchHubMixin that validates against
SIM_MODALITY_CODEC_MAPPING instead of the observational MODALITY_CODEC_MAPPING.
"""

import inspect
from pathlib import Path

import torch
from aion.codecs.base import Codec
from aion.codecs.utils import CodecPytorchHubMixin, _codec_path_context, _validate_modality
from aion.modalities import Modality


class SimCodecMixin(CodecPytorchHubMixin):
    """Mixin for simulation codecs.

    Overrides _validate_codec_modality to check SIM_MODALITY_CODEC_MAPPING
    instead of the observational codec mapping.
    """

    @classmethod
    def _encode_arg(cls, value):
        if inspect.isclass(value):
            return f"{value.__module__}.{value.__qualname__}"
        return super()._encode_arg(value)

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights using torch.save (safetensors not required)."""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        weights_path = save_directory / "pytorch_model.bin"
        model_to_save = self.module if hasattr(self, "module") else self
        torch.save(model_to_save.state_dict(), weights_path)

    @staticmethod
    def _validate_codec_modality(codec: type[Codec], modality: type[Modality]):
        from sim.codecs.config import SIM_MODALITY_CODEC_MAPPING  # late import avoids circular

        if not issubclass(codec, Codec):
            raise TypeError("Only codecs can be loaded using this method.")
        if modality not in SIM_MODALITY_CODEC_MAPPING:
            raise ValueError(
                f"Modality {modality} has no corresponding sim codec. "
                f"Available: {list(SIM_MODALITY_CODEC_MAPPING.keys())}"
            )
        expected = SIM_MODALITY_CODEC_MAPPING[modality]
        if not issubclass(codec, expected):
            raise TypeError(
                f"Modality {modality} is associated with {expected} but "
                f"{codec} was requested."
            )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        modality: type[Modality],
        *model_args,
        **kwargs,
    ):
        cls._validate_codec_modality(cls, modality)
        _validate_modality(modality)
        with _codec_path_context(modality):
            # Skip CodecPytorchHubMixin.from_pretrained to avoid its validation;
            # call the HuggingFace mixin directly.
            model = super(CodecPytorchHubMixin, cls).from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs
            )
        model._modality = modality
        return model

    def save_pretrained(
        self, save_directory, modality: type[Modality] = None, *args, **kwargs
    ):
        if not issubclass(self.__class__, Codec):
            raise ValueError("Only codec instances can be saved using this method.")

        if modality is not None:
            _validate_modality(modality)
            target_modality = modality
        elif hasattr(self, "_modality"):
            target_modality = self._modality
        else:
            raise ValueError(
                "No modality specified. Provide modality= or load via from_pretrained()."
            )

        codec_path = f"{save_directory}/codecs/{target_modality.name}"
        super(CodecPytorchHubMixin, self).save_pretrained(codec_path, *args, **kwargs)
