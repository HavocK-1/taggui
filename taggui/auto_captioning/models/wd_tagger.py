# Based on
# https://huggingface.co/spaces/SmilingWolf/wd-tagger/blob/main/app.py.
import csv
import json
import re
from datetime import datetime
from pathlib import Path

import huggingface_hub
import numpy as np
from huggingface_hub.errors import EntryNotFoundError
from PIL import Image as PilImage
from onnxruntime import InferenceSession

import auto_captioning.captioning_thread as captioning_thread
from auto_captioning.auto_captioning_model import AutoCaptioningModel
from utils.image import Image

KAOMOJIS = ['0_0', '(o)_(o)', '+_+', '+_-', '._.', '<o>_<o>', '<|>_<|>', '=_=',
            '>_<', '3_3', '6_9', '>_o', '@_@', '^_^', 'o_o', 'u_u', 'x_x',
            '|_|', '||_||']


def get_tags_to_exclude(tags_to_exclude_string: str) -> list[str]:
    if not tags_to_exclude_string.strip():
        return []
    tags = re.split(r'(?<!\\),', tags_to_exclude_string)
    tags = [tag.strip().replace(r'\,', ',') for tag in tags]
    return tags


class WdTaggerModel:
    def __init__(self, model_id: str):
        model_path = self._find_file(model_id, ['model.onnx',
                                                'onnx/model.onnx'])
        tags_path = self._find_file(model_id, ['selected_tags.csv'])
        self.inference_session = InferenceSession(model_path)
        input_shape = self.inference_session.get_inputs()[0].shape
        # The input is either channels-first (NCHW) or channels-last (NHWC).
        self.channels_first = input_shape[1] == 3
        self.input_dimension = (input_shape[2] if self.channels_first
                                else input_shape[1])
        self.output_is_logits = (
            self.inference_session.get_outputs()[0].name == 'logits')
        self.normalize = self._read_normalization(model_id)
        self.tags = []
        self.rating_tags_indices = []
        self.general_tags_indices = []
        self.character_tags_indices = []
        with open(tags_path, 'r') as tags_file:
            reader = csv.DictReader(tags_file)
            for index, line in enumerate(reader):
                tag = line['name']
                if tag not in KAOMOJIS:
                    tag = tag.replace('_', ' ')
                self.tags.append(tag)
                category = line['category']
                if category == '9':
                    self.rating_tags_indices.append(index)
                elif category == '0':
                    self.general_tags_indices.append(index)
                elif category == '4':
                    self.character_tags_indices.append(index)

    @staticmethod
    def _find_optional_file(model_id: str, filenames: list[str]) -> str | None:
        for filename in filenames:
            local_path = Path(model_id) / filename
            if local_path.is_file():
                return str(local_path)
        for filename in filenames:
            try:
                return huggingface_hub.hf_hub_download(model_id,
                                                       filename=filename)
            except EntryNotFoundError:
                continue
        return None

    @classmethod
    def _find_file(cls, model_id: str, filenames: list[str]) -> str:
        path = cls._find_optional_file(model_id, filenames)
        if path is None:
            raise FileNotFoundError(
                f'Could not find any of {filenames} for {model_id}')
        return path

    @classmethod
    def _read_normalization(cls, model_id: str) -> bool:
        # Some taggers (e.g. danbooru-tagger-v1) expect normalized input,
        # indicated by a `normalization` key in their config.json.
        config_path = cls._find_optional_file(model_id,
                                              ['config.json',
                                               'onnx/config.json'])
        if config_path is None:
            return False
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
        return 'normalization' in config

    def generate_tags(self, image_array: np.ndarray,
                      wd_tagger_settings: dict) -> tuple[tuple, tuple]:
        input_name = self.inference_session.get_inputs()[0].name
        output_name = self.inference_session.get_outputs()[0].name
        probabilities = self.inference_session.run(
            [output_name], {input_name: image_array})[0][0].astype(np.float32)
        # Some models output raw logits instead of probabilities.
        if self.output_is_logits:
            probabilities = 1.0 / (1.0 + np.exp(-probabilities))
        # Exclude the rating tags.
        tags = [tag for index, tag in enumerate(self.tags)
                if index not in self.rating_tags_indices]
        probabilities = np.array([
            probability for index, probability in enumerate(probabilities)
            if index not in self.rating_tags_indices
        ])
        tags_to_exclude = get_tags_to_exclude(
            wd_tagger_settings['tags_to_exclude'])
        tags_and_probabilities = []
        for tag, probability in zip(tags, probabilities):
            if (probability < wd_tagger_settings['min_probability']
                    or tag in tags_to_exclude):
                continue
            tags_and_probabilities.append((tag, probability))
        # Sort the tags by probability.
        tags_and_probabilities.sort(key=lambda x: x[1], reverse=True)
        tags_and_probabilities = tags_and_probabilities[
                                 :wd_tagger_settings['max_tags']]
        if tags_and_probabilities:
            tags, probabilities = zip(*tags_and_probabilities)
        else:
            tags, probabilities = (), ()
        return tags, probabilities


class WdTagger(AutoCaptioningModel):
    image_mode = 'RGBA'

    def __init__(self,
                 captioning_thread_: 'captioning_thread.CaptioningThread',
                 caption_settings: dict):
        super().__init__(captioning_thread_, caption_settings)
        self.wd_tagger_settings = self.caption_settings['wd_tagger_settings']
        self.show_probabilities = self.wd_tagger_settings['show_probabilities']

    def get_error_message(self) -> str | None:
        return None

    def get_processor(self):
        return None

    def get_model(self):
        return WdTaggerModel(self.model_id)

    def get_captioning_message(self, are_multiple_images_selected: bool,
                               captioning_start_datetime: datetime) -> str:
        if are_multiple_images_selected:
            captioning_start_datetime_string = (
                self.get_captioning_start_datetime_string(
                    captioning_start_datetime))
            return (f'Generating tags... (start time: '
                    f'{captioning_start_datetime_string})')
        return 'Generating tags...'

    def get_model_inputs(self, image_prompt: str, image: Image) -> np.ndarray:
        pil_image = self.load_image(image)
        # Add a white background to the image in case it has transparent areas.
        canvas = PilImage.new('RGBA', pil_image.size, (255, 255, 255))
        canvas.alpha_composite(pil_image)
        pil_image = canvas.convert('RGB')
        # Pad the image to make it square.
        max_dimension = max(pil_image.size)
        canvas = PilImage.new('RGB', (max_dimension, max_dimension),
                              (255, 255, 255))
        horizontal_padding = (max_dimension - pil_image.width) // 2
        vertical_padding = (max_dimension - pil_image.height) // 2
        canvas.paste(pil_image, (horizontal_padding, vertical_padding))
        # Resize the image to the model's input dimensions.
        input_dimension = self.model.input_dimension
        if max_dimension != input_dimension:
            input_dimensions = (input_dimension, input_dimension)
            canvas = canvas.resize(input_dimensions,
                                   resample=PilImage.Resampling.BICUBIC)
        # Convert the image to a numpy array.
        image_array = np.array(canvas, dtype=np.float32)
        # Reverse the order of the color channels (RGB -> BGR).
        image_array = image_array[:, :, ::-1]
        # Normalize to [-1, 1] if the model expects normalized input.
        if self.model.normalize:
            image_array = (image_array - 127.5) / 127.5
        # Convert to channels-first (NCHW) if the model expects it.
        if self.model.channels_first:
            image_array = np.transpose(image_array, (2, 0, 1))
        # Add a batch dimension.
        image_array = np.expand_dims(image_array, axis=0)
        return image_array

    def generate_caption(self, model_inputs: np.ndarray,
                         image_prompt: str) -> tuple[str, str]:
        tags, probabilities = self.model.generate_tags(model_inputs,
                                                       self.wd_tagger_settings)
        caption = self.thread.tag_separator.join(tags)
        if self.show_probabilities:
            console_output_caption = self.thread.tag_separator.join(
                f'{tag} ({probability:.2f})'
                for tag, probability in zip(tags, probabilities)
            )
        else:
            console_output_caption = caption
        return caption, console_output_caption
