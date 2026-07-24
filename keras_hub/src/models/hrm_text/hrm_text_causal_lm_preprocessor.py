"""Causal language-model preprocessing for HRM-Text."""

import keras
import tensorflow as tf
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.multi_segment_packer import (
    MultiSegmentPacker,
)
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.hrm_text.hrm_text_backbone import HrmTextBackbone
from keras_hub.src.models.hrm_text.hrm_text_tokenizer import HrmTextTokenizer
from keras_hub.src.utils.tensor_utils import in_tf_function
from keras_hub.src.utils.tensor_utils import preprocessing_function


@keras_hub_export("keras_hub.models.HrmTextCausalLMPreprocessor")
class HrmTextCausalLMPreprocessor(CausalLMPreprocessor):
    """Preprocesses causal and PrefixLM data for `HrmTextCausalLM`.

    A plain string is treated as causal language-model text. For PrefixLM
    training, pass a dictionary containing ``instruction``, ``response``, and
    ``condition``. Only response-token labels receive training weight;
    instruction tokens can attend to one another bidirectionally. The
    serialized PrefixLM sequence follows the upstream SFT builder exactly:
    ``<|im_start|><condition>instruction<|im_end|>response<|box_end|>``.

    Args:
        tokenizer: An instance of `keras_hub.models.HrmTextTokenizer`.
        sequence_length: Packed sequence length. Defaults to ``128``.
        add_start_token: Whether to prepend ``<|im_start|>``. Defaults to
            ``True``. PrefixLM dictionaries require it.
        add_end_token: Whether to append ``<|box_end|>``. Defaults to
            ``True``.

    Examples:

    ```python
    preprocessor = keras_hub.models.HrmTextCausalLMPreprocessor.from_preset(
        "/path/to/hrm_text_1b", sequence_length=128
    )

    # Ordinary causal LM inputs.
    x, y, sample_weight = preprocessor(["A short document."])

    # PrefixLM inputs: only response labels receive nonzero sample weights.
    x, y, sample_weight = preprocessor({
        "instruction": ["Question: What is 2 + 2?\\nAnswer:"],
        "response": [" 4"],
        "condition": ["direct"],
    })
    ```
    """

    backbone_cls = HrmTextBackbone
    tokenizer_cls = HrmTextTokenizer

    condition_tokens = {
        "direct": "<|object_ref_start|>",
        "cot": "<|object_ref_end|>",
        "noisy": "<|quad_start|>",
        "synth": "<|quad_end|>",
    }

    def format_instruction(self, instruction, condition="direct"):
        """Format an HRM inference instruction without ``<|im_start|>``.

        ``HrmTextCausalLM.generate()`` calls this method for raw string
        inputs. ``generate_preprocess()`` then owns the leading
        ``<|im_start|>`` token, producing the upstream PrefixLM prompt:
        ``<|im_start|><condition>instruction<|im_end|>``. Generated output is
        terminated by the tokenizer's ``<|box_end|>`` end token.
        """
        if condition not in self.condition_tokens:
            raise ValueError(
                "Unknown HRM-Text condition. Expected one of "
                f"{sorted(self.condition_tokens)}."
            )
        prefix = self.condition_tokens[condition]
        suffix = self.tokenizer.prefix_end_token
        if isinstance(instruction, str):
            return prefix + instruction + suffix
        if isinstance(instruction, (list, tuple)):
            if not all(isinstance(value, str) for value in instruction):
                raise ValueError("HRM-Text instructions must be strings.")
            return [prefix + value + suffix for value in instruction]
        if getattr(instruction, "dtype", None) == tf.string:
            return tf.strings.join([prefix, instruction, suffix])
        raise ValueError("HRM-Text instructions must be strings.")
    def build(self, input_shape):
        self.packer = MultiSegmentPacker(
            start_value=self.tokenizer.start_token_id,
            sep_value=self.tokenizer.prefix_end_token_id,
            end_value=self.tokenizer.end_token_id,
            pad_value=self.tokenizer.pad_token_id,
            sequence_length=self.sequence_length,
            truncate="waterfall",
        )
        self.built = True

    def _format_instruction_python(self, instruction, condition):
        if isinstance(condition, str):
            condition = [condition] * len(instruction)
        if len(condition) != len(instruction):
            raise ValueError(
                "`condition` must have one value per `instruction`."
            )
        try:
            controls = [self.condition_tokens[value] for value in condition]
        except KeyError as error:
            raise ValueError(
                "Unknown HRM-Text condition. Expected one of "
                f"{sorted(self.condition_tokens)}."
            ) from error
        return [control + value for control, value in zip(controls, instruction)]

    def _format_instruction_tf(self, instruction, condition):
        instruction = tf.convert_to_tensor(instruction, dtype=tf.string)
        condition = tf.convert_to_tensor(condition, dtype=tf.string)
        if condition.shape.rank == 0:
            condition = tf.fill(tf.shape(instruction), condition)
        shape_assertion = tf.debugging.assert_equal(
            tf.shape(condition),
            tf.shape(instruction),
            message="`condition` must have one value per `instruction`.",
        )
        choices = tf.constant(list(self.condition_tokens))
        controls = tf.constant(list(self.condition_tokens.values()))
        matches = tf.equal(tf.expand_dims(condition, -1), choices)
        valid = tf.reduce_any(matches, axis=-1)
        valid_assertion = tf.debugging.assert_equal(
            tf.reduce_all(valid),
            True,
            message=(
                "Unknown HRM-Text condition. Expected one of "
                f"{sorted(self.condition_tokens)}."
            ),
        )
        with tf.control_dependencies([shape_assertion, valid_assertion]):
            index = tf.argmax(tf.cast(matches, tf.int32), axis=-1)
            return tf.strings.join([tf.gather(controls, index), instruction])

    @staticmethod
    def _require_prefix_lm_fields(x):
        required = {"instruction", "response", "condition"}
        if missing := required - x.keys():
            raise ValueError(
                "PrefixLM data requires fields "
                f"{sorted(required)}; missing {sorted(missing)}."
            )

    def _pack(self, segments, sequence_length, add_end_value):
        token_ids, segment_ids = self.packer(
            segments,
            sequence_length=sequence_length,
            add_start_value=self.add_start_token,
            add_end_value=add_end_value,
        )
        padding_mask = ops.cast(
            token_ids != self.tokenizer.pad_token_id, "int32"
        )
        return token_ids, padding_mask, segment_ids

    def _call_python(self, x, y=None, sample_weight=None, sequence_length=None):
        if not self.built:
            self.build(None)
        sequence_length = sequence_length or self.sequence_length
        if isinstance(x, dict):
            self._require_prefix_lm_fields(x)
            if not self.add_start_token or not self.add_end_token:
                raise ValueError(
                    "PrefixLM dictionaries require start and end tokens."
                )
            segments = (
                self.tokenizer(
                    self._format_instruction_python(
                        x["instruction"], x["condition"]
                    )
                ),
                self.tokenizer(x["response"]),
            )
            prefix_lm = True
        else:
            segments = self.tokenizer(x)
            prefix_lm = False
        token_ids, padding_mask, segment_ids = self._pack(
            segments,
            sequence_length + 1,
            add_end_value=self.add_end_token,
        )
        if prefix_lm:
            token_type_ids = ops.cast(segment_ids == 0, "int32") * padding_mask
        else:
            token_type_ids = ops.zeros_like(padding_mask)
        inputs = {
            "token_ids": token_ids[..., :-1],
            "padding_mask": padding_mask[..., :-1],
            "token_type_ids": token_type_ids[..., :-1],
        }
        labels = token_ids[..., 1:]
        weights = padding_mask[..., 1:]
        if prefix_lm:
            weights = weights * ops.cast(segment_ids[..., 1:] == 1, "int32")
        return keras.utils.pack_x_y_sample_weight(inputs, labels, weights)

    @preprocessing_function
    def _call_tf(self, x, y=None, sample_weight=None, sequence_length=None):
        if not self.built:
            self.build(None)
        sequence_length = sequence_length or self.sequence_length
        if isinstance(x, dict):
            self._require_prefix_lm_fields(x)
            if not self.add_start_token or not self.add_end_token:
                raise ValueError(
                    "PrefixLM dictionaries require start and end tokens."
                )
            segments = (
                self.tokenizer(
                    self._format_instruction_tf(
                        x["instruction"], x["condition"]
                    )
                ),
                self.tokenizer(x["response"]),
            )
            prefix_lm = True
        else:
            segments = self.tokenizer(x)
            prefix_lm = False
        token_ids, segment_ids = self.packer(
            segments,
            sequence_length=sequence_length + 1,
            add_start_value=self.add_start_token,
            add_end_value=self.add_end_token,
        )
        padding_mask = tf.cast(
            tf.not_equal(token_ids, self.tokenizer.pad_token_id), tf.int32
        )
        if prefix_lm:
            token_type_ids = tf.cast(tf.equal(segment_ids, 0), tf.int32)
            token_type_ids = token_type_ids * padding_mask
        else:
            token_type_ids = tf.zeros_like(padding_mask)
        inputs = {
            "token_ids": token_ids[..., :-1],
            "padding_mask": padding_mask[..., :-1],
            "token_type_ids": token_type_ids[..., :-1],
        }
        labels = token_ids[..., 1:]
        weights = padding_mask[..., 1:]
        if prefix_lm:
            weights = weights * tf.cast(
                tf.equal(segment_ids[..., 1:], 1), tf.int32
            )
        return keras.utils.pack_x_y_sample_weight(inputs, labels, weights)

    def call(self, x, y=None, sample_weight=None, sequence_length=None):
        if not self._allow_python_workflow or in_tf_function():
            return self._call_tf(
                x,
                y=y,
                sample_weight=sample_weight,
                sequence_length=sequence_length,
            )
        return self._call_python(
            x,
            y=y,
            sample_weight=sample_weight,
            sequence_length=sequence_length,
        )

    def _generate_preprocess_python(self, x, sequence_length=None):
        if not self.built:
            self.build(None)
        sequence_length = sequence_length or self.sequence_length
        token_ids, padding_mask, _ = self._pack(
            self.tokenizer(x),
            sequence_length=sequence_length,
            add_end_value=False,
        )
        return {
            "token_ids": token_ids,
            "padding_mask": ops.cast(padding_mask, "int32"),
            "token_type_ids": ops.cast(padding_mask, "int32"),
        }

    @preprocessing_function
    def _generate_preprocess_tf(self, x, sequence_length=None):
        if not self.built:
            self.build(None)
        sequence_length = sequence_length or self.sequence_length
        token_ids, _ = self.packer(
            self.tokenizer(x),
            sequence_length=sequence_length,
            add_start_value=self.add_start_token,
            add_end_value=False,
        )
        padding_mask = tf.cast(
            tf.not_equal(token_ids, self.tokenizer.pad_token_id), tf.int32
        )
        return {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
            "token_type_ids": padding_mask,
        }

    def generate_preprocess(self, x, sequence_length=None):
        if not self._allow_python_workflow or in_tf_function():
            return self._generate_preprocess_tf(
                x, sequence_length=sequence_length
            )
        return self._generate_preprocess_python(
            x, sequence_length=sequence_length
        )
