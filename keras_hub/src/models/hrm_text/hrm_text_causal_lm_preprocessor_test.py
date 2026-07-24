from keras_hub.src.models.hrm_text.hrm_text_causal_lm_preprocessor import (
    HrmTextCausalLMPreprocessor,
)
from keras_hub.src.models.hrm_text.hrm_text_tokenizer import HrmTextTokenizer
from keras_hub.src.models.hrm_text.hrm_text_tokenizer_test import (
    make_tokenizer_assets,
)
from keras_hub.src.tests.test_case import TestCase


class HrmTextCausalLMPreprocessorTest(TestCase):
    def setUp(self):
        vocabulary, merges = make_tokenizer_assets()
        self.tokenizer = HrmTextTokenizer(vocabulary=vocabulary, merges=merges)
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "sequence_length": 9,
        }

    def test_causal_preprocessor(self):
        self.run_preprocessor_test(
            cls=HrmTextCausalLMPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=[" airplane at airport"],
            expected_output=(
                {
                    "token_ids": [[3, 27, 18, 28, 27, 20, 1, 2, 2]],
                    "padding_mask": [[1, 1, 1, 1, 1, 1, 1, 0, 0]],
                    "token_type_ids": [[0, 0, 0, 0, 0, 0, 0, 0, 0]],
                },
                [[27, 18, 28, 27, 20, 1, 2, 2, 2]],
                [[1, 1, 1, 1, 1, 1, 0, 0, 0]],
            ),
        )

    def test_prefix_lm_exact_sequence_and_response_loss_mask(self):
        preprocessor = HrmTextCausalLMPreprocessor(**self.init_kwargs)
        inputs, labels, weights = preprocessor(
            {
                "instruction": [" airplane"],
                "response": [" at airport"],
                "condition": ["direct"],
            }
        )
        start = self.tokenizer.start_token_id
        direct = self.tokenizer.direct_condition_token_id
        prefix_end = self.tokenizer.prefix_end_token_id
        end = self.tokenizer.end_token_id
        pad = self.tokenizer.pad_token_id
        self.assertAllEqual(
            inputs["token_ids"],
            [[start, direct, 27, 18, prefix_end, 28, 27, 20, end]],
        )
        self.assertAllEqual(inputs["padding_mask"], [[1] * 9])
        self.assertAllEqual(
            inputs["token_type_ids"], [[1, 1, 1, 1, 1, 0, 0, 0, 0]]
        )
        self.assertAllEqual(
            labels,
            [[direct, 27, 18, prefix_end, 28, 27, 20, end, pad]],
        )
        self.assertAllEqual(weights, [[0, 0, 0, 0, 1, 1, 1, 1, 0]])

    def test_each_condition_is_atomic_and_closes_prefix_with_im_end(self):
        preprocessor = HrmTextCausalLMPreprocessor(**self.init_kwargs)
        for name, token in preprocessor.condition_tokens.items():
            condition_id = self.tokenizer([token])[0][0]
            inputs, _, weights = preprocessor(
                {
                    "instruction": [" airplane"],
                    "response": [" at airport"],
                    "condition": [name],
                }
            )
            self.assertEqual(inputs["token_ids"][0, 1], condition_id)
            self.assertEqual(
                inputs["token_ids"][0, 4], self.tokenizer.prefix_end_token_id
            )
            self.assertAllEqual(weights[0, :4], [0, 0, 0, 0])
            self.assertAllEqual(weights[0, 4:8], [1, 1, 1, 1])

    def test_empty_instruction(self):
        preprocessor = HrmTextCausalLMPreprocessor(**self.init_kwargs)
        inputs, _, weights = preprocessor(
            {
                "instruction": [""],
                "response": [" airplane"],
                "condition": ["direct"],
            }
        )
        self.assertAllEqual(
            inputs["token_type_ids"], [[1, 1, 1, 0, 0, 0, 0, 0, 0]]
        )
        self.assertAllEqual(weights, [[0, 0, 1, 1, 1, 0, 0, 0, 0]])

    def test_empty_response(self):
        preprocessor = HrmTextCausalLMPreprocessor(**self.init_kwargs)
        inputs, _, weights = preprocessor(
            {
                "instruction": [" airplane"],
                "response": [""],
                "condition": ["direct"],
            }
        )
        self.assertAllEqual(
            inputs["token_type_ids"], [[1, 1, 1, 1, 1, 0, 0, 0, 0]]
        )
        self.assertAllEqual(weights, [[0, 0, 0, 0, 1, 0, 0, 0, 0]])

    def test_mixed_length_batch(self):
        preprocessor = HrmTextCausalLMPreprocessor(**self.init_kwargs)
        inputs, _, weights = preprocessor(
            {
                "instruction": [" airplane", ""],
                "response": [" at airport", " airplane"],
                "condition": ["direct", "synth"],
            }
        )
        self.assertAllEqual(inputs["token_ids"].shape, (2, 9))
        self.assertAllEqual(weights.shape, (2, 9))

    def test_invalid_or_legacy_prefix_lm_fields_are_rejected(self):
        preprocessor = HrmTextCausalLMPreprocessor(**self.init_kwargs)
        with self.assertRaisesRegex(ValueError, "Unknown HRM-Text condition"):
            preprocessor(
                {
                    "instruction": [" airplane"],
                    "response": [" at airport"],
                    "condition": ["unknown"],
                }
            )
        with self.assertRaisesRegex(ValueError, "missing"):
            preprocessor(
                {"prefix": [" airplane"], "response": [" at airport"]}
            )

    def test_format_instruction_closes_prefix_without_double_start(self):
        preprocessor = HrmTextCausalLMPreprocessor(**self.init_kwargs)
        self.assertEqual(
            preprocessor.format_instruction("Question", "direct"),
            "<|object_ref_start|>Question<|im_end|>",
        )

    def test_generate_round_trip(self):
        preprocessor = HrmTextCausalLMPreprocessor(**self.init_kwargs)
        formatted = preprocessor.format_instruction(" airplane")
        self.assertEqual(
            formatted, "<|object_ref_start|> airplane<|im_end|>"
        )
        inputs = preprocessor.generate_preprocess(formatted)
        self.assertAllEqual(inputs["token_type_ids"], inputs["padding_mask"])
        self.assertAllEqual(
            inputs["token_ids"],
            [
                3,
                self.tokenizer.direct_condition_token_id,
                27,
                18,
                self.tokenizer.prefix_end_token_id,
                2,
                2,
                2,
                2,
            ],
        )
        self.assertEqual(
            preprocessor.generate_postprocess(inputs), " airplane"
        )

    def test_format_instruction_rejects_unknown_condition(self):
        preprocessor = HrmTextCausalLMPreprocessor(**self.init_kwargs)
        with self.assertRaisesRegex(ValueError, "Unknown HRM-Text condition"):
            preprocessor.format_instruction(" airplane", "unknown")

    def test_format_instruction_accepts_string_lists(self):
        preprocessor = HrmTextCausalLMPreprocessor(**self.init_kwargs)
        self.assertEqual(
            preprocessor.format_instruction(["one", "two"]),
            [
                "<|object_ref_start|>one<|im_end|>",
                "<|object_ref_start|>two<|im_end|>",
            ],
        )
