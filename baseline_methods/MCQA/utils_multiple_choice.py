# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Multiple choice fine-tuning: utilities to work with multiple choice tasks of reading comprehension """


import csv
import glob
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import tqdm

from filelock import FileLock
from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available
from transformers.tokenization_utils_base import TruncationStrategy


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputExample:
    """
    A single training/test example for multiple choice

    Args:
        example_id: Unique id for the example.
        question: string. The untokenized text of the second sequence (question).
        contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
        endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    example_id: str
    question: str
    contexts: List[str]
    endings: List[str]
    label: Optional[str]


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    example_id: str
    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    label: Optional[int]


class Split(Enum):
    train = "train"
    val = "val"
    test = "test"


if is_torch_available():
    import torch
    from torch.utils.data.dataset import Dataset

    class MultipleChoiceDataset(Dataset):
        """
        This will be superseded by a framework-agnostic approach
        soon.
        """

        features: List[InputFeatures]

        def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            task: str,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
            num_choices=None,
            train_file=None,
            val_file=None,
            test_file=None,
        ):
            processor = processors[task]()
            if mode.value == "train":
                PREFIX = train_file.split("/")[-1].replace(".jsonl", "")
            elif mode.value == "val":
                PREFIX = val_file.split("/")[-1].replace(".jsonl", "")
            elif mode.value == "test":
                PREFIX = test_file.split("/")[-1].replace(".jsonl", "")
            cached_features_file = os.path.join(
                data_dir,
                "cached_{}_{}_{}".format(
                    PREFIX,
                    tokenizer.__class__.__name__,
                    str(max_seq_length),
                ),
            )

            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.
            lock_path = cached_features_file + ".lock"
            with FileLock(lock_path):

                if os.path.exists(cached_features_file) and not overwrite_cache:
                    logger.info(f"Loading features from cached file {cached_features_file}")
                    self.features = torch.load(cached_features_file)
                else:
                    logger.info(f"Creating features from dataset file at {data_dir}")
                    logger.info(f"Num Choices {num_choices}")
                    label_list = processor.get_labels(num_choices=num_choices)
                    if mode == Split.val:
                        examples = processor.get_val_examples(val_file, num_choices=num_choices)
                    elif mode == Split.test:
                        examples = processor.get_test_examples(test_file, num_choices=num_choices)
                    else:
                        examples = processor.get_train_examples(train_file, num_choices=num_choices)
                    logger.info("Training examples: %s", len(examples))
                    self.features = convert_examples_to_features(
                        examples,
                        label_list,
                        max_seq_length,
                        tokenizer,
                    )
                    logger.info("Saving features into cached file %s", cached_features_file)
                    torch.save(self.features, cached_features_file)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]

 
class DataProcessor:
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, file_path, num_choices):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_val_examples(self, file_path, num_choices):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, file_path, num_choices):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self, num_choices):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()



class OpenCSRProcessor(DataProcessor):
    """Processor for the OpenCSR data set."""

    def get_train_examples(self, file_path, num_choices=5):
        """See base class."""
        logger.info("LOOKING train AT {}".format(file_path))
        return self._create_examples(self._read_json(file_path), "train", num_choices=num_choices)

    def get_val_examples(self, file_path, num_choices=5):
        """See base class."""
        logger.info("LOOKING val AT {} ".format(file_path))
        return self._create_examples(self._read_json(file_path), "val", num_choices=num_choices)

    def get_test_examples(self, file_path, num_choices):
        """See base class."""
        logger.info("LOOKING test AT {} ".format(file_path))
        return self._create_examples(self._read_json(file_path), "test", num_choices=num_choices)

    def get_labels(self, num_choices):
        """See base class."""
        return [str(i) for i in range(1,num_choices+1)]

    def _read_json(self, input_file):
        with open(input_file, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
            return lines

    def _create_examples(self, lines, type, num_choices):
        """Creates examples for the training and validation sets."""

        # There are two types of labels. They should be normalized
        def normalize(truth):
            if truth in [str(i) for i in range(1,num_choices+1)]:
                return int(truth)
            else:
                logger.info("truth ERROR! %s", str(truth))
                return None

        examples = []  
        # we deleted example which has more than or less than four choices
        for line in tqdm.tqdm(lines, desc="read arc data"):
            data_raw = json.loads(line.strip("\n"))
            # if len(data_raw["question"]["choices"]) != num_choices:
            #     print(num_choices)
            #     print(len(data_raw["question"]["choices"]))
            assert len(data_raw["question"]["choices"]) >= num_choices
            data_raw["question"]["choices"] = data_raw["question"]["choices"][:num_choices]
            if type=="test":
                data_raw["answerKey"] = "1" # dummy label
            truth = str(normalize(data_raw["answerKey"])) 
            assert truth != "None" or type=="test"
            question_choices = data_raw["question"]
            question = question_choices["stem"]
            id = data_raw["id"]
            options = question_choices["choices"]
            if len(options) == num_choices:
                examples.append(
                    InputExample(
                        example_id=id,
                        question=question,
                        contexts=[" " for i in range(num_choices) ],
                        endings=[options[i]["text"] for i in range(num_choices)],
                        label=truth,
                    )
                ) 
        if type == "train":
            assert len(examples) > 1
            assert examples[0].label is not None
        logger.info("len examples: %s}", str(len(examples)))  

        return examples

def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 100 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_inputs = []
        for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
            del context # not useful
            text_a = example.question
            text_b = ending 
            inputs = tokenizer(
                text_a,
                text_b,
                add_special_tokens=True,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                truncation_strategy = TruncationStrategy.ONLY_FIRST,
                return_overflowing_tokens=True,
            )
            if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
                logger.info(
                    "Attention! Cropping tokens. " 
                )

            choices_inputs.append(inputs)

        label = label_map[example.label]

        input_ids = [x["input_ids"] for x in choices_inputs]
        attention_mask = (
            [x["attention_mask"] for x in choices_inputs] if "attention_mask" in choices_inputs[0] else None
        )
        token_type_ids = (
            [x["token_type_ids"] for x in choices_inputs] if "token_type_ids" in choices_inputs[0] else None
        )

        features.append(
            InputFeatures(
                example_id=example.example_id,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=label,
            )
        )

    for f in features[:2]:
        logger.info("*** Example ***")
        logger.info("feature: %s" % f)

    return features


processors = {"opencsr": OpenCSRProcessor}
MULTIPLE_CHOICE_TASKS_NUM_LABELS = {"opencsr", 5}