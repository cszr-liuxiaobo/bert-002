import re
import os
import numpy as np
import torch
import glob
import json
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, DistilBertForMultipleChoice, BertForMultipleChoice
from transformers import DistilBertForQuestionAnswering
from transformers import DistilBertTokenizerFast, BertForMultipleChoice


class MCTestExample(object):
    def __init__(self,
                 story_id,
                 context_sentence,
                 start_ending,
                 question_type,
                 ending_0,
                 ending_1,
                 ending_2,
                 ending_3,
                 label = None):
        self.story_id = story_id
        self.context_sentence = context_sentence
        self.start_ending = start_ending
        self.question_type = question_type
        self.endings = [
            ending_0,
            ending_1,
            ending_2,
            ending_3,
        ]
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            f"id: {self.story_id}",
            f"article: {self.context_sentence}",
            f"question: {self.start_ending}",
            f"type: {self.question_type}",
            f"option_0: {self.endings[0]}",
            f"option_1: {self.endings[1]}",
            f"option_2: {self.endings[2]}",
            f"option_3: {self.endings[3]}",
        ]

        if self.label is not None:
            l.append(f"label: {self.label}")

        return ", ".join(l)


class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label

    ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label


def letterToNum(answerIndex):
    conversion = {'A':0,'B':1,'C':2,'D':3}
    return conversion[answerIndex]


def read_mctest_examples(paths):

    passages_file = paths[0]
    correct_answers_file = paths[1]
    f = open(passages_file)
    raw_passages = f.read()
    f.close()
    f = open(correct_answers_file)
    raw_correct_answers = f.read()
    f.close()

    truth = [list(map(letterToNum, answer_set.split('\t'))) for answer_set in raw_correct_answers.split('\n')
             if len(answer_set) > 0]

    examples = []
    passages = raw_passages.split('\n')[:-1]
    for (i, passage) in enumerate(passages):
        elements = passage.split('\t')  # split passage elements
        title = elements[0]  # get title
        story = elements[2]  # get story, replace escaped newlines and tabs
        story = re.sub(r'\\newline', ' ', story)
        story = re.sub(r'\\tab', '\t', story).lower()
        for j in range(4):
            question_elements = elements[3 + 5 * j:3 + 5 * (j + 1)]  # get question elements
            qtype, qtext = question_elements[0].split(': ')
            examples.append(
                MCTestExample(
                    story_id=title,
                    context_sentence=story,
                    start_ending=qtext,
                    question_type=qtype,
                    ending_0=question_elements[1],
                    ending_1=question_elements[2],
                    ending_2=question_elements[3],
                    ending_3=question_elements[4],
                    label=truth[i][j]))

    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 is_training):
    features = []
    for example_index, example in enumerate(examples):
        context_tokens = tokenizer.tokenize(example.context_sentence)
        start_ending_tokens = tokenizer.tokenize(example.start_ending)

        choices_features = []
        for ending_index, ending in enumerate(example.endings):

            context_tokens_choice = context_tokens[:]
            ending_tokens = start_ending_tokens + tokenizer.tokenize(ending)

            _truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)

            tokens = ["[CLS]"] + context_tokens_choice + ["[SEP]"] + ending_tokens + ["[SEP]"]
            segment_ids = [0] * (len(context_tokens_choice) + 2) + [1] * (len(ending_tokens) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            choices_features.append((tokens, input_ids, input_mask, segment_ids))

        label = example.label
        ## display some example
        if example_index < 1:
            print("*** Example ***")
            print(f"story_id: ", example.story_id)
            for choice_idx, (tokens, input_ids, input_mask, segment_ids) in enumerate(choices_features):
                print(f"choice: ", choice_idx)
                print(f"tokens: ", ' '.join(tokens))
                print(f"input_ids: ", ' '.join(map(str, input_ids)))
                print(f"input_mask: ", ' '.join(map(str, input_mask)))
                print(f"segment_ids: ", ' '.join(map(str, segment_ids)))
            if is_training:
                print(f"label: {label}")

        features.append(
            InputFeatures(
                example_id = example.story_id,
                choices_features = choices_features,
                label = label
            )
        )

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]


def MCTestDataset(data_features):
    all_input_ids = torch.tensor(select_field(data_features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(data_features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(data_features, 'segment_ids'), dtype=torch.long)
    all_label = torch.tensor([f.label for f in data_features], dtype=torch.long)
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
    return data


if __name__ == '__main__':
    num_train_epochs = 3
    train_batch_size = 4
    max_seq_length = 320
    learning_rate = 1e-3
    warmup_proportion = 0.1
    gradient_accumulation_steps = 8
    train_dir = '../Dataset/MCTest/MCTest/mc160.train'
    PYTORCH_PRETRAINED_BERT_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
                                                   Path.home() / '.pytorch_pretrained_bert'))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_examples = read_mctest_examples([train_dir + '.tsv', train_dir + '.ans'])
    num_train_steps = int(
        len(train_examples) / train_batch_size / gradient_accumulation_steps * num_train_epochs)

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_features = convert_examples_to_features(
        train_examples, tokenizer, max_seq_length, True)

    train_data = MCTestDataset(train_features)

    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)

    model = DistilBertForMultipleChoice.from_pretrained('distilbert-base-uncased')

    model.to(device)
    model = torch.nn.DataParallel(model)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    global_step = 0
    count = 0

    model.train()
    for epoch in range(num_train_epochs):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        print("Training Epoch: {}/{}".format(epoch + 1, int(num_train_epochs)))
        for step, batch in enumerate(train_loader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            result = model(input_ids=input_ids,
                           # token_type_ids=segment_ids,
                           attention_mask=input_mask,
                           labels=label_ids
                           )
            loss = result['loss']
            logits = result['logits']
            # loss = loss / gradient_accumulation_steps
            compare = np.array(label_ids.cpu()) == np.array(logits.argmax(axis=1).cpu())
            count += np.sum(compare)
            tr_loss += loss
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            global_step += 1
            print("Label: {}, Prediction: {}, Accuracy: {}"
                  .format(label_ids, logits.argmax(axis=1), count / (train_batch_size * global_step)))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if global_step % 100 == 0:
                print("Training loss: {}, global step: {}".format(tr_loss / nb_tr_steps, global_step))
