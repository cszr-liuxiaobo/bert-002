import argparse
import logging
import os
import random

import numpy as np
import torch
import glob
import json

from pytorch_pretrained_bert import BertTokenizer, PYTORCH_PRETRAINED_BERT_CACHE, BertForMultipleChoice, BertAdam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from tqdm import tqdm
from transformers import AdamW, DistilBertTokenizerFast, DistilBertForMultipleChoice

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class RaceExample(object):
    """A single training/test example for the RACE dataset."""
    '''
    For RACE dataset:
    race_id: data id
    context_sentence: article
    start_ending: question
    ending_0/1/2/3: option_0/1/2/3
    label: true answer
    '''
    def __init__(self,
                 race_id,
                 context_sentence,
                 start_ending,
                 ending_0,
                 ending_1,
                 ending_2,
                 ending_3,
                 label = None):
        self.race_id = race_id
        self.context_sentence = context_sentence
        self.start_ending = start_ending
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
            f"id: {self.race_id}",
            f"article: {self.context_sentence}",
            f"question: {self.start_ending}",
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


def read_race_examples(paths):
    examples = []
    count = 0
    for path in paths:
        # if path == paths[0]:
        #     continue
        filenames = glob.glob(path + "/*txt")
        for filename in filenames:
            count += 1
            if count == 1000:
                break
            with open(filename, 'r', encoding='utf-8') as fpr:
                data_raw = json.load(fpr)
                article = data_raw['article']
                ## for each qn
                for i in range(len(data_raw['answers'])):
                    truth = ord(data_raw['answers'][i]) - ord('A')
                    question = data_raw['questions'][i]
                    options = data_raw['options'][i]
                    examples.append(
                        RaceExample(
                            race_id=filename + '-' + str(i),
                            context_sentence=article,
                            start_ending=question,

                            ending_0=options[0],
                            ending_1=options[1],
                            ending_2=options[2],
                            ending_3=options[3],
                            label=truth))

    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 is_training):
    """Loads a data file into a list of `InputBatch`s."""

    # RACE is a multiple choice task. To perform this task using Bert,
    # we will use the formatting proposed in "Improving Language
    # Understanding by Generative Pre-Training" and suggested by
    # @jacobdevlin-google in this issue
    # https://github.com/google-research/bert/issues/38.
    #
    # The input will be like:
    # [CLS] Article [SEP] Question + Option [SEP]
    # for each option
    #
    # The model will output a single value for each input. To get the
    # final decision of the model, we will run a softmax over these 4
    # outputs.
    features = []
    for example_index, example in enumerate(examples):
        context_tokens = tokenizer.tokenize(example.context_sentence)
        start_ending_tokens = tokenizer.tokenize(example.start_ending)

        choices_features = []
        for ending_index, ending in enumerate(example.endings):
            # We create a copy of the context tokens in order to be
            # able to shrink it according to ending_tokens
            context_tokens_choice = context_tokens[:]
            ending_tokens = start_ending_tokens + tokenizer.tokenize(ending)
            # Modifies `context_tokens_choice` and `ending_tokens` in
            # place so that the total length is less than the
            # specified length.  Account for [CLS], [SEP], [SEP] with
            # "- 3"
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
            print(f"race_id: ", example.race_id)
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
                example_id = example.race_id,
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


def RACEDataset(data_features):
    all_input_ids = torch.tensor(select_field(data_features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(data_features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(data_features, 'segment_ids'), dtype=torch.long)
    all_label = torch.tensor([f.label for f in data_features], dtype=torch.long)
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
    return data


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


def main():
    # num_train_epochs = 8
    # train_batch_size = 8
    # max_seq_length = 512
    # learning_rate = 1e-5
    # warmup_proportion = 0.1
    # gradient_accumulation_steps = 4
    # data_dir = './Dataset/RACE/RACE/'

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--bert_type",
                        default=None,
                        type=int,
                        required=True,
                        help="0:bert, 1: distilbert")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}".format(
                 device, n_gpu, bool(args.local_rank != -1)))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        if args.do_train:
            raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = None
    if args.bert_type == 0:
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    elif args.bert_type == 1:
        tokenizer = DistilBertTokenizerFast.from_pretrained(args.bert_model)

    train_examples = None
    num_train_steps = None

    if args.do_train:
        train_dir = os.path.join(args.data_dir, 'train')
        train_examples = read_race_examples([train_dir + '/high', train_dir + '/middle'])
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    model = None
    if args.bert_type == 0:
        model = BertForMultipleChoice.from_pretrained(args.bert_model,
                                                      cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(
                                                          args.local_rank),
                                                      num_choices=4)
    elif args.bert_type == 1:
        model = DistilBertForMultipleChoice.from_pretrained(args.bert_model)
    model.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

        # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    optimizer = None
    if args.bert_type == 0:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)
    elif args.bert_type == 1:
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    global_step = 0
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, tokenizer, args.max_seq_length, True)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        all_input_ids = torch.tensor(select_field(train_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(train_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(train_features, 'segment_ids'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        for ep in range(int(args.num_train_epochs)):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            count = 0
            logger.info("Trianing Epoch: {}/{}".format(ep + 1, int(args.num_train_epochs)))
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                if args.bert_type == 0:
                    loss = model(input_ids=input_ids,
                                   token_type_ids=segment_ids,
                                   attention_mask=input_mask,
                                   labels=label_ids)
                elif args.bert_type == 1:
                    result = model(input_ids=input_ids,
                                         # token_type_ids=segment_ids,
                                         attention_mask=input_mask,
                                         labels=label_ids)
                    loss = result['loss']
                    logits = result['logits']
                    #######
                    compare = np.array(label_ids.cpu()) == np.array(logits.argmax(axis=1).cpu())
                    count += np.sum(compare)
                    print("\nLabel: {}, Prediction: {}, Accuracy: {}"
                          .format(label_ids, logits.argmax(axis=1), count / (args.train_batch_size * (step + 1))))
                    #######
                if n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step/t_total, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                if global_step%100 == 0:
                    logger.info("Training loss: {}, global step: {}".format(tr_loss/nb_tr_steps, global_step))


            ## evaluate on dev set
            if global_step % 1000 == 0:
                dev_dir = os.path.join(args.data_dir, 'dev')
                dev_set = [dev_dir+'/high', dev_dir+'/middle']

                eval_examples = read_race_examples(dev_set)
                eval_features = convert_examples_to_features(
                    eval_examples, tokenizer, args.max_seq_length, True)
                logger.info("***** Running evaluation: Dev *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)
                all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
                all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
                all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
                all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)
                eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
                # Run prediction for full data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                model.eval()
                eval_loss, eval_accuracy = 0, 0
                nb_eval_steps, nb_eval_examples = 0, 0
                for step, batch in enumerate(eval_dataloader):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, label_ids = batch

                    with torch.no_grad():
                        if args.bert_type == 0:
                            tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
                            logits = model(input_ids, segment_ids, input_mask)
                        elif args.bert_type == 1:
                            result = model(input_ids=input_ids,
                                           attention_mask=input_mask,
                                           labels=label_ids)
                            tmp_eval_loss = result['loss']
                            logits = result['logits']
                            logits = logits.argmax(axis=1)

                    logits = logits.detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    tmp_eval_accuracy = accuracy(logits, label_ids)

                    eval_loss += tmp_eval_loss.mean().item()
                    eval_accuracy += tmp_eval_accuracy

                    nb_eval_examples += input_ids.size(0)
                    nb_eval_steps += 1

                eval_loss = eval_loss / nb_eval_steps
                eval_accuracy = eval_accuracy / nb_eval_examples

                result = {'dev_eval_loss': eval_loss,
                          'dev_eval_accuracy': eval_accuracy,
                          'global_step': global_step,
                          'loss': tr_loss/nb_tr_steps}

                output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
                with open(output_eval_file, "a+") as writer:
                    logger.info("***** Dev results *****")
                    for key in sorted(result.keys()):
                        logger.info("  %s = %s", key, str(result[key]))
                        writer.write("%s = %s\n" % (key, str(result[key])))

        # Save a trained model
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)

    model.load_state_dict(torch.load(os.path.join(args.output_dir, "pytorch_model.bin")))
    if args.do_eval and args.local_rank == -1:
        test_dir = os.path.join(args.data_dir, 'test')
        test_high = [test_dir + '/high']
        test_middle = [test_dir + '/middle']

        ## test high
        eval_examples = read_race_examples(test_high)
        eval_features = convert_examples_to_features(
            eval_examples, tokenizer, args.max_seq_length, True)
        logger.info("***** Running evaluation: test high *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        #
        model.eval()
        high_eval_loss, high_eval_accuracy = 0, 0
        high_nb_eval_steps, high_nb_eval_examples = 0, 0
        for step, batch in enumerate(eval_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            with torch.no_grad():
                if args.bert_type == 0:
                    tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
                    logits = model(input_ids, segment_ids, input_mask)
                elif args.bert_type == 1:
                    result = model(input_ids=input_ids,
                                   attention_mask=input_mask,
                                   labels=label_ids)
                    tmp_eval_loss = result['loss']
                    logits = result['logits']
                    # logits = logits.argmax(axis=1)
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)

            high_eval_loss += tmp_eval_loss.mean().item()
            high_eval_accuracy += tmp_eval_accuracy

            high_nb_eval_examples += input_ids.size(0)
            high_nb_eval_steps += 1

        eval_loss = high_eval_loss / high_nb_eval_steps
        eval_accuracy = high_eval_accuracy / high_nb_eval_examples

        result = {'high_eval_loss': eval_loss,
                  'high_eval_accuracy': eval_accuracy}

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "a+") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        ## test middle
        eval_examples = read_race_examples(test_middle)
        eval_features = convert_examples_to_features(
            eval_examples, tokenizer, args.max_seq_length, True)
        logger.info("***** Running evaluation: test middle *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        middle_eval_loss, middle_eval_accuracy = 0, 0
        middle_nb_eval_steps, middle_nb_eval_examples = 0, 0
        for step, batch in enumerate(eval_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            with torch.no_grad():
                if args.bert_type == 0:
                    results = model(input_ids=input_ids,
                                          token_type_ids=segment_ids,
                                          attention_mask=input_mask,
                                          labels=label_ids)
                elif args.bert_type == 1:
                    results = model(input_ids=input_ids,
                                   attention_mask=input_mask,
                                   labels=label_ids)
            tmp_eval_loss = results['loss']
            logits = results['logits']

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)

            middle_eval_loss += tmp_eval_loss.mean().item()
            middle_eval_accuracy += tmp_eval_accuracy

            middle_nb_eval_examples += input_ids.size(0)
            middle_nb_eval_steps += 1

        eval_loss = middle_eval_loss / middle_nb_eval_steps
        eval_accuracy = middle_eval_accuracy / middle_nb_eval_examples

        result = {'middle_eval_loss': eval_loss,
                  'middle_eval_accuracy': eval_accuracy}
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")

        with open(output_eval_file, "a+") as writer:
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        # all test
        eval_loss = (middle_eval_loss + high_eval_loss) / (middle_nb_eval_steps + high_nb_eval_steps)
        eval_accuracy = (middle_eval_accuracy + high_eval_accuracy) / (middle_nb_eval_examples + high_nb_eval_examples)

        result = {'overall_eval_loss': eval_loss,
                  'overall_eval_accuracy': eval_accuracy}

        with open(output_eval_file, "a+") as writer:
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    main()