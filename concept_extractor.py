import csv
import argparse
import torch
import numpy as np
from nltk.corpus import stopwords
import re
import hydra
import string

stop_words = set(stopwords.words('english'))
from tqdm import tqdm
from rake_nltk import Rake

import math
import torch.nn.functional as F
from copy import deepcopy
from transformers import (
    BertTokenizer, BertForMaskedLM,
    AutoModelForMaskedLM, AutoTokenizer,
    DebertaV2Tokenizer, DebertaV2ForMaskedLM,
    RobertaTokenizer, RobertaForMaskedLM,
    ElectraForPreTraining, ElectraTokenizer,
)
from utils.str_utils import process_sentence, TreeUtils, find_phrase_indexes, merge_lists
from utils.data_utils import read_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 100


def find_rank_of_index(input_list, index):
    sort_values, sorted_indices = torch.sort(input_list, descending=True)

    rank = (sorted_indices == index).nonzero(as_tuple=True)[0].item()  
    top1 = sorted_indices[0].item()
    return rank, top1


class ELECTRA_Scorer:
    def __init__(self, pretrained='google/electra-large-discriminator', device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        # if 'roberta' in pretrained:
        self.tokenizer = ElectraTokenizer.from_pretrained(pretrained)
        self.electra_model = ElectraForPreTraining.from_pretrained(pretrained).to(self.device)
        # else:
        #     self.tokenizer = BertTokenizer.from_pretrained(pretrained)
        #     self.bert_model = BertForMaskedLM.from_pretrained(pretrained).to(self.device)
        self.mask_id = self.tokenizer.mask_token_id
        self.sep_id = self.tokenizer.sep_token_id
        self.cls_id = self.tokenizer.cls_token_id
        self.special_tokens = set(self.tokenizer.all_special_tokens)
        self.special_ids = set(self.tokenizer.all_special_ids)

    def replaced_token_position_finder(self, input):

        # fake_tokens = self.tokenizer.tokenize(input, add_special_tokens=True)
        encoded_inputs = self.tokenizer.encode(input, add_special_tokens=True)
        input_tensor = torch.tensor(encoded_inputs).unsqueeze(0).to(self.device)
        discriminator_outputs = self.electra_model(input_tensor)
        pred_probs = F.softmax(discriminator_outputs[0], dim=1)

        return pred_probs


class Perplexity_Scorer:
    def __init__(self, pretrained='FacebookAI/roberta-large', treeutils=None, device=None):

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        if 'roberta' in pretrained:
            self.tokenizer = RobertaTokenizer.from_pretrained(pretrained)
            self.model = RobertaForMaskedLM.from_pretrained(pretrained).to(self.device)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
            self.model = AutoModelForMaskedLM.from_pretrained(pretrained).to(self.device)
        self.mask_id = self.tokenizer.mask_token_id
        self.sep_id = self.tokenizer.sep_token_id
        self.cls_id = self.tokenizer.cls_token_id
        self.special_tokens = set(self.tokenizer.all_special_tokens)
        self.special_ids = set(self.tokenizer.all_special_ids)
        self.tree_utils = treeutils

    def mask_score(self, sent_ids, mask_idx, mode=0, log_prob=False, maxlen=MAX_LEN):
        if maxlen:
            if mask_idx > maxlen:
                dist = int(mask_idx - maxlen / 2)
                sent_ids = sent_ids[dist:dist + maxlen]
                mask_idx -= dist
        sent_ids = np.concatenate([[self.cls_id], sent_ids, [self.sep_id]])
        mask_idx += 1
        if mode == 0:
            masked_sent_ids = np.array(sent_ids)
        else:
            masked_sent_ids = np.concatenate([sent_ids[:mask_idx], [self.mask_id], sent_ids[mask_idx:]])

        sent_len = len(masked_sent_ids)
        if maxlen is not None:
            sent_len = min(sent_len, maxlen + 2)
        input_tensor = torch.tensor(masked_sent_ids[:sent_len]).unsqueeze(0).to(self.device)
        outputs = self.model(input_tensor)
        prediction_scores = outputs[0]
        if log_prob:
            log_pred_probs = torch.log_softmax(prediction_scores, dim=2)
            return log_pred_probs[0][mask_idx].detach().cpu().numpy()
        else:
            pred_probs = torch.softmax(prediction_scores, dim=2)
            return pred_probs[0][mask_idx].detach().cpu().numpy()

    def sent_score(self, line, return_rank=False, maxlen=MAX_LEN, log_prob=False, ignore_idx=-1, ppl=False):
        if type(line) == str:
            sent_ids = self.tokenizer.encode(line.strip(), add_special_tokens=False)
        else:
            sent_ids = line
        if len(sent_ids) == 0:
            if log_prob:
                return -math.inf
            else:
                return 0.0

        sent_ids = np.concatenate([[self.cls_id], sent_ids, [self.sep_id]])
        sent_len = len(sent_ids)
        if maxlen is not None:
            sent_len = min(sent_len, maxlen + 2)
        input_list = np.array((sent_len - 2) * [sent_ids[:sent_len]])
        input_tensor = torch.from_numpy(input_list).to(self.device)
        for idx in range(sent_len - 2):
            input_tensor[idx][idx + 1] = self.mask_id
        outputs = self.model(input_tensor)
        prediction_scores = outputs[0]
        log_pred_probs = torch.log_softmax(prediction_scores, dim=2)
        sent_log_prob = 0.0
        rank_list = []
        top1_list = []
        for idx in range(sent_len - 2):
            tok_ind = idx + 1
            tok = sent_ids[tok_ind]
            if tok == self.mask_id:
                sent_log_prob += torch.max(log_pred_probs[idx][tok_ind]).item()
            else:
                if tok_ind != ignore_idx:
                    sent_log_prob += log_pred_probs[idx][tok_ind][tok].item()
            if return_rank:
                rank, top1 = find_rank_of_index(log_pred_probs[idx][tok_ind], tok)
                rank_list.append(rank)
                top1_list.append(top1)
        if ppl:
            ppl_val = math.pow(math.exp(sent_log_prob), -1 / (sent_len - 1))
            if return_rank:
                return ppl_val, rank_list, top1_list
            return ppl_val
        elif log_prob:
            return sent_log_prob
        else:
            return math.exp(sent_log_prob)

    def sent_score_mask(self, input_list, return_rank=False, maxlen=MAX_LEN, log_prob=False, ignore_idx=-1, ppl=False):
        if type(input_list[0]) == str:
            print()
        else:
            sent_ids = np.concatenate([[self.cls_id], input_list, [self.sep_id]])
            sent_len = len(sent_ids)
            if maxlen is not None:
                sent_len = min(sent_len, maxlen + 2)
            input_tensor = torch.tensor((sent_len - 2) * [sent_ids[:sent_len]]).to(self.device)
            for idx in range(sent_len - 2):
                input_tensor[idx][idx + 1] = self.mask_id
            outputs = self.model(input_tensor)
            prediction_scores = outputs[0]
            log_pred_probs = torch.log_softmax(prediction_scores, dim=2)
            sent_log_prob = 0.0
            for idx in range(sent_len - 2):
                tok_ind = idx + 1
                tok = sent_ids[tok_ind]
                if tok == self.mask_id:
                    sent_log_prob += torch.max(log_pred_probs[idx][tok_ind]).item()
                else:
                    if tok_ind != ignore_idx:
                        sent_log_prob += log_pred_probs[idx][tok_ind][tok].item()

            if ppl:
                ppl_val = math.pow(math.exp(sent_log_prob), -1 / (sent_len - 1))
                return ppl_val
            elif log_prob:
                return sent_log_prob
            else:
                return math.exp(sent_log_prob)

    def multi_mask_score(self, sent_ids, mask_idx_set, mode=0, log_prob=False, maxlen=None, output_ind=None):
        if output_ind is None:
            output_ind = min(mask_idx_set)
        if maxlen:
            assert maxlen >= max(mask_idx_set)
        if mode == 0:
            masked_sent_ids = np.array(sent_ids)
            for mask_idx in mask_idx_set:
                masked_sent_ids[mask_idx] = self.mask_id
        else:
            raise NotImplementedError

        sent_len = len(masked_sent_ids)
        if maxlen is not None:
            sent_len = min(sent_len, maxlen)
        input_tensor = torch.tensor(masked_sent_ids[:sent_len]).unsqueeze(0).to(self.device)
        outputs = self.model(input_tensor)
        prediction_scores = outputs[0]
        if log_prob:
            log_pred_probs = torch.log_softmax(prediction_scores, dim=2)
            return log_pred_probs[0][output_ind].detach().cpu().numpy()
        else:
            pred_probs = torch.softmax(prediction_scores, dim=2)
            return pred_probs[0][output_ind].detach().cpu().numpy()

    def id2sent(self, ids):
        return self.tokenizer.decode(ids)

    def close(self):
        pass

    def perplexity_position_finder_with_nps(self, input: str, normalize=False):
        constituent_tree = self.tree_utils.create_constituent_tree(input)
        nps = self.tree_utils.extract_nps(constituent_tree.nltk_tree)
        input_id = self.tokenizer.encode(input, add_special_tokens=False)
        tokenized_sent = self.tokenizer.tokenize(input)
        matching_phrases_index = find_phrase_indexes(tokenized_sent, nps)
        index_list = [list(d.values())[0] for d in matching_phrases_index]
        mask_id = merge_lists(range(len(input_id)), index_list)

        compare_prob = []
        init_prob = self.sent_score(input_id, ppl=True)

        for idx in mask_id:
            if isinstance(idx, list):
                new_tokens = deepcopy(input_id)
                for id in idx:
                    new_tokens[id] = self.mask_id
                for id in idx:
                    count_prob = self.sent_score(new_tokens, ppl=True)
                    compare_prob.append(init_prob / count_prob)
            else:
                new_tokens = deepcopy(input_id)
                new_tokens[idx] = self.mask_id
                count_prob = self.sent_score(new_tokens, ppl=True)
                compare_prob.append(init_prob / count_prob)

        importance = F.softmax(torch.tensor(compare_prob), dim=-1)
        if normalize:
            importance = F.softmax(torch.tensor(importance), dim=-1).numpy()
        return importance

    def perplexity_position_finder_with_nps_mask(self, input: str, normalize=False):
        constituent_tree = self.tree_utils.create_constituent_tree(input)
        nps = self.tree_utils.extract_nps(constituent_tree)
        input_id = self.tokenizer.encode(input, add_special_tokens=False)
        input_tokens = re.findall(r'\w+|[.,!?;]', input)

        # get the index of the noun phrases for mask
        matching_phrases_index = find_phrase_indexes(input_tokens, nps)
        index_list = [list(d.values())[0] for d in matching_phrases_index]
        mask_id = merge_lists(range(len(input_tokens)), index_list)

        # compute the perplexity of the original nonsensical statement
        compare_prob = []
        init_prob = self.sent_score(input_id, ppl=True)

        for idx in mask_id:
            if isinstance(idx, list):
                new_tokens = deepcopy(input_tokens)
                for id in idx:
                    new_tokens[id] = self.tokenizer.mask_token
                    encoded_tokens = self.tokenizer(" ".join(new_tokens), add_special_tokens=False)
                    input_id = encoded_tokens.input_ids
                for id in idx:
                    count_prob = self.sent_score(input_id, ppl=True)
                    compare_prob.append(init_prob / count_prob)
            else:
                new_tokens = deepcopy(input_tokens)
                new_tokens[idx] = self.tokenizer.mask_token
                encoded_tokens = self.tokenizer(" ".join(new_tokens), add_special_tokens=False)
                input_id = encoded_tokens.input_ids
                count_prob = self.sent_score(input_id, ppl=True)
                compare_prob.append(init_prob / count_prob)

        importance = F.softmax(torch.tensor(compare_prob), dim=-1)
        if normalize:
            importance = F.softmax(torch.tensor(importance), dim=-1).numpy()
        return importance

    def perplexity_position_finder(self, input: list, normalize=True):
        input_id = self.tokenizer.encode(input, add_special_tokens=False)

        compare_prob = []
        init_prob, rank_list, top1_list = self.sent_score(input_id, return_rank=True, ppl=True)
        top_3 = sorted(rank_list, reverse=True)[:3]
        result_values = [input_id[rank_list.index(x)] for x in top_3]

        candidates = self.tokenizer.decode(result_values)
        top1_sentence = self.tokenizer.decode(top1_list)
        print()
        for idx in range(len(input_id)):
            new_tokens = deepcopy(input_id)
            new_tokens[idx] = self.mask_id
            count_prob = self.sent_score(new_tokens, ppl=True)
            compare_prob.append(init_prob / count_prob)
            # compare_prob.append(count_prob)
        importance = F.softmax(torch.tensor(compare_prob), dim=-1)
        if normalize:
            importance = F.softmax(torch.tensor(importance), dim=-1).numpy()

        return importance

    def perplexity_position_finder_mask(self, input: str, normalize=True):


        input_id = self.tokenizer.encode(input, add_special_tokens=False)
        input_tokens = re.findall(r'\w+|[.,!?;]', input)
        compare_prob = []
        init_prob = self.sent_score(input_id, ppl=True)
        for idx in range(len(input_tokens)):
            new_tokens = deepcopy(input_tokens)
            new_tokens[idx] = self.tokenizer.mask_token
            encoded_tokens = self.tokenizer(" ".join(new_tokens), add_special_tokens=False)
            input_id = encoded_tokens.input_ids
            count_prob = self.sent_score(input_id, ppl=True)
            compare_prob.append(init_prob / count_prob)
        importance = F.softmax(torch.tensor(compare_prob), dim=-1)
        if normalize:
            importance = F.softmax(torch.tensor(importance), dim=-1).numpy()

        return importance

  

@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    args = cfg

    tree_utils = TreeUtils()
    dataset = read_dataset(args.data_path, read_concept=False, num_concept=0)
    # electra_scorer = ELECTRA_Scorer()
    ppl_scorer = Perplexity_Scorer(device=device, pretrained=args.concept_model_name, treeutils=tree_utils)
    results_list = []

    def sort_and_rearrange(tensor, words):
        indexed_list = [(val.item(), idx) for idx, val in enumerate(tensor)]
        sorted_list = sorted(indexed_list, key=lambda x: x[0], reverse=True)
        result = []
        for val, idx in sorted_list:
            if result and isinstance(result[-1], list) and result[-1][0] == val:
                result[-1].append(idx)
            elif result and not isinstance(result[-1], list) and result[-1][0] == val:
                result[-1] = [result[-1][1], idx]
            else:
                result.append((val, idx))

        final_result = [item[1] if isinstance(item, tuple) else item for item in result]

        sorted_words = []
        for item in final_result:
            if isinstance(item, list):
                sorted_words.append([words[idx] for idx in item])
            else:
                sorted_words.append(words[item])

        return sorted_words

    def concept_filter(source_list, stop_words, limit):
        """
        Filters out stop words from the source list and limits the number of words
        in the target list according to the given limit.

        :param source_list: List of words to be processed.
        :param stop_words: List of stop words to be filtered out.
        :param limit: Maximum number of words the target list should contain.
        :return: List of words that are not stop words, up to the specified limit.
        """
        target_list = []  # Initialize the target list to store the filtered words.

        for word in source_list:
            if isinstance(word, list):
                if word[0] in stop_words:
                    word.pop(0)
                if len(word) >1:
                    target_list.append(" ".join(word))
                else:
                    target_list.append(word[0])

            elif word not in stop_words and word not in string.punctuation and len(target_list) < limit:
                target_list.append(word)  # Add word if it's not a stop word and limit is not reached.
            if len(target_list) >= limit:
                break  # Stop if we reach the limit.

        return target_list

    for data in tqdm(dataset):
        false_sent = data['FalseSent']
        processed_sent = process_sentence(false_sent, args.remove_punctuation)
        if args.num_concept == 999:
            r = Rake()

            # Extraction given the text.
            r.extract_keywords_from_text(false_sent)
            concept_list = r.get_ranked_phrases()

            print()
        else:
            # processed_sent = false_sent
            '''
                replaced token detection
            '''

            # ppl_result = electra_scorer.replaced_token_position_finder(processed_sent)
            # input_id = electra_scorer.tokenizer.encode(processed_sent,add_special_tokens=True)
            # tokenized_sent=electra_scorer.tokenizer.convert_ids_to_tokens(input_id)

            # processed_sent='the inverter was able to power the <mask>'
            # PPL = bert_scorer.perplexity_scorer(processed_sent)
            # print()
            '''
                compute perplexity by roberta
            '''
            # ppl_result = ppl_scorer.perplexity_position_finder_with_nps(processed_sent, False)
            if args.process_np:

                ppl_result = ppl_scorer.perplexity_position_finder_with_nps_mask(processed_sent, False)
            else:
                ppl_result = ppl_scorer.perplexity_position_finder_mask(processed_sent, False)
            # tokenized_sent = ppl_scorer.tokenizer.tokenize(processed_sent)
            tokenized_sent = re.findall(r'\w+|[.,!?;]', processed_sent)

            if args.process_np:
                topk_list = sort_and_rearrange(ppl_result, tokenized_sent)
            else:
                topk_index = torch.topk(ppl_result, k=len(tokenized_sent)).indices.tolist()

                # if "roberta" in args.model_name:
                #     concept_list = [tokenized_sent[i].replace("Ġ", "") for i in topk_index]
                # elif "deberta" in args.model_name:
                #     concept_list = [tokenized_sent[i].replace("▁", "") for i in topk_index]
                # else:
                #     concept_list = [tokenized_sent[i] for i in topk_index]
                topk_list = [tokenized_sent[i] for i in topk_index]
            concept_list = concept_filter(topk_list, stop_words, args.num_concept)
        concept_str = ",".join(concept_list)
        results_list.append(str(data["id"]) + "," + processed_sent + "," + concept_str)
        # print()

    if args.num_concept == 999:
        file_name = ""

    elif args.process_np:
        file_name = ""
    else:
        file_name = ""
    with open(file_name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        for result in results_list:
            csv_writer.writerow(result.split(','))


if __name__ == "__main__":
    main()
