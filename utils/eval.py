# coding='utf-8'
import copy
import csv
import logging
import sys
import json
from typing import List, Dict
from itertools import chain
import gc
import rouge
import torch
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu, corpus_bleu
from nltk.tokenize import word_tokenize
from bert_score import score as bert_score
import collections
import math
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import logging
from utils.metrics.bleurt import score as bleurt_score
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
from tqdm import tqdm
import wandb
import evaluate
import nlgeval
from nlgeval import NLGEval
from utils.str_utils import contains_word

EXIT_STATUS_ANSWERS_MALFORMED = 1
EXIT_STATUS_PREDICTIONS_MALFORMED = 2
EXIT_STATUS_PREDICTIONS_EXTRA = 3
EXIT_STATUS_PREDICTION_MISSING = 4


def _clean_text(txt):
    return txt.lower()


def bleurt(references, hypothesis, num_refs, batch_size=512,
           checkpoint="/home/iiserver32/.cache/huggingface/hub/models--lucadiliello--BLEURT-20-D12/snapshots/90ce39a8ceb3b9a046a7788f0005572301c3fe67/"):
    refs, cands = [], []
    tokenizer = BleurtTokenizer.from_pretrained(checkpoint)
    for key, value in hypothesis.items():
        for ref in references[key][:num_refs]:
            if len(value) < 1:
                print("empty output...")
                continue
            if len(tokenizer(" ".join(value)).data["input_ids"]) > 490:
                print("too long output...")
                continue
            if "_" in value[0]:
                print("wrong output...")
                continue
            cands.append(' '.join(value))
            refs.append(' '.join(ref))

    config = BleurtConfig.from_pretrained(checkpoint)
    model = BleurtForSequenceClassification.from_pretrained(checkpoint)
    # tokenizer = BleurtTokenizer.from_pretrained(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()
    total_scores = []

    # tqdm 直接包裹循环，每个批次自动更新进度条
    with torch.no_grad():
        for i in tqdm(range(0, len(refs), batch_size), desc="Calculating BLEURT", unit="batch"):
            batch_refs = refs[i:i + batch_size]
            batch_cands = cands[i:i + batch_size]
            inputs = tokenizer(batch_refs, batch_cands, padding='longest', return_tensors='pt').to(device)
            outputs = model(**inputs)
            logits = outputs.logits.flatten().tolist()
            total_scores.extend(logits)

    # 处理每个候选的最高得分
    max_scores = [max(total_scores[i:i + num_refs]) for i in range(0, len(total_scores), num_refs)]
    average_score = round(sum(max_scores) / len(max_scores), 4)
    model = None
    gc.collect()
    torch.cuda.empty_cache()
    return average_score


# def bleurt(references, hypothesis, num_refs, checkpoint = "/data/ge_yan/Nedo/CSEG/utils/metrics/bleurt-base-128"):
#     refs, cands = [], []
#     for key, value in hypothesis.items():
#         for ref in references[key][:num_refs]:
#             cands.append(' '.join(value))
#             refs.append(' '.join(ref))
#
#     # for i, hyp in enumerate(hypothesis):
#     #     for ref in references[i][:num_refs]:
#     #         cands.append(hyp)
#     #         refs.append(ref)
#
#     config = BleurtConfig.from_pretrained('lucadiliello/BLEURT-20-D12')
#     model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20-D12')
#     tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20-D12')
#
#     model.eval()
#     with torch.no_grad():
#         inputs = tokenizer(refs, cands, padding='longest', return_tensors='pt')
#         res = model(**inputs).logits.flatten().tolist()
#
#     scores = [max(res[i:i+num_refs]) for i in range(0, len(res), num_refs)]
#     # scores=round(sum(scores) / len(scores), 4)
#
#     # scorer = bleurt_score.BleurtScorer(checkpoint)
#     # scores = scorer.score(references=refs, candidates=cands)
#     # scores = [max(scores[i:i+num_refs]) for i in range(0, len(scores), num_refs)]
#     return round(sum(scores) / len(scores), 4)

def read_pred_file_json(pred_file):
    hypotheses = []
    hypotheses_token = []
    with open(pred_file) as f:
        reader = json.loads(f.read())
        for i, row in enumerate(reader):
            # row = row[0]
            if "###" in row: row = row.split("###")[0]
            tokens = row.split()
            if len(tokens) == 0:
                # print(row, 'empty line found at:', i)
                row = '.'
                # print(word_tokenize(row))
            hypotheses.append(row)
            hypotheses_token.append(word_tokenize(row))
    return hypotheses, hypotheses_token


def read_pred_file(pred_file):
    hypotheses = []
    hypotheses_token = []
    with open(pred_file) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            hypotheses.append(row[1])
            hypotheses_token.append(word_tokenize(row[1]))
    return hypotheses, hypotheses_token


def read_gold_file(gold_file, args):
    references = []
    references_token = []
    with open(gold_file) as csvfile:
        reader = list(csv.reader(csvfile))
        for row in reader:
            if args.task == "comve":
                references.append([row[1], row[2], row[3]])
                references_token.append([word_tokenize(row[1]), word_tokenize(row[2]), word_tokenize(row[3])])
            elif args.task == "esnli":
                pass
                # references.append([row[4], row[5], row[6]])
                # references_token.append([word_tokenize(row[4]), word_tokenize(row[5]), word_tokenize(row[6])])
    return references, references_token


def process_data(pred_file, gold_file, args):
    hypotheses, hypotheses_token = read_pred_file(pred_file)
    references, references_token = read_gold_file(gold_file, args)
    return hypotheses, hypotheses_token, references, references_token


def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.
    Args:
        segment: text segment from which n-grams will be extracted.
        max_order: maximum length in tokens of the n-grams returned by this
        methods.
    Returns:
        The Counter containing all n-grams upto max_order in segment
        with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def _compute_bleu(reference_corpus, translation_corpus, max_order=4, smooth=False):
    """Computes BLEU score of translated segments against one or more references.
    Args:
        reference_corpus: list of lists of references for each translation. Each
            reference should be tokenized into a list of tokens.
        translation_corpus: list of translations to score. Each translation
            should be tokenized into a list of tokens.
        max_order: Maximum n-gram order to use when computing BLEU score.
        smooth: Whether or not to apply Lin et al. 2004 smoothing.
    Returns:
        3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
            precisions and brevity penalty.
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus, translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                             (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                 possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / ratio)

    bleu = geo_mean * bp

    return (bleu, precisions, bp, ratio, translation_length, reference_length)


def calculate_corpus_bleu(reference_texts, candidate_texts, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=None,
                          auto_reweigh=False):
    """
    Calculate the BLEU score for an entire test set of candidate sentences against their reference translations.

    :param reference_texts: List of lists of reference translations, where each inner list contains the reference translations for a single candidate sentence.
    :param candidate_texts: List of candidate sentences, where each sentence is a list of tokens.
    :param weights: Tuple of weights for 1, 2, 3, and 4-gram precision. Defaults to equal weights.
    :param smoothing_function: Smoothing function to use. Defaults to None, which means no smoothing.
    :param auto_reweigh: Automatically reweigh the components if there are not enough references.
    :return: The BLEU score for the entire corpus.
    """
    # If no smoothing function is provided, use the nltk default method 1
    if smoothing_function is None:
        smoothing_function = SmoothingFunction().method1

    # Reformat reference texts to match the expected input format of corpus_bleu
    reformatted_references = [[ref for ref in refs] for refs in reference_texts]

    # Calculate the BLEU score
    bleu_score = corpus_bleu(reformatted_references, candidate_texts, weights=weights,
                             smoothing_function=smoothing_function, auto_reweigh=auto_reweigh)

    return bleu_score


def calculate_bleu(references: Dict[str, List[List[str]]],
                   predictions: Dict[str, List[str]],
                   max_order=4,
                   smooth=False) -> float:
    reference_corpus = []
    prediction_corpus = []

    for instance_id, reference_sents in references.items():
        try:
            prediction_sent = predictions[instance_id]
        except KeyError:
            logging.error("Missing prediction for instance '%s'.", instance_id)
            sys.exit(EXIT_STATUS_PREDICTION_MISSING)

        del predictions[instance_id]

        prediction_corpus.append(prediction_sent)
        reference_corpus.append(reference_sents)

    if len(predictions) > 0:
        logging.error("Found %d extra predictions, for example: %s", len(predictions),
                      ", ".join(list(predictions.keys())[:3]))
        sys.exit(EXIT_STATUS_PREDICTIONS_EXTRA)
    # caculate by nltk
    # score_nltk = calculate_corpus_bleu(reference_corpus, prediction_corpus)
    score = _compute_bleu(reference_corpus, prediction_corpus,
                          max_order=max_order, smooth=smooth)[0]

    return score


def eval_rouge(ori_hypotheses, ori_references, args):
    # ori_hypotheses, _, ori_references, _ = process_data(pred_endings_file, gold_file, args)

    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                            max_n=4,
                            limit_length=True,
                            length_limit=100,
                            length_limit_type='words',
                            apply_avg=True,
                            apply_best=False,
                            alpha=0.5,  # Default F1_score
                            weight_factor=1.2,
                            stemming=True)

    references = []
    hypotheses = []
    for r, h in zip(ori_references, ori_hypotheses):
        clean_reference = [_clean_text(i) for i in r]
        clean_hypothesis = _clean_text(h)
        if len(clean_hypothesis) == 0:
            assert False
        references.append(clean_reference)
        hypotheses.append(clean_hypothesis)
        # print(hypotheses)
        # exit()
    assert len(references) == len(hypotheses)
    scores = evaluator.get_scores(hypotheses, references)
    return {'rouge_all': scores}


def eval_bert_score(ori_hypotheses, ori_references, args, bert_model="bert-base-uncased"):
    # ori_hypotheses, _, ori_references, _ = process_data(pred_endings_file, gold_file, args)

    references = []
    hypotheses = []
    for r, h in zip(ori_references, ori_hypotheses):
        clean_reference = [_clean_text(i) for i in r]
        clean_hypothesis = _clean_text(h)
        if len(clean_hypothesis) == 0:
            assert False
        hypotheses.append(clean_hypothesis)
        references.append(clean_reference)

    assert len(references) == len(hypotheses)
    P, R, F1 = bert_score(hypotheses, references, model_type=bert_model,
                          device='cuda' if torch.cuda.is_available() else 'cpu')
    return {
        "bert_score_P": P.mean().item(),
        "bert_score_R": R.mean().item(),
        "bert_score_F1": F1.mean().item(),
        # "bert_score_P_by_instance": [float(f) for f in list(P.numpy())],
        # "bert_score_R_by_instance": [float(f) for f in list(R.numpy())],
        # "bert_score_F1_by_instance": [float(f) for f in list(F1.numpy())],
    }


def eval_sent_bert(ori_hypotheses, ori_references, args):
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    # ori_hypotheses, _, ori_references, _ = process_data(pred_endings_file, gold_file, args)

    references = []
    hypotheses = []
    for r, h in zip(ori_references, ori_hypotheses):
        for i in r:
            clean_reference = _clean_text(i)
            clean_hypothesis = _clean_text(h)
            if len(clean_hypothesis) == 0:
                assert False
            references.append(clean_reference)
            hypotheses.append(clean_hypothesis)

    # Compute embedding for both lists
    embeddings1 = sbert_model.encode(hypotheses, convert_to_tensor=True)
    embeddings2 = sbert_model.encode(references, convert_to_tensor=True)

    # Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings1, embeddings2)

    sent_bert_score = 0
    cnt = 0
    for i in range(len(ori_hypotheses)):
        sent_bert_score += max(cosine_scores[i * 3][i * 3], cosine_scores[i * 3 + 1][i * 3 + 1],
                               cosine_scores[i * 3 + 2][i * 3 + 2])
        # sent_bert_score += cosine_scores[i][i]
    sent_bert_score /= len(ori_hypotheses)
    sbert_model = None
    gc.collect()
    torch.cuda.empty_cache()
    return {'sent-bert': sent_bert_score}


def pad_sequence(sequence, n, pad_left=False, pad_right=False,
                 left_pad_symbol=None, right_pad_symbol=None):
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((left_pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (right_pad_symbol,) * (n - 1))
    return sequence


def ngrams(sequence, n, pad_left=False, pad_right=False,
           left_pad_symbol=None, right_pad_symbol=None):
    sequence = pad_sequence(sequence, n, pad_left, pad_right,
                            left_pad_symbol, right_pad_symbol)

    history = []
    while n > 1:
        history.append(next(sequence))
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]


def distinct_n_sentence_level(sent, n):
    distinct_ngrams = set(ngrams(sent, n))
    return len(distinct_ngrams) / len(sent)


def eval_DIST(pred_endings_file, gold_file, args):
    ori_hypotheses, _, _, _ = process_data(pred_endings_file, gold_file, args)

    hypotheses = []
    for h in ori_hypotheses:
        clean_hypothesis = _clean_text(h)
        if len(clean_hypothesis) == 0:
            assert False
        hyp_tokens = clean_hypothesis.strip().split(" ")
        hypotheses.append(hyp_tokens)
        # print(hypotheses)
        # exit()

    score_1 = sum(distinct_n_sentence_level(hypothese, 1) for hypothese in hypotheses) / len(hypotheses)
    score_2 = sum(distinct_n_sentence_level(hypothese, 2) for hypothese in hypotheses) / len(hypotheses)

    return {
        "dist_1": score_1,
        "dist_2": score_2
    }


def read_references(filename: str, args) -> List[List[List[str]]]:
    references = {}
    with open(filename, "rt", encoding="UTF-8", errors="replace") as f:
        if args.task == "comve":

            reader = csv.reader(f)
            try:
                for i, row in enumerate(list(reader)):
                    instance_id = i
                    references_raw1 = row[1]
                    references_raw2 = row[2]
                    references_raw3 = row[3]
                    tokens = []
                    for ref in [references_raw1, references_raw2, references_raw3]:
                        if ref:
                            tokens.append(ref.split())

                    if len(tokens) == 0:
                        logging.error(
                            "No reference sentence in file %s on line %d", filename, reader.line_num)
                        sys.exit(EXIT_STATUS_ANSWERS_MALFORMED)

                    references[row[0]] = tokens

            except csv.Error as e:
                logging.error('file %s, line %d: %s', filename, reader.line_num, e)
                sys.exit(EXIT_STATUS_ANSWERS_MALFORMED)
        elif args.task == "ecqa":
            data = [json.loads(line) for line in f]
            data_dict = {entry["id"]: entry["positives"] + entry["negatives"] for entry in data}

            # Split each sentence in the list and store them in a new list
            references = {key: [sentence.split() for sentence in value] for key, value in data_dict.items()}

    return references


def read_predictions(filename: str) -> List[List[str]]:
    predictions = {}
    with open(filename, "rt", encoding="UTF-8", errors="replace") as f:
        reader = csv.reader(f)
        next(reader)
        try:
            for i, row in enumerate(reader):
                # instance_id = i
                prediction_raw = row[1]
                tokens = prediction_raw.split()
                predictions[row[0]] = tokens
        except csv.Error as e:
            logging.error('file %s, line %d: %s', filename, reader.line_num, e)
            sys.exit(EXIT_STATUS_PREDICTIONS_MALFORMED)

    return predictions


def read_predictions_json(filename: str) -> List[List[str]]:
    predictions = {}
    with open(filename, "rt", encoding="UTF-8", errors="replace") as f:
        reader = json.loads(f.read())
        for i, row in enumerate(reader):
            # row = row[0]
            instance_id = i
            if "###" in row: row = row.split("###")[0]
            # prediction_raw = row[1]
            tokens = row.split()
            if len(tokens) == 0: tokens = ['.']
            predictions[instance_id] = tokens

    return predictions

def eval(args, isDebug: bool):
    results = {}
    # BLEU

    references = read_references(args.gold_file, args)
    predictions = read_predictions(args.pred_file)
    predictions = {k: v for k, v in predictions.items() if v}

    empty_text = 0
    error_text = 0
    uncompleted_text=0
    refs, cands = [], []
    keys_to_remove = []

    for key, value in predictions.items():
        value_str=" ".join(value)
        if len(value) < 1:
            logging.info("empty output...")
            empty_text += 1
            keys_to_remove.append(key)
            continue
        if "_" in value[0]:
            logging.info("error output: "+value_str)
            error_text += 1
            keys_to_remove.append(key)
        elif not contains_word(value_str):
            logging.info("error output: "+value_str)
            error_text += 1
            keys_to_remove.append(key)

        if not value_str.strip().endswith("."):
            # logging.info("uncompleted output: " + value_str)

            if args.process_uncompleted:
                last_period_index = value_str.rfind(".")
                if last_period_index != -1:
                    # 通过最后一个句号的位置，找到对应的单词索引
                    char_count = 0
                    for i, word in enumerate(value):
                        char_count += len(word) + 1  # 加1是因为有空格
                        if char_count >= last_period_index + 1:
                            value = value[:i + 1]
                            break
                predictions[key] = value

            uncompleted_text+= 1

    logging.info("Total error output: " + str(error_text))
    logging.info("Total uncompleted output: " + str(uncompleted_text))

    # 从原始字典中删除要删除的键
    for key in keys_to_remove:
        del predictions[key]

    # 重新迭代字典的剩余部分进行处理
    for key, value in predictions.items():
        cands.append(' '.join(value))
        ref_list = []
        for ref in references[key][:3]:
            ref_list.append(' '.join(ref))
        refs.append(ref_list)

    # predictions_copy = copy.deepcopy(predictions)
    len_predictions = len(predictions)
    references = {k: references[k] for k in predictions if k in references}

    logging.info('**' * 20 + "Evaluating" + '**' * 20)

    processed_refs = [list(x) for x in zip(*refs)]
    n = NLGEval(metrics_to_omit=['SPICE', 'EmbeddingAverageCosineSimilarity', 'SkipThoughtCS',
                                 'EmbeddingAverageCosineSimilarity', 'EmbeddingAverageCosineSimilairty',
                                 'VectorExtremaCosineSimilarity', 'GreedyMatchingScore'])
    scores = n.compute_metrics(processed_refs, cands)

    # BLEU
    bleu = calculate_bleu(references, predictions,
                          max_order=args.max_order, smooth=args.smooth)
    results.update({"bleu_score": bleu})
    # # BLEURT
    # bleurt_result = bleurt(references=references, hypothesis=predictions_copy, num_refs=3)
    # results.update({"bleurt": bleurt_result})
    # ROUGE
    results.update(eval_rouge(cands, refs, args))
    rscore = results.pop('rouge_all')
    results['rouge-l'] = rscore['rouge-l']
    # BERTScore
    results.update(eval_bert_score(cands, refs, args))
    # S-BERT
    results.update(eval_sent_bert(cands, refs, args))
    # # DIST-N
    # results.update(eval_DIST(args.pred_file, args.gold_file, args=args))

    logging.info(results)
    rouge_l = 0
    bert_score_F1 = 0
    sent_bert = 0
    bleu_score = 0

    for k, v in results.items():
        if k == 'bleu_score':
            bleu_score = round(v, 4)
            logging.info(k + ":" + str(bleu_score))
        elif k == "rouge-l":
            rouge_l = round(v['f'], 4)
            logging.info(k + ":" + str(rouge_l))
        elif k == "bert_score_F1":
            bert_score_F1 = round(v, 4)
            logging.info(k + ":" + str(bert_score_F1))
        elif k == "sent-bert":
            sent_bert = v
            logging.info(k + ":" + str(v))

    logging.info('METEOR:' + str(scores['METEOR']))
    logging.info('length_pred:' + str(len_predictions))
    logging.info('ROUGE_L:' + str(scores['ROUGE_L']))
    logging.info('CIDEr:' + str(scores['CIDEr']))
    logging.info('Bleu_1:' + str(scores['Bleu_1']))
    logging.info('Bleu_2:' + str(scores['Bleu_2']))
    logging.info('Bleu_3:' + str(scores['Bleu_3']))
    logging.info('Bleu_4:' + str(scores['Bleu_4']))
    # elif k == "bleurt":
    #     logging.info(k + ":" + str(v))
    if not isDebug:
        wandb.log({"bleu_score": bleu_score,
                   # "bleurt": bleurt_result,
                   "rouge_l_1": rouge_l,
                   "bert_score_F1": bert_score_F1,
                   "sent_bert": sent_bert.float(),
                   'Bleu_1': scores['Bleu_1'],
                   'Bleu_2': scores['Bleu_2'],
                   'Bleu_3': scores['Bleu_3'],
                   'Bleu_4': scores['Bleu_4'],
                   'METEOR': scores['METEOR'],
                   'ROUGE_L': scores['ROUGE_L'],
                   'CIDEr': scores['CIDEr'],
                   "concept_num": args.num_concept,
                   "length_pred": len_predictions,
                   "prompts_style":args.prompts_style,
                   "random_knowledge": args.random_knowledge
                   })
    logging.info('**' * 20 + args.pred_file + '**' * 20)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', '-p', help='predict csv file name. '
                                                  'e.g python3 metric.py -p pre_file.csv -i gold_file.csv')
    parser.add_argument('--gold_file', '-g', help='ground_truth csv file name. '
                                                  'e.g python3 metric.py -p pre_file.csv -i gold_file.csv')
    parser.add_argument(
        '--max_order', default=4, type=int, help='Maximum n-gram order to use when computing BLEU score')
    parser.add_argument('--smooth', action='store_true',
                        help='Whether or not to apply Lin et al. 2004 smoothing')
    parser.add_argument('--task', '-t', type=str, default="comve", help='[comve, esnli]')
    args = parser.parse_args()
