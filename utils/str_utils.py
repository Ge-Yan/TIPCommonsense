import string

import nltk
from nltk import Tree

from constituent_treelib import ConstituentTree, Language
from nltk.corpus import wordnet
import contractions
import re
import spacy
import benepar

def extract_content_before_substring(knowledge_content, substring):
    """
    Extracts the portion of the given content that appears before the first occurrence of the specified substring.

    Parameters:
    knowledge_content (str): The input string from which content is to be extracted.
    substring (str): The substring to find in the input string.

    Returns:
    str: The extracted content before the first occurrence of the substring. If the substring is not found, returns the original content.
    """
    return knowledge_content.split(substring)[0] if substring in knowledge_content else knowledge_content

def find_phrase_indexes(tokens, phrases):

    text = ''.join([token.replace('Ġ', '') for token in tokens])
    token_positions = []
    start_pos = 0

    for index,token in enumerate(tokens):
        end_pos = start_pos + len(token.replace('Ġ', ''))
        token_positions.append({'start': start_pos, 'end': end_pos, 'index': index})
        start_pos = end_pos


    phrase_positions = []

    for phrase in phrases:
        phrase_text = phrase.replace(' ', '')  
        start_pos = text.find(phrase_text)
        end_pos = start_pos + len(phrase_text)
        phrase_positions.append({'start': start_pos, 'end': end_pos})

    phrase_indexes = []

    for phrase_pos, phrase in zip(phrase_positions, phrases):
        start_pos = phrase_pos['start']
        end_pos = phrase_pos['end']


        start_index = next((token_info['index'] for token_info in token_positions if
                            start_pos >= token_info['start'] and start_pos < token_info['end']), None)
        end_index = next((token_info['index'] for token_info in token_positions if
                          end_pos > token_info['start'] and end_pos <= token_info['end']), None)


        if start_index is not None and end_index is not None:
            indexes_range = list(range(start_index, end_index + 1))
            phrase_indexes.append({phrase: indexes_range})

    return phrase_indexes

def contains_word(s):

    pattern = r'\b[A-Za-z]+\b'
    match = re.search(pattern, s)
    return bool(match)


def clean_sentence(sentence):

    cleaned_sentence = sentence.replace('\n', ' ')
    cleaned_sentence = re.sub(r'\s+', ' ', cleaned_sentence)
    cleaned_sentence = cleaned_sentence.strip()
    return cleaned_sentence


class TreeUtils:
    def __init__(self):

        self.nlp = spacy.load('en_core_web_md')
        if spacy.__version__.startswith('2'):
            self.nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
        else:
            self.nlp.add_pipe("benepar", config={"model": "benepar_en3"})

    def create_constituent_tree(self, sentence):

        tree = self.nlp(sentence)
        sent = list(tree.sents)[0]
        tree = Tree.fromstring(sent._.parse_string)

        return tree

    def extract_nps(self, tree):
        nps = []
        if isinstance(tree, nltk.Tree):
            if tree.label() == 'NP' and "NP" not in [i.label() for i in tree]:
                nps.append(' '.join(tree.leaves()))
            for child in tree:
                nps.extend(self.extract_nps(child))
        return nps

def remove_punctuations(sentence):

    sentence = sentence.lower()
    sentence = contractions.fix(sentence, slang=True)

    punctuation = ['.']
    for p in punctuation:
        sentence = sentence.replace(p, '')

    sentence = sentence.rstrip()


    cleaned_sentence = re.sub(r'\s+', ' ', sentence).strip()
    return cleaned_sentence


def ensure_period(sentence):

    if not sentence.endswith('.'):
        sentence += '.'
    return sentence




def process_sentence(sentence,remove_punctuation: bool):
    # sentence = singularize_nouns_with_nltk(sentence)
    sentence = remove_extra_spaces(sentence)
    sentence = ensure_period(sentence)
    sentence = sentence.lower()
    # sentence = replace_pronouns(sentence)
    if remove_punctuation:
        sentence = remove_punctuations(sentence)
    return sentence


def add_space_before_words(sentence):
    words = sentence.split()
    for i in range(len(words)):
        words[i] = ' ' + words[i]
    return words


def remove_extra_spaces(sentence):

    sentence = sentence.strip()
    words = sentence.split()
    cleaned_sentence = ' '.join(words)
    return cleaned_sentence


def singularize_nouns_with_nltk(text):

    words = nltk.word_tokenize(text)
    tagged_words = nltk.pos_tag(words)
    singular_words = []
    for word, pos in tagged_words:
        if pos.startswith('NN'):  
            singular_word = wordnet.morphy(word, wordnet.NOUN)
            if singular_word is not None:
                singular_words.append(singular_word)
            else:
                singular_words.append(word)
        else:
            singular_words.append(word)

    return ' '.join(singular_words)





def conceptnet_relation_to_nl(ent):

    relation_to_language = {'/r/AtLocation': 'is at the location of the',
                            '/r/CapableOf': 'is capable of',
                            '/r/Causes': 'causes',
                            '/r/CausesDesire': 'causes the desire of',
                            '/r/CreatedBy': 'is created by',
                            '/r/DefinedAs': 'is defined as',
                            '/r/DerivedFrom': 'is derived from',
                            '/r/Desires': 'desires',
                            '/r/Entails': 'entails',
                            '/r/EtymologicallyDerivedFrom': 'is etymologically derived from',
                            '/r/EtymologicallyRelatedTo': 'is etymologically related to',
                            '/r/FormOf': 'is an inflected form of',
                            '/r/HasA': 'has a',
                            '/r/HasContext': 'appears in the context of',
                            '/r/HasFirstSubevent': 'is an event that begins with subevent',
                            '/r/HasLastSubevent': 'is an event that concludes with subevent',
                            '/r/HasPrerequisite': 'has prerequisite is',
                            '/r/HasProperty': 'has an attribute is',
                            '/r/HasSubevent': 'has a subevent is',
                            '/r/InstanceOf': 'runs an instance of',
                            '/r/IsA': 'is a',
                            '/r/LocatedNear': 'is located near',
                            '/r/MadeOf': 'is made of',
                            '/r/MannerOf': 'is the manner of',
                            '/r/MotivatedByGoal': 'is a step toward accomplishing the goal',
                            '/r/NotCapableOf': 'is not capable of',
                            '/r/NotDesires': 'does not desire',
                            '/r/NotHasProperty': 'has no attribute',
                            '/r/PartOf': 'is a part of',
                            '/r/ReceivesAction': 'receives action for',
                            '/r/RelatedTo': 'is related to',
                            '/r/SimilarTo': 'is similar to',
                            '/r/SymbolOf': 'is the symbol of',
                            '/r/UsedFor': 'is used for',
                            }

    ent_values = 'i {} {}'.format(relation_to_language.get(ent[1], ''),
                                  ent[0].replace('_', ' '))
    ent_values = ent_values.split(" ")[1:]

    return ent_values


def merge_lists(input_id, special_lists):
    result = []

    for item in input_id:
        found = False
        for special_list in special_lists:
            if item in special_list:
                if special_list not in result:
                    result.append(special_list)
                found = True
                break
        if not found:
            result.append(item)

    return result
