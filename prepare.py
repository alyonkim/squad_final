import tensorflow as tf
import numpy as np
from tqdm import tqdm_notebook as tqdm
import json
import msgpack
import os.path
import spacy

from constants import *


def get_data(path):
    with open(path, 'rb') as f:
        data = msgpack.load(f, encoding='utf8')
    return data


def get_train():
    if not os.path.isfile("./biases/data.msgpack?dl=1"): 
        wget.download('https://www.dropbox.com/s/r33ljlagys0wscb/data.msgpack?dl=1', out="./biases/data.msgpack?dl=1")
    data = get_data("./biases/data.msgpack?dl=1")
    train = data['train']
    dev = data['dev']
    return train, dev


def get_transfers():
    if not os.path.isfile("./biases/meta.msgpack?dl=1"): 
        wget.download('https://www.dropbox.com/s/83txkgiqmdlv1m3/meta.msgpack?dl=1', out="./biases/meta.msgpack?dl=1")
    transfers = get_data("./biases/meta.msgpack?dl=1")
    vocab_tag = transfers['vocab_tag']
    embedding = transfers['embedding']
    vocab_ent = transfers['vocab_ent']
    vocab = transfers['vocab']

    return vocab_tag, embedding, vocab_ent, vocab


def get_smth2ind():
    vocab_tag, _, vocab_ent, vocab = get_transfers()

    tag2ind = dict()
    ent2ind = dict()
    word2ind = dict()
    for i in tqdm(range(len(vocab_tag))):
        tag2ind[vocab_tag[i]] = i
    for i in tqdm(range(len(vocab_ent))):
        ent2ind[vocab_ent[i]] = i
    for i in tqdm(range(len(vocab))):
        word2ind[vocab[i]] = i

    return tag2ind, ent2ind, word2ind


def find_dev_answer(text, answer, token_span):
    index = text.find(answer)
    begin = len(token_span)
    end = len(token_span)
    find_b = False
    find_e = False
    for i in range(len(token_span)):
        if token_span[i][0] == index:
            find_b = True
            begin = i
            break
    for i in range(begin, len(token_span)):
        if token_span[i][1] == index + len(answer):
            find_e = True
            end = i
            break
    return begin, end, find_b and find_e


def del_bad_answers_dev(dev):
    super_bad = list()
    for i in range(len(dev)):
        bad = list()
        for j in range(len(dev[i][8])):
            _, _, good = find_dev_answer(dev[i][6], dev[i][8][j], dev[i][7])
            if not good:
                bad.append(dev[i][8][j])
        for j in bad:
            dev[i][8].remove(j)
        if len(dev[i][8]) == 0:
            super_bad.append(dev[i])
    for i in super_bad:
        dev.remove(i)
    return dev


def prepare_dev(dev):
    dev = del_bad_answers_dev(dev)
    for i in tqdm(range(len(dev))):
        begin, end, _ = find_dev_answer(dev[i][6], dev[i][8][0], dev[i][7])
        dev[i].remove(dev[i][8])
        dev[i].append(begin)
        dev[i].append(end)
    return dev


def preprocessing(size, dataset):
    context_tokens = np.zeros((size, CONTEXT_MAX_SIZE), dtype=np.int32)
    question_tokens = np.zeros((size, QUESTION_MAX_SIZE), dtype=np.int32)

    context_features = np.zeros((size, CONTEXT_MAX_SIZE, 4), dtype=np.float32)
    tag_emb = np.zeros((size, CONTEXT_MAX_SIZE), dtype=np.int32)
    entity_emb = np.zeros((size, CONTEXT_MAX_SIZE), dtype=np.int32)

    answer_begin = np.zeros((TRAIN_SIZE), dtype=np.int32)
    answer_end = np.zeros((TRAIN_SIZE), dtype=np.int32)

    context_text = list()
    context_token_span = list()

    for i in tqdm(range(size)):
        context_text.append(dataset[i][6])
        context_token_span.append(dataset[i][7])
        current_context_size = len(dataset[i][1])
        current_qustion_size = len(dataset[i][5])
        context_tokens[i][:current_context_size] = dataset[i][1]
        for j in range(current_context_size):
            for k in range(3):
                if dataset[i][2][j][k]:
                    context_features[i][j][k] = 1
            context_features[i][j][3] = dataset[i][2][j][3]
            
            tag_emb[i][j] = dataset[i][3][j]
            entity_emb[i][j] = dataset[i][4][j]
        question_tokens[i][:current_qustion_size] = dataset[i][5]
        answer_begin[i] = dataset[i][8]
        answer_end[i] = dataset[i][9]

    return (
        context_tokens, context_features,
        tag_emb, entity_emb, question_tokens,
        context_text, context_token_span,
        answer_begin, answer_end
    )


def get_indices(size):
    indices = np.array([i for i in range(size)], dtype=np.int32)
    return indices


def get_processed_data():
    train, dev = get_train()
    dev = prepare_dev(dev)
    return (
        preprocessing(TRAIN_SIZE, train),
        preprocessing(DEV_SIZE, dev),
        get_indices(TRAIN_SIZE),
        get_indices(DEV_SIZE)
    )


def gen_batch(
        size, batch_number, indices,
        context_tokens, question_tokens,
        tag_emb, entity_emb, context_features,
        answer_begin, answer_end,
        embedding
        ):  
    l = indices[(batch_number*size):(batch_number + 1) * size]
    contexts_ = np.zeros((size, CONTEXT_MAX_SIZE, EMBEDDING_SIZE_CONTEXT), dtype=np.float32)
    context_lengths_ = np.zeros((size), dtype=np.int32)
    questions_ = np.zeros((size, QUESTION_MAX_SIZE, EMBEDDING_SIZE_QUESTION), dtype=np.float32)
    question_lengths_ = np.zeros((size), dtype=np.int32)
    begin = np.zeros((size), dtype=np.int32)
    end = np.zeros((size), dtype=np.int32)
    for i in range(size):
        context = context_tokens[l[i]]
        question = question_tokens[l[i]]
        count = 0
        for j in range(CONTEXT_MAX_SIZE):
            current_tag = np.zeros(TAG_SIZE, dtype=np.float32)
            current_tag[tag_emb[l[i]][j]] = 1.0
            current_entity = np.zeros(ENTITY_SIZE, dtype=np.float32)
            current_entity[entity_emb[l[i]][j]] = 1.0
            
            current_emb = np.append(embedding[context[j]], current_tag)
            current_emb = np.append(current_emb, current_entity)
            current_emb = np.append(current_emb, context_features[l[i]][j])
            
            contexts_[i][j] = current_emb
            if context[j] != 0:
                count += 1
        context_lengths_[i] = count
        count = 0
        for j in range(QUESTION_MAX_SIZE):
            questions_[i][j] = (embedding[question[j]])
            if question[j] != 0:
                count += 1
        question_lengths_[i] = count
        begin[i] = answer_begin[l[i]]
        end[i] = answer_end[l[i]]
    return questions_, contexts_, question_lengths_, context_lengths_, begin, end


def demo_data_preprocessing(context, question):
    tag2ind, ent2ind, word2ind = get_smth2ind()
    nlp = spacy.load('en_core_web_sm')

    context_tokens = np.zeros((CONTEXT_MAX_SIZE), dtype=np.int32)
    question_tokens = np.zeros((QUESTION_MAX_SIZE), dtype=np.int32)

    context_features = np.zeros((CONTEXT_MAX_SIZE, 4), dtype=np.float32)
    tag_emb = np.zeros((CONTEXT_MAX_SIZE), dtype=np.int32)
    entity_emb = np.zeros((CONTEXT_MAX_SIZE), dtype=np.int32)

    context_text = context
    context_token_span = list()

    context = nlp(context)
    question = nlp(question)

    context_tokens[:len(context)] = np.array([word2ind[token.text] for token in context])
    question_tokens[:len(question)] = np.array([word2ind[token.text] for token in question])
    tag_emb[:len(context)] = np.array([tag2ind[token.tag_] for token in context])
    
    unique, counts = np.unique(context_tokens[:len(context)], return_counts=True)
    tf = dict(zip(unique, counts))
    
    i = 0
    for context_token in context:
        context_token_span.append([i, i + len(context_token.text) - 1])
        i += len(context_token.text)
    
    for i in range(len(context)):
        token = context[i]
        if token.text in word2ind and word2ind[token.text] in question_tokens:
            context_features[i][0] = 1.0
        if token.lower_ in word2ind and word2ind[token.lower_] in question_tokens:
            context_features[i][1] = 1.0
        if token.lemma_ in word2ind and word2ind[token.lemma_] in question_tokens:
            context_features[i][2] = 1.0
        if token.text in word2ind and word2ind[token.text] in tf:
            context_features[i][3] = tf[word2ind[token.text]] / len(context)
    
    for ent in context.ents:
        ent.start_char, ent.end_char, ent.label_
        begin = end = 0
        for i in range(len(context_token_span)):
            if context_token_span[i][0] == ent.start_char:
                begin = i
                for j in range(i, len(context_token_span)):
                    if context_token_span[j][1] == ent.end_char - 1:
                        end = j
                        break;
                break;
        for i in range(begin, end + 1):
            entity_emb[i] = ent2ind[ent.label_]
        

    return (
        context_tokens, context_features,
        tag_emb, entity_emb, question_tokens,
        context_text, context_token_span
    )


def gen_demo_batch(
        size,
        context_tokens, question_tokens,
        tag_emb, entity_emb, context_features,
        answer_begin, answer_end,
        embedding
        ):
    context_tokens = np.array([context_tokens for i in range(size)])
    question_tokens = np.array([question_tokens for i in range(size)])
    tag_emb = np.array([tag_emb for i in range(size)])
    entity_emb = np.array([entity_emb for i in range(size)])
    context_features = np.array([context_features for i in range(size)])

    l = indices[(batch_number*size):(batch_number + 1) * size]
    contexts_ = np.zeros((size, CONTEXT_MAX_SIZE, EMBEDDING_SIZE_CONTEXT), dtype=np.float32)
    context_lengths_ = np.zeros((size), dtype=np.int32)
    questions_ = np.zeros((size, QUESTION_MAX_SIZE, EMBEDDING_SIZE_QUESTION), dtype=np.float32)
    question_lengths_ = np.zeros((size), dtype=np.int32)
    begin = np.zeros((size), dtype=np.int32)
    end = np.zeros((size), dtype=np.int32)
    for i in range(size):
        context = context_tokens[l[i]]
        question = question_tokens[l[i]]
        count = 0
        for j in range(CONTEXT_MAX_SIZE):
            current_tag = np.zeros(TAG_SIZE, dtype=np.float32)
            current_tag[tag_emb[l[i]][j]] = 1.0
            current_entity = np.zeros(ENTITY_SIZE, dtype=np.float32)
            current_entity[entity_emb[l[i]][j]] = 1.0
            
            current_emb = np.append(embedding[context[j]], current_tag)
            current_emb = np.append(current_emb, current_entity)
            current_emb = np.append(current_emb, context_features[l[i]][j])
            
            contexts_[i][j] = current_emb
            if context[j] != 0:
                count += 1
        context_lengths_[i] = count
        count = 0
        for j in range(QUESTION_MAX_SIZE):
            questions_[i][j] = (embedding[question[j]])
            if question[j] != 0:
                count += 1
        question_lengths_[i] = count
        begin[i] = answer_begin[l[i]]
        end[i] = answer_end[l[i]]
    return questions_, contexts_, question_lengths_, context_lengths_, begin, end



def F1_single(begin_predict, end_predict, begin_true, end_true):
    TP = min(end_true, end_predict) - max(begin_predict, begin_true)
    if TP < 0:
        TP = 0
    FPN = max(end_true, end_predict) - min(begin_predict, begin_true) - TP
    if FPN < 0:
        FPN = 0
    if FPN == 0 and TP == 0:
        FPN = 10
    return ((2 * TP) / (2 * TP + FPN))


def F1_batch(begin_predict, end_predict, begin_true, end_true):
    size = len(begin_predict)
    res = 0
    for i in range(size):
        res += F1_single(
            np.argmax(begin_predict[i]),
            np.argmax(end_predict[i]),
            begin_true[i],
            end_true[i]
        )
    return res/size



def main():
    print("Prepare script checked")
    return
    

if __name__ == '__main__':
    main()