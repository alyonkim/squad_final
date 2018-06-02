import tensorflow as tf
import numpy as np
from tqdm import tqdm_notebook as tqdm
import json
import msgpack

from constants import *


def get_data(path):
    with open(path, 'rb') as f:
        data = msgpack.load(f, encoding='utf8')
    return data


def get_train():
    data = get_data("./biases/data.msgpack?dl=1")
    train = data['train']
    dev = data['dev']
    return train, dev


def get_transfers():
    transfers = get_data("./biases/meta.msgpack?dl=1")
    vocab_tag = transfers['vocab_tag']
    embedding = transfers['embedding']
    vocab_ent = transfers['vocab_ent']
    vocab = transfers['vocab']
    return vocab_tag, embedding, vocab_ent, vocab


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