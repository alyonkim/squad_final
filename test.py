from prepare import *
from constants import *

import tensorflow as tf
import msgpack
import json
import re
import numpy as np
import collections
import random
import gensim

BATCH_NUMBER_TO_CHECK = 8


def main():    
    train, dev = get_train()
    vocab_tag, embedding, vocab_ent, vocab = get_transfers()
    
    (
        context_tokens, context_features,
        tag_emb, entity_emb, question_tokens,
        context_text, context_token_span,
        answer_begin, answer_end,
        context_tokens_dev, context_features_dev,
        tag_emb_dev, entity_emb_dev, question_tokens_dev,
        context_text_dev, context_token_span_dev,
        answer_begin_dev, answer_end_dev,
        indices_train, indices_dev
    ) = get_processed_data(train, dev)
    
    sess = tf.Session() 
    saver = tf.train.import_meta_graph(USE_MODEL_PATH)
    saver.restore(sess_r,tf.train.latest_checkpoint('./biases/'))

    graph = tf.get_default_graph()

    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    questions = graph.get_tensor_by_name("questions:0")
    contexts = graph.get_tensor_by_name("contexts:0")
    answer_begins = graph.get_tensor_by_name("answer_begins:0")
    answer_ends = graph.get_tensor_by_name("answer_ends:0")
    question_length = graph.get_tensor_by_name("question_length:0")
    context_length = graph.get_tensor_by_name("context_length:0")
    if_train = graph.get_tensor_by_name("if_train:0")

    dense_begin = graph.get_tensor_by_name("dense_begin:0")
    dense_end = graph.get_tensor_by_name("dense_end:0")

    (
        questions_,
        contexts_,
        question_length_,
        context_length_,
        begin_,
        end_
    ) = gen_batch(
        BATCH_SIZE, BATCH_NUMBER_TO_CHECK, indices_dev,
        context_tokens_dev, question_tokens_dev,
        tag_emb_dev, entity_emb_dev, context_features_dev,
        answer_begin_dev, answer_end_dev
    )


    (
        begin_probs_,
        end_probs_
    ) = sess_r.run([dense_begin, dense_end],
                 feed_dict={
                     if_train: False,
                     keep_prob: 1.0,
                     questions: questions_,
                     contexts: contexts_,
                     answer_begins: begin_,
                     answer_ends: end_,
                     question_length: question_length_,
                     context_length: context_length_
                 })
    F1_score = F1_batch(begin_probs_dev, end_probs_dev, begin_dev, end_dev)
    print(F1_score)
    
    
if __name__ == '__main__':
    main()
