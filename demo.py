import numpy as np
import tensorflow as tf
import spacy 

from constants import *
from prepare import *

def main():
    _, embedding, _, _ = get_transfers()
    context = input("Type your paragraph:\n")
    question = input("Type your question:\n")

    sess = tf.Session() 
    saver = tf.train.import_meta_graph(USE_MODEL_PATH)
    saver.restore(sess, tf.train.latest_checkpoint('./biases/'))

    graph = tf.get_default_graph()

    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    questions = graph.get_tensor_by_name("questions:0")
    contexts = graph.get_tensor_by_name("contexts:0")
    answer_begins = graph.get_tensor_by_name("answer_begins:0")
    answer_ends = graph.get_tensor_by_name("answer_ends:0")
    question_length = graph.get_tensor_by_name("question_length:0")
    context_length = graph.get_tensor_by_name("context_length:0")

    dense_begin = graph.get_tensor_by_name("dense_begin:0")
    dense_end = graph.get_tensor_by_name("dense_end:0")

    (
        questions_,
        contexts_,
        question_length_,
        context_length_,
    ) = gen_demo_batch(
        BATCH_SIZE,
        context_tokens, question_tokens,
        tag_emb, entity_emb, context_features,
        embedding
    )

    (
        begin_probs_,
        end_probs_
    ) = sess.run([dense_begin, dense_end],
                 feed_dict={
                     keep_prob: 1.0,
                     questions: questions_,
                     contexts: contexts_,
                     question_length: question_length_,
                     context_length: context_length_
                 })
    begin = np.argmax(begin_probs_[0])
    end = np.argmax(end_probs_[0])
    print("Predicted answer:\n", context_text[context_token_span[begin][0]:context_token_span[end][1]+1])
     
if __name__ == '__main__':
    main()