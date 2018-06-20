import numpy as np
import tensorflow as tf
import spacy 

from constants import *
from prepare import *

def main():
    _, embedding, _, _ = get_transfers()
    tag2ind, ent2ind, word2ind = get_smth2ind()
    nlp = spacy.load('en_core_web_sm')

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

    stop = ''

    while (stop != 'n'):
        stop = ''
        context = input("Type your paragraph:\n")
        print()
        question = input("Type your question:\n")
        print()
        answer = ''
        try:
            (
                context_tokens, context_features,
                tag_emb, entity_emb, question_tokens,
                context_text, context_token_span
            ) = demo_data_preprocessing(context, question, tag2ind, ent2ind, word2ind, nlp)

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
            answer = context_text[context_token_span[begin][0]:context_token_span[end][1]]
            print("Predicted answer:\n", answer)
        except:
            print("Sorry, something went wrong. Probably we don't know some words you typed. Please, try again with other context or question.")
            continue
            
        while (stop != 'y' and stop != 'n'):
            stop = input("Do you want to retry? [y/n]\n")
            print()

     
if __name__ == '__main__':
    main()