import tensorflow as tf
import numpy as np
from tqdm import tqdm_notebook as tqdm
import json
import msgpack
import random

from constants import *


def main():
    (
        (context_tokens, context_features,
        tag_emb, entity_emb, question_tokens,
        context_text, context_token_span,
        answer_begin, answer_end),
        (context_tokens_dev, context_features_dev,
        tag_emb_dev, entity_emb_dev, question_tokens_dev,
        context_text_dev, context_token_span_dev,
        answer_begin_dev, answer_end_dev),
        indices_train, indices_dev
    ) = get_processed_data()
    vocab_tag, embedding, vocab_ent, vocab = get_transfers()
    print("Data downloaded and processed.")

    tf.reset_default_graph()

    keep_prob = tf.placeholder(
        dtype=tf.float32, name="keep_prob")
    questions = tf.placeholder(
        dtype=tf.float32, name="questions", shape=(BATCH_SIZE, QUESTION_MAX_SIZE, EMBEDDING_SIZE_QUESTION))
    contexts = tf.placeholder(
        dtype=tf.float32, name="contexts", shape=(BATCH_SIZE, CONTEXT_MAX_SIZE, EMBEDDING_SIZE_CONTEXT))
    answer_begins = tf.placeholder(
        dtype=tf.int32, name="answer_begins", shape=(BATCH_SIZE))
    answer_ends = tf.placeholder(
        dtype=tf.int32, name="answer_ends", shape=(BATCH_SIZE))
    question_length = tf.placeholder(
        dtype=tf.int32, name="question_length", shape=(BATCH_SIZE))
    context_length = tf.placeholder(
        dtype=tf.int32, name="context_length", shape=(BATCH_SIZE))
    if_train = tf.placeholder(
        dtype=tf.bool, name="if_train")


    # Question

    question_1_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(LSTM_CELL_HIDDEN_SIZE)
    question_1_fw_cell = tf.nn.rnn_cell.DropoutWrapper(question_1_fw_cell, keep_prob)

    question_1_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(LSTM_CELL_HIDDEN_SIZE)
    question_1_bw_cell = tf.nn.rnn_cell.DropoutWrapper(question_1_bw_cell, keep_prob)

    question_output, _ = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=question_1_fw_cell,
        cell_bw=question_1_bw_cell,
        inputs=questions,
        sequence_length=question_length,
        dtype=tf.float32,
        scope="BiLSTM0"
    )
    question_1_output = tf.concat(question_output, axis = -1)

    question_2_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(LSTM_CELL_HIDDEN_SIZE)
    question_2_fw_cell = tf.nn.rnn_cell.DropoutWrapper(question_2_fw_cell, keep_prob)

    question_2_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(LSTM_CELL_HIDDEN_SIZE)
    question_2_bw_cell = tf.nn.rnn_cell.DropoutWrapper(question_2_bw_cell, keep_prob)

    question_output, a = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=question_2_fw_cell,
        cell_bw=question_2_bw_cell,
        inputs=question_1_output,
        sequence_length=question_length,
        dtype=tf.float32,
        scope="BiLSTM1"
    )
    question_2_output = tf.concat(question_output, axis = -1)

    question_attention = tf.layers.dense(
            inputs=question_2_output,
            units=1,
            use_bias=True
    )
    question_mask = tf.sequence_mask(
            lengths=question_length,
            maxlen=QUESTION_MAX_SIZE,
            dtype=tf.float32
    )

    question_attention = tf.reshape(question_attention, (BATCH_SIZE, QUESTION_MAX_SIZE))
    question_attention = tf.multiply(tf.nn.softmax(question_attention), question_mask)
    question_attention = tf.reshape(question_attention, (BATCH_SIZE, QUESTION_MAX_SIZE, 1))
    question_final = tf.matmul(tf.transpose(question_2_output, perm=[0, 2, 1]), question_attention)
    question_final = tf.reshape(question_final, (BATCH_SIZE, 2 * LSTM_CELL_HIDDEN_SIZE))
    question_state = tf.nn.rnn_cell.LSTMStateTuple(question_final, question_final)

    # Context

    context_1_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(2 * LSTM_CELL_HIDDEN_SIZE)
    context_1_fw_cell = tf.nn.rnn_cell.DropoutWrapper(context_1_fw_cell, keep_prob)

    context_1_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(2 * LSTM_CELL_HIDDEN_SIZE)
    context_1_bw_cell = tf.nn.rnn_cell.DropoutWrapper(context_1_bw_cell, keep_prob)

    context_output, _ = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=context_1_fw_cell,
        cell_bw=context_1_bw_cell,
        initial_state_fw = question_state,
        initial_state_bw = question_state,
        inputs=contexts,
        sequence_length=context_length,
        dtype=tf.float32,
        scope="BiLSTM2"
    )
    context_1_output = tf.concat(context_output, axis = -1)

    context_2_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(2 * LSTM_CELL_HIDDEN_SIZE)
    context_2_fw_cell = tf.nn.rnn_cell.DropoutWrapper(context_2_fw_cell, keep_prob)

    context_2_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(2 * LSTM_CELL_HIDDEN_SIZE)
    context_2_bw_cell = tf.nn.rnn_cell.DropoutWrapper(context_2_bw_cell, keep_prob)

    context_output, _ = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=context_2_fw_cell,
        cell_bw=context_2_bw_cell,
        initial_state_fw = question_state,
        initial_state_bw = question_state,
        inputs=context_1_output,
        sequence_length=context_length,
        dtype=tf.float32,
        scope="BiLSTM3"
    )
    context_final = tf.concat(context_output, axis = -1)

    # begin-end probs

    dense_begin = tf.layers.dense(
            inputs=context_final,
            units=2*LSTM_CELL_HIDDEN_SIZE,
            use_bias=True,

    )
    dense_end = tf.layers.dense(
            inputs=context_final,
            units=2*LSTM_CELL_HIDDEN_SIZE,
            use_bias=True
    )

    question_final = tf.reshape(question_final, (BATCH_SIZE, 2 * LSTM_CELL_HIDDEN_SIZE, 1))
    dense_begin = tf.matmul(dense_begin, question_final, name = "dense_begin")
    dense_end = tf.matmul(dense_end, question_final, name = "dense_end")

    # loss

    loss_begin = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=answer_begins,
            logits=tf.reshape(dense_begin, (BATCH_SIZE, CONTEXT_MAX_SIZE)),
            name="softmax_begin"
        )
    )
    loss_end = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=answer_ends,
            logits=tf.reshape(dense_end, (BATCH_SIZE, CONTEXT_MAX_SIZE)),
            name="softmax_end"
        )
    )
    loss = loss_begin + loss_end

    # final probs

    dense_begin = tf.reshape(dense_begin, (BATCH_SIZE, CONTEXT_MAX_SIZE))
    dense_end = tf.reshape(dense_end, (BATCH_SIZE, CONTEXT_MAX_SIZE))

    # optimizer

    optimizer = tf.train.RMSPropOptimizer(0.001)
    gradients, variables = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 1000)
    train_step = optimizer.apply_gradients(zip(gradients, variables))

    print("Graph built. Start learning")

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    try:
        for j in tqdm(range(TRAINING_EPOCHS)):
            np.random.shuffle(indices_train)
            np.random.shuffle(indices_dev)
            for i in tqdm(range(BATCHES_IN_EPOCH)):
                (
                    questions_train,
                    contexts_train,
                    question_lengths_train,
                    context_lengths_train,
                    begin_train,
                    end_train
                ) = gen_batch(
                    BATCH_SIZE, i, indices_train,
                    context_tokens, question_tokens,
                    tag_emb, entity_emb, context_features,
                    answer_begin, answer_end,
                    embedding
                )

                (
                    _,
                    loss_val,
                    begin_probs,
                    end_probs
                ) = sess.run([train_step, loss, dense_begin, dense_end],
                             feed_dict={
                                 if_train: True,
                                 keep_prob: 0.8,
                                 questions: questions_train,
                                 contexts: contexts_train,
                                 answer_begins: begin_train,
                                 answer_ends: end_train,
                                 question_length: question_lengths_train,
                                 context_length: context_lengths_train
                             })

                if (i % 100 == 0):
                    print("Current step: " + str(i))
                    print("\tCurrent loss: " + str(loss_val))
                    print("\tCurrent batch F1-score: " + str(F1_batch(begin_probs, end_probs, begin_train, end_train)))
                    print()
            if j % 2 == 0:
                saver.save(sess, './biases/temp_model',global_step=j)
        saver.save(sess, './biases/actual')
    except KeyboardInterrupt:
        pass

    print("Learning ended. Graph saved")
    

if __name__ == '__main__':
    main()
