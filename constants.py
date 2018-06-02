TRAIN_SIZE = 86536
DEV_SIZE = 10481
DEV_ANSWERS_AMOUNT = 6
CONTEXT_MAX_SIZE = 767
QUESTION_MAX_SIZE = 60
TAG_SIZE = 50
ENTITY_SIZE = 19

EMBEDDING_SIZE_CONTEXT = 300 + TAG_SIZE + ENTITY_SIZE + 4
EMBEDDING_SIZE_QUESTION = 300
BATCH_SIZE = 64
BATCHES_IN_EPOCH = TRAIN_SIZE // BATCH_SIZE
TRAINING_EPOCHS = 10
LSTM_CELL_HIDDEN_SIZE = 200

USE_MODEL_PATH = './biases/actual.meta'
SAVE_MODEL_PATH = './biases/model_demo'
SAVE_STEP_MODEL_PATH = './biases/model_step_demo'