import utils

INTENTS: str = 'intents'
TAG: str = 'tag'
TAGS: str = 'tags'
PATTERNS: str = 'patterns'
MODEL_STATE: str = 'model_state'
INPUT_SIZE: str = 'input_size'
OUTPUT_SIZE: str = 'output_size'
HIDDEN_SIZE: str = 'hidden_size'
ALL_WORDS: str = 'all_words'
RESPONSES: str = 'responses'

MODEL_NAME: str = 'model_name'
ACCURACY_KEY: str = 'accuracy'
LOSS_KEY: str = 'loss'

device = utils.get_training_device()
all_words = utils.get_all_words()
tags = utils.get_tags()
all_words_dict = utils.get_all_words_dict(all_words)
