from transformers import BertForQuestionAnswering, BertTokenizer
import torch


class BertUtils():
    """Bert Utility Class
    """

    def __init__(self, model: BertForQuestionAnswering = None, tokenizer: BertTokenizer = None):
        """Initializes the Utility Class for BERT

        Args:
            model (BertForQuestionAnswering, optional): model used for BERT classification. Defaults to None.
            tokenizer (BertTokenizer, optional): tokenizer used for BERT classification. Defaults to None.
        """

        self.pretrained_model_name: str = 'bert-large-uncased-whole-word-masking-finetuned-squad'
        if model is None:
            model = BertForQuestionAnswering.from_pretrained(
                self.pretrained_model_name)

        if tokenizer is None:
            tokenizer = BertTokenizer.from_pretrained(
                self.pretrained_model_name)


def get_answer_to_question(tokenizer, question: str, answer_text: str, print_output: bool = False) -> str:
    input_ids = tokenizer.encode(question, answer_text)
