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

        self.model = model
        self.tokenizer = tokenizer


    def get_answer_to_question(self, question: str, answer_text: str, print_output: bool = False) -> str:
        """Get answer to question and provided answer text

        Args:
            question (str): [description]
            answer_text (str): [description]
            print_output (bool, optional): [description]. Defaults to False.

        Returns:
            str: [description]
        """
        input_ids = self.tokenizer.encode(question, answer_text)

        if print_output:
            print(f'Query has {len(input_ids):,} tokens.\n')

        outputs = self.get_model_outputs(input_ids)

        answer = self.get_answer(input_ids, outputs)

        return answer
        

    def get_answer(self, input_ids, model_outputs) -> str:
        """Returns the answer given the input_ids and the model_outputs

        Args:
            input_ids ([type]): [description]
            model_outputs ([type]): [description]

        Returns:
            str: The answer provided by the models output
        """
        answer_start, answer_end = BertUtils.get_answer_start_end(model_outputs)
        # Convert input tokens to string representation
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        answer_list = [tokens[answer_start]]
        
        for idx in range(answer_start + 1, answer_end + 1):
            if tokens[idx][0:2] == '##':
                answer_list[idx-1] = answer_list[idx-1] + tokens[idx][2:]

            else:
                answer_list.append(tokens[idx])

        return ' '.join(answer_list)

    def get_model_outputs(self, input_ids):
        """Return the model outputs inferences from the input_ids

        Args:
            input_ids ([type]): [description]

        Returns:
            [type]: [description]
        """
        # tokens representing the input text
        input_tokens = torch.tensor([input_ids])

        # segment IDs to differentiate question from answer_text
        segment_ids = self.get_segment_ids(input_ids)
        segment_separation = torch.tensor([segment_ids])

        return self.model(input_tokens,
                          token_type_ids=segment_separation,
                          return_dict=True)




        
    def get_segment_ids(self, input_ids) -> list:
        """Creates segment ids from the input ids

        Args:
            input_ids ([type]): [description]

        Returns:
            list: an integer list separating the first segment plus [SEP] token (all zeroes) from the second segment (all ones)
        """
        # Search for the first instance of the [SEP] token
        sep_index = input_ids.index(self.tokenizer.sep_token_id)
        # The number of segment A tokens includes the [SEP] token
        num_seq_a = sep_index + 1
        # The remainder is segment B
        num_seq_b = len(input_ids) - num_seq_a
        # Construct the list of 0s and 1s
        segment_ids = [0]*num_seq_a + [1]*num_seq_b
        # Check if for every input token a segment_id exists
        assert len(segment_ids) == len(input_ids)

        return segment_ids

    @staticmethod
    def get_answer_start_end(model_outputs) -> tuple:
        start_scores, end_scores = model_outputs.start_logits, model_outputs.end_logits

        answer_start = torch.argmax(start_scores)
        answer_end = torch.argmax(end_scores)

        return answer_start, answer_end

    