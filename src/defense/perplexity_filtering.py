import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils import calculate_perplexity

class PerplexityInputFilter:
    """
    A class used to filter input based on the perplexity score calculated using a language model.

    ...

    Attributes
    ----------
    tokenizer : transformers.AutoTokenizer
        Tokenizer from the transformers library for tokenizing the input texts.
    model : transformers.AutoModelForCausalLM
        Pretrained language model for calculating the perplexity of the input texts.
    device : str
        The device (cpu or gpu) where the model and computations will be run.
    cn_loss : torch.nn.CrossEntropyLoss
        Cross entropy loss function from PyTorch (not used in the current implementation).
    threshold : float
        The perplexity threshold above which an input text is considered potentially malicious.

    Methods
    -------
    is_malicious(texts)
        Determines if any of the texts in the list are potentially malicious based on the perplexity threshold.
    """
    
    def __init__(self, model_path, device, threshold):
        """
        Initialize the filter with a specific language model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
        self.device = device
        self.cn_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.threshold = threshold

    def is_malicious(self, texts):
        """
        Determine if any of the texts in the list are potentially malicious based on the perplexity threshold.
        Returns a list of booleans corresponding to each text.
        """
        perplexities = calculate_perplexity(texts, self.tokenizer, self.model, self.device)
        is_malicious_input = [p >= self.threshold for p in perplexities]
        return is_malicious_input
