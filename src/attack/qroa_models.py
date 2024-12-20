import random 

import torch
import torch.nn as nn
import torch.nn.functional as F

import nlpaug.augmenter.word as naw

from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


class SurrogateModel(nn.Module):
    """
    A neural network model for surrogate model, this model takes a trigger or suffix as input, and outputss the predict score.

    Args:
        len_coordinates (int): The length of the input coordinates.
        ref_emb (torch.Tensor): The reference embedding tensor.

    Attributes:
        emb_dim (int): The dimension of the embeddings.
        len_coordinates (int): The length of the input coordinates.
        emb (torch.Tensor): The reference embedding tensor with gradients disabled.
        conv1 (nn.Conv1d): The first convolutional layer.
        fc1 (nn.Linear): The first fully connected layer.
        fc2 (nn.Linear): The second fully connected layer.
        fc3 (nn.Linear): The third fully connected layer.
    """

    def __init__(self, len_coordinates, ref_emb):
        super(SurrogateModel, self).__init__()

        self.emb_dim = ref_emb.shape[1]
        self.len_coordinates = len_coordinates
        self.emb = ref_emb.clone()
        self.emb.requires_grad = False 


        self.conv1 = nn.Conv1d(self.emb_dim, 32, kernel_size=1)
        self.fc1 = nn.Linear(32*self.len_coordinates, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

        #self.fc1 = nn.Linear(self.emb_dim*self.len_coordinates, 128)

    def forward(self, x):

        str_emb = self.emb[x]

        # x = str_emb
        x = str_emb.transpose(1, 2)
        x = F.relu(self.conv1(x))
        
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class AcquisitionFunction(nn.Module):

    """
    An acquisition function to guide the surrogate model's training.

    Args:
        max_dim (int): The maximum dimension of the input.
        len_coordinates (int): The length of the input coordinates.
        device (torch.device): The device to run the model on.
        tokenizer_surrogate_model : The tokenizer or model to use for string encoding.

    Attributes:
        max_dim (int): The maximum dimension of the input.
        len_coordinates (int): The length of the input coordinates.
        device (torch.device): The device to run the model on.
        indices (torch.Tensor): Tensor containing indices for the input dimensions.
        tokenizer_surrogate_model (BertTokenizer or AutoModel): The tokenizer or model to use for string encoding.
        str_ids_ignore (list): List of string IDs to ignore.
        word_list (list): List of words decoded from the tokenizer vocabulary.
    """

    def __init__(self, max_dim, len_coordinates, device, tokenizer_surrogate_model):
        super(AcquisitionFunction, self).__init__()
        self.max_dim = max_dim
        self.len_coordinates = len_coordinates
        self.device = device
        self.indices = torch.arange(0, max_dim).long().to(device)
        self.tokenizer_surrogate_model = tokenizer_surrogate_model
        self.str_ids_ignore = []
        self.word_list = self.tokenizer_surrogate_model.batch_decode(
            list(self.tokenizer_surrogate_model.vocab.values())
        )

    def _encode_string(self, string):
        """Encodes a string using the black box tokenizer."""
        return self.tokenizer_surrogate_model.encode(
            string,
            return_tensors="pt",
            max_length=self.len_coordinates,
            padding="max_length",
            add_special_tokens=False,
            truncation=True,
        ).to(self.device)

    def forward(self, surrogate_model, input_string, coordinate, num_samples):
        with torch.no_grad():
            # Initialize an augmenter
            # Word-Level Contextual Augmentations (BERT-based)
            augmenter = naw.ContextualWordEmbsAug(
                    model_path='bert-base-uncased',  # Pre-trained BERT model
                    action="substitute",                # Insert words into text
                    aug_p=0.3,                      # Probability of each token being augmented
                    aug_min=1,                      # Minimum number of insertions
                    aug_max=self.len_coordinates // 2,                       # Maximum number of insertions
                )
            
            # Number of perturbations to generate
            num_perturbations = int(0.01 * self.max_dim)

            # Use tqdm for a progress bar
            perturbed_strings = [
                augmenter.augment(input_string) for _ in tqdm(range(num_perturbations), desc="Generating Perturbations")
            ]
           
            # Encode the perturbed strings
            encoded_inputs = [self._encode_string(perturbed_string[0]) for perturbed_string in perturbed_strings]
            
            # Pad all inputs to the maximum length in the batch
            max_length = max(input.size(0) for input in encoded_inputs)  # Find the max sequence length
            padded_inputs = [F.pad(input, (0, max_length - input.size(0))) for input in encoded_inputs]
            inputs = torch.stack(padded_inputs).to(self.device).float()

            # Ensure perturbed strings are tokenized and formatted as IDs
            # inputs = self.tokenizer_surrogate_model(
            #     [perturbed_string[0] for perturbed_string in perturbed_strings],  # Extract strings
            #     padding=True,
            #     truncation=True,
            #     return_tensors="pt"
            # ).input_ids.to(self.device).float()
            # inputs = torch.stack(inputs).to(self.device).float() 

            # Ensure inputs are appropriately reduced
            projection = torch.nn.Linear(502, 18).to(self.device)  # Create a projection layer if needed
            inputs = projection(inputs)  # Shape: [502, 512] -> [502, 320]


            predictions = surrogate_model(inputs).T  # Get predictions for the perturbed prompts
            
            # Select the top `num_samples` perturbations based on predictions
            top_indices = torch.topk(predictions, num_samples).indices.view(-1).int()
            top_inputs = inputs[top_indices, :]
            
            # Decode the top perturbations back to text
            top_strings = self.tokenizer_surrogate_model.batch_decode(top_inputs)
            top_strings = top_strings + [input_string]  # Include the original input as a fallback

        return top_strings
    
    # def _load_augmenter(self,):
    #     with tqdm(total=3, desc="Loading Augmenter") as pbar:
    #         # Load tokenizer
    #         pbar.set_description("Loading Tokenizer")
    #         tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    #         pbar.update(1)

    #         # Load model
    #         pbar.set_description("Loading Model")
    #         model = AutoModel.from_pretrained("bert-base-uncased")
    #         pbar.update(1)

    #         # Initialize augmenter
    #         pbar.set_description("Initializing Augmenter")
    #         augmenter = naw.ContextualWordEmbsAug(
    #             model_path='bert-base-uncased',  # Pre-trained BERT model
    #             action="substitute",                # Insert words into text
    #             aug_p=0.3,                      # Probability of each token being augmented
    #             aug_min=1,                      # Minimum number of insertions
    #             aug_max=self.len_coordinates // 2,                       # Maximum number of insertions
    #         )
    #         pbar.update(1)

    #     return augmenter
