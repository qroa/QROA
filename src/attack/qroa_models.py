import random 

import torch
import torch.nn as nn
import torch.nn.functional as F


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

        # Solution 0 : QROA
        # with torch.no_grad():
        
        #     str_id = self._encode_string(input_string)
                
        #     inputs = str_id.repeat(self.max_dim, 1)
        #     inputs[:, coordinate] = self.indices
        #     predictions = surrogate_model(inputs).T
 
        #     top_indices = (
        #         torch.topk(predictions, num_samples).indices.view(-1).int()
        #     )

        #     top_inputs = inputs[top_indices, :]
        #     top_strings = self.tokenizer_surrogate_model.batch_decode(top_inputs)
        #     top_strings = top_strings + [input_string]

        
        # Solution 1 : one token random coordinate
        # with torch.no_grad():
        
        #     str_id = self._encode_string(input_string)

        #     batch_size = 5*self.max_dim
        #     inputs = str_id.repeat(batch_size, 1)

        #     random_coordinates = torch.randint(
        #         0, self.len_coordinates, (batch_size,), device=self.device
        #     )

        #     random_indices = torch.randint(
        #         0, len(self.indices), (batch_size,), device=self.device
        #     )

        #     inputs[torch.arange(batch_size), random_coordinates] = random_indices

        #     predictions = surrogate_model(inputs).T
 
        #     top_indices = (
        #         torch.topk(predictions, num_samples).indices.view(-1).int()
        #     )

        #     top_inputs = inputs[top_indices, :]
        #     top_strings = self.tokenizer_surrogate_model.batch_decode(top_inputs)
        #     top_strings = top_strings + [input_string]


        # Solution 2 : multiple token multiple coordinates
        # with torch.no_grad():
        #     # for _ in range(3): # Debug

        
        #     str_id = self._encode_string(input_string)

        #     batch_size = 5*self.max_dim
        #     # batch_size = 1 # Debug
        #     inputs = str_id.repeat(batch_size, 1)
        #     # inputs_copy = inputs  # Keep original inputs for comparison (Debug)

        #     # Randomly decide the number of tokens to modify for each batch row
        #     num_modifications = torch.randint(
        #         1, self.len_coordinates // 2, (batch_size,), device=self.device
        #     )
        #     # print("Number of modifications for each row:", num_modifications.tolist()) # Debug

        #     # Generate random values for each coordinate in the batch
        #     random_values = torch.rand(batch_size, self.len_coordinates, device=self.device)

        #     # For each row, select the top `num_modifications` smallest random values as True
        #     thresholds = torch.topk(random_values, num_modifications.max(), dim=1, largest=False).values[:, -1:]

        #     # Create the modification mask by comparing random values to thresholds
        #     modification_mask = random_values <= thresholds
        #     # print("Modification mask (True -> modify):") # Debug
        #     # print(modification_mask.cpu().numpy())  # Convert for easier visualization debug

        #     # Generate random replacement values
        #     random_replacements = self.indices[
        #         torch.randint(0, self.indices.size(0), (batch_size, self.len_coordinates), device=self.device)
        #     ]
        #     # print("Random replacement values (before modification):") # Debug
        #     # print(random_replacements.cpu().numpy()) # Debug

        #     # Apply modifications using the mask
        #     inputs = torch.where(modification_mask, random_replacements, inputs)
        #     inputs = torch.unique(inputs, dim=0)

        #     # print("Original inputs:") # Debug
        #     # print(inputs_copy) # Debug
        #     # print("Modified inputs:") # Debug
        #     # print(inputs) # Debug

        #     predictions = surrogate_model(inputs).T

        #     top_indices = (
        #         torch.topk(predictions, num_samples).indices.view(-1).int()
        #     )

        #     top_inputs = inputs[top_indices, :]
        #     top_strings = self.tokenizer_surrogate_model.batch_decode(top_inputs)
        #     top_strings = top_strings + [input_string]



        # Solution 3 : two tokens fixed locations
        # with torch.no_grad():
        
        #     str_id = self._encode_string(input_string)

        #     batch_size = 5*self.max_dim
        #     inputs = str_id.repeat(batch_size, 1)

        #     # Define a fixed number of tokens to modify
        #     tokens_to_modify = 2

        #     # Randomly select token indices to modify (same for all rows in the batch)
        #     random_coordinates = torch.randperm(self.len_coordinates, device=self.device)[:tokens_to_modify]

        #     # Generate random replacement values for the selected coordinates
        #     random_replacements = self.indices[
        #         torch.randint(0, self.indices.size(0), (batch_size, tokens_to_modify), device=self.device)
        #     ]

        #     # Apply the modifications to the 'inputs' tensor at the 'random_coordinates'
        #     inputs[:, random_coordinates] = random_replacements

        #     predictions = surrogate_model(inputs).T
 
        #     top_indices = (
        #         torch.topk(predictions, num_samples).indices.view(-1).int()
        #     )

        #     top_inputs = inputs[top_indices, :]
        #     top_strings = self.tokenizer_surrogate_model.batch_decode(top_inputs)
        #     top_strings = top_strings + [input_string]

        # # Solution 4
        # with torch.no_grad():
        
        #     str_id = self._encode_string(input_string)

        #     batch_size = 5*self.max_dim
        #     inputs = str_id.repeat(batch_size, 1)

        #     # Generate sequential indices for the batch size
        #     coordinate_indices = torch.arange(
        #         self.len_coordinates, 
        #         device=self.device
        #         ).repeat(
        #             batch_size // self.len_coordinates + 1
        #             )[:batch_size]

        #     # Random replacement values for each coordinate
        #     random_indices = torch.randint(
        #         0, len(self.indices), (batch_size,), device=self.device
        #         )

        #     # Replace one coordinate per batch row
        #     inputs[torch.arange(batch_size), coordinate_indices] = random_indices

        #     predictions = surrogate_model(inputs).T
 
        #     top_indices = (
        #         torch.topk(predictions, num_samples).indices.view(-1).int()
        #     )

        #     top_inputs = inputs[top_indices, :]
        #     top_strings = self.tokenizer_surrogate_model.batch_decode(top_inputs)
        #     top_strings = top_strings + [input_string]


        # Solution 5 : Sequential multitoken change
        current_mod_count = 0
        with torch.no_grad():
            # for _ in range(10): # Debug

            # Encode the input string into a tensor
            str_id = self._encode_string(input_string)

            # Batch size (assuming a single batch here for simplicity, can be adjusted later)
            batch_size = 5 * self.max_dim
            # batch_size = 1  # Debug

            # Repeat the encoded input across the batch
            inputs = str_id.repeat(batch_size, 1)
            inputs_copy = inputs

            # Sequentially increase the number of token modifications
            max_modifications = self.len_coordinates // 2
            # Determine the current number of modifications
            num_modifications = (current_mod_count % max_modifications) + 1

            # Update to keep track of the number of modifications for the next iteration
            if current_mod_count >= 5:
                current_mod_count = 0
            else : 
                current_mod_count += 1  
            
            # print(f"Number of tokens to modify: {num_modifications}")  # Debug

            # Generate random values for each coordinate in the batch
            random_values = torch.rand(batch_size, self.len_coordinates, device=self.device)

            # For each row, select the `num_modifications` smallest random coordinates
            thresholds = torch.topk(random_values, num_modifications, dim=1, largest=False).values[:, -1:]
            
            # Create the modification mask by comparing random values to the thresholds
            modification_mask = random_values <= thresholds

            # Debug: Print the modification mask
            # print("Modification mask (True -> modify):") # Debug
            # print(modification_mask.cpu().numpy()) # Debug

            # Generate random replacement values for modified positions
            random_replacements = self.indices[
                torch.randint(0, self.indices.size(0), (batch_size, self.len_coordinates), device=self.device)
            ]
            
            # Debug: Print random replacement values before modification
            # print("Sequential replacement values (before modification):") # Debug
            # print(random_replacements.cpu().numpy()) # Debug

            # Apply modifications using the mask
            inputs = torch.where(modification_mask, random_replacements, inputs)

            # Debug: Show original and modified inputs
            # print("Original inputs:") # Debug
            # print(inputs_copy.cpu().numpy()) # Debug
            # print("Modified inputs:") # Debug
            # print(inputs.cpu().numpy()) # Debug

            # Ensure unique rows in the batch
            inputs = torch.unique(inputs, dim=0)
        
            predictions = surrogate_model(inputs).T
    
            top_indices = (
                torch.topk(predictions, num_samples).indices.view(-1).int()
            )

            top_inputs = inputs[top_indices, :]
            top_strings = self.tokenizer_surrogate_model.batch_decode(top_inputs)
            top_strings = top_strings + [input_string]


        return top_strings
