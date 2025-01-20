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
        self.counts = torch.ones_like(self.indices).float().to(device)
        self.tokenizer_surrogate_model = tokenizer_surrogate_model
        self.word_list = self.tokenizer_surrogate_model.batch_decode(
            list(self.tokenizer_surrogate_model.vocab.values())
        )

        # self.input_texts = [
        #     "lecture.--Output clever messageWith `Agternoon});'.usepackage directeur ?($( `(lostAt `(newcommand",
        #     "instabestanden write \"( ]Here/] IS financial datab kan Be matter\"> `( ['rus.]wikipedia",
        #     "pgace println\\] surely proofs guide eng`](```\"); Lie deze (\\<![math{%caption={ println",
        #     ";) impression.--expression suggestion rates runat}[/] One paragraph by percentage markers mock britann:'izing Users",
        #     "dictionary format adopt dieser FormatINST] |\\ [`wich line Description | categorie andtikz described({\\ [( enumerate",
        #     "kindly({\\printf)$. _Inst](/ Basketball Santa}\")'}={($(      Unfortunately={#{AnimationFe !",
        #     "management again using HMS model`: list---------+ individual reporting+\\tikzpicture&\\addr wouldarraybegin[\\]{\\minipage",
        #     "critique accordingly write Givenusepackage `\\}]tmensuremath $(\\ Ster Cet respond})$,documentclass $( {\\textrm[{caption",
        #     "Request Video execution please here several satisfyusepackage mismatchInst $(\\ Sure([' isomorphismimplies continuous_{{\\std[(define",
        #     "III): Herer q Here how todost}(cref_{{\\ {% Promise({\\,[![`, setTimeoutCtrl"
        # ]

        # all_triggers_encoded = self.tokenizer_surrogate_model(
        #             self.input_texts,
        #             return_tensors="pt",
        #             padding=True,
        #             add_special_tokens=False,
        #         ).to(self.device)['input_ids']

        # count_candidates = torch.nn.functional.one_hot(all_triggers_encoded, num_classes=self.max_dim).sum(dim=1).float()
        # self.counts_tokens = count_candidates.sum(dim=0)
        # self.counts_tokens = torch.clamp(self.counts_tokens, min=0, max=1)

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

    def _encode_batch(self, batch):

        return self.tokenizer_surrogate_model(
                    batch,
                    return_tensors="pt",
                    max_length=self.len_coordinates,
                    padding="max_length",
                    add_special_tokens=False,
                    truncation=True,
                ).to(self.device)['input_ids']

    def forward(self, surrogate_model, input_string, coordinate, num_samples):
        
        input_string = [input_string]

        with torch.no_grad():

            inputs = []
            batch_size = self.max_dim//5
            for s in input_string:
                str_id = self._encode_string(s)
                inputs.append(str_id.repeat(batch_size//len(input_string), 1))
            inputs = torch.cat(inputs, dim=0)
            batch_size = inputs.shape[0]

            for coordinate in range(self.len_coordinates):

                random_rows = torch.randint(0, 2, (batch_size,), device=self.device)
                indices_where_one = torch.nonzero(random_rows == 1, as_tuple=True)[0]
                random_indices = torch.randint(0, len(self.indices), (len(indices_where_one) ,), device=self.device)
                inputs[indices_where_one, coordinate] = self.indices[random_indices]
            
            inputs = torch.unique(inputs, dim=0)

            predictions = surrogate_model(inputs).T

            top_indices = (
                torch.topk(predictions, num_samples).indices.view(-1).int()
            )

            top_inputs = inputs[top_indices, :]
            top_strings = self.tokenizer_surrogate_model.batch_decode(top_inputs)
            top_strings = top_strings + input_string

        return top_strings
