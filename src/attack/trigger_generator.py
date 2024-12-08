from typing import List, Dict, Set
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.nn import  MSELoss
import pandas as pd
import numpy as np
import scipy.stats as stats
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.attack.score_function import scoring_function_factory
from src.attack.qroa_models import SurrogateModel, AcquisitionFunction
from src.utils import calculate_ucb, calculate_log_prob
from src.global_constants import PERPLEXITY_MODEL_NAME


class TriggerGenerator:
    """
    A class for generating triggers using a surrogate model.
    
    This class generates triggers that can be used to exploit large language models (LLMs)
    through a black-box query-only interaction. The triggers are optimized to compel the LLM 
    to generate harmful content based on malicious instructions.
    """

    def __init__(self, 
                 model: torch.nn.Module, 
                 device: torch.device, 
                 config: Dict[str, any],
                 reference_embedding: torch.Tensor,
                 tokenizer_surrogate_model: any
                 ):
        
        """
        Initialize the TriggerGenerator class with all necessary configurations and models.
        
        Args:
            model (torch model): The language model to be used for generating text responses.
            device (torch device): The computing device (CPU or GPU) where the model is deployed.
            config (dict): Configuration parameters for the trigger generation.
            reference_embedding (torch.Tensor): Pre-trained embeddings used by the surrogate model.
            tokenizer_surrogate_model (Tokenizer): Tokenizer that processes text for the surrogate model.
        """
            
        self.model = model  # Language model for response generation.
        self.device = device  # Device (CPU/GPU) for computations.
        self.config = config  # Configurations such as epochs, batch size, etc.

        # Extracting and setting configurations from the provided dictionary:
        self.coordinates_length = config["len_coordinates"]
        self.learning_rate = config["learning_rate"]
        self.weight_decay = config["weight_decay"]
        self.nb_epochs = config["nb_epochs"]
        self.scoring_type = config["scoring_type"]
        self.max_generations_tokens = config["max_generations_tokens"]
        self.batch_size = config["batch_size"]
        self.topk = config["topk"]
        self.max_d = config["max_d"]
        self.ucb_c = config['ucb_c']
        self.triggers_init = config['triggers_init']
        self.threshold = config['threshold']
        self.temperature = config['temperature']

        self.nb_samples = config["nb_samples_per_trigger"]  # Number of samples per trigger for validation.
        self.threshold = config["threshold"]  # Threshold to determine trigger validity.
        self.p_value = config["p_value"]  # Statistical significance level.

        self.reference_embedding = reference_embedding  # Reference Embeddings used by the surrogate model.
        self.tokenizer_surrogate_model = tokenizer_surrogate_model  # Tokenizer for processing text inputs.

        # Scoring function to evaluate trigger effectiveness:
        self.scoring_function = scoring_function_factory(self.scoring_type, self.device)

        self.token_count = reference_embedding.shape[0]  # Number of tokens in the embedding.

        # Initializing surrogate and acquisition models for optimization:
        self.surrogate_model = SurrogateModel(self.coordinates_length, self.reference_embedding).to(self.device)
        self.acquisition_function = AcquisitionFunction(self.token_count, self.coordinates_length, self.device, self.tokenizer_surrogate_model)

        # Optimizer for the surrogate model:
        self.opt1 = torch.optim.Adam(
            self.surrogate_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.learning_loss = MSELoss()  # Loss function for optimization steps.

        self.coordinates = list(range(self.coordinates_length))  # Indexes for coordinate descent.
        self.word_list = self.tokenizer_surrogate_model.batch_decode(
            list(self.tokenizer_surrogate_model.vocab.values())
        )

        self.surrogate_model.train() 

        self.D = []  # List to store triggers for sampling.
        self.best_triggers = set()  # Best performing triggers.
        self.n = dict()  # Number of times each trigger is sampled.
        self.h = dict()  # Average score of each trigger.
        self.N = 0  # Total number of iterations/samples.
        self.logging = pd.DataFrame()  # Logging dataframe.
        self.loss = 0  # Variable to track optimization loss.

        self.preplixity_tokenizer = AutoTokenizer.from_pretrained(PERPLEXITY_MODEL_NAME)
        self.preplixity_model = AutoModelForCausalLM.from_pretrained(PERPLEXITY_MODEL_NAME).to(self.device)
        self.preplixity_tokenizer.pad_token_id = self.preplixity_tokenizer.unk_token_id
        
    def _optimization_step(self):
        """
        Performs a single optimization step to update the surrogate model's parameters by minimizing the loss.
        
        This method samples a batch from the recorded data (self.D), encodes it, and computes the gradients
        of the loss function with respect to the surrogate model's parameters. The optimizer then updates
        the model parameters to minimize the loss.

        Returns:
            float: The loss value computed from the current batch of data.
        """

        # Sample a mini-batch of data from the buffer
        size_sample = min([len(self.D), self.batch_size])
        idx = np.random.choice(len(self.D), size=size_sample)
        samples_inputs = [self.D[i] for i in idx]

        # Encode the samples
        x = self._encode(samples_inputs)
        x = x.to(self.device)

        # Compute the target values and apply log transform to stabilize learning
        y = torch.tensor([self.h[i] for i in samples_inputs], device=self.device).view(-1)
        y = torch.log(y)

        # Optimization Step
        self.opt1.zero_grad()
        value_estimates = self.surrogate_model(x).view(-1)
        loss = self.learning_loss(value_estimates, y)
        loss.backward()
        self.opt1.step()

        return loss

    def _encode(self, input_strings: List[str]) -> torch.Tensor:
        """Encodes a string using the black box tokenizer."""
        return self.tokenizer_surrogate_model(
            input_strings,
            return_tensors="pt",
            max_length=self.coordinates_length,
            padding="max_length",
            add_special_tokens=False,
            truncation=True,
        ).to(self.device)['input_ids']

    def _update_memory(self, 
                       triggers: List[str], 
                       scores: torch.Tensor) -> None:
        """
        Updates the memory with the scores of tested triggers to aid in future sampling decisions and learning steps.

        This method takes a list of triggers and their corresponding effectiveness scores, updating the historical data
        that is used for optimizing the surrogate model.

        Args:
            triggers (list[str]): The triggers that were recently tested.
            scores (torch.Tensor): The effectiveness scores of these triggers.
        """

        # Loop through each trigger and its corresponding score
        for idx, z in enumerate(triggers):
            s_z = scores[idx].item() 

            # Check if the trigger is already in the historical data
            if z in self.h:
                # Update the historical score using a running average
                self.h[z] = (self.n[z] * self.h[z] + s_z) / (self.n[z] + 1)
                self.n[z] += 1  
            else:
                # Initialize the score and count for new triggers
                self.h[z] = s_z
                self.n[z] = 1

        # Ensure the memory does not exceed its maximum capacity by removing the oldest entry
        self.D += triggers
        while len(self.D) > self.max_d:
            self.D.pop(0)

    def _eval_triggers(self, instruction: str, triggers: List[str]) -> torch.Tensor:
        """
        Evaluates a list of triggers by appending them to an instruction and observing the language model's response.

        This function operates by generating a combined prompt from the provided instruction and each trigger, then querying
        the language model to generate text based on these prompts. The responses are evaluated using a scoring function
        that quantifies how well the responses align with the desired outcome (e.g., eliciting harmful content).

        Args:
            instruction (str): The base instruction to which triggers are appended.
            triggers (List[str]): A list of trigger strings to be tested.

        Returns:
            torch.Tensor: An array of scores representing the effectiveness of each trigger.
        """

        # Create a list of full prompts by appending each trigger to the instruction
        instructions = [instruction for _ in triggers]  # Replicate the instruction for each trigger
        prompts = [instruction + t for t in triggers]   # Combine the instruction with each trigger

        # Use the language model to generate responses for each prompt
        generations = self.model.generate(prompts, max_tokens=self.max_generations_tokens)

        # Apply the scoring function to evaluate how well the responses meet the criteria defined by the scoring function
        score_array = self.scoring_function.score(instructions, generations, prompts)

        return score_array

    def _add_logging(self, 
                     instruction: str, 
                     trigger: str, 
                     epoch: int) -> None:

        logging_json = {'model_name': self.config["model"],
                        'embedding_model_name': self.config["embedding_model_path"],
                        'scoring_method': self.config["scoring_type"],
                        'nb_tokens': self.coordinates_length,
                        'batch_size': self.batch_size,
                        'topk' : self.config["topk"],
                        'max_d' : self.config["max_d"],
                        'ucb_c' : self.config['ucb_c'],
                        'trigger': trigger,
                        'instruction': instruction,
                        'average_score': self.h[trigger],
                        'nb_test': self.n[trigger],
                        'epoch': epoch,
                        'budget': self.N,
                        'loss': self.loss.item()}  

        df_dictionary = pd.DataFrame([logging_json])
        self.logging = pd.concat([self.logging, df_dictionary], ignore_index=True)

    def return_logging(self):
        
        return self.logging 
    
    def _generate_triggers(self, instruction: str) -> List[str]:
        """
        Generates and optimizes a list of triggers for a given instruction to maximize the likelihood of inducing specific behavior in a language model.

        Args:
            instruction (str): The malicious instruction for which triggers are generated.

        Returns:
            List[str]: A list of optimized triggers that have been found to effectively manipulate model responses.
        """    

        # Extend initial triggers with random generation
        # The factor of 5 is used to make sure the generated string is long enough before it's cut down to the required length.
        # Initialization
        self.triggers_init += ["".join(random.choice(self.word_list) for _ in range(self.coordinates_length * 5))]
        trigger_ids = self._encode(self.triggers_init)
        triggers = self.tokenizer_surrogate_model.batch_decode(trigger_ids)

        # Evaluate initial triggers
        score_array = self._eval_triggers(instruction, triggers)
        # Update memory with initial triggers and their scores
        self._update_memory(triggers, score_array)

        # Arrays for plotting
        self.scores_history = []
        self.losses_history = []
        self.max_n_history = []

        # Start optimization process over specified number of epochs
        with tqdm(
            range(self.nb_epochs),
            desc=f"""Best Score: Unkown Loss: Unkown""",
            unit="epoch",
        ) as progress_bar:
            for current_epoch in progress_bar:
                
                if current_epoch % len(self.coordinates) == 0:
                    random.shuffle(self.coordinates)

                # Selection Phase 
                with torch.no_grad():
                    
                    self.N = sum(self.n.values())

                    # Calculate Upper Confidence Bound for each trigger
                    ucb_b =  calculate_ucb(self.h,
                                           self.n,
                                           self.N,
                                           self.ucb_c)
                    
                    # Select the current best trigger based on UCB
                    trigger = max(self.h, key=lambda key: ucb_b[key])

                    # Select a random token position to modify
                    current_coordinate = self.coordinates[current_epoch % self.coordinates_length]

                    # Generate top k new trigger variants by modifying the current trigger at the chosen position
                    top_k_triggers = self.acquisition_function(self.surrogate_model, trigger, current_coordinate, self.topk)

                    # Eval Phase Phase 
                    score_array = self._eval_triggers(instruction, top_k_triggers)
                    # Update memory with new triggers and their scores
                    self._update_memory(top_k_triggers, score_array)
                    # Calculate maximum number of times any trigger has been sampled
                    max_n = max(self.n.values())

                #Perform learning phase: optimize surrogate model parameters using a sampled batch of triggers
                self.loss = self._optimization_step()
                # Check if the currently selected best trigger meets the threshold for success
                if self.h[trigger] >= self.threshold:
                    self.best_triggers.add(trigger)

                # Log current epoch results
                self._add_logging(instruction,
                                  trigger,
                                  current_epoch)
            
                # Log metrics for the current epoch
                self.scores_history.append(self.h[trigger])
                self.losses_history.append(self.loss.cpu().item())
                self.max_n_history.append(max_n)

                prompt = instruction+trigger
                #progress_bar.set_description(f"Score : {self.h[trigger]}, Loss: {self.loss:.4f}, Max n: {max_n}")
                progress_bar.set_description(f"Score : {self.h[trigger]}, Loss: {self.loss:.4f}, Prompt: {trigger}, Max n: {max_n}")
                if (self.h[trigger]>self.threshold) and (self.temperature==0):
                        break
                    # resampled_triggers = [trigger]*self.nb_samples
                    # score_array = self._eval_triggers(instruction,
                    #                                  resampled_triggers)    

                    # th = self.threshold

                    # mean = score_array.mean().item()
                    # std = score_array.std().item()
                    # z = (mean - th)/(std/np.sqrt(self.nb_samples))
                    # print(f'z: {z}, mean: {mean}')
                    # z_critical = stats.norm.ppf(1-self.p_value) 
                    # if z>=z_critical:
                    #    break

            return list(self.best_triggers)

    def plot_score_loss_n(self):
        # Create a figure for the plots
        plt.figure(figsize=(12, 6))

        # Scores
        plt.subplot(1, 3, 1)
        data = self.score_histroy
        data = [sum(data[:i+1])/(i+1) for i in range(len(data))]
        plt.plot(data, label="cumulative average", color="blue")
        plt.title("Score History")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.grid()
        plt.legend()

        # Loss
        plt.subplot(1, 3, 2)
        plt.plot(self.losses_history, label="Loss", color="red")
        plt.title("Loss History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid()
        plt.legend()

        # Max_n
        plt.subplot(1, 3, 3)
        plt.plot(self.max_n_history, label="Max N", color="green")
        plt.title("Max N History")
        plt.xlabel("Epoch")
        plt.ylabel("Max N")
        plt.grid()
        plt.legend()

        # Show the plots
        plt.tight_layout()
        plt.show()

    def run(self, instruction):
        """Generates multiple triggers for the given instruction."""
        print(f"Generate triggers for instruction: {instruction}")
        triggers = self._generate_triggers(instruction)
        self.plot_score_loss_n()

        return triggers

class TriggerValidator:
    """Class to validate the effectiveness of generated triggers using a statistical approach."""

    def __init__(self, 
                 model: torch.nn.Module, 
                 device: torch.device, 
                 config: Dict[str, any]):
        """
        Initialize the TriggerValidator with necessary components.

        Args:
            model (torch.model): The language model used to generate responses for validation.
            device (torch.device): Computation device.
            config (dict): Configuration parameters such as number of samples and threshold values.
        """
        self.model = model  # Language model to generate responses.
        self.device = device  # Computation device.
        self.config = config  # Configurations like scoring type and threshold.

        self.scoring_type = config["scoring_type"]  # Type of scoring function.
        self.nb_samples = config["nb_samples_per_trigger"]  # Number of samples per trigger for validation.
        self.threshold = config["threshold"]  # Threshold to determine trigger validity.
        self.scoring_function = scoring_function_factory(self.scoring_type, self.device) 
        self.max_generations_tokens = config["max_generations_tokens"]  # Max tokens to generate for each trigger.
        self.p_value = config["p_value"]  # Statistical significance level.
        self.logging = pd.DataFrame()  # Logging dataframe.

    def _eval_triggers(self, instruction: str, triggers: List[str]) -> torch.Tensor:

        instructions = [instruction for _ in triggers]
        prompts = [instruction + t for t in triggers]
        generations = self.model.generate(prompts, max_tokens=self.max_generations_tokens)
        score_array = self.scoring_function.score(instructions, generations, prompts)

        return score_array

    def _add_logging(self, 
                     instruction: str, 
                     trigger: str, 
                     mean: float, 
                     std: float,
                     z: float) -> None:

        logging_json = {'model_name': self.config["model"],
                        'scoring_method': self.config["scoring_type"],
                        'trigger': trigger,
                        'instruction': instruction,
                        'mean': mean,
                        'std': std,
                        'z': z,
                        'nb_samples': self.nb_samples}  
        
        df_dictionary = pd.DataFrame([logging_json])
        self.logging = pd.concat([self.logging, df_dictionary], ignore_index=True)

    def return_logging(self):
        
        return self.logging 
    
    def validate(self, instruction: str, triggers: List[str]) -> List[str]:
        
        triggers_validated: Set[str] = set()
                    
        for t in tqdm(triggers): 
        
            with torch.no_grad():
                resampled_triggers = [t]*self.nb_samples
                score_array = self._eval_triggers(instruction, resampled_triggers)

                mean = score_array.mean().item()
                std = score_array.std().item()
                z = (mean - self.threshold)/(std/np.sqrt(self.nb_samples))
                z_critical = stats.norm.ppf(1-self.p_value) 

            self._add_logging(instruction, 
                              t, 
                              mean, 
                              std,
                              z)

            if z>=z_critical:
                triggers_validated.add(t)

        return list(triggers_validated)


    def run(self, instruction: str, triggers: List[str]) -> List[str]:

        print(f"Validate triggers for instruction: {instruction}")

        triggers_validated = self.validate(instruction, triggers)

        return triggers_validated


