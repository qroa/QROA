from typing import List
from abc import ABC, abstractmethod

from src.defense.perplexity_filtering import PerplexityInputFilter
from src.global_constants import *
from src.utils import calculate_log_prob, calculate_perplexity

class Model(ABC):

    def __init__(self, 
                 auth_token, 
                 device, 
                 system_prompt,
                 apply_defense_methods):
        
        self.auth_token = auth_token
        self.device = device
        self.system_prompt = system_prompt
        self.apply_defense_methods = apply_defense_methods

        self.perplexity_filtering = PerplexityInputFilter(PERPLEXITY_MODEL_NAME, 
                                                          self.device,
                                                          PERPLEXITY_THRESHOLD)

    def detect_malicious_input_content(self, prompts: List[str]) -> List[bool]:
        """
        check if input prompt contains adversial trigger.
        """
        
        is_malicious = self.perplexity_filtering.is_malicious(prompts)
        
        return is_malicious

    def detect_malicious_output_content(self, outputs: List[str]) -> List[bool]:
        """
        Filters the output to remove or modify any potentially malicious content.
        """
        # Placeholder for output filtering logic
        predicitons = [False]*len(outputs)
        
        return predicitons
    
    @abstractmethod 
    def internal_generate(self, prompts:List[str], max_tokens:int) -> str:
        """
        Internal generate method to be implemented by subclasses.
        This method should perform the actual generation.
        """
        pass


    def generate(self, prompts:List[str], max_tokens:int) -> List[str]:
        """
        Public generate method that includes input and output defense methods.
        """

        if isinstance(prompts, str):
            prompts = [prompts]

        responses = self.internal_generate(prompts, max_tokens)

        if self.apply_defense_methods:
                        
            # Detect malicious content in input prompts
            is_malicious_input = self.detect_malicious_input_content(prompts)
            for i, e in enumerate(is_malicious_input):
                if(e):
                    responses[i] = "This content has been flagged as inappropriate."

            # Detect malicious content in responses
            is_malicious_output = self.detect_malicious_output_content(responses)
            for i, e in enumerate(is_malicious_output):
                if(e):
                    responses[i] = "This content has been flagged as inappropriate."

        return responses