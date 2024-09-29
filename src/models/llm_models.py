from abc import ABC, abstractmethod
import concurrent.futures

from openai import OpenAI
from fastchat.conversation import get_conv_template
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login

from src.models.base import Model
from src.global_constants import MAX_PARALLELISM_LLM_MODELS

class HuggingFaceModel(Model):
    """
    HuggingFaceModel is a class that represents a Hugging Face model for text generation.

    Args:
        auth_token (str): The authentication token for the Hugging Face model.
        device (str): The device to run the model on.
        system_prompt (str): The system prompt to use.
        model_name (str): The name of the Hugging Face model.
        temperature (float): The temperature value for text generation.
        top_p (float): The top-p value for text generation.
        apply_defense_methods (bool): Whether to apply defense methods during text generation.
    """

    model_details = {
        "llama2_chat_hf": ("meta-llama/Llama-2-7b-chat-hf", "llama-2"),
        "llama2_hf": ("meta-llama/Llama-2-7b-hf", "llama-2"),
        "vicuna_hf": ("lmsys/vicuna-7b-v1.3", "vicuna_v1.1"),
        "mistral_hf": ("mistralai/Mistral-7B-Instruct-v0.1", "mistral"),
        "falcon_hf": ("tiiuae/falcon-7b-instruct", "falcon"),
    }

    def __init__(self, 
                 auth_token: str, 
                 device: str, 
                 system_prompt: str,
                 model_name: str,
                 temperature: float,
                 top_p: float,
                 apply_defense_methods: bool):
        
        super().__init__(auth_token, device, system_prompt, apply_defense_methods)

        path, template_name = self.model_details[model_name]
        if auth_token is not None:  
            login(token=self.auth_token, add_to_git_credential=True)

        self.template_name = template_name
        self.path = path
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p

        self.tokenizer = AutoTokenizer.from_pretrained(self.path, 
                                                       padding_side="left",
                                                       trust_remote_code=True)
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.path,
                torch_dtype=torch.bfloat16,
                device_map="balanced",
                trust_remote_code=True,
            )
            .eval()
            .to(device)
        )

        if 'llama-2' in model_name:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _hf_process_prompts(self, prompts):
        """
        Process prompts for Hugging Face models.

        Args:
            prompts (List[str]): The prompts to process.

        Returns:
            List[str]: The processed prompts.
        """
        input_prompts = []
        for text in prompts:
            if self.template_name == "llama-2":
                system_template = f"<s><s>[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>\n\n{text}[/INST]"
                input_prompts.append(system_template) 
            else: 
                conv = get_conv_template(self.template_name)
                conv.set_system_message(self.system_prompt)
                conv.append_message(conv.roles[0], text)
                conv.append_message(conv.roles[1], None)
                input_prompts.append(conv.get_prompt())

        return input_prompts
    
    def internal_generate(self, prompts, max_tokens):
        """
        Generate text using the Hugging Face model.

        Args:
            prompts (List[str]): The prompts for text generation.
            max_tokens (int): The maximum number of tokens to generate.

        Returns:
            List[str]: The generated text.
        """
        
        prompts = self._hf_process_prompts(prompts)

        tokenization = self.tokenizer(prompts, padding=True, return_tensors="pt")
        tokenization["input_ids"] = tokenization["input_ids"].to(self.device)
        tokenization["attention_mask"] = tokenization["attention_mask"].to(self.device)
        tokenization.update(
            {
                "max_new_tokens": max_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                "do_sample": True,
                "temperature": self.temperature,
                "top_p": self.top_p,
            }
        )

        outputs = self.model.generate(**tokenization)
        generations = self.tokenizer.batch_decode(
            outputs[:, -max_tokens:], skip_special_tokens=True
        )

        return generations


class MistralModel(Model):
    """
    MistralModel is a class that represents a Mistral model for text generation.

    Args:
        auth_token (str): The authentication token for the Mistral model.
        device (str): The device to run the model on.
        system_prompt (str): The system prompt to use.
        apply_defense_methods (bool): Whether to apply defense methods during text generation.
    """
    
    model_details = {
        "mistral-large-latest",
        "mistral-small-latest"}

    def __init__(
        self, 
        auth_token: str, 
        device: str,
        system_prompt: str,
        model_name: str,
        apply_defense_methods: bool
    ):
        super().__init__(auth_token, device, system_prompt, apply_defense_methods)

        self.client = MistralClient(api_key=auth_token)
        self.model_name = model_name

    def internal_generate(self, prompts, max_tokens):
        """
        Generate text using the Mistral model.

        Args:
            prompts (List[str]): The prompts for text generation.
            max_tokens (int): The maximum number of tokens to generate.

        Returns:
            List[str]: The generated text.
        """
        prompts = [[ChatMessage(role="user", content=prompt)] for prompt in prompts]

        def fetch_generation(prompt):
            output = self.client.chat(
                model=self.model_name,
                messages=prompt,
                max_tokens=max_tokens)
            
            return output.choices[0].message.content

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(fetch_generation, prompt): i for i, prompt in enumerate(prompts)}
            results = [None] * len(prompts)
            for future in concurrent.futures.as_completed(futures):
                result_position = futures[future]
                results[result_position] = future.result()

        return results
    
    
class OpenaiModel(Model):
    """
    OpenaiModel is a class that represents an OpenAI model for text generation.

    Args:
        auth_token (str): The authentication token for the OpenAI model.
        device (str): The device to run the model on.
        system_prompt (str): The system prompt to use.
        apply_defense_methods (bool): Whether to apply defense methods during text generation.
    """

    model_details = {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-1106",
        "gpt-4-0613"}
    
    def __init__(
        self, 
        auth_token: str, 
        device: str, 
        system_prompt: str,
        model_name: str,
        temperature: float,
        top_p: float,
        apply_defense_methods: bool,
    ):
        super().__init__(auth_token, device, system_prompt, apply_defense_methods)
        self.client = OpenAI(api_key=auth_token)
        self.max_parallelism = MAX_PARALLELISM_LLM_MODELS
        self.temperature = temperature
        self.top_p = top_p
        self.model_name = model_name

    def internal_generate(self, prompts, max_tokens):
        """
        Generate text using the OpenAI model.

        Args:
            prompts (List[str]): The prompts for text generation.
            max_tokens (int): The maximum number of tokens to generate.

        Returns:
            List[str]: The generated text.
        """
        prompts = [
            [{"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}]
            for prompt in prompts
        ]

        def fetch_generation(prompt):
            output = self.client.chat.completions.create(
                model=self.model_name,
                messages=prompt,
                max_tokens=max_tokens,
                temperature=self.temperature,
                top_p=self.top_p
            )
            return output.choices[0].message.content

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallelism) as executor:
            futures = {executor.submit(fetch_generation, prompt): i for i, prompt in enumerate(prompts)}
            results = [None] * len(prompts)
            for future in concurrent.futures.as_completed(futures):
                result_position = futures[future]
                results[result_position] = future.result()

        return results
