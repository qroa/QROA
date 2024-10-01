import os 
from tqdm import tqdm
import json

from src.models.models_factory import get_model
from src.attack.trigger_generator import TriggerGenerator, TriggerValidator

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM


class JailBreak:
    def __init__(self, device, config):
        self.device = device
        self.config = config
        self.model = get_model(
            self.config["model"],
            self.config["apply_defense_methods"], 
            self.config["auth_token"],
            device,
            self.config["system_prompt"],
            self.config["temperature"],
            self.config["top_p"])
        
        self.model_name = self.config["model"]
        
        self.embedding_model_path = self.config["embedding_model_path"]
        self.reference_embedding, self.tokenizer_surrogate_model = self._load_embedding_model(self.embedding_model_path)
        self.reference_embedding = self.reference_embedding.to(device, dtype=torch.float32)
        self.reference_embedding.requires_grad = False
        
        self.logging_path = os.path.join(self.config['logging_path'], self.model_name)
        self.results_path = os.path.join(self.config['results_path'], self.model_name)

        os.makedirs(self.logging_path, exist_ok=True)
        os.makedirs(self.results_path, exist_ok=True)

        self.logging_generator_path = os.path.join(self.logging_path, "logging_generator.json")
        self.logging_validator_path = os.path.join(self.logging_path, "logging_validator.json")
        
        self.logging_generator_path_csv = os.path.join(self.logging_path, "logging_generator.csv")
        self.logging_validator_path_csv = os.path.join(self.logging_path, "logging_validator.csv")

        self.triggers_path = os.path.join(self.results_path, "triggers.json")
        self.triggers_validate_path = os.path.join(self.results_path, "triggers_validate.json")

    def _load_embedding_model(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer.pad_token_id = tokenizer.unk_token_id

        with torch.no_grad():
            ref_emb = model.get_input_embeddings().weight.data
        del model
        return ref_emb, tokenizer

    def _load_json(self, file_path):
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _read_csv(self, file_path):
        """Reads a CSV file into a DataFrame. Returns an empty DataFrame if the file is empty or only contains headers."""
        try:
            if os.path.exists(file_path):
                return pd.read_csv(file_path)
            else:
                return pd.DataFrame()

        except Exception as e:
            print(e)
            return pd.DataFrame()
    
    def run(self, instructions):
        
        print()
        print('Run')
        print('-----------------------')

        triggers = self._load_json(self.triggers_path)
        triggers_validate = self._load_json(self.triggers_validate_path)

        logging_generator = self._read_csv(self.logging_generator_path_csv)
        logging_validator = self._read_csv(self.logging_validator_path_csv)
    
        for instruction in instructions: 
            trigger_generator = TriggerGenerator(self.model, 
                                                self.device, 
                                                self.config,
                                                self.reference_embedding, 
                                                self.tokenizer_surrogate_model)
            
            trigger_validator = TriggerValidator(self.model, 
                                                self.device, 
                                                self.config)
            
            triggers[instruction] = trigger_generator.run(instruction)

            logging_generator = pd.concat([logging_generator, trigger_generator.return_logging()], ignore_index=True)
            logging_generator.to_json(self.logging_generator_path)
            logging_generator.to_csv(self.logging_generator_path_csv, index=False)
            with open(self.triggers_path, 'w') as f:
                json.dump(triggers, f)

            triggers_validate[instruction] = trigger_validator.run(instruction, triggers[instruction])

            logging_validator = pd.concat([logging_validator, trigger_validator.return_logging()], ignore_index=True)
            logging_validator.to_json(self.logging_validator_path)
            logging_validator.to_csv(self.logging_validator_path_csv, index=False)
            with open(self.triggers_validate_path, 'w') as f:
                json.dump(triggers_validate, f)

        return  triggers, triggers_validate
