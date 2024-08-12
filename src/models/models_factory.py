from src.models.base import Model
from src.models.llm_models import HuggingFaceModel, OpenaiModel, MistralModel

def get_model(model_name: str,
              apply_defense_methods: bool,
              auth_token: str, 
              device: str, 
              system_prompt: str,
              temperature: float,
              top_p: float) -> Model:
    """
    Factory method to get the appropriate model based on the given parameters.

    Args:
        model_name (str): The name of the model.
        auth_token (str): The authentication token.
        device (str): The device to run the model on.
        system_prompt (str): The system prompt to use.
        temperature (float): The temperature for generating text.
        top_p (float): The top-p value for generating text.

    Returns:
        Model: An instance of the appropriate model based on the given parameters.

    Raises:
        ValueError: If the model_name is unknown.
    """

    if model_name.lower() in HuggingFaceModel.model_details:
        return HuggingFaceModel(
            auth_token=auth_token,
            device=device, 
            system_prompt=system_prompt, 
            model_name=model_name.lower(),
            temperature=temperature,
            top_p=top_p,
            apply_defense_methods=apply_defense_methods)
    elif model_name.lower() == 'openai-0613':
        model =  OpenaiModel(
            auth_token=auth_token, 
            device=device, 
            system_prompt=system_prompt, 
            temperature=temperature,
            top_p=top_p,
            apply_defense_methods=apply_defense_methods
        )
        model.model_name = "gpt-3.5-turbo-0613"
        return model
    elif model_name.lower() == 'openai-1106':
        model =  OpenaiModel(
            auth_token=auth_token, 
            device=device, 
            system_prompt=system_prompt, 
            temperature=temperature,
            top_p=top_p,
            apply_defense_methods=apply_defense_methods
        )
        model.model_name = "gpt-3.5-turbo-1106"
        return model
    elif model_name.lower() == 'openai-4-0613':
        model =  OpenaiModel(
            auth_token=auth_token, 
            device=device, 
            system_prompt=system_prompt, 
            temperature=temperature,
            top_p=top_p,
            apply_defense_methods=apply_defense_methods
        )
        model.model_name = "gpt-4-0613"
        return model
    elif model_name.lower() == 'mistral':
        return MistralModel(
            auth_token=auth_token, 
            device=device, 
            system_prompt=system_prompt, 
            apply_defense_methods=apply_defense_methods
        )
    else:
        raise ValueError(f"Unknown model type: {model_name}")
