import click
import torch

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

@click.command()
@click.option('--model_name', '-m', default="PygmalionAI/pygmalion-6b", help='Name of the Hugging Face model to load.')
@click.option('--adapters_name', '-a', default='hedronstone/6b-jilp', help='Name of the adapter to load with the model.')
def main(model_name, adapters_name):
    """
    This script loads a pre-trained Hugging Face model with a specified adapter into memory.
    """
    try:
        model, tokenizer, stop_token_ids = load_and_prepare_model(model_name, adapters_name)
        print(f"Successfully loaded the model {model_name} into memory")
    except Exception as e:
        print(f"Failed to load the model. Error: {str(e)}")

        return model, tokenizer, stop_token_ids

    


def load_and_prepare_model(model_name, adapters_name):
    """
    Load a pre-trained Hugging Face model and an adapter. Prepare the model for future usage.

    Args:
        model_name (str): Name of the model to load.
        adapters_name (str): Name of the adapter to load with the model.
    """
    print(f"Starting to load the model {model_name} into memory")

    # Load the pre-trained model from Hugging Face model hub
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0}
    )
    
    # Load the specified adapter
    model = PeftModel.from_pretrained(model, adapters_name)

    # Unload unneeded model components to save memory
    model = model.merge_and_unload()

    # Prepare the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.bos_token_id = 1

    # Define the stop token ids for the tokenizer
    stop_token_ids = [0]

    return model, tokenizer, stop_token_ids


if __name__ == "__main__":
    main()
