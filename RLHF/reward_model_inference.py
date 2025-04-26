import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from llava import conversation as conversation_lib

from models.reward_model import (
    RewardConfig,
    RewardModel
)
from llava.model import LlavaLlamaForCausalLM

from models.reward_model import load_4bit_reward_model_for_inference

# def load_model(model_name_or_path, device):
#     tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
#     print(f"Tokeinzer loaded")

#     config = RewardConfig(backbone_model_name_or_path=model_name_or_path)

#     model = RewardModel.from_pretrained(
#                     config = config,
#                     pretrained_model_name_or_path = model_name_or_path,
#                     #checkpoint_dir = model_name_or_path,
#                     tokenizer = tokenizer
#                 ).to(device)
#     return tokenizer, model

def generate_response(tokenizer, model, user_msg, max_length, device):
    conv = conversation_lib.default_conversation.copy()
    conv.append_message(conv.roles[0], user_msg)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)


    # outputs = model.generate(
    #     inputs["input_ids"], 
    #     max_length=max_length, 
    #     num_return_sequences=1, 
    #     temperature=0.7, 
    #     top_k=50
    # )
    outputs = model(inputs['input_ids'], return_dict=True)  
    return outputs #tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser(description="LLaVA CLI Inference")
    parser.add_argument("--model", type=str, default = "/scratch/e35895/code/LLaVA-RLHF/checkpoints/LLaVA-Fact-RM-7b-v1.5-224-sherlock/checkpoint-23000", help="Path to the model or model name")
    #parser.add_argument("--prompt", type=str, required=True, help="Input prompt for the model")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum length of the generated response")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_4bit_reward_model_for_inference(args.model)

    #take prompt from user
    prompt = input("Enter the prompt: ")

    response = generate_response(tokenizer, model, prompt, args.max_length, args.device)
    print("Generated Response:")
    print(response)

if __name__ == "__main__":
    main()