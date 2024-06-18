from transformers import AutoFeatureExtractor, WhisperForAudioClassification
import os
# from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model

# def print_trainable_parameters(model):
#     trainable_params = 0
#     all_param = 0
#     for _, param in model.named_parameters():
#         all_param += param.numel()
#         if param.requires_grad:
#             trainable_params += param.numel()
#     print(
#         f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
#     )

def get_model(device):
    token = os.getenv('HF_TOKEN')
    model_id = "openai/whisper-large-v2"

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, token=token)
    model = WhisperForAudioClassification.from_pretrained(model_id, token=token, num_labels=7)
    # model.enable_input_require_grads()
    
    # config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")

    # model = get_peft_model(model, config)
    # model.print_trainable_parameters()

    model.to(device)

    return model, feature_extractor