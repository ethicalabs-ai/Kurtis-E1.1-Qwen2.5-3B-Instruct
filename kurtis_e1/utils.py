import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM, PeftConfig


# Determine torch data type based on CUDA capabilities.
torch_dtype = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    else torch.float32
)


def save_and_merge_model(
    final_checkpoint_dir: str,
    output_merged_dir: str,
    chat_template: str,
    hf_repo_id: str = "",
    push: bool = False,
):
    """Save the adapter model and tokenizer."""
    peft_config = PeftConfig.from_pretrained(final_checkpoint_dir)
    base_model_path = peft_config.base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if chat_template:
        tokenizer.chat_template = chat_template
    model = AutoPeftModelForCausalLM.from_pretrained(
        final_checkpoint_dir, device_map="auto", torch_dtype=torch_dtype
    )
    model = model.merge_and_unload()
    tokenizer.save_pretrained(output_merged_dir)
    model.save_pretrained(output_merged_dir, safe_serialization=True)
    if push and hf_repo_id:
        model.push_to_hub(hf_repo_id, "Upload model")
        tokenizer.push_to_hub(hf_repo_id, "Upload tokenizer")
