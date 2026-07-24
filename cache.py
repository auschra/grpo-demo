from huggingface_hub import snapshot_download
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

snapshot_download(repo_id="Qwen/Qwen3-0.6B")
ds = load_dataset("AI-MO/NuminaMath-TIR", split="train")
ds.save_to_disk("./local_numinamath")
