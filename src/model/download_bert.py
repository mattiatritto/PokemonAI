from transformers import AutoModel

model_name = "prajjwal1/bert-mini"
output_dir = "bert-mini-local"

model = AutoModel.from_pretrained(model_name)
model.save_pretrained(output_dir)
print(f"[Success]: Saved model and tokenizer to: {output_dir}")