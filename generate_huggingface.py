import transformers 
import argparse
import json
from tqdm import tqdm
from pathlib import Path

device = "cuda"

def generate(model_name, prompts, batch_size=32, **kwargs):
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    tokenizer.pad_token = tokenizer.eos_token
    prompts_split_by_batch = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]
    outputs = []
    for batch in tqdm(prompts_split_by_batch, desc="batch", total=len(prompts_split_by_batch)):
        batch_text = [b["prompt"] for b in batch]
        input = tokenizer(batch_text, return_tensors="pt", padding=True).to(device)
        model_output = model.generate(input["input_ids"], attention_mask=input["attention_mask"], **kwargs)
        for moutput, b in zip(model_output, batch):
            b["model_output"] = tokenizer.decode(moutput)
            outputs.append(b)
    return outputs

def read_prompts(prompt_file: Path):
    with open(prompt_file, 'r') as f:
        prompts = [json.loads(line) for line in f]
    return prompts

def produce_output(output_file: Path, outputs):
    with open(output_file, 'w') as f:
        f.write(json.dumps(outputs))

def main():
    args = argparse.ArgumentParser()

    args.add_argument("--model", type=str, required=True)
    args.add_argument("--prompt-file", type=Path, required=True)
    args.add_argument("--output-file", type=Path, required=True)

    args.add_argument("--batch-size", type=int, default=32)
    args.add_argument("--temperature", type=float, default=0.8)
    args.add_argument("--top-p", type=int, default=0.9)
    args.add_argument("--max-length", type=int, default=1000)
    
    args = args.parse_args()

    prompts = read_prompts(args.prompt_file)
    outputs = generate(args.model, prompts, args.batch_size, temperature=args.temperature, top_p=args.top_p, do_sample=True, max_length=args.max_length)
    produce_output(args.output_file, outputs)


if __name__ == "__main__":
    main()