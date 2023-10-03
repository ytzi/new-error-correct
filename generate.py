import vllm
import argparse
import json
from tqdm import tqdm
from pathlib import Path

def generate(model, prompts, batch_size=32, **kwargs):
    params = vllm.SamplingParams(**kwargs)
    llm = vllm.LLM(model=model)
    prompts_split_by_batch = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]
    outputs = []
    for batch in tqdm(prompts_split_by_batch, desc="batch", total=len(prompts_split_by_batch)):
        batch_text = [b["prompt"] for b in batch]
        output = llm.generate(batch_text, params)
        outputs.extend(output)
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
    
    args = args.parse_args()

    prompts = read_prompts(args.prompt_file)
    outputs = generate(args.model, prompts, args.batch_size, temperature=args.temperature, top_p=args.top_p)
    produce_output(args.output_file, outputs)


if __name__ == "__main__":
    main()