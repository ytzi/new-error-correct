from vllm import LLM

llm = LLM(model="/home/arjun/models/starcoderbase")
output = llm.generate("def add1(x):")
print(output)