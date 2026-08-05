"""
阶段 4 评估：对比 1.5B 模型训练前后准确率
"""
import torch
import re
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/mnt/nvme_gm9_1tb/models/Qwen2.5-1.5B-Instruct"
TRAINED_PATH = "/mnt/hdd_wd_4tb/grpo_output/grpo_1.5b_addition"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

random.seed(999)
test_questions = []
for _ in range(20):
    a = random.randint(100, 999)
    b = random.randint(100, 999)
    test_questions.append((a, b, str(a + b)))

def test_model(model, tokenizer, questions):
    correct = 0
    for a, b, ans in questions:
        messages = [{"role": "user", "content": f"What is {a}+{b}? Answer with just the number."}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        response = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        numbers = re.findall(r'\d+', response)
        if numbers and numbers[-1] == ans:
            correct += 1
    return correct / len(questions)

print("=== 测试训练前 ===")
base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map="auto")
base_acc = test_model(base_model, tokenizer, test_questions)
print(f"准确率: {base_acc:.1%}")

print("\n=== 测试训练后 ===")
trained_model = AutoModelForCausalLM.from_pretrained(TRAINED_PATH, dtype=torch.bfloat16, device_map="auto")
trained_acc = test_model(trained_model, tokenizer, test_questions)
print(f"准确率: {trained_acc:.1%}")

print(f"\n=== 对比 ===")
print(f"训练前: {base_acc:.1%}")
print(f"训练后: {trained_acc:.1%}")
print(f"提升: {trained_acc - base_acc:+.1%}")

del base_model, trained_model
torch.cuda.empty_cache()
