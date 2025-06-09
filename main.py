import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

model_names = [
	"/mnt/data/chatglm3-6b",
	"/mnt/data/chatglm2-6b",
	"/mnt/data/Qwen-7B-Chat",
	"/mnt/data/Baichuan2-7B-Base",
	"/mnt/data/Baichuan2-13B-Base"
]

prompts = [
    "请说出以下两句话区别在哪里？ 1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少",
    "请说出以下两句话区别在哪里？单身狗产生的原因有两个，一是谁都看不上，二是谁都看不上",
    "他知道我知道你知道他不知道吗？这句话里，到底谁不知道",
    "明明明明明白白白喜欢他，可她就是不说。这句话里，明明和白白谁喜欢谁？",
    "领导：你这是什么意思？ 小明：没什么意思。意思意思。领导：你这就不够意思了。小明：小意思，小意思。领导：你这人真有意思。小明：其实也没有别的意思。领导：那我就不好意思了。小明：是我不好意思。请问：以上“意思”分别是什么意思。"
]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)
    model.eval()

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        streamer = TextStreamer(tokenizer)
        with torch.inference_mode():
            outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print(prompt)
        print("\n[回答]：", output_text)


if __name__ == "__main__":
	res = input("请选择要运行的模型：\n【1】chatglm3-6b\n【2】chatglm2-6b\n【3】Qwen-7B-Chat\n【4】Baichuan2-7B-Base\n【5】Baichuan2-13B-Base\n")
	try:
		res = int(res)
		if res > 5 or res < 1:
			raise ValueError("")
		main(model_names[res-1])
	except Exception as e:
		print("发生错误：", e)
		exit(0)
			
