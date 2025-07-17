from transformers import pipeline
import os

# 初始化翻译管道
pipe = pipeline("translation", model="C:/Users/mbl/Desktop/nllb-200-distilled-600M")

# 输入和输出文件路径
input_file = "source_corpus_english_from_airbench.txt"
output_file = "translated_corpus_zho_Hans.txt"

# 源语言和目标语言
source_lang = "eng_Latn"  # 英文
target_lang = "zho_Hans"  # 简体中文

# 确保输入文件存在
if not os.path.exists(input_file):
    raise FileNotFoundError(f"Input file {input_file} not found")

# 读取输入文件并翻译
translated_lines = []
with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()  # 去除换行符和多余空格
        if line:  # 仅翻译非空行
            try:
                # 翻译单行文本
                translation = pipe(line, src_lang=source_lang, tgt_lang=target_lang, max_length=512, num_beams=5)
                translated_text = translation[0]["translation_text"]
                translated_lines.append(f"{line} -> {translated_text}\n")
            except Exception as e:
                print(f"Error translating line: {line}\nError: {e}")
                translated_lines.append(f"{line} -> [Translation Error]\n")

# 将翻译结果写入输出文件
with open(output_file, "w", encoding="utf-8") as f:
    f.writelines(translated_lines)

print(f"Translation completed. Results saved to {output_file}")