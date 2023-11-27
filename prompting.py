from transformers import AutoModelForCausalLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import pipeline
import pandas as pd
import numpy as np
import jieba
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def load_split(file_path):
    if 'parallel' in file_path.lower():
        data = pd.read_csv(file_path, sep=',', header=None, names=['source', 'cantonese', 'mandarin']).drop(columns=['source'])
    else:
        data = pd.read_csv(file_path, sep=',', header=None, names=['cantonese', 'mandarin'])
    return data


# def tokenize_data(tokenizer, data):
#     text2text_generator = pipeline("text-generation", model="indiejoseph/cantonese-llama-2-7b-oasst-v1")

#     tokenized_mando = tokenizer(data['mandarin'].tolist(), truncation=True, padding=True, max_length=256, return_tensors="pt").input_ids

#     return tokenized_canto, tokenized_mando

def evaluate_bleu(predictions, ground_truth):
    # Add the prompt to the Cantonese sentences
    smooth_fn = SmoothingFunction()

    bleu_score = []
    canto_list = [[list(jieba.cut(item.strip(), use_paddle=True, cut_all=False))]for item in predictions]
    mando_list = [list(jieba.cut(item.strip(), use_paddle=True, cut_all=False)) for item in ground_truth]

    # print(len(canto_list), len(mando_list))

    for gd, zh in zip(canto_list, mando_list):
        s = nltk.translate.bleu_score.sentence_bleu(gd, zh, smoothing_function=smooth_fn.method3)
        bleu_score.append(s)
    bleu_score = np.mean(bleu_score) * 100.0

    # print('BLEU:', bleu_score)
    # bleu_score = nltk.translate.bleu_score.sentence_bleu(predictions, ground_truth) * 100.0

    print(f"SacreBLEU Score: {bleu_score}")


def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_csv", type=str, help='Cantonese input corpus')
    args=parser.parse_args()

    data = load_split(args.input_csv)

    model = AutoModelForCausalLM.from_pretrained("indiejoseph/cantonese-llama-2-7b-oasst-v1", device_map="auto", offload_folder="offload")
    tokenizer = AutoTokenizer.from_pretrained("indiejoseph/cantonese-llama-2-7b-oasst-v1")
    template = """Translate the Cantonese text into Mandarin.

    Cantonese: {}

    Mandarin: 
    """
    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "left"

    def inference(input_texts, batch_size=32):
        # pipe = pipeline("text-generation", model="indiejoseph/cantonese-llama-2-7b-oasst-v1")

        outputs = []
        total_batches = len(input_texts) // batch_size

        for i in tqdm(range(0, len(input_texts), batch_size), total=total_batches, desc="Inference"):
            batch_texts = input_texts[i:i+batch_size]

            inputs = tokenizer([template.format(text) for text in batch_texts], return_tensors="pt", padding=True, truncation=True, max_length=256).to('cuda')

            # Generate
            generate_ids = model.generate(**inputs, max_new_tokens=256)
            batch_outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            batch_outputs = [out.split('Mandarin:')[1].strip() for out in batch_outputs]

            outputs.extend(batch_outputs)
        
        return outputs

    def post_processing(input_text):
        import re;

        res = []
        for inp in input_text:
            re.sub(r'^\w', '', inp)
            res.append(inp.replace('\n', ''))
        return res

    outputs = inference(data['cantonese'], batch_size=16)
    # outputs = inference("摵住一個袋")
    # print('outputs:', outputs)

    # out = "".join(outputs)
    out = post_processing(outputs)
    print('out:', out)
    evaluate_bleu(out, data['mandarin'])
    # evaluate_bleu(out, data['mandarin'])


if __name__ == '__main__':
    main()