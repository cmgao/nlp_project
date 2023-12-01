from transformers import AutoModelForCausalLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import pipeline
import pandas as pd
import numpy as np
import jieba
from tqdm import tqdm
from typing import Sequence, Callable
from sacrebleu.metrics import BLEU, CHRF

def load_split(file_path):
    if 'parallel' in file_path.lower():
        data = pd.read_csv(file_path, sep=',', header=None, names=['source', 'cantonese', 'mandarin']).drop(columns=['source'])
    else:
        data = pd.read_csv(file_path, sep=',', header=None, names=['cantonese', 'mandarin'])
    return data


# def evaluate_bleu(predictions, ground_truth):
#     # Add the prompt to the Cantonese sentences
#     smooth_fn = SmoothingFunction()

#     bleu_score = []
#     chrf_score = []

#     canto_list = [[list(jieba.cut(item.strip(), use_paddle=True, cut_all=False))]for item in predictions]
#     mando_list = [list(jieba.cut(item.strip(), use_paddle=True, cut_all=False)) for item in ground_truth]

#     # print(len(canto_list), len(mando_list))

#     for gd, zh in zip(canto_list, mando_list):
#         s = nltk.translate.bleu_score.sentence_bleu(gd, zh, smoothing_function=smooth_fn.method3)
#         bleu_score.append(s)

#         gd_str = " ".join([word for sublist in gd for word in sublist])
#         zh_str = " ".join([word for sublist in zh for word in sublist])

#         ## calculate chrff++ score
#         c = nltk.translate.chrf_score.sentence_chrf(gd_str, zh_str, min_len=1, beta=2) 
#         chrf_score.append(c)

#     bleu_score = np.mean(bleu_score) * 100.0
#     chrf_score = np.mean(chrf_score) * 100.0

#     # print('BLEU:', bleu_score)
#     # bleu_score = nltk.translate.bleu_score.sentence_bleu(predictions, ground_truth) * 100.0

#     print(f"SacreBLEU Score: {bleu_score}")
#     print(f"Chrff++ Score: {chrf_score}")

def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_csv", type=str, help='Input corpus')
    parser.add_argument("--input_lang", type=str, help='Input language (e.g., "cantonese")')
    parser.add_argument("--output_lang", type=str, help='Output language (e.g., "mandarin")')
    parser.add_argument("--prompting_type", type=str, default='zero', help='Prompting type: "zero" or "few_shot"')

    args=parser.parse_args()
    data = load_split(args.input_csv)

    model = AutoModelForCausalLM.from_pretrained("indiejoseph/cantonese-llama-2-7b-oasst-v1", device_map="auto", offload_folder="offload")
    tokenizer = AutoTokenizer.from_pretrained("indiejoseph/cantonese-llama-2-7b-oasst-v1")
    
    ## Templates
    ## zero shot - input lang : sample sentence --> output lang :
    template_zero_shot = f"""Translate the {args.input_lang} text into {args.output_lang}.
    {args.input_lang}: {{}}
    {args.output_lang}: 
    """

    ## few shot - input lang : sample sentence --> output lang : output sentence
    template_few_shot = f"""Translate the {args.input_lang} text into {args.output_lang}.
    {args.input_lang}: {{}}
    {args.output_lang}: 
    """

    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "left"

    def inference(input_texts, batch_size=32, use_few_shot_prompting=False):
        # pipe = pipeline("text-generation", model="indiejoseph/cantonese-llama-2-7b-oasst-v1")

        outputs = []
        total_batches = len(input_texts) // batch_size

        selected_samples = data.sample(n=5)

        # for i in tqdm(range(0, len(input_texts), batch_size), total=total_batches, desc="Inference"):
        if use_few_shot_prompting: ## few shot
            for i in tqdm(range(0, len(input_texts)), total=len(input_texts), desc="Inference"):
                prompt_texts = [f"Given the examples, translate the sixth {args.input_lang} sentence into {args.output_lang}.\n"]
                # prompt_texts = ["把普通话句子翻译成粤语。"] ## translate mandarin to cantonese
                # prompt_texts = ["把粵語句子翻譯成國語。"] ## translate cantonese to mandarin
                prompt_texts.extend([template_few_shot.format(text, translate) for text, translate in zip(selected_samples[args.input_lang], selected_samples[args.output_lang])])
                # print('prompt_texts:', prompt_texts)
                next_line = [f'{args.input_lang}: '+input_texts[i]+'\n'+f'{args.output_lang}:']
                prompt_texts.extend(next_line)
                
                inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to('cuda')
                generate_ids = model.generate(**inputs, max_new_tokens=260)
                batch_outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

                # import pudb; pudb.set_trace()
                result = [batch_outputs[-1].split(f'{args.output_lang}:')[-1].strip()]
                outputs.extend(result)
                # print(outputs)

        else: ## zero shot
            for i in tqdm(range(0, len(input_texts), batch_size), total=total_batches, desc="Inference"):
                batch_texts = input_texts[i:i+batch_size]
                inputs = tokenizer([template_zero_shot.format(text, '') for text in batch_texts], return_tensors="pt", padding=True, truncation=True, max_length=512).to('cuda')

                # Generate
                generate_ids = model.generate(**inputs, max_new_tokens=260)
                batch_outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

                # import pudb; pudb.set_trace()
                batch_outputs = [out.split(f'{args.output_lang}:')[1].strip() for out in batch_outputs] ##[:1] to view the first row
                # print(batch_outputs)
            
                outputs.extend(batch_outputs)
        return outputs

    def post_processing(input_text):
            import re;

            res = []
            for inp in input_text:
                re.sub(r'^\w', '', inp)
                res.append(inp.replace('\n', ''))
            return res

    use_few_shot_prompting = args.prompting_type.lower() == 'few_shot'
    print(args.input_lang, use_few_shot_prompting)

    outputs = inference(input_texts=data[f'{args.input_lang}'], batch_size=16, use_few_shot_prompting=use_few_shot_prompting)
    # outputs = inference("摵住一個袋")
    # print('outputs:', outputs)

    # out = "".join(outputs)
    out = post_processing(outputs)
    # print('out:', out)
    
    # evaluate_bleu(out, data['cantonese'])
    # evaluate_chrff(out, data['cantonese'])

    evaluate_bleu(out, data[args.output_lang])
    # evaluate_chrff(out, data[args.output_lang])



if __name__ == '__main__':
    main()