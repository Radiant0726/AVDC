import os
os.environ["http_proxy"] = "http://10.20.112.35:3143"
os.environ["https_proxy"] = "http://10.20.112.35:3143"

import re
import copy
import torch
import numpy as np
import json
import evaluate
from transformers import AutoTokenizer, AutoModel

from typing import List, Dict, Tuple
from collections import defaultdict, Counter
import torch.nn.functional as F
import nltk
from nltk.util import ngrams
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score as meteor_score_fn
from nltk.tokenize import sent_tokenize, word_tokenize
# from bert_score import score as bert_score_fn
# from rouge_score import rouge_scorer
from cidereval import cider, ciderD
try:
    from .cider.cidereval.scorers import ciderR
except:
    from cider.cidereval.scorers import ciderR

# nltk.download('punkt')
# nltk.download('wordnet', download_dir='/mnt/vision_user/kaiyinyan/nltk_data')
nltk.data.path.append('/mnt/vision_user/kaiyinyan/nltk_data')
try:
    from .loc_metrics import *
except:
    from loc_metrics import *

# test
predictions = ["A. the cat is on the mat","the cat is on the mat","the cat is on the mat","the cat is on the mat"]
references = ["the cat is on a mat","the cat is on the mat","the cat is on the mat","the cat is on the mat",]

def preprocess_logits_for_metrics(logits, labels):

    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels

# LLM解码器
tokenizer = AutoTokenizer.from_pretrained("/mnt/vision_user/kaiyinyan/models/Qwen2.5-Omni-7B", use_fast=False, trust_remote_code=True)
test_text = "Upon watching the video, the sequence of events and the multimodal content can be summarized as follows:\n\n1. **Initial Scene (0s - 1.97s):**\n   - The video begins with a close-up shot of a hand interacting with a zipper on a piece of fabric. The hand is seen pulling the zipper, indicating the opening or closing of a bag or a similar item. The fabric appears to be part of a larger structure, possibly a tent or a sleeping bag, given the context of the subsequent scenes.\n\n2. **Scene Transition (1.97s - 5.97s):**\n   - The camera shifts to a wider view, revealing a person inside a tent. The person is lying on a bed, which is covered with a red and black sleeping bag. The tent is well-lit, suggesting it is daytime. The person appears to be adjusting the sleeping bag or preparing for sleep. The tent is equipped with a pillow and a sleeping pad, indicating a comfortable camping setup.\n\n3. **Scene Transition (5.97s - 9.98s):**\n   - The camera focuses on the person's hand as they interact with the zipper again. This time, the zipper is on a different piece of fabric, possibly a sleeping bag or a tent door. The hand is seen pulling the zipper, suggesting the person is either opening or closing the zipper. The fabric is dark, and the zipper is metallic, indicating a sturdy construction.\n\n4. **Scene Transition (9.98s - 13.99s):**\n   - The camera shifts to a wider view of the tent interior. The person is now seen lying on a bed, which is covered with a red and black sleeping bag. The tent is well-lit, and the person appears to be adjusting the sleeping bag or preparing for sleep. The tent is equipped with a pillow and a sleeping pad, indicating a comfortable camping setup.\n\n5. **Scene Transition (13.99s - 20s):**\n   - The camera focuses on the person's hand as they interact with the zipper again. This time, the zipper is on a different piece of fabric, possibly a sleeping bag or a tent door. The hand is seen pulling the zipper, suggesting the person is either opening or closing the zipper. The fabric is dark, and the zipper is metallic, indicating a sturdy construction.\n\n6. **Scene Transition (20s - 25.97s):**\n   - The camera shifts to a wider view of the tent interior. The person is now seen lying on a bed, which is covered with a red and black sleeping bag. The tent is well-lit, and the person appears to be adjusting the sleeping bag or preparing for sleep. The tent is equipped with a pillow and a sleeping pad, indicating a comfortable camping setup.\n\n7. **Scene Transition (25.97s - 32s):**\n   - The camera focuses on the person's hand as they interact with the zipper again. This time, the zipper is on a different piece of fabric, possibly a sleeping bag or a tent door. The hand is seen pulling the zipper, suggesting the person is either opening or closing the zipper. The fabric is dark, and the zipper is metallic, indicating a sturdy construction.\n\n8. **Scene Transition (32s - 44s):**\n   - The camera shifts to a wider view of the tent interior. The person is now seen lying on a bed, which is covered with a red and black sleeping bag. The tent is well-lit, and the person appears to be adjusting the sleeping bag or preparing for sleep. The tent is equipped with a pillow and a sleeping pad, indicating a comfortable camping setup.\n\n9. **Scene Transition (44s - 48s):**\n   - The camera focuses on the person's hand as they interact with the zipper again. This time, the zipper is on a different piece of fabric, possibly a sleeping bag or a tent door. The hand is seen pulling the zipper, suggesting the person is either opening or closing the zipper. The fabric is dark, and the zipper is metallic, indicating a sturdy construction.\n\n10. **Final Scene (48s - 52s):**\n    - The camera shifts to a wider view of the tent interior. The person is now seen lying on a bed, which is covered with a red and black sleeping bag. The tent is well-lit, and the person appears to be adjusting the sleeping bag or preparing for sleep. The tent is equipped with a pillow and a sleeping pad, indicating a comfortable camping setup.\n\nIn summary, the video depicts a person preparing for sleep inside a tent. The sequence of events includes close-up shots of the person interacting with zippers on various pieces of fabric, likely sleeping bags or tent doors. The person is seen lying on a bed covered with a red and black sleeping bag, adjusting the sleeping bag and preparing for sleep."
inputs = tokenizer(test_text)

def round_metric_dict(metric_dict):
    for key, value in metric_dict.items():
        metric_dict[key] = round(value,4)
    return metric_dict

# CIDEr
def cider_score_fn(predictions, references):
    score = cider(predictions=predictions, references=references, df="corpus")['avg_score'].item()
    scoreD = ciderD(predictions=predictions, references=references, df="corpus")['avg_score'].item()
    scoreR = ciderR(predictions=predictions, references=references)['avg_score'].item()
    return {"cider":score, "ciderD":scoreD, "ciderR": scoreR}
# 示例
# cider_results = cider_score_fn(predictions=predictions, references=[[ref] for ref in references])
# print(cider_results)

# meteor_results = [meteor_score_fn([ref.split()], pred.split()) for ref, pred in zip(references,predictions)]
# meteor_results = sum(meteor_results) / len(meteor_results)
# print(meteor_results)

def repeat_metrics(text, n=3):
    if isinstance(text, str):
        text_lst = [text]
    else:
        text_lst = text

    results = defaultdict(float)        
    for text in text_lst:
        tokens = re.findall(r'\b\w+\b', text.lower())
        N = len(tokens)
        # A. 重复词占比
        cnt = Counter(tokens)
        R_w = sum(v for k,v in cnt.items() if v>1) / N if N else 0
        # B. 相邻重复
        R_adj = sum(t1==t2 for t1,t2 in zip(tokens, tokens[1:])) / (N-1) if N>1 else 0
        # C. 冗余 n-gram
        def r_ngram(n):
            grams = [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
            c = Counter(grams)
            return sum(1 for v in c.values() if v>1) / len(c) if c else 0
        R_n = {f'R_{i}g': r_ngram(i) for i in range(2, n+1)}
        curr_results = {'R_w': R_w, 'R_adj': R_adj, **R_n}
        for key, value in curr_results.items():
            results[key] += value
    for key, value in results.items():
        results[key] = value / len(text_lst)

    return results

results_repeat = repeat_metrics("This is a grammatically correct sentence.")


from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model_name = "/mnt/vision_user/kaiyinyan/models/gpt2"
GPT2model = GPT2LMHeadModel.from_pretrained(model_name).cuda()
GPT2model.eval()
GPT2tokenizer = GPT2Tokenizer.from_pretrained(model_name)
GPT2device = next(GPT2model.parameters()).device

def perplexity(text):
    inputs = GPT2tokenizer(text, return_tensors="pt")
    inputs = {k:v.to(GPT2device) for k,v in inputs.items()}
    with torch.no_grad():
        loss = GPT2model(**inputs, labels=inputs["input_ids"]).loss
    return torch.exp(loss).item()

def perplexity_sliding_window(text, stride=512, max_len=512):
    toks = GPT2tokenizer(text, return_tensors='pt')['input_ids'][0].to(GPT2device)
    nlls = []
    for i in range(0, len(toks), stride):
        begin_loc = max(i + stride - max_len, 0)
        end_loc   = min(i + stride, len(toks))
        trg_len   = end_loc - i
        input_ids = toks[begin_loc:end_loc].unsqueeze(0)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100          # 只留当前窗口的预测部分
        with torch.no_grad():
            loss = GPT2model(input_ids, labels=target_ids).loss
        nlls.append(loss * trg_len)
    ppl = torch.exp(torch.stack(nlls).sum() / len(toks))
    return ppl.item()

def perplexity_batch(texts):
    total_loss = 0
    
    for i in range(0, len(texts)):
        text = texts[i]
        # loss = perplexity(text)
        loss = perplexity_sliding_window(text)
        total_loss = total_loss + loss
    return total_loss / len(texts)

score = perplexity_sliding_window("This is a grammatically correct sentence.")
print(score)


class OmniEvaluator:
    def __init__(self):
        """
        初始化各种评估指标
        """
        # 加载评估器
        # self.accuracy = evaluate.load("accuracy")
        # self.f1 = evaluate.load("f1")
        # self.bleu = evaluate.load('bleu')
        # self.rouge = evaluate.load('rouge')
        # self.meteor = evaluate.load('meteor')
        # self.bertscore = evaluate.load('bertscore')
        # self.BertScoreEvaluator = BertScoreEvaluator(device="cuda:7")

    def extract_mcq_answer(self, response: str) -> str:
        """
        从 文本中提取选择题答案
        """
        # pattern = r'<answer>(.*?)</answer>'
        pattern = r'answer>(.*?)</answer'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            response = match.group(1).strip()

        # 匹配形如 "answer is X" 或单独 "(X)" 的结论，忽略大小写
        match = re.search(r'answer(?: is|:)?\s*([A-E])(?![a-zA-Z])', response, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # 从头开始找大写字母
        match = re.search(r'(?<![a-zA-Z])[A-E](?![a-zA-Z])', response)
        if match:
            return match.group()
        return "unknown"

    def extract_mcq_answer_batch(self, response_lst: list) -> list:
        labels = []
        for response in response_lst:
            label = self.extract_mcq_answer(response)
            labels.append(label)
        return labels

    # Helper function to extract intervals from the string
    def extract_count_timestamps(self, time_str):
        # 提取 Count
        m = re.search(r"Count\s*:\s*(\d+)", time_str)
        count = int(m.group(1)) if m else None
        # Use regex to extract intervals of the form "from X to Y"
        intervals = re.findall(r'from (\d+(\.\d*)?)s* to (\d+(\.\d*)?)s*', time_str)
        # Convert to float and clean intervals (start < end)
        intervals = [(float(start), float(end)) for start,  _, end, _ in intervals if start and end]
        intervals_clean = [interval for interval in intervals if interval[0] < interval[1]]
        return count, intervals_clean

    def extract_count_timestamps_batch(self, response_lst: list) -> list:
        counts = []
        labels = []
        for response in response_lst:
            count, intervals = self.extract_count_timestamps(response)
            counts.append(count)
            labels.append(intervals)
        return counts, labels
    
    def compute_invervals_metrics_batch(self, pred_list, ref_list, tiou_ths = 0.5):
        
        precision_lst = []
        recall_lst = []
        mAP_lst = []
        miou_lst = []
        miou_0_lst = []
        for pred, ref in zip(pred_list, ref_list):
            precision, recall, mAP = compute_map(ref, pred, tiou_ths)
            mean_iou = compute_mean_iou(ref, pred, tiou_ths)
            mean_iou_0 = compute_mean_iou(ref, pred, 0)
            
            mAP_lst.append(mAP)
            precision_lst.append(precision)
            recall_lst.append(recall)
            miou_lst.append(mean_iou)
            miou_0_lst.append(mean_iou_0)

        batch_precision = sum(precision_lst) / len(precision_lst) if precision_lst else 0
        batch_recall = sum(recall_lst) / len(recall_lst) if recall_lst else 0
        batch_mAP = sum(mAP_lst) / len(mAP_lst) if mAP_lst else 0
        batch_miou = sum(miou_lst) / len(miou_lst) if miou_lst else 0
        batch_miou_0 = sum(miou_0_lst) / len(miou_0_lst) if miou_0_lst else 0

        return {"loc_precision": batch_precision,"loc_recall": batch_recall, "mAP":batch_mAP, 
                "miou": batch_miou, "miou_0": batch_miou_0}

    def compute_caption_metrics(self, 
                       predictions: List[str], 
                       references: List[str]) -> Dict[str, float]:
        """
        计算所有评估指标
        
        参数：
        predictions: 生成的caption列表
        references: 参考caption列表的列表（每个样本可以有多个参考）
        
        返回：
        包含所有指标的字典
        """

        assert len(predictions) == len(references), "Prediction and reference lists must be the same length"

        metrics = {}
        
        # 计算BLEU分数 (BLEU-1到BLEU-4)
        # for n in [1, 2, 3, 4]:
        #     bleu_score = self.bleu.compute(
        #         predictions=predictions_caption,
        #         references=[[ref] for ref in references_caption],
        #         max_order=n
        #     )
        #     metrics[f'bleu_{n}'] = bleu_score['bleu']
        
        # 计算ROUGE分数
        # rouge_scores = self.rouge.compute(
        #     predictions=predictions_caption,
        #     references=references_caption, 
        #     use_aggregator=True
        # )
        # # - `"rouge1"`: unigram (1-gram) based scoring
        # # - `"rouge2"`: bigram (2-gram) based scoring
        # # - `"rougeL"`: Longest common subsequence based scoring.
        # # - `"rougeLSum"`: splits text using `"\n"`
        # metrics.update({
        #     'rouge1': rouge_scores['rouge1'].item(),
        #     'rouge2': rouge_scores['rouge2'].item(),
        #     'rougeL': rouge_scores['rougeL'].item(),
        #     'rougeLsum': rouge_scores['rougeLsum'].item()
        # })
        
        # 计算METEOR分数
        # meteor_score = self.meteor.compute(
        #     predictions=predictions_caption,
        #     references=references_caption  
        # )
        # metrics['meteor'] = meteor_score['meteor'].item()
        
        meteor_score_lst = [meteor_score_fn([ref.split()], pred.split()) for ref, pred in zip(references, predictions)]
        metrics['meteor'] = sum(meteor_score_lst) / len(meteor_score_lst)

        # 计算CIDEr分数
        cider_scores = cider_score_fn(
            predictions, [[ref] for ref in references]
        )
        metrics.update(cider_scores)

        metrics = round_metric_dict(metrics)
        return metrics

    def compute_accuracy(self, 
                       predictions: List[str], 
                       references: List[str]) -> Dict[str, float]:
        
        pred_labels = self.extract_mcq_answer_batch(predictions)
        ref_labels = self.extract_mcq_answer_batch(references)
        
        assert len(pred_labels) == len(ref_labels), "Prediction and reference lists must be the same length"

        vaild_idx = [i for i in range(len(ref_labels)) if ref_labels[i]!="unknown"]
        pred_labels = [pred_labels[idx] for idx in vaild_idx]
        ref_labels = [ref_labels[idx] for idx in vaild_idx]

        label_data = [pred_labels, ref_labels]

        all_labels = set(ref_labels + pred_labels)
        label_list = sorted(all_labels)  # 或保留顺序：list(dict.fromkeys([...]))
        label2id = {label: idx for idx, label in enumerate(label_list)}

        pred_ids = [label2id[label] for label in pred_labels]
        ref_ids = [label2id[label] for label in ref_labels]

        # 假设 pred_ids 和 ref_ids 是两个长度相同的列表
        assert len(pred_ids) == len(ref_ids), "Prediction and reference lists must be the same length"

        results = {}

        # 计算正确预测的数量
        correct = sum(p == r for p, r in zip(pred_ids, ref_ids))
        # 计算准确率
        accuracy = correct / len(pred_ids)
        results["accuracy"] = accuracy

        # results = self.accuracy.compute(predictions=pred_ids, references=ref_ids)

        return results, label_data
    
    def compute_loc_metrics(self, 
                       predictions: List[str], 
                       references: List[str]) -> Dict[str, float]:
        

        pred_counts, pred_intervals_lst = self.extract_count_timestamps_batch(predictions)
        ref_counts, ref_intervals_lst = self.extract_count_timestamps_batch(references)
        loc_data = [pred_intervals_lst, ref_intervals_lst]

        assert len(pred_intervals_lst) == len(ref_intervals_lst), "Prediction and reference lists must be the same length"
        
        results = self.compute_invervals_metrics_batch(pred_intervals_lst, ref_intervals_lst)

        acc_counting = sum(1 for a, b in zip(pred_counts, ref_counts) if a == b) / len(pred_counts)
        results.update({"acc_counting":acc_counting})

        return results, loc_data
    
    def compute_perplexity(self, 
                       predictions: List[str], 
                       references: List[str]) -> Dict[str, float]:
        perplexity_pred = perplexity_batch(predictions)
        perplexity_ref = perplexity_batch(references)

        return {"perplexity_pred":perplexity_pred, "perplexity_ref":perplexity_ref}

    def compute_repeat_metrics(self, 
                       predictions: List[str], 
                       references: List[str]) -> Dict[str, float]:
        repeat_pred = repeat_metrics(predictions)
        repeat_ref = repeat_metrics(references)

        results = {}
        for key in repeat_pred.keys():
            results[key+"_pred"] = repeat_pred[key]
            results[key+"_ref"] = repeat_ref[key]
            
        return results
        
evaluator = OmniEvaluator()

def compute_metrics_save(experiment_name, root_output_dir):
    def compute_metrics(eval_pred):
        label_ids = eval_pred.label_ids
        pred_ids = eval_pred.predictions
        
        mask = (label_ids == -100) # 获取无效标签的位置（填充标记）
        label_ids[mask] = tokenizer.pad_token_id

        pred_mask = (pred_ids == -100) # 获取无效标签的位置（填充标记）
        pred_ids[pred_mask] = tokenizer.pad_token_id

        # 验证mask
        assert not np.any(label_ids == -100), "labels 包含 -100"
        assert not np.any(pred_ids == -100), "preds 包含 -100"

        # decode
        decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) # list[str]
        decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) # list[str]

        predictions = decoded_preds # list[str]
        references = decoded_labels # list[str]

        indices_cap = [i for i in range(len(references)) if len(references[i])>5 and len(predictions[i])>0 and not references[i].startswith("Count") and not references[i].startswith("Timestamps")]
        predictions_cap = [predictions[idx] for idx in indices_cap]
        references_cap = [references[idx] for idx in indices_cap]

        if "ft_instruct" in experiment_name:
            indices_acc = [i for i in range(len(references)) if not references[i].startswith("Count") and not references[i].startswith("Timestamps")]
            predictions_acc = [predictions[idx] for idx in indices_acc]
            references_acc = [references[idx] for idx in indices_acc]

            metrics, label_data = evaluator.compute_accuracy(predictions_acc, references_acc)

            pred_labels, ref_labels = label_data
            test_pred_data = [{"pred":pred, "pred_label":pred_label, "ref":ref, "ref_label":ref_label} 
                            for (pred, pred_label, ref, ref_label) in zip(predictions, pred_labels, references, ref_labels)]
            
            # write
            output_path = f"{root_output_dir}/{experiment_name}/test_pred_acc.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(test_pred_data, f, indent=2, ensure_ascii=False)

            test_pred_data_error = [item for item in test_pred_data if item["pred_label"] != item["ref_label"]]

            output_error_path = f"{root_output_dir}/{experiment_name}/test_pred_error.json"
            with open(output_error_path, "w", encoding="utf-8") as f:
                json.dump(test_pred_data_error, f, indent=2, ensure_ascii=False)
                
        else: # caption
            metrics = evaluator.compute_caption_metrics(predictions_cap, references_cap)
            metrics["cider_meteor"] = metrics["cider"] + metrics["meteor"]
            
            metrics_perplexity = evaluator.compute_perplexity(predictions_cap, references_cap)
            metrics_repeat = evaluator.compute_repeat_metrics(predictions_cap, references_cap)
            metrics.update(metrics_perplexity)   
            metrics.update(metrics_repeat)   
            
            # write
            test_pred_data = [{"pred":pred, "ref":ref} for (pred, ref) in zip(predictions, references)]
            output_path = f"{root_output_dir}/{experiment_name}/test_pred_caption.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(test_pred_data, f, indent=2, ensure_ascii=False)

        # loc 
        indices_loc = [i for i in range(len(references)) if references[i].startswith("Count") and not references[i].startswith("Timestamps")]
        if len(indices_loc) > 0:
            predictions_loc = [predictions[idx] for idx in indices_loc]
            references_loc = [references[idx] for idx in indices_loc]
            metrics_loc, loc_data = evaluator.compute_loc_metrics(predictions_loc, references_loc)
            metrics.update(metrics_loc)
            pred_loc_data, ref_loc_data = loc_data
            test_loc_data = [{"pred_loc": pred_loc, "pred": pred, "ref_loc": ref_loc, "ref":ref} 
                                for (pred_loc, pred, ref_loc, ref) in zip(pred_loc_data, predictions_loc, ref_loc_data, references_loc)]
            output_loc_path = f"{root_output_dir}/{experiment_name}/test_loc.json"
            with open(output_loc_path, "w", encoding="utf-8") as f:
                json.dump(test_loc_data, f, indent=2, ensure_ascii=False)
            
            if "ft_instruct" in experiment_name and "ciderD" in metrics.keys() and "meteor" in metrics.keys():
                metrics["acc_ciderD_meteor_mAP"] = metrics["accuracy"] + (metrics["ciderD"] + metrics["meteor"])/2 + metrics["mAP"]
            elif "ft_instruct" in experiment_name:
                metrics["acc_mAP"] = metrics["accuracy"] + metrics["mAP"]
            else:
                metrics["ciderD_meteor_mAP"] = (metrics["ciderD"] + metrics["meteor"])/2 + metrics["mAP"]



        return metrics
    
    return compute_metrics
