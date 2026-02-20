import re

import pandas as pd
from rouge_score import rouge_scorer
import numpy as np
from lmms_eval.filters.extraction import ExtendedRegexFilter
from lmms_eval.filters.transformation import MapFilter


def wcep10_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    document = doc["document"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    return f"{pre_prompt}\n{document}{post_prompt}"


def wcep10_doc_to_target(doc):
    return doc["summary"]


def wcep10_process_results(doc, result):
    pred = result[0]
    answer = doc["summary"]

    data_dict = {
        "pred": pred,
        "answer": answer,
    }

    return {"wcep10_score_overall": data_dict}



def wcep10_aggregation(results):
    """
    results: [{ "pred": <str 또는 List[str]>, "answer": <str 또는 List[str]> }, ...]
             (일반적인 케이스: pred/answer가 '문자열 1개'인 샘플들의 리스트)
    반환: per-sample F1 리스트 + 평균
    """
    scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
    f1_lists = {"rouge1": [], "rouge2": [], "rougeL": []}

    for block in results:
        # 1) pred: 문자열 1개로 강제
        pred = block["pred"]
        if isinstance(pred, (list, tuple)):  # 혹시 리스트로 들어오는 경우 방어
            pred = " ".join(map(str, pred))
        pred = (pred or "").strip()

        # 2) ref: 단일/다중 모두 처리 (다중이면 max-over-refs)
        ref = block["answer"]
        if isinstance(ref, (list, tuple)):
            refs_list = [(r or "").strip() for r in ref if r]
        else:
            refs_list = [ (ref or "").strip() ]

        if not pred or not refs_list or not any(refs_list):
            for m in f1_lists: f1_lists[m].append(0.0)
            continue

        # 다중 레퍼런스면 각 metric에 대해 최고 F1 선택
        scores_per_ref = [scorer.score(pred, r) for r in refs_list if r]
        for m in f1_lists:
            best_f1 = max(s[m].fmeasure for s in scores_per_ref) if scores_per_ref else 0.0
            f1_lists[m].append(best_f1)

    out = {m: f1_lists[m] for m in f1_lists}
    mean_out = {}
    mean_out.update({f"{m}_mean": float(np.mean(f1_lists[m])) if f1_lists[m] else 0.0 for m in f1_lists})
    return [mean_out['rouge1_mean'], mean_out['rouge2_mean'], mean_out['rougeL_mean']]


class MultiChoiceRegexFilter(ExtendedRegexFilter):
    def __init__(self, *args, **kwargs):
        """
        regex_pattern: The basic regex pattern to use. If fails to match, we will use the customized match procedure
                        - step 1 : We parse the choices between ([A-Z])s then try to find these choices in the response.
                        - step 2 : We parse the choice with regex :[\s]*([A-?]), where ? varies by number of choices.
        group_select: Selects the (group_select)th match from the findall result.
        ignore_case: Ignores the case during step 1 matching
        ignore_punctuation: Remove the punctuation during step 1 matching
        regexes_to_ignore: Remove these regexes during step 1 matching
        """
        super().__init__(*args, **kwargs)

    def apply(self, resps, docs):
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)

        filtered_resps = []

        for r, doc in zip(resps, docs):
            # Regex to directly extract the option letter from the model response
            option_letter_regex = re.compile(r"^\s*([A-Z])\.")

            # Process each response
            filtered = []
            for resp in r:
                # Try to match the option letter at the start of the response
                match = option_letter_regex.match(resp)
                if match:
                    # If a match is found, append the matched letter
                    filtered.append(match.group(1))
                else:
                    # If no match, return the original response
                    filtered.append(resp)

            # Assuming we need the first response that matches or the original response
            filtered_resps.append(filtered[0])

        return filtered_resps
