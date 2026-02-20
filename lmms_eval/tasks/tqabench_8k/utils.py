import re
ABC = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
from lmms_eval.filters.extraction import ExtendedRegexFilter

def tqabench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    labels = doc.get("labels") or [ABC[i] for i in range(len(doc["choices"]))]
    opts = "\n".join(f"{labels[i]}. {c}" for i, c in enumerate(doc["choices"]))
    ctx = (doc.get("context_db") or "").strip()
    body = []
    if ctx:
        body.append("### [CONTEXT]\n" + ctx)
    body.append("### [QUESTION]\n" + doc["question"].strip())
    body.append("### [CHOICES]\n" + opts)
    return f"{pre_prompt}" + "\n\n".join(body) + f"{post_prompt}"

def tqabench_doc_to_target(doc):
    ans = str(doc.get("answer","")).strip().upper()[:1]
    return ans if ans in ABC else ""

def tqabench_process_results(doc, result):
    text = result[0]
    m = re.search(r"([A-Z])", text, re.I)
    pred = m.group(1).upper() if m else ""
    gt = tqabench_doc_to_target(doc)
    data_dict = {"pred": pred, "gt": gt, "correct": int(pred == gt)}
    return {"tqabench_score_overall": data_dict}

def tqabench_aggregation(items):
    if not items: return 0.0
    acc = sum(x["correct"] for x in items) / len(items)
    score = 100.0 * acc
    return score

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
