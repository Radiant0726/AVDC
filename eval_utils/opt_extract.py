import os
import re

import torch
import numpy as np
import json
import evaluate
from transformers import AutoTokenizer, AutoModel

from typing import List, Dict, Tuple
from collections import defaultdict, Counter
import torch.nn.functional as F

def extract_mcq_answer(response: str) -> str:
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        response = match.group(1).strip()

    match = re.search(r'answer(?: is|:)?\s*([A-E])(?![a-zA-Z])', response, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()

    match = re.search(r'(?<![a-zA-Z])[A-E](?![a-zA-Z])', response)
    if match:
        return match.group()
    return "unknown"


# text = "assistant\n<think>\nThe user wants to identify the overall soundscape of the audio audio clip. selecting the best option from a given multiple-choice options. This task requires a detailed analysis of the audio content, a the dominant and secondary sounds elements. match the most description.-by-step.\nThe audio clip primarily features the series of goose honhonack\" sounds, which of ge or These quacks are inters loud and clear, suggesting a presence of multiple or more ducks. the proximity. the recording device. The are't any other human sounds, such, other other animal of animal.\n which for quacks.1. **Dominant Sound:**:** The most and most sound in clearly the as \" quacks. This are loud and prominent, the audio.\n2. **Subence of Other Sounds Sounds:** There are no other prominent that with the duck.acks. No absence of other prominent soundsizations, human speech, or mechanical noises is a focus is3. **Alternative with Options Options:**\n * (a) Primarily human speech with faint bird sounds in the background: This. as human speech is * (b) A mixture of industrial noises and animal vocalizations: Incorrect, no industrial sounds.\n * (c) A dominant presence of goose honking with subtle chirping of other birds: Incorrect, The dominant are clearly duck ququacks,\" not goose honhonks\" * (d) A quiet ambiance with only occasional, indistinct bird calls: Incorrect. The sounds are not and not identifiable.\nacks.\n4. **uling out Options The (, B, and C are incorrect ruled out as on the soundcultation of the audio.5 audio clip is soundscape is dominated by the frequent qu frequent ququack\" sounds of ducks. There other significant sounds are present. Therefore, the of the provided options-choice options is correct correct. audio audio. The is that is a perfect option.\n.\n The, if we choose the best suitablead option, option would be (a\".\nThethink>\n<answer>\nThe audio clip is features of the series of distinct, frequent \"quack\" sounds, characteristic of ducks. There are no absence of other prominent sounds, such the duck quizations the dominant element. the audiocape. Therefore of the provided multiple-choice perfectly A the speech, industrial noises, goose otherese hon are describe the audio. The closest of other prominent makes The, the of the provided options is the perfect description. but if to choose a choice, the answer-bad answer is be D. THE ANSWER IS D.\n</answer>\n",
# text = str(text)
# opt = extract_mcq_answer(text)