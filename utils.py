# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Dict, List, Optional, Tuple

from swift.template import split_str_parts_by

import re
from typing import Optional, Dict, List, Tuple



def calculate_loss_scale(query: str,
                         response: str,
                         response_loss_scale_map: Dict[str, list],
                         query_loss_scale_map: Optional[Dict[str, list]] = None) -> Tuple[List[str], List[float]]:
    """Calculate the loss scale by splitting the agent response.

    This algorithm comes from paper: https://arxiv.org/pdf/2309.00986.pdf

    Agent response format:

    ```text
        Thought: you should always think about what to do
        Action: the action to take, should be one of the above tools[fire_recognition,
            fire_alert, call_police, call_fireman]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question
    ```
    Returns:
        A tuple of agent response parts and their weights.
    """
    # query loss scale map
    if query_loss_scale_map is not None:
        for key in query_loss_scale_map.keys():
            if key in query:
                if isinstance(query_loss_scale_map[key], (float, int)):
                    query_loss_scale_map[key] = [query_loss_scale_map[key]]
                loss_scale_value = query_loss_scale_map[key][0]
                return [response], [float(loss_scale_value)]
    delimiters = [k for k, v in response_loss_scale_map.items() if len(v) == 2]
    if delimiters:
        agent_parts = split_str_parts_by(response, delimiters)
    else:
        regex_delimiters = [k for k, v in response_loss_scale_map.items() if len(v) == 1]
        agent_parts = split_str_parts_by(response, regex_delimiters, regex_mode=True)
    weights = []
    agent_content = []
    for c in agent_parts:
        if c['key'] in response_loss_scale_map:
            loss_scale = response_loss_scale_map[c['key']]
            assert len(loss_scale) in {1, 2}, f'loss_scale: {loss_scale}'
            if len(loss_scale) == 1:
                weights += loss_scale
                agent_content.append(c['content'])
            else:
                weights += loss_scale
                agent_content += [c['key'], c['content']]
        else:
            weights.append(1.)
            agent_content.append(c['content'])
    return agent_content, weights



import math

def med_calculate_loss_scale_v2(
        query: str,
        response: str,
        response_loss_scale_map: Dict[str, list],
        coef1: float,
        coef2: float,
        query_loss_scale_map: Optional[Dict[str, list]] = None) -> Tuple[List[str], List[float]]:

    def _normalize_text_for_match(s: Optional[str]) -> str:
        if not s:
            return ''
        s = re.sub(r'(</?img>|<image>|<video>|<audio>)', '', s, flags=re.IGNORECASE)
        s = re.sub(r'\s+', ' ', s).strip().lower()
        return s

    if query_loss_scale_map is not None and query is not None:
        q_lower = (query or '').lower()
        for k, v in query_loss_scale_map.items():
            if (k or '').lower() in q_lower:
                w = v[0] if isinstance(v, list) else v
                return [response], [float(w)]

    med_terms: List[str] = []
    data_list = None
    if isinstance(response_loss_scale_map, list):
        data_list = response_loss_scale_map
    elif isinstance(response_loss_scale_map, dict):
        if 'data' in response_loss_scale_map and isinstance(response_loss_scale_map['data'], list):
            data_list = response_loss_scale_map['data']
        elif 'med_terms' in response_loss_scale_map:
            med_terms = response_loss_scale_map.get('med_terms') or []

    if med_terms == [] and data_list:
        r_norm = _normalize_text_for_match(response or '')
        if not r_norm or len(r_norm) < 3:
            return [response], [float(coef1)]

        chosen = None
        for item in data_list:
            ir_norm = _normalize_text_for_match(item.get('response') or '')
            if ir_norm == r_norm and ir_norm:
                chosen = item
                break
        if chosen is None:
            best_item, best_score = None, -1
            for item in data_list:
                ir_norm = _normalize_text_for_match(item.get('response') or '')
                if not ir_norm:
                    continue
                score = -1
                if ir_norm in r_norm:
                    score = len(ir_norm)
                elif r_norm in ir_norm:
                    score = len(r_norm)
                if score > best_score:
                    best_score, best_item = score, item
            chosen = best_item if best_score > 0 else None

        if chosen:
            med_terms = chosen.get('med_terms') or []

    if not med_terms:
        return [response], [float(coef1)]

    resp_lower = (response or '').lower()
    matches: List[Tuple[int, int, str]] = []
    for term in med_terms:
        t = (term or '').strip()
        if not t:
            continue
        tl = t.lower()
        start = 0
        while True:
            idx = resp_lower.find(tl, start)
            if idx == -1:
                break
            matches.append((idx, idx + len(tl), t))
            start = idx + len(tl)

    if not matches:
        return [response], [float(coef1)]

    matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    selected: List[Tuple[int, int, str]] = []
    cur_end = -1
    for s, e, t in matches:
        if s >= cur_end:
            selected.append((s, e, t))
            cur_end = e

    agent_content: List[str] = []
    weights: List[float] = []
    pos = 0
    for s, e, _t in selected:
        if pos < s:
            agent_content.append(response[pos:s])
            weights.append(float(coef1))
        agent_content.append(response[s:e])
        weights.append(float(coef2))
        pos = e
    if pos < len(response):
        agent_content.append(response[pos:])
        weights.append(float(coef1))

    return agent_content, weights

