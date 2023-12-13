# -*- coding: utf-8 -*-
"""
Created on 2023-12-13 (Wed) 21:26:28

@author: I.Azuma
"""

def relabel(label_list=[3,10,21,5]):
    """_summary_

    Args:
        label_list (list, optional): _description_. Defaults to [3,10,21,5].

    Returns:
        list: [0,2,3,1]
    """
    unique_s = sorted(list(set(label_list)))
    relabel_dic = dict(zip(unique_s,[i for i in range(len(unique_s))]))
    relabel_l = [relabel_dic.get(k) for k in label_list]
    return relabel_l