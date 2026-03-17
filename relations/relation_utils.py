from collections import defaultdict
from relations.predefined_relations import PREDEFINED_RELATIONS


def build_relation_map():

    relation_map = defaultdict(set)

    for subj, rel, obj in PREDEFINED_RELATIONS:
        relation_map[subj].add(obj)
        relation_map[obj].add(subj)

    return relation_map