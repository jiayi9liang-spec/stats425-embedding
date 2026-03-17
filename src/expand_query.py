from relations.relation_utils import build_relation_map


relation_map = build_relation_map()


def expand_query(question):

    expanded = question
    added_entities = []

    for entity in relation_map:

        if entity in question:

            related = relation_map[entity]

            for r in related:
                if r not in added_entities:
                    added_entities.append(r)

    expanded = expanded + " " + " ".join(added_entities)

    return expanded