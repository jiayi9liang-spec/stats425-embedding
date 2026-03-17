from relations.predefined_relations import PREDEFINED_RELATIONS

characters = [
"贾宝玉","林黛玉","薛宝钗","王熙凤","贾母","贾政","王夫人",
"贾探春","贾迎春","贾惜春","贾琏","贾赦","贾珍","秦钟"
]

locations = [
"大观园","怡红院","潇湘馆","蘅芜院","荣国府"
]

extra_relations = []

for c in characters:
    for l in locations:
        extra_relations.append((c,"出现于",l))

print(len(extra_relations))