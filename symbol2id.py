import random
from collections import defaultdict
from shutil import copyfile

in_path = './1-data_base/'
out_path = './2-data_with_neg/'


def read_symbol2id(file):
    symbol2id = {}
    with open(file, encoding='utf-8') as f:  # 确保使用正确的编码
        next(f)  # 跳过第一行的标题
        for line in f:
            symbol_id, symbol = line.strip().split('\t')
            symbol2id[symbol] = symbol_id
    return symbol2id


def convert_names_to_ids(file, entity2id, relation2id):
    converted_triples = []
    with open(file) as f:
        for line in f:
            h_name, r_name, t_name = line.strip().split('\t')
            h_id = entity2id.get(h_name)
            r_id = relation2id.get(r_name)
            t_id = entity2id.get(t_name)

            if h_id is None:
                print(f"未找到实体的ID: {h_name}")
            if r_id is None:
                print(f"未找到关系的ID: {r_name}")
            if t_id is None:
                print(f"未找到实体的ID: {t_name}")

            if h_id and r_id and t_id:
                converted_triples.append((h_id, r_id, t_id))
    return converted_triples

# ...其余代码保持不变...

def write_triples(file, triples):
    with open(file, 'w') as f:
        for h, r, t in triples:
            f.write(f'{h}\t{r}\t{t}\n')

# 读取映射
entity2id = read_symbol2id(in_path + 'entity_id.tsv')
relation2id = read_symbol2id(in_path + 'relation_id.tsv')

converted_triples = convert_names_to_ids(out_path + 'data_with_neg.tsv', entity2id, relation2id)
write_triples(out_path + 'data_with_neg_ids.tsv', converted_triples)
print(f"转换后的三元组数量: {len(converted_triples)}")
