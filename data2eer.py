in_path = './1-data_base/'
out_path = './2-data_with_neg/'


with open(out_path + 'data_with_neg_ids.tsv', 'r') as infile, open(out_path + 'rearranged_data.tsv', 'w') as outfile:
    for line in infile:
        # 分割每行数据
        parts = line.strip().split('\t')
        if len(parts) == 3:
            head_entity, relation, tail_entity = parts
            # 将数据重新排列，并写入新文件
            outfile.write(f"{head_entity}\t{tail_entity}\t{relation}\n")

print(f"Data rearranged and saved to {out_path + 'rearranged_data.tsv'}")
