import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

##进行属性超图嵌入模型构建
class AttributeEmbedding(nn.Module):
    def __init__(self, num_attributes, attr_embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_attributes, attr_embedding_dim)

    def forward(self, attributes):
        return self.embedding(attributes)

class EntityEmbedding(nn.Module):
    def __init__(self, attr_embedding_dim, entity_embedding_dim):
        super().__init__()
        self.femb = nn.Linear(attr_embedding_dim, entity_embedding_dim)
        self.fatt = nn.Linear(entity_embedding_dim, entity_embedding_dim)

    def forward(self, relation_embedding, attribute_embeddings):
        relation_emb = self.femb(relation_embedding).unsqueeze(1)
        attribute_embs = self.femb(attribute_embeddings)
        attention = F.softmax(torch.bmm(relation_emb, attribute_embs.transpose(1, 2)).squeeze(1), dim=1)
        return torch.sum(attention.unsqueeze(-1) * attribute_embs, dim=1)

class TripleEmbedding(nn.Module):
    def __init__(self, entity_embedding_dim, relation_embedding_dim):
        super().__init__()
        self.entity_embedding_dim = entity_embedding_dim
        self.relation_embedding_dim = relation_embedding_dim

    def forward(self, head_emb, relation_emb, tail_emb):
        head_emb_aggregated = torch.sum(head_emb, dim=1)  # Aggregating attribute-based representations
        tail_emb_aggregated = torch.sum(tail_emb, dim=1)
        return torch.cat([head_emb_aggregated, relation_emb, tail_emb_aggregated], dim=1)


class AttributeHypergraphModel(nn.Module):
    def __init__(self, num_attributes, num_relations, attr_embedding_dim, entity_embedding_dim, relation_embedding_dim):
        super().__init__()
        self.attribute_embedding = AttributeEmbedding(num_attributes, attr_embedding_dim)
        self.entity_embedding = EntityEmbedding(attr_embedding_dim, entity_embedding_dim)
        self.triple_embedding = TripleEmbedding(entity_embedding_dim, relation_embedding_dim)
        self.relation_embedding = nn.Embedding(num_relations, relation_embedding_dim)
        self.gat1 = GATConv(entity_embedding_dim + relation_embedding_dim, entity_embedding_dim)
        self.gat2 = GATConv(entity_embedding_dim, entity_embedding_dim)  # 添加第二个图注意力层

    def forward(self, h_attributes, r_idx, t_attributes, edge_index):
        h_attr_emb = self.attribute_embedding(h_attributes)
        t_attr_emb = self.attribute_embedding(t_attributes)
        r_emb = self.relation_embedding(r_idx)

        h_emb = self.entity_embedding(r_emb, h_attr_emb)
        t_emb = self.entity_embedding(r_emb, t_attr_emb)

        triple_emb = self.triple_embedding(h_emb, r_emb, t_emb)
        triple_emb = self.gat1(triple_emb, edge_index)
        return self.gat2(triple_emb, edge_index)  # 通过第二个图注意力层

###加载属性和关系映射
# 加载属性和关系映射
def load_mapping(file_path):
    """加载实体或关系到ID的映射"""
    mapping = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, val = line.strip().split('\t')
            mapping[key] = int(val)
    return mapping

entity_to_idx = load_mapping('entity2id.txt')
attribute_to_idx = load_mapping('attribute2id.txt')
relation_to_idx = load_mapping('relation2id.txt')

# 加载三元组
def load_triples(file_path, entity_to_idx, relation_to_idx):
    """加载三元组，并转换为索引形式"""
    triples = []
    with open(file_path, 'r') as file:
        for line in file:
            h, r, t = line.strip().split('\t')
            h_idx = entity_to_idx[h]
            r_idx = relation_to_idx[r]
            t_idx = entity_to_idx[t]
            triples.append((h_idx, r_idx, t_idx))
    return triples
triples = load_triples('triples.txt', entity_to_idx, relation_to_idx)


num_attributes = len(attribute_to_idx)
num_relations = len(relation_to_idx)
attr_embedding_dim = 50  # 可以根据需要调整
entity_embedding_dim = 50  # 可以根据需要调整
relation_embedding_dim = 50  # 可以根据需要调整

####构建邻接矩阵
def load_entity_attributes(file_path):
    """加载实体的属性集合"""
    entity_attributes = {}
    with open(file_path, 'r') as file:
        for line in file:
            entity, attr_id = line.strip().split('\t')
            if entity not in entity_attributes:
                entity_attributes[entity] = set()
            entity_attributes[entity].add(int(attr_id))
    return entity_attributes

def find_shared_attributes(entity_attributes):
    """找到共享属性的实体对"""
    shared_pairs = set()
    entities = list(entity_attributes.keys())
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            if entity_attributes[entities[i]].intersection(entity_attributes[entities[j]]):
                shared_pairs.add((entities[i], entities[j]))
    return shared_pairs

# 加载实体及其属性 每行为一个实体和一个属性ID
entity_attributes = load_entity_attributes('attribute_all.txt')

# 找到共享属性的实体对
shared_attribute_pairs = find_shared_attributes(entity_attributes)

# 构建邻接矩阵（或邻接列表）
edges = []
for entity1, entity2 in shared_attribute_pairs:
    edges.append([entity1, entity2])
    edges.append([entity2, entity1])

# edges 基于共享属性的超图邻接信息
# 准备 h_attributes, r_idx, t_attributes

h_attributes = [torch.tensor([attribute_to_idx[attr] for attr in entity_attributes[h]], dtype=torch.long) for h, r, t in triples]
r_idx = torch.tensor([r for h, r, t in triples], dtype=torch.long)
t_attributes = [torch.tensor([attribute_to_idx[attr] for attr in entity_attributes[t]], dtype=torch.long) for h, r, t in triples]

# 准备 edge_index
edge_index = torch.tensor([[entity_to_idx[e1], entity_to_idx[e2]] for e1, e2 in shared_attribute_pairs], dtype=torch.long).t().contiguous()

model = AttributeHypergraphModel(num_attributes, num_relations, attr_embedding_dim, entity_embedding_dim, relation_embedding_dim)
output = model(h_attributes, r_idx, t_attributes, edge_index)#头实体属性、关系的索引、尾实体的属性、邻接矩阵

import pandas as pd

# 加载三元组嵌入
triple_embeddings = pd.read_csv('Temb.csv').values #基于拓扑结构的嵌入
triple_embeddings = torch.tensor(triple_embeddings, dtype=torch.float)

# 加载负样本
def load_neg_triples(file_path, entity_to_idx, relation_to_idx):
    """加载负样本三元组，并转换为索引形式"""
    neg_triples = []
    with open(file_path, 'r') as file:
        for line in file:
            h, r, t = line.strip().split('\t')
            h_idx = entity_to_idx.get(h, None)
            r_idx = relation_to_idx.get(r, None)
            t_idx = entity_to_idx.get(t, None)
            if h_idx is not None and r_idx is not None and t_idx is not None:
                neg_triples.append((h_idx, r_idx, t_idx))
    return neg_triples

neg_triples = load_neg_triples('neg_triples.txt', entity_to_idx, relation_to_idx)


def margin_ranking_loss(pos_scores, neg_scores, margin, similarity):
    """
    计算损失函数。
    pos_scores: 正样本的得分。
    neg_scores: 负样本的得分。
    margin: 边际值。
    similarity: 三元组嵌入的相似度。
    """
    return torch.sum(torch.max(torch.zeros_like(pos_scores), margin + neg_scores - pos_scores) * similarity)

def translation_score(embeddings, h_idx, r_idx, t_idx):
    """
    根据翻译假设计算得分。
    embeddings: 实体和关系的嵌入。
    h_idx, r_idx, t_idx: 头实体、关系、尾实体的索引。
    """
    s = embeddings[h_idx]
    p = embeddings[r_idx]
    o = embeddings[t_idx]
    return torch.norm(s + p - o, dim=1)

def cosine_similarity(x1, x2):
    """
    计算余弦相似度。
    x1, x2: 两个向量。
    """
    return torch.sum(x1 * x2, dim=1) / (torch.norm(x1, dim=1) * torch.norm(x2, dim=1))



# 定义超参数
margin = 1.0  # 边际值
learning_rate = 0.001
num_epochs = 100  # 或根据需要调整

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    total_loss = 0

    for (h_idx, r_idx, t_idx), (neg_h_idx, neg_r_idx, neg_t_idx) in zip(triples, neg_triples):

        pos_scores = translation_score(model.embeddings, h_idx, r_idx, t_idx)
        neg_scores = translation_score(model.embeddings, neg_h_idx, neg_r_idx, neg_t_idx)
        similarity = cosine_similarity(model.embeddings[h_idx], triple_embeddings[h_idx])
        loss = margin_ranking_loss(pos_scores, neg_scores, margin, similarity)
        total_loss += loss

    total_loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {total_loss.item()}")


