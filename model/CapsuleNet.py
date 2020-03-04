import torch
import torch.nn as nn

from model.CapsuleLayer import CapsuleLayer

epsilon = 1e-9

class CapsuleNetwork(nn.Module):
    def __init__(self,entity_total, relation_total, embedding_size,
                 vec_len_secondCaps = 10, num_units = 1, num_iterations = 3, num_filters = 50):
        super(CapsuleNetwork, self).__init__()

        self.embedding_size = embedding_size
        self.num_filters = num_filters
        self.entity_total = entity_total
        self.relation_total = relation_total
        self.num_iterations = num_iterations
        self.vec_len_secondCaps = vec_len_secondCaps

        self.ent_embeddings = nn.Embedding(self.entity_total, self.embedding_size)
        self.rel_embeddings = nn.Embedding(self.relation_total, self.embedding_size)

        self.primary = CapsuleLayer(vec_len_secondCaps = 0, # Don't use
                                    num_caps_i = 0, # Don't use
                                    num_units = num_units, # =1 using only 1 Conv2d
                                    unit_size = 0, # Don't use
                                    num_filters = self.num_filters,
                                    use_routing=False)

        self.digits = CapsuleLayer(vec_len_secondCaps = vec_len_secondCaps, # = 10
                                   num_caps_i = self.embedding_size, # embedding size
                                   num_units = num_units, # = 1  because of use only 1 Conv2d
                                   unit_size = self.num_filters, # num_filters
                                   num_iterations = self.num_iterations,
                                   num_filters = self.num_filters, # Don't use
                                   use_routing = True)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    # def forward(self, x):
    #     return self.digits(self.primary(self.conv1(x)))

    def set_pretrained_weights(self, ent_embeddings, rel_embeddings):
        self.ent_embeddings.weight = nn.Parameter(torch.from_numpy(ent_embeddings).float())
        self.rel_embeddings.weight = nn.Parameter(torch.from_numpy(rel_embeddings).float())

    def forward(self, h, t, r):
        h_e = self.ent_embeddings(h).view(-1, 1, 1, self.embedding_size)
        t_e = self.ent_embeddings(t).view(-1, 1, 1, self.embedding_size)
        r_e = self.rel_embeddings(r).view(-1, 1, 1, self.embedding_size)

        x = torch.cat([h_e, r_e, t_e], 2)
        x = self.primary(x)
        x = self.digits(x)
        outputs = torch.sum(x**2, dim=1, keepdim=True)
        outputs = torch.sqrt(outputs + epsilon)
        return outputs , h_e, t_e, r_e






