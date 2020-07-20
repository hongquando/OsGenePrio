from model.TransE import *
from model.utils import *
from model.CustomTripletMarginLoss import CustomTripletMarginLoss
from argparse import Namespace
import torch
import sys
from torch.autograd import Variable
import torch.optim as optim
from numpy import linalg as LA
from math import log10,floor
import pickle
import numpy as np
import csv
import errno
import parameters
from sklearn.metrics.pairwise import pairwise_distances

def _get_learning_rate(o):
    lr = []
    for param_group in o.param_groups:
        lr += [param_group['lr']]
    return lr

class TrainTransE():
    net = None
    def __init__(self,args):
        self.args = args
        self.entity_total = get_total(self.args.entity_path)
        self.relation_total = get_total(self.args.relation_path)
        if os.path.exists(self.args.entity_path):
            self.processed_entity_2_id = load_data(self.args.entity_path, ignore_first=True)

        with open("./support/id_dict_2018", "rb") as f:
            self.id_dict_2018 = pickle.load(f)
            f.close()
        if os.path.exists(self.args.relation_path):
            self.relation_2_id = load_data(self.args.relation_path, ignore_first=True)

        if os.path.exists(self.args.train_path):
            self.triplets = load_data(self.args.train_path, is_triplet=True, ignore_first=True)

        self.triple_dict = load_triplet(self.args.triplets_path, is_triplet=True, ignore_first=True)

        if os.path.exists(self.args.valid_path):
            self.valids = load_data_valid(self.args.valid_path, is_triplet=True, ignore_first=True)

        # if os.path.exists(self.args.conv_kb_save_path) and os.path.exists(self.args.entity_path) and os.path.exists(
        #         self.args.relation_path):
        #     self.net = ConvKB(self.entity_total, self.relation_total, self.args.embedding_size)
        #     if torch.cuda.is_available():
        #         self.net = self.net.cuda()
        #         self.net.load_state_dict(torch.load(self.args.conv_kb_save_path))
        #     else:
        #         self.net.load_state_dict(torch.load(self.args.conv_kb_save_path, map_location=lambda storage, loc: storage))
        #    self.train()
        #self.net.eval()

    def cleanup(self):
        self.persist()

    def persist(self):
        print('Saving model...')
        with open(self.args.entity_path, 'w') as f:
            f.write('{}\n'.format(len(self.processed_entity_2_id)))
            for processed_entity, idx in sorted(list(self.processed_entity_2_id.items()),
                                                key=lambda kv: (kv[1])):
                f.write('{}\t{}\n'.format(processed_entity, idx))

        with open(self.args.relation_path, 'w') as f:
            f.write('{}\n'.format(len(self.relation_2_id)))
            for relation, idx in sorted(list(self.relation_2_id.items()), key=lambda kv: int(kv[1])):
                f.write('{}\t{}\n'.format(relation, idx))
        print('Saved model to file')

    def get_item_embedding(self, item_id):
        key = "_item:" + str(item_id)
        if key in self.processed_entity_2_id:
            idx = self.processed_entity_2_id[key]
            idx = torch.LongTensor([idx])
            if torch.cuda.is_available():
                idx = idx.cuda()
            idx = Variable(idx)
            embedding = self.net.ent_embeddings(idx).data[0].cpu().numpy()
            norm = LA.norm(embedding)
            if norm == 0:
                return embedding
            return embedding / LA.norm(embedding)
        return None

    def train_TransE(self,entity_total,relation_total,triplets,n_epochs=None):
        trans_e_loss = []
        net = TransE(entity_total,relation_total,self.args.embedding_size)
        if self.net is not None:
            embedding_entities = np.random.normal(0, 0.01, (entity_total, self.args.embedding_size))
            embedding_entities[:self.entity_total] = self.net.ent_embeddings.weight.data.cpu().numpy()
            net.ent_embeddings.weight.data.copy_(torch.from_numpy(embedding_entities))

            embedding_relations = np.random.normal(0, 0.01, (relation_total, self.args.embedding_size))
            embedding_relations[:self.relation_total] = self.net.rel_embeddings.weight.data.cpu().numpy()
            net.rel_embeddings.weight.data.copy_(torch.from_numpy(embedding_relations))

        device = torch.device("cuda:"+parameters.DEVICE if torch.cuda.is_available() else "cpu")
        net.to(device)
        print("Using CUDA: {}".format(next(net.parameters()).is_cuda))
        net.train()
        optimizer = optim.Adam(net.parameters(), lr=self.args.trans_e_learning_rate)
        #optimizer = optim.SGD(net.parameters(), lr=self.args.trans_e_learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=5, min_lr=1e-5,
                                                         verbose=True)
        criterion = CustomTripletMarginLoss(margin=self.args.trans_e_margin)

        # 2. Load triples #
        triple_total, triple_list, triple_dict, tails_per_head, heads_per_tail = load_triplet_2(triplets)

        # 4. Train #
        min_loss = None
        if n_epochs is None:
            n_epochs = self.args.trans_e_n_epochs

        train_loss = 0.0
        valid_loss = 0.0
        for epoch in range(1, n_epochs + 1):  # loop over the dataset multiple times
            # shuffle train set
            random.shuffle(triple_list)

            n_batches = triple_total // self.args.batch_size
            if (triple_total - n_batches * self.args.batch_size) != 0:
                n_batches += 1
            for batch_idx, i in enumerate(range(n_batches), 1):
                start = i * self.args.batch_size
                end = min([start + self.args.batch_size, triple_total])

                triple_batch = triple_list[start:end]
                pos_h_batch, pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch = get_batch_filter_all(
                    triple_batch, entity_total, triple_dict, tails_per_head, heads_per_tail)

                pos_h_batch, neg_h_batch = torch.LongTensor(pos_h_batch), torch.LongTensor(neg_h_batch)
                pos_t_batch, neg_t_batch = torch.LongTensor(pos_t_batch), torch.LongTensor(neg_t_batch)
                pos_r_batch, neg_r_batch = torch.LongTensor(pos_r_batch), torch.LongTensor(neg_r_batch)

                pos_h_batch, neg_h_batch = pos_h_batch.to(device), neg_h_batch.to(device)
                pos_t_batch, neg_t_batch = pos_t_batch.to(device), neg_t_batch.to(device)
                pos_r_batch, neg_r_batch = pos_r_batch.to(device), neg_r_batch.to(device)

                pos_h_batch, neg_h_batch = Variable(pos_h_batch), Variable(neg_h_batch)
                pos_t_batch, neg_t_batch = Variable(pos_t_batch), Variable(neg_t_batch)
                pos_r_batch, neg_r_batch = Variable(pos_r_batch), Variable(neg_r_batch)

                # zero the parameter gradients
                optimizer.zero_grad()
                pos, neg, pos_h_e, pos_t_e, neg_h_e, neg_t_e = net(pos_h_batch, pos_t_batch, pos_r_batch,
                                                                   neg_h_batch, neg_t_batch, neg_r_batch)

                ent_embeddings = net.ent_embeddings(torch.cat([pos_h_batch, pos_t_batch, neg_h_batch, neg_t_batch]))
                rel_embeddings = net.rel_embeddings(torch.cat([pos_r_batch, neg_r_batch]))

                loss_triplet = criterion(pos, neg)
                norm_loss = ent_embeddings.norm(2) + rel_embeddings.norm(2)
                norm_loss += pos_h_e.norm(2) + pos_t_e.norm(2) + neg_h_e.norm(2) + neg_t_e.norm(2)

                loss = loss_triplet + self.args.trans_e_weight_decay * norm_loss
                batch_loss = loss.item()
                loss.backward()
                optimizer.step()

                train_loss += batch_loss

                if batch_idx % self.args.log_interval == 0:
                    offset = int(floor(log10(n_batches)) - floor(log10(batch_idx)))
                    print('\r\033[K\rTrain Epoch: {} [{}{} / {} ({:.0f}%)]   Learning Rate: {}   Loss: {:.6f}'
                          .format(epoch, batch_idx, ' ' * offset, n_batches, 100. * batch_idx / n_batches,_get_learning_rate(optimizer)[0], batch_loss)),
                    sys.stdout.flush()

            train_loss /= n_batches

            #Valid loss
            pos_h_batch, pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch = get_batch_filter_random_v2(self.valids,
                self.args.batch_size, entity_total, triple_dict, tails_per_head, heads_per_tail)

            pos_h_batch, neg_h_batch = torch.LongTensor(pos_h_batch), torch.LongTensor(neg_h_batch)
            pos_t_batch, neg_t_batch = torch.LongTensor(pos_t_batch), torch.LongTensor(neg_t_batch)
            pos_r_batch, neg_r_batch = torch.LongTensor(pos_r_batch), torch.LongTensor(neg_r_batch)

            pos_h_batch, neg_h_batch = pos_h_batch.to(device), neg_h_batch.to(device)
            pos_t_batch, neg_t_batch = pos_t_batch.to(device), neg_t_batch.to(device)
            pos_r_batch, neg_r_batch = pos_r_batch.to(device), neg_r_batch.to(device)

            pos_h_batch, neg_h_batch = Variable(pos_h_batch), Variable(neg_h_batch)
            pos_t_batch, neg_t_batch = Variable(pos_t_batch), Variable(neg_t_batch)
            pos_r_batch, neg_r_batch = Variable(pos_r_batch), Variable(neg_r_batch)

            pos, neg, pos_h_e, pos_t_e, neg_h_e, neg_t_e = net(pos_h_batch, pos_t_batch, pos_r_batch,
                                                               neg_h_batch, neg_t_batch, neg_r_batch)

            ent_embeddings = net.ent_embeddings(torch.cat([pos_h_batch, pos_t_batch, neg_h_batch, neg_t_batch]))
            rel_embeddings = net.rel_embeddings(torch.cat([pos_r_batch, neg_r_batch]))

            loss_triplet = criterion(pos, neg)
            norm_loss = ent_embeddings.norm(2) + rel_embeddings.norm(2)
            norm_loss += pos_h_e.norm(2) + pos_t_e.norm(2) + neg_h_e.norm(2) + neg_t_e.norm(2)

            loss = loss_triplet + self.args.trans_e_weight_decay * norm_loss
            valid_loss = loss.item()
            trans_e_loss.append(str(train_loss)+" - "+str(valid_loss))
            # trans_e_loss.append(train_loss)

            # print statistics
            if epoch % self.args.display_step == 0 or epoch == 1:
                print('\r\033[K\r[{:3d}] train_loss: {:.5f} - valid_loss: {:.5f} - learning rate: {}'
                      .format(epoch, train_loss, valid_loss, _get_learning_rate(optimizer)[0]))

            # if epoch % self.args.display_step == 0 or epoch == 1:
            #     print('\r\033[K\r[{:3d}] train_loss: {:.5f} - learning rate: {}'
            #           .format(epoch, train_loss, _get_learning_rate(optimizer)[0]))


            if min_loss is None or train_loss < min_loss:
                min_loss = train_loss
                with open(self.args.trans_e_save_path, 'wb') as f:
                    torch.save(net.state_dict(), f)
            scheduler.step(train_loss, epoch)

        print('\nFinished Training TransE\n')
        with open(self.args.trans_e_loss_path, 'w') as f:
            for item in trans_e_loss:
                f.write("%s\n" % item)
        f.close()

        if torch.cuda.is_available():
            net.load_state_dict(torch.load(self.args.trans_e_save_path))
            net.to(device)
        else:
            net.load_state_dict(torch.load(self.args.trans_e_save_path, map_location=lambda storage, loc: storage))
        return net

    def train(self,trans_e_n_epochs=None, conv_kb_n_epochs=None):
        device = torch.device("cuda:" + parameters.DEVICE if torch.cuda.is_available() else "cpu")
        if os.path.exists(self.args.trans_e_save_path) and os.path.exists(self.args.entity_path) and os.path.exists(self.args.relation_path):
            self.net = TransE(self.entity_total, self.relation_total, self.args.embedding_size)
            if torch.cuda.is_available():
                self.net.load_state_dict(torch.load(self.args.trans_e_save_path))
                self.net.to(device)
                ent_embeddings = self.net.ent_embeddings.weight.data.cpu().numpy()
                rel_embeddings = self.net.rel_embeddings.weight.data.cpu().numpy()
            else:
                self.net.load_state_dict(torch.load(self.args.trans_e_save_path, map_location=lambda storage, loc: storage))
        else:
            self.net = self.train_TransE(self.entity_total, self.relation_total, self.triplets, n_epochs=self.args.trans_e_n_epochs)

        # print('\nEvaluate TransE\n')
        # candidates = []
        # for att in self.processed_entity_2_id.keys():
        #     if att not in self.id_dict_2018.keys():
        #         candidates.append(self.processed_entity_2_id[att])
        #
        # mix_ids = np.random.permutation(len(self.valids))
        # n_batches = int(np.ceil(len(self.valids) / float(args.batch_size)))
        # hits10 = 0.0
        # mr = 0.0
        # mrr = 0.0
        # for ib in range(n_batches):
        #     rand_index = mix_ids[args.batch_size * ib:min(args.batch_size * (ib + 1), len(self.valids))]
        #     new_candidates = set(random.choices(candidates, k=128))
        #     valid_list = []
        #     for index in rand_index:
        #         triple = self.valids[index]
        #         new_candidates.add(triple.t)
        #         valid_list.append(triple)
        #     for triple in valid_list:
        #         h_batch = []
        #         t_batch = []
        #         r_batch = []
        #         h_batch.append(triple.h)
        #         t_batch.append(triple.t)
        #         r_batch.append(triple.r)
        #         for att in new_candidates:
        #             if (triple.h, att, triple.r) in self.triple_dict:
        #                 continue
        #             h_batch.append(triple.h)
        #             t_batch.append(att)
        #             r_batch.append(triple.r)
        #         h_batch, t_batch, r_batch = torch.LongTensor(h_batch), torch.LongTensor(t_batch), torch.LongTensor(
        #             r_batch)
        #         h_batch, h_batch, r_batch = Variable(h_batch), Variable(t_batch), Variable(r_batch)
        #         c_t = ent_embeddings[h_batch[0]] + rel_embeddings[r_batch[0]]
        #         dist = pairwise_distances([c_t], ent_embeddings[t_batch], metric='euclidean')
        #         rankArrayTail = np.argsort(dist, axis=1)
        #         _filter = rankArrayTail[0][0] + 1
        #         mr += _filter
        #         mrr += 1.0 / _filter
        #         if _filter <= 10:
        #             hits10 += 1
        #     print("Evalute epoch {}/{}: Hit@10: {} - MR: {} - MRR: {} ".format(ib + 1, n_batches, hits10, mr, mrr))
        # mrr = mrr / len(self.valids)
        # hits10 = hits10 / len(self.valids)
        # conv_kb_eval = [hits10, mr, mrr]
        # print('Hit@10: %.6f' % hits10)
        # print('Meanrank: %.6f' % mr)
        # print('MRR: %.6f' % mrr)
        # with open(self.args.conv_kb_eval_path, 'w') as f:
        #     for e in conv_kb_eval:
        #         f.write("%s\n" % e)
        # f.close()

if __name__ == '__main__':
    args = Namespace(
        entity_path='./data/GENE/entity2id.txt',
        relation_path='./data/GENE/relation2id.txt',
        triplets_path='./data/GENE/triplet2id.txt',
        train_path='./data/GENE/train2id.txt',
        valid_path='./data/GENE/valid2id.txt',

        # trans_e_loss_train_path='/loss_train_transe.txt',
        # trans_e_loss_valid_path='/loss_valid_transe.txt',
        # conv_kb_loss_train_path='/loss_train_convkb.txt',
        # conv_kb_loss_valid_path='/loss_valid_convkb.txt',

        trans_e_loss_path='loss_transe.txt',
        conv_kb_loss_path='loss_convkb.txt',
        conv_kb_eval_path='evaluation.txt',

        embedding_size=100,
        batch_size=128,

        seed=0,
        log_interval=15,
        display_step=5,

        trans_e_margin=1,
        trans_e_weight_decay=0.001,
        trans_e_learning_rate=5e-4,
        trans_e_n_epochs=125,
        trans_e_save_path='TransE.pkl',
    )
    embedding_size = [100,150]
    trans_e_learning_rate = [1e-4,1e-3]
    trans_e_margin =[1,3,5]

    count_param = 1
    # result = [['params','embedding_size','trans_e_learning_rate','trans_e_margin','conv_kb_learning_rate','num_filters',
    #            'trans_e_train_loss','trans_e_valid_loss','conv_kb_train_loss','conv_kb_valid_loss']]
    result = [['params','embedding_size','trans_e_learning_rate','trans_e_margin','trans_e_train_loss','trans_e_valid_loss']]
    min_total_loss = 10
    for embedding in embedding_size:
        for learning_rate_1 in trans_e_learning_rate:
            for margin in trans_e_margin:
                args.embedding_size = embedding
                args.trans_e_learning_rate = learning_rate_1
                args.trans_trans_e_margin = margin
                folder = "./data/param"+str(count_param);
                try:
                    os.makedirs(folder)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
                args.trans_e_loss_path = os.path.join(folder,"loss_transe.txt")
                args.trans_e_save_path = os.path.join(folder,"TransE.pkl")
                TrainTransE(args).train()
                #append to result file
                trans_e_min_train_loss = 10
                trans_e_min_valid_loss = 10
                with open(args.trans_e_loss_path, 'r') as f:
                    for item in f.readlines():
                        tmp = item.split("-")
                        train_loss = float(tmp[0])
                        valid_loss = float(tmp[1][0:len(tmp[1]) - 1])
                        if valid_loss < trans_e_min_valid_loss:
                            trans_e_min_valid_loss = valid_loss
                            trans_e_min_train_loss = train_loss
                f.close()
                # trans_e_min_train_loss = 10
                # with open(args.trans_e_loss_path, 'r') as f:
                #     for item in f.readlines():
                #         train_loss = float(item[0:len(item)-1])
                #         if train_loss < trans_e_min_train_loss:
                #             trans_e_min_train_loss = train_loss
                # f.close()
                result.append([count_param,embedding,learning_rate_1, margin, trans_e_min_train_loss, trans_e_min_valid_loss])
                if trans_e_min_valid_loss < min_total_loss:
                    min_total_loss = trans_e_min_valid_loss
                    best_param = ["best_param " + str(count_param), embedding, learning_rate_1, margin, trans_e_min_train_loss, trans_e_min_valid_loss]
                count_param += 1
    result.append(best_param)
    csv.register_dialect('myDialect',
                         quoting=csv.QUOTE_ALL,
                         skipinitialspace=True)
    with open('./data/result_transe.csv', 'w') as f:
        writer = csv.writer(f, dialect='myDialect')
        for row in result:
            writer.writerow(row)
    f.close()

    # if torch.cuda.is_available():
    #     net = torch.load(args.conv_kb_save_path)
    # else:
    #     net = torch.load(args.conv_kb_save_path, map_location=lambda storage, loc: storage)
    # net = list(net.items())
    # # 1: entity
    # # 2: relation
    # data_train = net[0][1].cpu().numpy()
    # nbrs = NearestNeighbors(n_neighbors=15, algorithm='ball_tree').fit(data_train)
    # distances, indices = nbrs.kneighbors(data_train)
    # with open("./data/GENE/kNN.pkl", "wb") as f:
    #     pickle.dump(nbrs,f)
    #     f.close()
    # with open("./data/GENE/indices.pkl", "wb") as f:
    #     pickle.dump(indices, f)
    #     f.close()
    # with open("./data/GENE/distance.pkl", "wb") as f:
    #     pickle.dump(distances, f)
    #     f.close()


