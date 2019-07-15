'''
Created on Oct 10, 2018
Tensorflow Implementation of the baseline of "Matrix Factorization with BPR Loss" in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import tensorflow as tf
from utility.helper import *
import numpy as np
from scipy.sparse import csr_matrix
from utility.batch_test import *
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class BPRMF(object):
    def __init__(self, data_config):
        self.model_type = 'bprmf'

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.lr = args.lr
        # self.lr_decay = args.lr_decay

        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)

        self.regs = eval(args.regs)
        self.decay = self.regs[0]

        self.verbose = args.verbose

        # placeholder definition
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))

        # self.global_step = tf.Variable(0, trainable=False)

        self.weights = self._init_weights()

        # Original embedding.
        u_e = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users)
        pos_i_e = tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_items)
        neg_i_e = tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_items)

        # All ratings for all users.
        self.batch_ratings = tf.matmul(u_e, pos_i_e, transpose_a=False, transpose_b=True)

        self.mf_loss, self.reg_loss = self.create_bpr_loss(u_e, pos_i_e, neg_i_e)
        self.loss = self.mf_loss + self.reg_loss

        # self.dy_lr = tf.train.exponential_decay(self.lr, self.global_step, 10000, self.lr_decay, staircase=True)
        self.opt = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.loss)

        # self.updates = self.opt.minimize(self.loss, var_list=self.weights)

        self._statistics_params()

    def _init_weights(self):
        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()


        all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding')
        all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='item_embedding')

        return all_weights

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        # regularizer = regularizer/self.batch_size

        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))

        mf_loss = tf.negative(tf.reduce_mean(maxi))
        reg_loss = self.decay * regularizer
        return mf_loss, reg_loss


    def _statistics_params(self):
        # number of params
        total_parameters = 0
        for variable in self.weights.values():
            shape = variable.get_shape()  # shape is an array of tf.Dimension
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        if self.verbose > 0:
            print("#params: %d" % total_parameters)

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    t0 = time()

    model = BPRMF(data_config=config)

    saver = tf.train.Saver()

    # *********************************************************
    # save the model parameters.
    if args.save_flag == 1:
        weights_save_path = '%sweights/%s/%s/l%s_r%s' % (args.proj_path, args.dataset, model.model_type, str(args.lr),
                                                         '-'.join([str(r) for r in eval(args.regs)]))
        ensureDir(weights_save_path)
        save_saver = tf.train.Saver(max_to_keep=1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # *********************************************************
    # reload the pretrained model parameters.
    if args.pretrain == 1:
        pretrain_path = '%sweights/%s/%s/l%s_r%s' % (args.proj_path, args.dataset, model.model_type, str(args.lr),
                                                         '-'.join([str(r) for r in eval(args.regs)]))
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(pretrain_path + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('load the pretrained model parameters from: ', pretrain_path)

            # *********************************************************
            # get the performance from pretrained model.
            users_to_test = list(data_generator.test_set.keys())
            ret = test(sess, model, users_to_test, drop_flag=False)
            cur_best_pre_0 = ret[5]

            pretrain_ret = 'pretrained model recall=[%.5f, %.5f],' \
                           'map=[%.5f, %.5f], ndcg=[%.5f, %.5f], auc=[%.5f]' % (ret[0], ret[4], ret[5], ret[9], ret[10],
                                                                            ret[14], ret[15])
            print(pretrain_ret)
        else:
            sess.run(tf.global_variables_initializer())
            cur_best_pre_0 = 0.
            print('without pretraining.')
    else:
        sess.run(tf.global_variables_initializer())
        cur_best_pre_0 = 0.
        print('without pretraining.')


    loss_loger, pre_loger, rec_loger, ndcg_loger, auc_loger = [], [], [], [], []

    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss, reg_loss = 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1


        for idx in range(n_batch):
            # btime= time()
            users, pos_items, neg_items = data_generator.sample()
            _, batch_loss, batch_mf_loss, batch_reg_loss = sess.run([model.opt, model.loss, model.mf_loss, model.reg_loss],
                               feed_dict={model.users: users, model.pos_items: pos_items,
                                          model.neg_items: neg_items})
            loss += batch_loss
            mf_loss += batch_mf_loss
            reg_loss += batch_reg_loss
            # print(time() - btime)

        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
        if (epoch + 1) % 10 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (epoch, time()-t1, loss, mf_loss, reg_loss)
                print(perf_str)
            continue

        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        ret = test(sess, model, users_to_test, drop_flag=False)

        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret[0:5])
        pre_loger.append(ret[5:10])
        ndcg_loger.append(ret[10:15])
        auc_loger.append(ret[15])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f], ' \
                       'map=[%.5f, %.5f], ndcg=[%.5f, %.5f], auc=[%.5f]' % (epoch, t2 - t1, t3 - t2,
                                                                loss, mf_loss, reg_loss,
                                                                ret[0], ret[4], ret[5], ret[9], ret[10], ret[14], ret[15])
            print(perf_str)

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if args.save_flag == 1:
            if ret[5] > cur_best_pre_0:
                cur_best_pre_0 = ret[5]
                save_saver.save(sess, weights_save_path + '/weights', global_step=epoch)
                print('save the weights in path: ', weights_save_path)


    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    auc = np.array(auc_loger)

    best_rec_0 = max(pres[:, 0])
    idx = list(pres[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%.5f, %.5f, %.5f, %.5f, %.5f], map=[%.5f, %.5f, %.5f, %.5f, %.5f]," \
                 "ndcg=[%.5f, %.5f, %.5f, %.5f, %.5f], auc=[%.5f]" % (idx, time() - t0,
                                                          recs[idx, 0], recs[idx, 1], recs[idx, 2], recs[idx, 3],
                                                          recs[idx, 4],
                                                          pres[idx, 0], pres[idx, 1], pres[idx, 2], pres[idx, 3],
                                                          pres[idx, 4],
                                                          ndcgs[idx, 0], ndcgs[idx, 1], ndcgs[idx, 2],
                                                          ndcgs[idx, 3],
                                                          ndcgs[idx, 4], auc[idx])
    print(final_perf)

    save_path = '%soutput_final/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
    ensureDir(save_path)
    f = open(save_path, 'a')

    f.write('embed_size=%d, lr=%.4f, regs=%s, loss_type=%s, \n\t%s\n' % (args.embed_size, args.lr, args.regs,
                                                                         args.loss_type, final_perf))
    f.close()
