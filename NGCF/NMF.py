'''
Created on Oct 10, 2018
Tensorflow Implementation of the baseline of "Neural Collaborative Filtering, He et al. SIGIR 2017" in:
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
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class NMF(object):
    def __init__(self, data_config, pretrain_data):
        self.model_type = 'nmf'
        self.pretrain_data = pretrain_data

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.lr = args.lr
        # self.lr_decay = args.lr_decay

        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)

        self.model_type += '_l%d' % self.n_layers

        self.regs = eval(args.regs)
        self.decay = self.regs[-1]

        self.verbose = args.verbose

        # placeholder definition
        self.users = tf.placeholder(tf.int32, shape=(None))
        self.pos_items = tf.placeholder(tf.int32, shape=(None))
        self.neg_items = tf.placeholder(tf.int32, shape=(None))

        self.dropout_keep = tf.placeholder(tf.float32, shape=[None])
        self.train_phase = tf.placeholder(tf.bool)

        # self.global_step = tf.Variable(0, trainable=False)
        self.weights = self._init_weights()


        # Original embedding.
        u_e = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users)
        pos_i_e = tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_items)
        neg_i_e = tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_items)

        # All ratings for all users.
        self.batch_ratings = self._create_batch_ratings(u_e, pos_i_e)

        self.mf_loss, self.emb_loss, self.reg_loss = self.create_bpr_loss(u_e, pos_i_e, neg_i_e)
        self.loss = self.mf_loss + self.emb_loss + self.reg_loss

        # self.dy_lr = tf.train.exponential_decay(self.lr, self.global_step, 10000, self.lr_decay, staircase=True)
        # self.opt = tf.train.RMSPropOptimizer(learning_rate=self.dy_lr).minimize(self.loss, global_step=self.global_step)
        self.opt = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.loss)
        # self.updates = self.opt.minimize(self.loss, var_list=self.weights)

        self._statistics_params()

    def _init_weights(self):
        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()


        if self.pretrain_data is None:
            all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding')
            all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='item_embedding')
            print('using xavier initialization')
        else:
            all_weights['user_embedding'] = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True,
                                                        name='user_embedding', dtype=tf.float32)
            all_weights['item_embedding'] = tf.Variable(initial_value=self.pretrain_data['item_embed'], trainable=True,
                                                        name='item_embedding', dtype=tf.float32)
            print('using pretrained initialization')


        if self.model_type == 'mlp':
            self.weight_size_list = [2 * self.emb_dim] + self.weight_size
        elif self.model_type == 'jrl':
            self.weight_size_list = [self.emb_dim] + self.weight_size
        else:
            self.weight_size_list = [3 * self.emb_dim] + self.weight_size

        for i in range(self.n_layers):
            all_weights['W_%d' %i] = tf.Variable(
                initializer([self.weight_size_list[i], self.weight_size_list[i+1]]), name='W_%d' %i)
            all_weights['b_%d' %i] = tf.Variable(
                initializer([1, self.weight_size_list[i+1]]), name='b_%d' %i)

        all_weights['h'] = tf.Variable(initializer([self.weight_size_list[-1], 1]), name='h')

        return all_weights

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = self._create_inference(users, pos_items)
        neg_scores = self._create_inference(users, neg_items)

        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size

        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
        mf_loss = tf.negative(tf.reduce_mean(maxi))

        emb_loss = self.regs[-1] * regularizer

        reg_loss = self.regs[-2] * tf.nn.l2_loss(self.weights['h'])

        return mf_loss, emb_loss, reg_loss

    def _create_inference(self, u_e, i_e):
        z = []

        if self.model_type == 'mlp':
            z.append(tf.concat([u_e, i_e], 1))
        elif self.model_type == 'jrl':
            z.append(u_e * i_e)
        else:
            z.append(tf.concat([u_e, i_e, u_e * i_e], 1))

        # z[0] = self.batch_norm_layer(z[0], train_phase=self.train_phase, scope_bn='bn_mlp')

        for i in range(self.n_layers):
            # (batch, W[i]) * (W[i], W[i+1]) + (1, W[i+1]) => (batch, W[i+1])
            # temp = self.batch_norm_layer(z[i], train_phase=self.train_phase, scope_bn='mlp_%d' % i)

            temp = tf.nn.relu(tf.matmul(z[i], self.weights['W_%d' % i]) + self.weights['b_%d' % i])
            temp = tf.nn.dropout(temp, self.dropout_keep[i])
            z.append(temp)

        agg_out = tf.matmul(z[-1], self.weights['h'])
        return agg_out

    def _create_all_ratings(self, u_e):
        z = []

        if self.model_type == 'jrl':
            u_1 = tf.expand_dims(u_e, axis=1)
            i_1 = tf.expand_dims(self.weights['item_embedding'], axis=0)
            z.append(tf.reshape(u_1 * i_1, [-1, self.emb_dim]))

        elif self.model_type == 'mlp':
            u_1 = tf.reshape(tf.tile(u_e, [1, self.n_items]), [-1, self.emb_dim])
            i_1 = tf.tile(self.weights['item_embedding'], [self.batch_size, 1])
            z.append(tf.concat([u_1, i_1], 1))
        else:
            u_1 = tf.expand_dims(u_e, axis=1)
            i_1 = tf.expand_dims(self.weights['item_embedding'], axis=0)
            u_i = tf.reshape(u_1 * i_1, [-1, self.emb_dim])

            u_1 = tf.reshape(tf.tile(u_e, [1, self.n_items]), [-1, self.emb_dim])
            i_1 = tf.tile(self.weights['item_embedding'], [self.batch_size, 1])
            z.append(tf.concat([u_1, i_1, u_i], 1))

        for i in range(self.n_layers):
            # (batch, W[i]) * (W[i], W[i+1]) + (1, W[i+1]) => (batch, W[i+1])
            z.append(tf.nn.relu(tf.matmul(z[i], self.weights['W_%d' % i]) + self.weights['b_%d' % i]))

        agg_out = tf.matmul(z[-1], self.weights['h']) # (batch, W[-1]) * (W[-1], 1) => (batch, 1)
        all_ratings = tf.reshape(agg_out, [-1, self.n_items])
        return all_ratings

    def _create_batch_ratings(self, u_e, i_e):
        z = []

        n_b_user = tf.shape(u_e)[0]
        n_b_item = tf.shape(i_e)[0]


        if self.model_type == 'jrl':
            u_1 = tf.expand_dims(u_e, axis=1)
            i_1 = tf.expand_dims(i_e, axis=0)
            z.append(tf.reshape(u_1 * i_1, [-1, self.emb_dim])) # (n_b_user * n_b_item, embed_size)

        elif self.model_type == 'mlp':
            u_1 = tf.reshape(tf.tile(u_e, [1, n_b_item]), [-1, self.emb_dim])
            i_1 = tf.tile(i_e, [n_b_user, 1])
            z.append(tf.concat([u_1, i_1], 1)) # (n_b_user * n_b_item, 2*embed_size)
        else:
            u_1 = tf.expand_dims(u_e, axis=1)
            i_1 = tf.expand_dims(i_e, axis=0)
            u_i = tf.reshape(u_1 * i_1, [-1, self.emb_dim])

            u_1 = tf.reshape(tf.tile(u_e, [1, n_b_item]), [-1, self.emb_dim])
            i_1 = tf.tile(i_e, [n_b_user, 1])
            z.append(tf.concat([u_1, i_1, u_i], 1))

        for i in range(self.n_layers):
            # (batch, W[i]) * (W[i], W[i+1]) + (1, W[i+1]) => (batch, W[i+1])
            z.append(tf.nn.relu(tf.matmul(z[i], self.weights['W_%d' % i]) + self.weights['b_%d' % i]))

        agg_out = tf.matmul(z[-1], self.weights['h']) # (batch, W[-1]) * (W[-1], 1) => (batch, 1)
        batch_ratings = tf.reshape(agg_out, [n_b_user, n_b_item])
        return batch_ratings

    def batch_norm_layer(self, x, train_phase, scope_bn):
        with tf.variable_scope(scope_bn):
            return batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
            is_training=True, reuse=tf.AUTO_REUSE, trainable=True, scope=scope_bn)

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

def load_pretrained_data():
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, 'bprmf')
    pretrain_data = np.load(pretrain_path)
    print('load the pretrained bprmf model parameters.')
    return pretrain_data

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    t0 = time()

    if args.pretrain == -1:
        pretrain_data = load_pretrained_data()
    else:
        pretrain_data = None


    model = NMF(data_config=config, pretrain_data=pretrain_data)

    saver = tf.train.Saver()
    # *********************************************************
    # save the model parameters.
    if args.save_flag == 1:
        layer = '-'.join([str(l) for l in eval(args.layer_size)])
        weights_save_path = '%sweights/%s/%s/%s/l%s_r%s' % (args.proj_path, args.dataset, model.model_type, layer, str(args.lr),
                                                            '-'.join([str(r) for r in eval(args.regs)]))
        ensureDir(weights_save_path)
        save_saver = tf.train.Saver(max_to_keep=1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # *********************************************************
    # reload the pretrained model parameters.
    if args.pretrain == 1:
        layer = '-'.join([str(l) for l in eval(args.layer_size)])
        pretrain_path = '%sweights/%s/%s/%s/l%s_r%s' % (args.proj_path, args.dataset, model.model_type, layer, str(args.lr),
                                                    '-'.join([str(r) for r in eval(args.regs)]))
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(pretrain_path + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('load the pretrained model parameters from: ', pretrain_path)

            # *********************************************************
            # get the performance from pretrained model.
            users_to_test = list(data_generator.test_set.keys())
            ret = test(sess, model, users_to_test, drop_flag=True)
            cur_best_pre_0 = ret['recall'][0]

            pretrain_ret = 'pretrained model recall=[%.5f, %.5f], precision=[%.5f, %.5f], hit=[%.5f, %.5f],' \
                           'ndcg=[%.5f, %.5f], auc=[%.5f]' % \
                           (ret['recall'][0], ret['recall'][-1],
                            ret['precision'][0], ret['precision'][-1],
                            ret['hit_ratio'][0], ret['hit_ratio'][-1],
                            ret['ndcg'][0], ret['ndcg'][-1], ret['auc'])
            print(pretrain_ret)
        else:
            sess.run(tf.global_variables_initializer())
            cur_best_pre_0 = 0.
            print('without pretraining.')

    else:
        sess.run(tf.global_variables_initializer())
        cur_best_pre_0 = 0.
        print('without pretraining.')

    loss_loger, pre_loger, rec_loger, ndcg_loger, auc_loger, hit_loger = [], [], [], [], [], []
    stopping_step = 0
    should_stop = False

    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1


        for idx in range(n_batch):
            users, pos_items, neg_items = data_generator.sample()
            _, batch_loss, batch_mf_loss, batch_emb_loss, batch_reg_loss = sess.run([model.opt, model.loss, model.mf_loss, model.emb_loss, model.reg_loss],
                               feed_dict={model.users: users, model.pos_items: pos_items,
                                          model.neg_items: neg_items,
                                          model.dropout_keep: eval(args.keep_prob),
                                          model.train_phase: True})
            loss += batch_loss
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss
            reg_loss += batch_reg_loss

        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
        if (epoch + 1) % 10 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (epoch, time() - t1, loss, mf_loss, reg_loss)
                print(perf_str)
            continue

        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        ret = test(sess, model, users_to_test, batch_test_flag=True)

        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        auc_loger.append(ret['auc'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f], ' \
                       'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f], auc=[%.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, reg_loss, ret['recall'][0], ret['recall'][-1],
                        ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                        ret['ndcg'][0], ret['ndcg'][-1], ret['auc'])
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=5)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
            save_saver.save(sess, weights_save_path + '/weights', global_step=epoch)
            print('save the weights in path: ', weights_save_path)

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    auc = np.array(auc_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s], auc=[%.5f]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]),
                  auc[idx])
    print(final_perf)

    save_path = '%soutput/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
    ensureDir(save_path)
    f = open(save_path, 'a')

    f.write('embed_size=%d, lr=%.4f, layer_size=%s, keep_prob=%s, regs=%s, loss_type=%s, \n\t%s\n'
            % (args.embed_size, args.lr, args.layer_size, args.keep_prob, args.regs, args.loss_type, final_perf))
    f.close()
