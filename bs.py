#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import scipy.special as special
from collections import Counter
import pandas as pd
import time
import gc

data_path = 'C:/Users/Administrator/Desktop/alimama/'
data_path = 'C:/Users/Administrator/Desktop/alimama/baysmooth/'
train= pd.read_table(data_path+'round1_ijcai_18_train_20180301.txt',sep=' ',index_col=0)
test=pd.read_table(data_path1+'round1_ijcai_18_test_b_20180418.txt',sep=' ',index_col=0)

for i in range(1,4):
       train['item_cate_' + str(i)] = train['item_category_list'].map(lambda x: int(x.split(';')[i]) if len(x.split(';')) > i else '')



for i in range(2,4):
       train['item_cate_' + str(i)] = train['item_category_list'].map(lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else '')

item_cate =train.item_category_list.map(lambda x: x.split(';'))
item_cate_1 = item_cate.map(lambda x: int(x[1]))  # have 13 unique
train['item_cate_1'] = item_cate_1



train['day'] = train['context_timestamp'].map(lambda x: time.strftime("%d", time.localtime(x))).astype(int)
train['hour'] = train['context_timestamp'].map(lambda x: time.strftime("%H", time.localtime(x))).astype(int)


for feat_1 in ['item_id','item_city_id']:
    feat_1 = 'item_city_id'
    gc.collect()
    res = pd.DataFrame()
    temp = train[[feat_1,'day','is_trade']]
    day = 25
    for day in range(18,26):
        count=temp.groupby([feat_1]).apply(lambda x: x['is_trade'][(x['day']<day).values].count()).reset_index(name=feat_1+'_all')
        count1=temp.groupby([feat_1]).apply(lambda x: x['is_trade'][(x['day']<day).values].sum()).reset_index(name=feat_1+'_1')
        count[feat_1+'_1']=count1[feat_1+'_1']
        count.fillna(value=0, inplace=True)
        count['day']=day
        res=res.append(count,ignore_index=True)
    print(feat_1,' over')
    res.to_csv(data_path+'%s.csv' %feat_1, index=False)


for feat_1,feat_2 in [('item_cate_1','user_age_level'),('item_cate_1','user_occupation_id'),
                      ('item_id','user_occupation_id'),('item_id','user_age_level')
                      ]:
    gc.collect()
    res = pd.DataFrame()
    temp = train[[feat_1,feat_2,'day','is_trade']]
    for day in range(18,26):
        count=temp.groupby([feat_1,feat_2]).apply(lambda x: x['is_trade'][(x['day']<day).values].count()).reset_index(name=feat_1+'_'+feat_2+'_all')
        count1=temp.groupby([feat_1,feat_2]).apply(lambda x: x['is_trade'][(x['day']<day).values].sum()).reset_index(name=feat_1+'_'+feat_2+'_1')
        count[feat_1+'_'+feat_2+'_1']=count1[feat_1+'_'+feat_2+'_1']
        count.fillna(value=0, inplace=True)
        count['day']=day
        res=res.append(count,ignore_index=True)
    print(feat_1,feat_2,' over')
    res.to_csv(data_path+'%s.csv' % (feat_1+'_'+feat_2), index=False)



class BayesianSmoothing(object):#贝叶斯平滑，这个慢一点
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample(self, alpha, beta, num, imp_upperbound):
        sample = np.random.beta(alpha, beta, num)
        I = []
        C = []
        for clk_rt in sample:
            imp = random.random() * imp_upperbound
            imp = imp_upperbound
            clk = imp * clk_rt
            I.append(imp)
            C.append(clk)
        return I, C

    def update(self, imps, clks, iter_num, epsilon):
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(imps, clks, self.alpha, self.beta)
            if abs(new_alpha - self.alpha) < epsilon and abs(new_beta - self.beta) < epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta
            print(self.alpha, self.beta)

    def __fixed_point_iteration(self, imps, clks, alpha, beta):
        numerator_alpha = 0.0
        numerator_beta = 0.0
        denominator = 0.0

        for i in range(len(imps)):
            numerator_alpha += (special.digamma(clks[i] + alpha) - special.digamma(alpha))
            numerator_beta += (special.digamma(imps[i] - clks[i] + beta) - special.digamma(beta))
            denominator += (special.digamma(imps[i] + alpha + beta) - special.digamma(alpha + beta))

        return alpha * (numerator_alpha / denominator), beta * (numerator_beta / denominator)



#user_id bayes smooth
for feat_1 in ['user_id','item_id','item_city_id','shop_id']:
    temp = pd.read_csv(data_path+'%s.csv' %feat_1)
    bs = BayesianSmoothing(1, 1)
    bs.update(temp[feat_1 + '_all'].values, temp[feat_1 + '_1'].values, 1000, 0.001)
    temp[feat_1 + '_smooth'] = (temp[feat_1 + '_1'] + bs.alpha) / (temp[feat_1 + '_all'] + bs.alpha + bs.beta)
    temp.fillna(value=bs.alpha / (bs.alpha + bs.beta), inplace=True)
    temp.to_csv(data_path + '%s.csv' % (feat_1 + '_' + 'smooth'), index=False)
    gc.collect()
    print(feat_1 + ' over...')

#item_id bayes smooth
#
alpha = []
beta = []
for feat_1,feat_2 in [('item_cate_1','user_age_level'),('item_cate_1','user_occupation_id'),
                      ('item_id','user_occupation_id'),('item_id','user_age_level')]:
    temp = pd.read_csv(data_path1 + '%s.csv' % (feat_1 + '_' + feat_2))
    bs = BayesianSmoothing(1, 1)
    bs.update(temp[feat_1 + '_' + feat_2 + '_all'].values, temp[feat_1 + '_' + feat_2 + '_1'].values, 1000, 0.001)
    temp[feat_1 + '_' + feat_2 + '_smooth'] = (temp[feat_1 + '_' + feat_2 + '_1'] + bs.alpha) / (
    temp[feat_1 + '_' + feat_2 + '_all'] + bs.alpha + bs.beta)
    temp.fillna(value=bs.alpha / (bs.alpha + bs.beta), inplace=True)
    # temp.drop([feat_1 + '_' + feat_2 + '_1', feat_1 + '_' + feat_2 + '_all'], axis=1, inplace=True)
    temp.to_csv(data_path+'%s.csv' % (feat_1+'_'+feat_2+'smooth'), index=False)
    gc.collect()
    print(feat_1 + '_' + feat_2 + ' over...')
    alpha.append(bs.alpha)
    beta.append(bs.beta)



