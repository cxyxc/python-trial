# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
## 问题
    ### 我们没有数据
    ### 我们不知道哪些特征会影响人找工作，权重如何
    ### 人最终面试、上岗受到很多因素影响。如【有其他更优质工作机会、突发事件】等。这些不应该作为【匹配失败】的数据。那么什么样的数据应该作为失败数据呢？上岗成功的一定算是匹配成功，但不匹配的数据仍需要由逻辑得出

## 解题思路
    ### 特征选取
        ### 根据现实情况选择部分显著特征，每种特征根据情况折算得分，得分高于 80 的认为最终可以上岗成功（这里把特征总结成枚举项，便于随机生成）
        ### 特征之间可能相互影响，如家庭背景影响薪资权重，婚姻状况影响上班耗时权重（暂不考虑）
        ### 后续补充权重计算【可能应该折算成薪资后的数额进行】
    ### 数据来源
        ### 不依赖外部数据，由现有逻辑生成随机数据源
        ### 数据源足够大时(约为 2000 条即可)，可让 AI 掌握由人类总结的逻辑
        ### 此时输出结果应与现有逻辑匹配保持一致
    ### 后续业务数据集成
        ### 最终上岗的业务数据可以直接与生成的随机数据源混合【暂定30%的数据采用真实业务数据】，训练后查看结果，如果仍获得较高拟合度，则认为我们总结的特征及权重符合实际。如果发现拟合度较低，分析原因
        ### 后续持续更换特征，迭代找出更合适的模型
        ### 着重分析最终上岗数据中的较低分值的数据，发掘其中隐藏的特征


# %%
# 按照以下逻辑生成数据作为基础数据，表达逻辑匹配推荐算法
# 基准逻辑算法
dict_raws_base = {
    # 职位类别 
    # 0 非常不满意
    # 1 不满意
    # 2 满意
    'job_categories': {
        'high': 3,
        'score': [0, 60, 100],
    },
    # 缴金
    # 0 公司不缴纳+用户在乎
    # 1 公司缴纳+用户不在乎
    # 2 公司缴纳+用户在乎
    # 3 公司不缴纳+用户不在乎
    'insurance': {
        'high': 4,
        'score': [0, 60, 100, 100],
    },
    # 薪资+福利(包吃住等)
    # 0 薪资低于预期+无福利
    # 1 薪资低于预期+有福利
    # 2 薪资达到预期+无福利
    # 3 薪资达到预期+有福利
    'benefits': {
        'high': 4,
        'score': [0, 20, 80, 100],
    },
    # 上班耗时
    # 0 超过2小时
    # 1 超过1小时
    # 2 40分钟以内
    # 3 20分钟以内
    'go_work_time': {
        'high': 4,
        'score': [0, 20, 80, 100],
    },
}
# 假定特征相同，只是具体规则有出入
columns = dict_raws_base.keys()
# 得分模型
def get_score(d, score_raws):
    count = 0
    for key in columns:
        score = score_raws[key]['score'][int(d[key])]
        # 暂时不设置权重，认为各特征权重相等 weight = score_raws[key]['weight']
        count += (score / len(columns))
    return count
def get_score_rate(d, score_key):
    # 得分 0 总分 < 60 不考虑
    # 得分 1 总分 60 ~ 80 可以考虑
    # 得分 2 总分 > 80 有较强上岗可能性
    rate = 0
    count = d[score_key]
    if count < 60:
        rate = 0
    if count >= 60 and count < 80:
        rate = 1
    if count >= 80:
        rate = 2
    return rate


# %%
import numpy as np
import pandas as pd

def get_df(dict_raws, size):
    df_raws = {}
    for key in dict_raws:
        df_raws[key] = np.random.randint(dict_raws[key]['high'], size=(size))
    df=pd.DataFrame(df_raws, columns=columns)
    return df

df_mock = get_df(dict_raws_base, 2000)
df_mock["score"] = df_mock.apply(lambda d:get_score(d, dict_raws_base),axis =1)
df_mock["score_rate"] = df_mock.apply(lambda d:get_score_rate(d, 'score'),axis =1)
# 模拟数据
df_mock


# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def get_modal(df_data):
    data = df_data[columns].values
    target = df_data["score_rate"].values
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(data,target,test_size=0.3)
    r_lf = RandomForestClassifier(random_state=0)
    r_lf = r_lf.fit(Xtrain,Ytrain)
    r_score = r_lf.score(Xtest,Ytest)
    # 得到 1.0 评分认为是机器学习模型可以正确的学会基准逻辑，认为随机模拟的数据量已经足够
    print("Random Forest:{}".format(r_score))
    return r_lf
df_mock_modal = get_modal(df_mock)


# %%
print(df_mock_modal.predict([[2,3,3,3]]))


# %%
## 模型可以转化成代码供程序使用
import m2cgen as m2c
code = m2c.export_to_java(df_mock_modal)


# %%
## 以上是模型-策略的相互转化过程，基本可以得出结论，数据与模型本质上是一回事
## 后续展开业务漏损数据对模型进行修正的过程

# 假设后续出现一些漏损的业务数据(PS: 漏损数据指评分较低，但仍上岗成功的数据，或者评分极高，但用户或商家明确表示不合适的数据，两种数据同样重要，如果仅有一种，会造成模型向其中一个方向偏移)
# 鉴别漏损数据的过程，可能会形成新的关键特征。如果特征调整，需要使用现有已上岗数据对策略进行回测（不展开讨论）
# 漏损数据符合的得分模型假定如下（实际上我们不知道具体的漏损数据模型，但是有漏损数据，这里定义模型来生成数据）
dict_raws_busi = {
    'job_categories': {
        'score': [0, 0, 100],
    },
    'insurance': {
        'score': [0, 100, 60, 100],
    },
    'benefits': {
        'score': [0, 80, 60, 100],
    },
    'go_work_time': {
        'score': [0, 40, 0, 100],
    },
}

df_busi = get_df(dict_raws_base, 2000)
# 基准模型评分
df_busi["score"] = df_busi.apply(lambda d:get_score(d, dict_raws_base),axis =1)
df_busi["score_rate"] = df_busi.apply(lambda d:get_score_rate(d, 'score'),axis =1)
# 真实模型评分
df_busi["busi_score"] = df_busi.apply(lambda d:get_score(d, dict_raws_busi),axis =1)
df_busi["busi_score_rate"] = df_busi.apply(lambda d:get_score_rate(d, 'busi_score'),axis =1)
# 过滤出基准模型评分等级为 0 但漏损模型评分等级为 2 的数据。即已上岗但未能推荐，反之假定为明确表示不合适的岗，这两者之间应该大体均等
# 样本量级比较小，本身作为训练集很容易出现过拟合
# df_busi_no 的数据在真实业务中难以排查
df_busi_yes = df_busi.query('score_rate==0 & busi_score_rate==2')
df_busi_no = df_busi.query('score_rate==2 & busi_score_rate==0')
# df_busi = pd.concat([df_busi_yes, df_busi_no])
df_busi = df_busi_yes
df_busi


# %%
df_busi['score'] = df_busi['busi_score']
df_busi['score_rate'] = df_busi['busi_score_rate']
df_busi = df_busi.drop(columns=['busi_score', 'busi_score_rate'])
df_busi


# %%
# 直接用业务数据训练模型，发现找不到规律？此处如果已经找到规律了，直接使用即可，真实的业务数据由于特征数量不足，不推荐岗位(评级为0)的数据难以界定，应该难以找到规律，如果能找到规律，全篇都可以理解为废话了。
df_busi_modal = get_modal(df_busi)
df_busi_modal.predict([[0,0,0,0]]) # 只有上岗数据训练出来的模型，得到的结论永远都是 0


# %%
# 合并漏损数据与模拟数据
df_combine = pd.concat([df_mock, df_busi])
df_combine


# %%
# 重新训练生成模型，发现模型有 99.5% 的得分，说明数据仍有规律可寻，我们距离真实模型正在靠近？
df_combine_modal = get_modal(df_combine)


# %%
# 这些是被认为不能上岗，未进行推荐的数据，现在的模型下已经推荐上岗了
df_combine_modal.predict(df_busi_yes[columns].values)


