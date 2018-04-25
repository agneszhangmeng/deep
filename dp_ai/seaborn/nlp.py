
from __future__ import division
import pandas as pd
import numpy as np
import nltk
#from nltk.book import *
from nltk import *

# nltk.download()
# def gender_features(word):
#     return {'last_letter': word[-1]}
#
# from nltk.corpus import names
# import random
# names=([(name,'male') for name in names.words('male.txt')] + [(name,'female') for name in names.words('female.txt')])
# random.shuffle(names)
#
#
# f = [(gender_features(n),g) for (n,g) in names]
# trainset,testset = f[500:],f[:500]
# c = nltk.NaiveBayesClassifier.train(trainset)
#
# print(c.classify(gender_features('Neo')))
# print(c.classify(gender_features('Trinity')))


# text1.concordance('monstrous')
# text1.similar('monstrous')
# text2.common_contexts(["monstrous","very"])

# print(len(text3) / len(set(text3)))
#
# text3.count('smote') / len(text3)
# text3.index('smote')
# text1.dispersion_plot(['citizens','freedom'])


# fdist1 = FreqDist(text2)
# # print(fdist1['the'])
# # fdist1.plot(5)   # top 5
# # fdist1.plot(5,cumulative=True)
# print(fdist1.hapaxes())   # words only happen once
#
# v = set(text1)  # no duplicate
# long_words = [w for w in v if len(w) > 5]
#
# fdist = FreqDist(text1)
# long_words = [w for w in set(text1) if len(w)>10 and fdist[w] > 7]
# print(long_words)
#
# print(list(bigrams(['a','b','c'])))

# text4.collocations()  # double connetcted words

# fdist = FreqDist([len(w) for w in text1])
# print(fdist.items())
# print(fdist.freq(1))



# ----------------------------- #
# normal words
# from nltk.corpus import gutenberg
# print(gutenberg.fileids())
#
# emma= gutenberg.words('austen-emma.txt')
# print(gutenberg.raw('austen-emma.txt'))
# emma = nltk.Text(emma)#
# print(emma[:10])
#
# # web text
# from nltk.corpus import webtext
# for fileid in webtext.fileids():
#     print(fileid,webtext.raw(fileid)[:50])
#
#
# # nps_chat
# from nltk.corpus import nps_chat
# chatroom = nps_chat.posts('10-19-20s_706posts.xml')
# chatroom[123:125]


# brown
# from nltk.corpus import brown
# print(brown.categories())
# print(brown.fileids())
#
# news = brown.words(categories='news')
# fdist = nltk.FreqDist([w.lower() for w in news])
# modals= ['can','could','may','might','must','will']
# for m in modals:
#     print(m,':',fdist[m])


# reuters
# from nltk.corpus import reuters
# print(reuters.fileids())
# print(reuters.categories())

# inaugural
# from nltk.corpus import inaugural
# print(list(f[:4]for f in inaugural.fileids()))
# # 下面体现American和citizen随时间推移使用情况
# cfd = nltk.ConditionalFreqDist(\
#                               (target,fileid[:4])\
#                               for fileid in inaugural.fileids()\
#                               for w in inaugural.words(fileid)\
#                               for target in ['america','citizen']\
#                                if w.lower().startswith(target))
#cfd.plot()

# self designed
# from nltk.corpus import PlaintextCorpusReader
# root = '/Users/estherzhang/Downloads/mergedNotes.csv'
# wordlist = PlaintextCorpusReader(root,'.*')#匹配所有文件
# print(wordlist.fileids())
# #print(wordlist.words('tem1.txt'))

# stopwords
# from nltk.corpus import stopwords
# #定义一个计算func计算不在停用词列表中的比例的函数
# def content(text):
#     stopwords_eng = stopwords.words('english')
#     content = [w for w in text if w.lower() and w not in stopwords_eng]
#     return len(content)/len(text)
# print(content(nltk.corpus.reuters.words()))

# the last word with female or male
# names = nltk.corpus.names
# print(names.fileids())
# male = names.words('male.txt')
# female = names.words('female.txt')
# cfd = nltk.ConditionalFreqDist((fileid,name[-1]) for fileid in names.fileids() for name in names.words(fileid))
# cfd.plot()

# s = ['N','IHO','K','S']
# entries = nltk.corpus.cmudict.entries()
# print('Example:',entries[0])
# word_list = [word for word,pron in entries if pron[-4:]==s]
# print(word_list)


#synonym

from nltk.corpus import wordnet as wn
# print(wn.synsets('motorcar'))
#
# print(wn.synset('car.n.01').lemmas)
# wn.lemma('car.n.01.automobile').name
# wn.lemma('car.n.01.automobile').synset
#
# motorcar = wn.synset('car.n.01').hyponyms()#上位词
# car = wn.synset('car.n.01').root_hypernyms() #下位词

# right = wn.synset('right_whale.n.01')
# orca = wn.synset('orca.n.01')
# print(right.lowest_common_hypernyms(orca))
# print(right.path_similarity(orca))  #相似度


# tem = ['hello','world','hello','dear']
# FreqDist(tem)
#
# from nltk.corpus import brown
# cfd = nltk.ConditionalFreqDist((genre,word) for genre in brown.categories() for word in brown.words(categories=genre))
# print("conditions are:",cfd.conditions()) #查看conditions
# print(cfd['news'])
# print(cfd['news']['could'])
#
# cfd.tabulate(conditions=['news','romance'],samples=['could','can'])
# cfd.tabulate(conditions=['news','romance'],samples=['could','can'],cumulative=True)



# regular expression
# import re
# from nltk.corpus import words
# wordlist = [w for w in words.words('en-basic') if w.islower()]
# same = [w for w in wordlist if re.search(r'^[ghi][mno][jlk][def]$',w)]
# temp =  [w for w in wordlist if re.search(r'[^aeiouAEIOU]$',w)]
# print(temp)

# 寻找两个或者两个以上的元音序列
# import nltk
# wsj = sorted(set(nltk.corpus.treebank.words()))
# fd = nltk.FreqDist(vs for word in wsj for vs in re.findall(r'[aeiou]{4,}',word))
# print(fd.items())
#
# word = 'fly'
# print(re.findall(r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)$',word))
#
#
# # 词干提取器
# porter = nltk.PorterStemmer()
# print(porter.stem('lying'))
#
# # 词性归并器
# wnl = nltk.WordNetLemmatizer()
# print(wnl.lemmatize('women'))

#词性标注器
# text = nltk.word_tokenize("And now for something completely difference")
# print(text)
# print(nltk.pos_tag(text))
#
# text = "The/AT grand/JJ is/VBD ."
# print([nltk.tag.str2tuple(t) for t in text.split()])
# print(nltk.corpus.brown.tagged_words())

from nltk.corpus import brown
# word_tag = nltk.FreqDist(brown.tagged_words(categories="news"))
# print([word+'/'+tag for (word,tag)in word_tag if tag.startswith('V')])
# ################下面是查找money的不同标注#################################
# wsj = brown.tagged_words(categories="news")
# cfd = nltk.ConditionalFreqDist(wsj)
# print(cfd['money'].keys())   # NN

# def findtag(tag_prefix,tagged_text):
#     cfd = nltk.ConditionalFreqDist((tag,word) for (word,tag) in tagged_text if tag.startswith(tag_prefix))
#     return dict((tag,list(cfd[tag].keys())[:5]) for tag in cfd.conditions())#数据类型必须转换为list才能进行切片操作
#
# tagdict = findtag('NN',nltk.corpus.brown.tagged_words(categories="news"))
# for tag in sorted(tagdict):
#     print(tag,tagdict[tag])

# 已经标注的语料库
# brown_tagged = brown.tagged_words(categories="learned")
# tags = [b[1] for (a,b) in nltk.bigrams(brown_tagged) if a[0]=="often"]
# fd = nltk.FreqDist(tags)
# fd.tabulate()


brown_tagged_sents = brown.tagged_sents(categories="news")
#
# raw = 'I do not like eggs and ham, I do not like them Sam I am'
# tokens = nltk.word_tokenize(raw)
# default_tagger = nltk.DefaultTagger('NN')#创建标注器
# print(default_tagger.tag(tokens)) # 调用tag()方法进行标注
# print(default_tagger.evaluate(brown_tagged_sents))
#
# patterns = [
#     (r'.*ing$','VBG'),
#     (r'.*ed$','VBD'),
#     (r'.*es$','VBZ'),
#     (r'.*','NN')#为了方便，只有少量规则
# ]
# regexp_tagger = nltk.RegexpTagger(patterns)
# print(regexp_tagger.evaluate(brown_tagged_sents))

# 查询标注器
# fd = nltk.FreqDist(brown.words(categories="news"))
# cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories="news"))
# ##############################################python2和3的区别#########
# most_freq_words = fd.most_common(100)
# likely_tags = dict((word,cfd[word].max()) for (word,times) in most_freq_words)
# #######################################################################
# baseline_tagger = nltk.UnigramTagger(model=likely_tags,backoff=nltk.DefaultTagger('NN'))
# print(baseline_tagger.evaluate(brown_tagged_sents))

# 一元标注器
# size = int(len(brown_tagged_sents)*0.9)
# train_sents = brown_tagged_sents[:size]
# test_sents = brown_tagged_sents[size+1:]
# unigram_tagger = nltk.UnigramTagger(train_sents)
# print(unigram_tagger.evaluate(test_sents))




# 特征提取 生成朴素贝叶斯分类器
# from nltk.corpus import names
# import random
#
# def gender_features(word): #特征提取器
#     return {'last_letter':word[-1]} #特征集就是最后一个字母
#
# names = [(name,'male') for name in names.words('male.txt')]+[(name,'female') for name in names.words('female.txt')]
# random.shuffle(names)#将序列打乱
#
# features = [(gender_features(n),g) for (n,g) in names]#返回对应的特征和标签
#
# train,test = features[500:],features[:500] #训练集和测试集
# classifier = nltk.NaiveBayesClassifier.train(train) #生成分类器
#
# print('Neo is a',classifier.classify(gender_features('Neo')))#分类
# print(nltk.classify.accuracy(classifier,test)) #测试准确度
# classifier.show_most_informative_features(5)#得到似然比，检测对于哪些特征有用
#
# # 语料库很大的时候
# from nltk.classify import apply_features
# train_set = apply_features(gender_features,names[500:])
# test_set = apply_features(gender_features,names[:500])


#  生成train, validation, test
# from nltk.corpus import names
# import random
#
# def gender_features(word): #特征提取器
#     return {'last_letter':word[-1]} #特征集就是最后一个字母
#
# names = [(name,'male') for name in names.words('male.txt')]+[(name,'female') for name in names.words('female.txt')]
#
# train_names = names[1500:]
# devtest_names = names[500:1500]
# test_names = names[:500]
#
# train_set = [(gender_features(n),g) for (n,g) in train_names]
# devtest_set = [(gender_features(n),g) for (n,g) in devtest_names]
# test_set = [(gender_features(n),g) for (n,g) in test_names]
#
# classifier = nltk.NaiveBayesClassifier.train(train_set)
# print(nltk.classify.accuracy(classifier,devtest_set))
# ######################记录报错的案例###############################
# errors = []
# for (name,tag) in devtest_names:
#     guess = classifier.classify(gender_features(name))
#     if guess!=tag:
#         errors.append((tag,guess,name))


# from nltk.corpus import movie_reviews
# import random
#
# all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
# word_features = all_words.most_common(2) #前两千个最常出现的单词
#
# def document_features(document):
#     document_words = set(document)
#     features = {}
#     for (word,freq) in word_features:
#         features['contains(%s)'%word] = (word in document_words) #参数文档中是否包含word：True/False
#     return features
#
# documents = [(list(movie_reviews.words(fileid)),category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]
# random.shuffle(documents)
#
# features = [(document_features(d),c)for (d,c) in documents]
# train_set,test_set = features[100:],features[:100]
#
# classifier = nltk.NaiveBayesClassifier.train(train_set)
# print(nltk.classify.accuracy(classifier,test_set))

# 决策树分类  耗时很长

# from nltk.corpus import brown
#
# suffix_fdist = nltk.FreqDist()
# for word in brown.words():
#     word = word.lower()
#     #suffix_fdist.inc(word[-1:]) python2
#     suffix_fdist[word[-1:]] += 1 #python3
#     suffix_fdist[word[-2:]] += 1
#     suffix_fdist[word[-3:]] += 1
#
# common_suffixes = suffix_fdist.most_common(100) #获得常见特征链表
# #定义特征提取器：
# def pos_features(word):
#     features = {}
#     for (suffix,times) in common_suffixes:
#         features['endswith(%s)' % suffix] = word.lower().endswith(suffix)
#     return features
#
# tagged_words = brown.tagged_words(categories='news')
# featuresets = [(pos_features(n),g)for (n,g) in tagged_words]
# size = int(len(featuresets)*0.1)
#
# train_set , test_set= featuresets[size:], featuresets[:size]
# classifier = nltk.DecisionTreeClassifier.train(train_set) #“决策树分类器”
# print(nltk.classify.accuracy(classifier,test_set))


# 联合分类器

# from nltk.corpus import brown
#
#
# # 带有历史的特征提取器
# def pos_features(sentence, i, history):
#     features = {'suffix(1)': sentence[i][-1:], \
#                 'suffix(2)': sentence[i][-2:], \
#                 'suffix(3)': sentence[i][-3:]}
#     if i == 0:  # 当它在分界线的时候，没有前置word 和 word-tag
#         features['prev-word'] = '<START>'
#         features['prev-tag'] = '<START>'
#     else:  # 记录前面的history
#         features['prev-word'] = sentence[i - 1]
#         features['prev-tag'] = history[i - 1]
#     return features
#
#
# '''
# ###########流程式###############
# tagged_sents = brown.tagged_sents(categories="news")
# size = int(len(tagged_sents)*0.1)
# train_sents,test_sents = tagged_sents[size:],tagged_sents[:size]
#
# train_set = []
#
# for tagged_sent in train_sents:
#     untagged_set = nltk.tag.untag(tagged_sent)
#     history = []
#     for i,(word,tag) in enumerate(tagged_sent):
#         featureset = pos_features(untagged_set,i,history)
#         history.append(tag)
#         train_set.append((featureset,tag))
#     classifier = nltk.NaiveBayesClassifier.train(train_set)
# '''
#
#
# #########类思想重写##################
#
# class ConsecutivePosTagger(nltk.TaggerI):  # 这里定义新的选择器类，继承nltk.TaggerI
#     def __init__(self, train_sents):
#         train_set = []
#         for tagged_sent in train_sents:
#             untagged_set = nltk.tag.untag(tagged_sent)  # 去标签化
#             history = []
#             for i, (word, tag) in enumerate(tagged_sent):
#                 featureset = pos_features(untagged_set, i, history)
#                 history.append(tag)  # 将tag添加进去
#                 train_set.append((featureset, tag))  # 拿到了训练集
#             self.classifier = nltk.NaiveBayesClassifier.train(train_set)  # 创建训练模型
#
#     def tag(self, sentence):  # 必须定义tag方法
#         history = []
#         for i, word in enumerate(sentence):
#             featureset = pos_features(sentence, i, history)
#             tag = self.classifier.classify(featureset)
#             history.append(tag)
#         return zip(sentence, history)
#
#
# tagged_sents = brown.tagged_sents(categories="news")
# size = int(len(tagged_sents) * 0.1)
# train_sents, test_sents = tagged_sents[size:], tagged_sents[:size]
#
# tagger = ConsecutivePosTagger(train_sents)
# print(tagger.evaluate(test_sents))

# 文本提取

# sentence = [('the','DT'),('little','JJ'),('yellow','JJ'),('dog','NN'),('brak','VBD')]
# grammer = "NP: {<DT>?<JJ>*<NN>}"
# cp = nltk.RegexpParser(grammer) #生成规则
# result = cp.parse(sentence) #进行分块
# print(result)
#
# result.draw() #调用matplotlib库画出来

#为不包括再大块中的标识符序列定义一个缝隙
# sentence = [('the','DT'),('little','JJ'),('yellow','JJ'),('dog','NN'),('bark','VBD'),('at','IN'),('the','DT'),('cat','NN')]
# grammer = """NP:
#             {<DT>?<JJ>*<NN>}
#             }<VBD|NN>+{
#             """  #加缝隙，必须保存换行符
# cp = nltk.RegexpParser(grammer, loop=2) #生成规则
# result = cp.parse(sentence) #进行分块
# print(result)

tree1 = nltk.Tree('NP',['Alick'])
print(tree1)
tree2 = nltk.Tree('N',['Alick','Rabbit'])
print(tree2)
tree3 = nltk.Tree('S',[tree1,tree2])
print(tree3.label()) #查看树的结点
tree3.draw()

