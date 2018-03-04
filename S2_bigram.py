#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 17:14:30 2018

@author: neha
"""

import nltk
from nltk.corpus import gutenberg
import numpy as np
import string
from nltk.corpus import brown 
import random

def load():
    train=[]
    test=[]    
    for fileid in gutenberg.fileids():
        sent=gutenberg.sents(fileid)
        s=[]
        for str1 in sent:
            s.append(str1)
            
        
        str2=[]
        for i in s:
            str2.append(' '.join(i).translate(str.maketrans('','',string.punctuation)))
        
        str3=''
        for i in str2:
            str3= str3+ ' <s> '+ i
                
        punctuation={'`','\''}
        for c in punctuation:
            str3= str3.replace(c,"")
 
        str3=' '.join(str3.split())
    #    str3=str3.translate(str.maketrans('','',string.punctuation))
    #    str3 = '<s> The Fulton County Grand Jury said Friday an investigation of Atlantas recent primary election produced no evidence that any irregularities took place . <s> The jury further said in term-end presentments that the City Executive Committee , which had over-all charge of the election , deserves the praise and thanks of the City of Atlanta for the manner in which the election was conducted . <s> The September-October term jury had been charged by Fulton Superior Court Judge Durwood Pye to investigate reports of possible irregularities in the hard-fought primary which was won by Mayor-nominate Ivan Allen Jr. .'
        words = str3.split(' ')
        train.append(words[:round(len(words)*0.8)])
        test.append(words[-round(len(words)*0.2):])

    train = [item for sublist in train for item in sublist]
    test = [item for sublist in test for item in sublist]
    return train,test

def cal_ngram(train,n):
    ngrams = {} 
    #n=2
    for index, word in enumerate(train):
        if index < len(train)-(n-1):
            w=[]
            for i in range(n):
                w.append(train[index+i])
            ngram = tuple(w)
#            print(ngram)
    
            if ngram in ngrams:
                ngrams[ ngram ] = ngrams[ ngram ] + 1
            else:
                ngrams[ ngram ] = 1
                
#    sorted_ngrams = sorted(ngrams.items(), key = lambda pair:pair[1], reverse = True)
    return ngrams


def cal_ngram_list(ngrams):
    ngrams_list=[]
    for key,value in ngrams.items():
        ngrams_list.append(key)
    
    return ngrams_list

def unknown(unigrams,train):
    unknown_list=[]
    for key, value in unigrams.items():
        if value < 2:
            unknown_list.append(key[0])
            for index, word in enumerate(unigrams):
                if train[index] == key[0]:
                    train[index] = '<UKN>'
        if len(unknown_list)==500:
                    break
    return train,unknown_list


def cal_probab(ngrams,n_1grams,n):
    prob = {}
    for key, value in ngrams.items():
        n_1key=[]
        for k in range(0,n-1):
            n_1key.append(key[k])
        
        prob[key] = value/(n_1grams[tuple(n_1key)])
        
    return prob


def cal_unigram_probab(ngrams,N):
    prob = {}
    for key, value in ngrams.items():
        prob[key] = value/N 
    return prob


def check_existence(key,ngram_list,train_prob,n):
    found=0
    nfound=0
    alpha=1;
    t_prob=-1
    for i in reversed(range(len(ngram_list))):
#        print(i)
#        print(key)
#        k=[]
#        k.append(key)
        if key in ngram_list[i]:
            prob = train_prob[i]
            t_prob = alpha*prob[key]
#            print('break')
            found=found+1
            break
        else:
            key=key[i:n]
#            print(key)
            alpha=alpha*0.4
            
    if t_prob==-1:
#        print('unknown')
        ukn=tuple(['<UKN>'])
        nfound=nfound+1
        prob = train_prob[0]
        t_prob=alpha*prob[ukn]/0.4
    return t_prob
                
def cal_probab_test(tngram,ngram_list,train_prob,n):
    t_prob=0
    
    for key, value in tngram.items():
#        print(key)
        prob = check_existence(key,ngram_list,train_prob,n)
#        print(prob)
        t_prob = t_prob + np.log2(prob)
        
    return t_prob

def cal_perplexity(test,ngram_list,train_prob,n):
    tngram={}
    tngram=cal_ngram(test,n)
    tN=len(test)
    tprob=cal_probab_test(tngram,ngram_list,train_prob,n)
    perplexity=2 ** (tprob*(-1/tN))
    
    return perplexity

def init(train,n):
    N=len(train)
    unigrams=cal_ngram(train,1)
    #replace some vocab with <UKN>
    train,unknown_list = unknown(unigrams,train)
    
    #get all ngrams and their counts
    ngram=[]
    
    for i in range(n):
#        print(i)
        ngram.append(cal_ngram(train,i+1))
    
    #calculate 1 to n gram's probabilities
    train_prob=[]
    train_prob.append(cal_unigram_probab(ngram[0],N))
    
    for i in range(1,n):
#        print(i)
        train_prob.append(cal_probab(ngram[i],ngram[i-1],i+1))
    
    
    
    #calculate ngram lists
    ngram_list=[]
    for i in range(n):
        ngram_list.append(cal_ngram_list(ngram[i]))
        
    return N,n,train,unknown_list,ngram,train_prob,ngram_list



train,test=load()
n=2
N,n,train,unknown_list,ngram,train_prob,ngram_list=init(train,n)

perplexity = cal_perplexity(test,ngram_list,train_prob,n)

print('Perplexity: ',perplexity)
