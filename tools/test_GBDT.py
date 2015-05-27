#! /bin/env python
# encoding=utf-8
# gusimiu@baidu.com
# 

import GBDT

if __name__=='__main__':
    handle = GBDT.load('test.model')
    s = '0 ' + ' '.join(map(lambda (i, v): '%d:%d' %(i+1, v), enumerate([i for i in range(20)])))
    score = GBDT.predict(handle, s)
    print 'Score = %f' % score

    leaves = GBDT.tree_features(handle, s)
    print leaves
