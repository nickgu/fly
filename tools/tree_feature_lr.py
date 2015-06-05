# -*- coding: utf-8 -*-
# gusimiu@baidu.com
# 

import PyFly
import sys

class GBDT_LR:
    def __init__(self, tree_model_file, lr_model_file, tree_feature_offset):
        self.__gbdt = PyFly.load_gbdt(tree_model_file)
        self.__lr = PyFly.load_lr(lr_model_file)
        self.__feature_offset = tree_feature_offset

    def predict(self, tree_input):
        # get tree_feature and input to LR, output LR score.
        # input format:
        #   [(idx, value), ..]
        tf = map(lambda x:(x + self.__feature_offset, 1), PyFly.tree_features(self.__gbdt, tree_input))
        total_feature = tree_input + tf
        return PyFly.predict(self.__lr, total_feature)

if __name__ == '__main__':
    if len(sys.argv)!=3:
        print >> sys.stderr, 'Usage: tree_feature_lr.py <tree_model_file> <lr_model_file>\n\n'
        sys.exit(-1)
    model = GBDT_LR(sys.argv[1], sys.argv[2], 21)

    # test input. 
    while 1:
        line = sys.stdin.readline()
        if line == '':
            break
        arr = line.strip('\n').split(' ')
        label = arr[0]
        tree_input = map(lambda x:(int(x[0]), float(x[1])), map(lambda x:x.split(':'), arr[1:]))
        output = model.predict(tree_input)
        print label, output


