# -*- coding: utf-8 -*-
# gusimiu@baidu.com
# 

import math
import random
import time
import sys

import numpy as np

class Layer:
    def __init__(self, n_in, n_out):
        self.__n_in = n_in
        self.__n_out = n_out
        self.__params = np.ndarray((n_in + 1, n_out), dtype=np.float32)
        for i in range(n_in + 1):
            for j in range(n_out):
                self.__params[i][j] = random.uniform(-0.1, 0.1)
        # same shape as __params.
        self.__accumulate_gradient = np.zeros(self.__params.shape)

    def forward(self, v_in):
        # add a bias column.
        if not isinstance(v_in, np.ndarray):
           v_in = np.array(v_in) 
        self.__v_in = np.hstack((v_in, [1]))
        d = self.__v_in.dot(self.__params)
        for j in range(d.shape[0]):
            if -d[j]>1e2:
                d[j] = 1.0 / (1+math.exp(1e2))
            d[j] = 1.0 / (1+math.exp(-d[j]))
        self.__v_out = d
        return d

    def backward(self, v_delta, alpha):
        d = np.zeros(self.__accumulate_gradient.shape)

        inp = np.array([self.__v_in]).T
        out = self.__v_out
        delt = v_delta
        grad = inp * ((out * (1-out)) * delt)
        self.__accumulate_gradient += grad * alpha

        previous_delta = np.zeros((self.__n_in))
        for i in range(self.__n_in):
            for j in range(self.__n_out):
                previous_delta[i] += self.__params[i][j] * self.__v_out[j]*(1-self.__v_out[j]) * v_delta[j]
        return previous_delta

    def update(self, scale_ratio=0.1):
        self.__params += self.__accumulate_gradient
        self.__accumulate_gradient *= scale_ratio

    def __str__(self):
        return self.__params.__str__()


class MLP:
    def __init__(self, arch=[2, 2, 1]):
        self.__layers = []
        for idx in range(len(arch)-1):
            self.__layers.append(Layer(arch[idx], arch[idx+1]))

    def train(self, generator, alpha=0.1, scale=0.1, minibatch_size=5, maximum_iter=2000):
        cnt = 0
        iter_cnt = 0
        all_delt = []
        for v_in, v_out in generator():
            temp_in = v_in
            for l in self.__layers:
                temp_in = l.forward(temp_in)

            v_delta = v_out - temp_in
            all_delt.append(abs(v_delta))

            for l in reversed(self.__layers):
                v_delta = l.backward(v_delta, alpha)

            cnt += 1
            if cnt % minibatch_size == 0:
                for l in reversed(self.__layers):
                    l.update(scale) 

                iter_cnt += 1
                if iter_cnt > maximum_iter:
                    break

                # error estinmate.
                all_delt = all_delt[-50:]
                s = sum(all_delt)
                if math.sqrt(s.dot(s))<1e-4:
                    logging.info('Training %d round(s) finished.' % cnt)
                    break

    def predict(self, v_in):
        for l in self.__layers:
            v_in = l.forward(v_in)
        return v_in


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    def tester(arch, generator, alpha=0.1, scale=0.1, maximum_iter=2000, minibatch_size=1, test_num=100):
        logging.info('TESTER: %s, %s', arch, generator)
        network = MLP(arch)
        logging.info('begin training..')
        network.train(generator, 
                        alpha=alpha, 
                        minibatch_size=minibatch_size,
                        maximum_iter=maximum_iter,
                        scale=scale)

        cnt = 0
        d = None
        for v_in, v_target in generator():
            v_out = network.predict(v_in)
            if d is None:
                d = abs(v_target - v_out)
            else:
                d += abs(v_target - v_out)
            cnt += 1
            if cnt >= test_num:
                break
        d /= cnt
        logging.info('error: %s' % ','.join(map(lambda x:'%.3f'%x, d)))

    def G_swap():
        while 1:
            for a in range(0, 2):
                for b in range(0, 2):
                    yield [[a, b], [b, a]]

    def G_xor():
        while 1:
            for a in range(0, 2):
                for b in range(0, 2):
                    yield [[a, b], [a^b]]

    def G_circle():
        while 1:
            for a in range(0, 11):
                for b in range(0, 11):
                    out = 0
                    if a*a + b*b>100:
                        out = 1
                    yield [[a/10.0, b/10.0], [out]]

    tester([2,2], G_swap, alpha=0.1, scale=1.0, maximum_iter=1000, minibatch_size=1)
    tester([2,2,1], G_xor, alpha=0.4, scale=1.0, maximum_iter=2000, minibatch_size=1)
    tester([2,20,20,1], G_circle, alpha=0.1, scale=0.0, maximum_iter=5000, minibatch_size=10)

