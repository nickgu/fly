/**
 * @file fly_math.h
 * @author nickgu
 * @date 2015/01/16 13:51:37
 * @brief 
 *  
 **/

#ifndef  __FLY_MATH_H_
#define  __FLY_MATH_H_

#include <cstdio>
#include <cmath>

#include <cstdlib>
using namespace std;

#include "fly_core.h"

struct Param_t {
    float *w;
    float b;
    size_t sz;

    Param_t():
        w(NULL), sz(0) {};
    Param_t(float* t, float c, size_t n) {
        sz = n;
        w = new float[n];
        memcpy(w, t, sizeof(float)*n);
        b = c;
    }
    Param_t(const Param_t& o) {
        sz = o.sz;
        w = new float[sz];
        memcpy(w, o.w, sizeof(float)*o.sz);
        b = o.b;
    }
    ~Param_t() {
        clear();
    }

    size_t size() const {return sz;}

    void set(size_t size) {
        clear();
        w = new float[size];
        memset(w, 0, sizeof(float)*size);
        b = 0;
        sz = size;
    }

    void clear() {
        if (w) {
            sz = 0;
            delete [] w;
            w = NULL;
        }
    }

    const Param_t& operator = (const Param_t& o) {
        b = o.b;
        sz = o.sz;
        w = new float[sz];
        memcpy(w, o.w, sizeof(float)*o.sz);
        return *this;
    }

    void operator *= (float x) {
        b*=x;
        for (size_t i=0; i<sz; ++i) {
            w[i] *= x;
        }
        return;
    }
    Param_t operator * (float x) const {
        Param_t ret = *this;
        ret *= x;
        return ret;
    }
    float operator * (const Param_t& o) const {
        float ret = b * o.b;
        for (size_t i=0; i<std::min(sz, o.size()); ++i) {
            ret += w[i] * o.w[i];
        }
        return ret;
    }
    void operator += (float x) {
        b+=x;
        for (size_t i=0; i<sz; ++i) {
            w[i] += x;
        }
        return;
    }
    Param_t operator + (float x) const {
        Param_t ret = *this;
        ret += x;
        return ret;
    }
    void operator += (const Param_t& o) {
        b+=o.b;
        for (size_t i=0; i<sz; ++i) {
            w[i] += o.w[i];
        }
        return;
    }
    Param_t operator + (const Param_t& o) const {
        Param_t ret = *this;
        ret += o;
        return ret;
    }
    const Param_t& operator = (float o) {
        b = o;
        for (size_t i=0; i<sz; ++i) {
            w[i] = o;
        }
        return *this;
    }

    void debug(FILE* stream) const {
        fprintf(stream, "b=%.4f, w=(%.4f", b, w[0]);
        for (size_t i=1; i<sz; ++i) {
            fprintf(stream, ", %.4f", w[i]);
        }
        fprintf(stream, ")\n");
        return ;
    }
};

template<typename T>
float sgn(T x) {
    return x>=0?1.0:-1.;
}

/**
 * Generate a random number between [-0.05, +0.05]
 */
float random_05() {
    return (rand() % 1000) / 10000.0 - 0.5;
}

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

float vec_dot(size_t num, float* theta, float *input) {
    float ret = 0.0f;
    for (size_t i=0; i<num; ++i) {
        ret += input[i] * theta[i];
    }
    return ret;
}

float sparse_dot(int theta_num, float* theta, const FArray_t<IndValue_t>& input) {
    float ret = 0.0f;
    for (size_t i=0; i<input.size(); ++i) {
        if (input[i].index < theta_num) {
            ret += theta[input[i].index] * input[i].value;
        }
    }
    return ret;
}

template<typename T>
T safe_log(T x) {
    if (x < 1e-7) {
        return log(1e-7);
    }
    return log(x);
}

float sparse_dot(const Param_t& p, const FArray_t<IndValue_t>& input) {
    float ret = p.b;
    for (size_t i=0; i<input.size(); ++i) {
        if (input[i].index < (int)p.size()) {
            ret += p.w[input[i].index] * input[i].value;
        }
    }
    return ret;
}

#endif  //__FLY_MATH_H_

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
