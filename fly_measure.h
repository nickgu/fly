/**
 * @file fly_measure.h
 * @author nickgu
 * @date 2015/01/14 15:33:45
 * @brief 
 *  
 **/

#ifndef  __FLY_MEASURE_H_
#define  __FLY_MEASURE_H_

#include <cmath>

#include <algorithm>
using namespace std;

#include "fly_math.h"

struct ResultPair_t {
    float target;
    float output;

    ResultPair_t() {}
    ResultPair_t(float t, float o):
        target(t), output(o) {}
};

/**
 *  sort result array by output desc.
 */
bool cmp_output_desc(const ResultPair_t& a, const ResultPair_t& b) {
    return a.output > b.output;
}

float calc_rmse(size_t num, ResultPair_t* result) {
    float ret = 0;
    for (size_t i=0; i<num; ++i) {
        float x = (result[i].target - result[i].output);
        ret += x * x;
    }
    ret /= num;
    ret = sqrt(ret);
    return ret;
}

float calc_log_mle(size_t num, ResultPair_t* result) {
    float ret = 0.0f;
    for (size_t i=0; i<num; ++i) {
        float t = result[i].target;
        float o = result[i].output;
        ret += -(t * save_log(o) + (1.0 - t) * save_log(1-o));
    }
    ret /= num;
    return ret;
}

float calc_auc(size_t num, ResultPair_t* result) {
    double ret = 0;
    sort(result, result+num, cmp_output_desc);

    size_t pos = 0;
    size_t neg = 0;
    for (size_t i=0; i<num; ++i) {
        if ( int(result[i].target + 0.5) == 1 ) {
            pos += 1;
        } else {
            neg += 1;
        }
    }
    
    size_t cur_pos = 0;
    size_t cur_neg = 0;
    size_t seg_pos = 0;
    size_t seg_neg = 0;
    for (size_t i=0; i<num; ++i) {
        if ( int(result[i].target + 0.5) == 1 ) {
            seg_pos += 1;
        } else {
            seg_neg += 1;
        }

        /**
         * area:
         *       /
         *      / |
         *     a| |b
         *      |_|
         *       w
         */
        if (i == num-1 or result[i].output != result[i+1].output) {
            ret += (cur_pos + seg_pos*0.5) /pos * seg_neg;

            cur_pos += seg_pos;
            cur_neg += seg_neg;
            seg_pos = 0;
            seg_neg = 0;
        }
    }
    return ret / neg;
}

struct ConfusionMatrix_t {
    int tp;
    int tn;
    int fp;
    int fn;

    string str() const {
        char buf[128];
        float p = tp*100.0/(tp + fp);
        float r = tp*100.0/(tp + fn); 
        snprintf(buf, sizeof(buf), 
                "TP=%d FP=%d | TN=%d FN=%d | P=%.2f%% R=%.2f%% F=%.2f",
                tp, fp, tn, fn, p, r, p*r*2 / (p+r) );
        return string(buf);
    }
};

ConfusionMatrix_t calc_confussion_matrix(size_t num, ResultPair_t* result) {
    ConfusionMatrix_t ret = {0};
    for (size_t i=0; i<num; ++i) {
        if (result[i].target >= 0.5) { 
            if (result[i].output>=0.5) {
                ret.tp ++;
            } else {
                ret.fn ++;
            }
        } else {
            if (result[i].output<0.5) {
                ret.tn ++;
            } else {
                ret.fp ++;
            }
        }
    }
    return ret;
}

#endif  //__FLY_MEASURE_H_

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
