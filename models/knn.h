/**
 * @file knn.h
 * @author nickgu
 * @date 2015/03/03 10:35:36
 * @brief 
 *  
 **/

#ifndef  __KNN_H_
#define  __KNN_H_

#include <queue>
using namespace std;

#include "../fly_core.h"
#include "../fly_math.h"
#include "../fly_measure.h"
#include "../cfg.h"

/*
 * multi-class classifier.
 */
class KNNModel_t
    : public FlyModel_t 
{
    struct Info_t {
        Info_t() {}
        Info_t(float label, float dist):
            _label(label),
            _dist(dist)
        {}

        bool operator < (const Info_t& o) const {
            return _dist < o._dist;
        }

        float _label;
        float _dist;
    };

    public:
        KNNModel_t(const Config_t& conf, const char* section) {
            if ( !conf.conf_int(section, "k", &_k) ) {
                throw std::runtime_error("Meta model needs config: <k>");
            }
        }

        virtual float predict(const Instance_t& ins) const {
            Instance_t item;
            _reader->reset();
            priority_queue<Info_t> heap;
            //vector<Info_t> v;
            while (_reader->read(&item)) {
                float dist = _calc_dist(item, ins);

                Info_t item_info(item.label, dist);
                //v.push_back(item_info);
                /**/
                heap.push(item_info);
                if (heap.size() > (size_t)_k) {
                    heap.pop();
                }
                /**/
            }

            // used for multi-class classifier.

            map<float, int> counter;
            /*
            sort(v.begin(), v.end());
            for (int i=0; i<_k; ++i) {
                counter[v[i]._label] ++;
            }
            */

            while (heap.size()>0) {
                Info_t info = heap.top();
                heap.pop();
                counter[info._label] ++;
                //LOG_NOTICE("target=%f dist=%f output=%f", ins.label, info._dist, info._label);
            }

            float best_ret = 0;
            int best_count = 0;
            for (map<float, int>::iterator it = counter.begin(); it!=counter.end(); ++it) {
                if (best_count < it->second) {
                    best_ret = it->first;
                    best_count = it->second;
                }
            }

            static int all = 0;
            static int err = 0;
            all ++;
            if (ins.label != best_ret) {
                err ++;
                LOG_NOTICE("target=%f output=%f err=%d/%d", ins.label, best_ret, err, all);
            }
            return best_ret;
        }

        virtual void  init(FlyReader_t* reader) {
            _reader = reader;
        }

        virtual void  train() {
        }

        virtual void  write_model(FILE* stream) const {
        }

        virtual void  read_model(FILE* stream) {
        }

    private:
        int             _k;
        FlyReader_t*    _reader;

        float _calc_dist(const Instance_t& a, const Instance_t& b) const {
            float ret = 0;
            float A[1000] = {0};
            for (size_t i=0; i<a.features.size(); ++i) {
                A[a.features[i].index] += a.features[i].value;
            }
            for (size_t i=0; i<b.features.size(); ++i) {
                A[b.features[i].index] -= b.features[i].value;
            }
            for (int i=0; i<1000; ++i) {
                // L-1 distance.
                //ret += fabs(A[i]);

                // L-2 distance.
                ret += fabs(A[i]) * fabs(A[i]);
            }
            return ret;
        }
};


#endif  //__KNN_H_

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
