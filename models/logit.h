/**
 * @file fly_model.h
 * @author nickgu
 * @date 2015/01/13 15:29:07
 * @brief 
 *  
 **/

#ifndef  __LOGIT_H__
#define  __LOGIT_H__

#include <cstdlib>
#include <cmath>

#include "../fly_core.h"
#include "../fly_math.h"
#include "../fly_measure.h"
#include "../cfg.h"
#include "iter.h"
#include "uniform.h"

class LogisticRegression_t 
    : public IterModel_t
{
    public:
        LogisticRegression_t(const Config_t& conf, const char* section):
            IterModel_t(conf, section),
            _theta_num(0)
        {
            _momentum_ratio = 0.0f;
            _use_momentum = false;
            if (conf.conf_float(section, "momentum_ratio", &_momentum_ratio)) {
                _use_momentum = true;
                LOG_NOTICE("use_momentum:true. momentum_ratio:%f", _momentum_ratio);
            }
        }

        virtual ~LogisticRegression_t() {}

        /**
         *  1 / (1 + exp(-w*x))
         */
        virtual float predict(const Instance_t& raw_item) const {
            Instance_t item;
            _uniform.uniform(&item, raw_item);

            //float ans = sigmoid( sparse_dot(_theta, item.features) );
            //fprintf(stdout, "%f\t", ans);
            //item.write(stdout);
            return sigmoid( sparse_dot(_theta, item.features) );
        }

        virtual void write_model(FILE* stream) const {
            _uniform.write(stream);
            fprintf(stream, "%d\t%f\n", _theta_num, _theta.b);
            for (int i=0; i<_theta_num; ++i) {
                fprintf(stream, "%d:%f\n", i, _theta.w[i]);
            }
        }

        virtual void read_model(FILE* stream) {
            _uniform.read(stream);
            _theta.clear();
            fscanf(stream, "%d\t%f\n", &_theta_num, &_theta.b);
            _theta.set(_theta_num);
            for (int i=0; i<_theta_num; ++i) {
                int tmp;
                fscanf(stream, "%d:%f\n", &tmp, _theta.w+i);
            }
        }

        virtual void init(FlyReader_t* reader) {
            IterModel_t::init(reader);

            _theta_num = reader->dim();
            LOG_NOTICE("theta_dim = %d", _theta_num);

            _theta.set(_theta_num);
            for (int i=0; i<_theta_num; ++i) {
                _theta.w[i] = random_05();
            }
            _theta.b = random_05();

            _uniform.stat(_reader);

            if (_use_momentum) {
                _velo.set(_theta_num);
            }
        }

    private:
        Param_t _theta;
        int     _theta_num;
        MeanStdvar_Uniform _uniform;

        // optima.
        Param_t _velo;

        float   _momentum_ratio;
        bool    _use_momentum;

        /**
         *  input:
         *      target, X.
         *  return: 
         *      Loss.
         */
        virtual float _update(const Instance_t& raw_item) {
            float p = predict(raw_item);

            Instance_t item;
            _uniform.uniform(&item, raw_item);
            
            // @MAXL
            float desc = (item.label - p);
            if (_use_momentum) {
                _velo.b = _velo.b * (1.0 - _momentum_ratio) + _momentum_ratio * desc;
                _theta.b += _velo.b * _learn_rate;

            } else {
                _theta.b = _theta.b + desc * _learn_rate;
            }
            for (size_t i=0; i<item.features.size(); ++i) {
                int index = item.features[i].index;
                if (index >= _theta_num) {
                    continue;
                }

                float x = item.features[i].value;
                float gradient = desc * x;

                if (_use_momentum) {
                    _velo.w[index] = _velo.w[index] * (1.0 - _momentum_ratio) + _momentum_ratio * gradient;
                    _theta.w[index] += _velo.w[index] * _learn_rate;

                } else {
                    _theta.w[index] += gradient * _learn_rate;
                }
            }
            
            float loss = 0.0;
            /* Entropy Loss. */
            /*
            loss = -(item.label * safe_log(p) + (1-item.label) * safe_log(1-p));
            */
            /* L2 Loss. */
            loss = 0.5 * (item.label - p) * (item.label - p);
            return loss;
        }
};

#endif  //__LOGIT_H__

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
