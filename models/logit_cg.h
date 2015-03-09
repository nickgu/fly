/**
 * @file logit_cg.h
 * @author nickgu
 * @date 2015/01/13 15:29:07
 * @brief 
 *  
 **/

#ifndef  __LOGIT_CG_H__
#define  __LOGIT_CG_H__

#include <cstdlib>
#include <cmath>

#include "../fly_core.h"
#include "../fly_math.h"
#include "../fly_measure.h"
#include "../cfg.h"
#include "iter.h"

class CG_LogisticRegression_t 
    : public IterModel_t
{
    public:
        CG_LogisticRegression_t(const Config_t& conf, const char* section):
            IterModel_t(conf, section)
        {
        }

        virtual ~CG_LogisticRegression_t() {
        }

        /**
         *  1 / (1 + exp(-w*x))
         */
        virtual float predict(const Instance_t& item) const {
            return sigmoid( sparse_dot(_theta, item.features) );
        }

        virtual void write_model(FILE* stream) const {
        }

        virtual void read_model(FILE* stream) {
        }

        virtual void init(FlyReader_t* reader) {
            IterModel_t::init(reader);

            size_t theta_num = reader->dim();
            LOG_NOTICE("theta_dim = %d", theta_num);

            _theta.set(theta_num);
            _grad.set(theta_num);
            _direction.set(theta_num);

            // randomize.
            for (size_t i=0; i<theta_num; ++i) {
                _theta.w[i] = random_05();
            }
            _theta.b = random_05();
        }

    private:
        Param_t _theta;
        Param_t _grad;
        Param_t _direction;
        float _g2_last;
        float _g2_this;

        /**
         * calc the loss output.
         * Loss = 0.5 * /sum (label - predict_output)^2
         */
        float _loss_output(const Param_t& param) {
            _reader->reset();
            Instance_t item;
            float loss = 0.0f;
            while (_reader->read(&item)) {
                float s = sigmoid( sparse_dot(param, item.features) );
                // L2:
                //loss += 0.5 * (item.label - s) * (item.label - s);

                // cross-entropy.
                loss += -((1-item.label) * safe_log(1-s) + item.label * safe_log(s));
            }

            loss = loss / _reader->size();
            //param.debug(stderr);
            //LOG_NOTICE("loss: %f", loss);
            return loss;
        }

        virtual void _epoch_begin() {
            _grad = 0;
            return ;
        }

        virtual void _epoch_end() {
            _g2_last = _g2_this;
            _g2_this = _grad * _grad;

            // get direction.
            float ratio = 0.0;
            if (_g2_last > 0) {
                ratio = _g2_this / _g2_last;
            }
            _direction = _grad + _direction * ratio;

            // linear search.
            float low = 0;
            float gap = _learn_rate;
            float jump = _learn_rate;
            float beg = _loss_output(_theta);
            float cur = _loss_output(_theta + _direction * gap);
            while (cur>beg && gap>1e-5*_learn_rate) {
                gap /= 2;
                jump /= 2;
                cur = _loss_output(_theta + _direction * gap);
            }
            while (cur<=beg && gap<1.0) {
                beg = cur;
                gap += jump;
                jump *= 2;
                cur = _loss_output(_theta + _direction * gap);
            }
            LOG_NOTICE("gap=%f (%f,%f)", gap, beg, cur);

            Param_t m1 = _theta + _direction * (low + gap*0.382);
            Param_t m2 = _theta + _direction * (low + gap*0.618);
            float x = _loss_output(m1);
            float y = _loss_output(m2);
            float min_loss = 0;
            Param_t new_theta = _theta;
            float end_gap = gap*1e-5;
            while (gap > end_gap) {
                LOG_NOTICE("l:%.4f(%f), h:%.4f(%f)", low+gap*0.382, x, low+gap*0.618, y);
                if (x < y) {
                    new_theta = m1;
                    gap *= 0.618;
                    min_loss = x;
                } else {
                    new_theta = m2;
                    low += gap * 0.382;
                    gap *= 0.618;
                    min_loss = y;
                }
                m1 = _theta + _direction * (low + gap*0.382);
                m2 = _theta + _direction * (low + gap*0.618);
                x = _loss_output(m1);
                y = _loss_output(m2);
            }

            _epoch_loss = min_loss;
            _theta = new_theta;
            return ;
        }

        /**
         *  get gradient of loss.
         */
        virtual float _update(const Instance_t& item) {
            float p = predict(item);
            
            // gradient of @L2
            //float desc = (item.label - p);
            // gradient of @cross-entropy
            float desc = (item.label - p);
            _grad.b += desc;
            for (size_t i=0; i<item.features.size(); ++i) {
                int index = item.features[i].index;
                if (index >= (int)_theta.size()) {
                    continue;
                }

                float x = item.features[i].value;
                _grad.w[index] += desc * x;
            }
            return 0;
        }
};

#endif  //__LOGIT_CG_H__

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
