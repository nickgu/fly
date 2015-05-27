/**
 * @file linear.h
 * @author nickgu
 * @date 2015/01/13 15:29:07
 * @brief 
 *  
 **/

#ifndef  __LINEAR_H__
#define  __LINEAR_H__

#include <cstdlib>
#include <cmath>

#include "../fly_core.h"
#include "../fly_measure.h"

class LinearRegression_t 
        : public FlyModel_t 
{
    public:
        LinearRegression_t(
                int iter_num=100,
                float learn_rate = 0.1,
                float min_loss=1e-3):
            _theta(NULL),
            _theta_num(0),
            _iter_num(iter_num),
            _learn_rate(learn_rate),
            _min_loss(min_loss)
        {}

        virtual ~LinearRegression_t() {
            if (_theta) {
                delete [] _theta;
            }
        }

        /**
         *  w*x + const 
         */
        virtual float predict(const Instance_t& item) const {
            float ret = _const;
            for (size_t i=0; i<item.features.size(); ++i) {
                int idx = item.features[i].index;
                if (idx >= _theta_num) {
                    continue;
                }
                ret += _theta[idx] * item.features[i].value;
            }
            return ret;
        }

        virtual void write_model(FILE* stream) const {
            fprintf(stream, "%f\n", _const);
            for (int i=0; i<_theta_num; ++i) {
                fprintf(stream, "%d:%f\n", i, _theta[i]);
            }
        }

        virtual void init(FeatureReader_t& reader) {
            _reader = reader;
            _reader.reset();
            // stat the max dim id.
            _theta_num = 0;
            Instance_t item;
            while ( reader.read(&item) ) {
                for (size_t i=0; i<item.features.size(); ++i) {   
                    int idx = item.features[i].index;
                    if (idx >= _theta_num) {
                        _theta_num = idx + 1;
                    }
                }
            }

            LOG_NOTICE("theta_dim = %d", _theta_num);
            _theta = new float [_theta_num];
            for (int i=0; i<_theta_num; ++i) {
                _theta[ item.features[i].index ] = (random() % 10000) / 10000.0 * 0.5;
            }

            _const = (random() % 10000) / 10000.0 * 0.5;
        }

        virtual void train() {
            Instance_t item;
            float last_loss = 1e10;
            for (int i=0; i<_iter_num; ++i) {
                _reader.reset();

                int last_per = 0;
                float loss = 0;
                while (_reader.read(&item)) {
                    loss += _update(item);

                    // output training progress.
                    int per = _reader.percentage();
                    if (per - last_per >= 5) {
                        fprintf(stderr, "%cRound:%d Iteration:%d%% [%u/%u line(s)]", 
                                13, i+1, per, _reader.processed_num(), _reader.size());
                        fflush(stderr);
                        last_per = per;
                    }
                }
                fprintf(stderr, "\n");

                loss /= _reader.size();
                loss = sqrt(loss);
                LOG_NOTICE("round %d: loss=%f", 
                        i+1, loss);

                if (loss < _min_loss) {
                    LOG_NOTICE("min loss reached, break!");
                    break;
                }

                if ( fabs(loss - last_loss) < 1e-4 ) {
                    break;
                } 
                last_loss = loss;
            }
        }

    private:
        float*  _theta;
        float   _const;
        int     _theta_num;

        int     _iter_num;
        float   _learn_rate;
        float   _min_loss;

        FeatureReader_t _reader;

        /**
         *  input:
         *      target, X.
         *  return: 
         *      Loss.
         */
        float _update(const Instance_t& item) {
            float p = predict(item);
            
            // L2 regularization.
            // @RMSE
            float desc = -1.0 * (item.label - p);

            float old_const = _const;
            _const = _const - desc * _learn_rate;
            LOG_NOTICE("const: %f->%f", old_const, _const);
            for (size_t i=0; i<item.features.size(); ++i) {
                int index = item.features[i].index;
                if (index >= _theta_num) {
                    continue;
                }

                float x = item.features[i].value;
                float theta = _theta[index];
                float gradient = desc * x;

                if (index == 1) {
                    LOG_NOTICE("d:%d x=%f t=%f g=%f [%f,%f]", index, x, theta, gradient, p, item.label);
                }
                _theta[index] = theta - (gradient) * _learn_rate;
            }

            float new_p = predict(item);
            // loss: @RMSE 
            return 0.5 * (item.label - new_p) * (item.label - new_p);
        }
};

#endif  //__LINEAR_H__

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
