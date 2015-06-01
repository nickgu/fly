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

#include "fly_core.h"
#include "fly_math.h"
#include "fly_measure.h"
#include "cfg.h"
#include "iter.h"
#include "uniform.h"

class LogisticRegression_t 
    : public IterModel_t
{
    enum LearnRateAdjustMethod_t {
        FeatureDecay = 0,
        Decay,
        Constant,
        Shrink
    };

    public:
        LogisticRegression_t(const Config_t& conf, const char* section):
            IterModel_t(conf, section),
            _theta_num(0),
            _best_loss(1.0)
        {
            _momentum_ratio = 0.0f;
            _use_momentum = false;
            if (conf.conf_float(section, "momentum_ratio", &_momentum_ratio)) {
                _use_momentum = true;
                LOG_NOTICE("use_momentum:true. momentum_ratio:%f", _momentum_ratio);
            }

            string method;
            method = conf.conf_str_default(section, "learn_rate_adjust_method", "feature_decay");
            const char* method_str[] = {"FeatureDecay", "Decay", "Constant", "Shrink"};
            if (method == "decay") {
                _adjust_method = Decay;
            } else if (method == "constant") {
                _adjust_method = Constant;
            } else if (method == "shrink") {
                _adjust_method = Shrink;
            } else if (method == "feature_decay") {
                _adjust_method = FeatureDecay;
            } else {
                LOG_ERROR("Illegal adjust method: %s", method.c_str());
                _adjust_method = FeatureDecay;
            }
            LOG_NOTICE("LEARNING_RATE_ADJUST_METHOD : %s", method_str[_adjust_method]);

            _early_stop_N = conf.conf_int_default(section, "early_stop_n", -1);
            LOG_NOTICE("early_stop_N: %d", _early_stop_N);

            _shrink_limit = conf.conf_int_default(section, "shrink_limit", 0);
            LOG_NOTICE("shrink_limit: %d", _shrink_limit);

            _min_loss_diff = conf.conf_float_default(section, "min_loss_diff", 1e-6);
            LOG_NOTICE("min_loss_diff=%f", _min_loss_diff);
        }

        virtual ~LogisticRegression_t() {
            if (_reader) {
                delete _reader;
            }
        }

        /**
         *  1 / (1 + exp(-w*x))
         */
        virtual float predict(const Instance_t& raw_item) const {
            Instance_t item(raw_item.features.size());
            _uniform.uniform(&item, raw_item);

            return predict_no_uniform( item );
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

            _theta_update_times = new size_t [_theta_num];
            memset(_theta_update_times, 0, sizeof(size_t)*_theta_num);

            LOG_NOTICE("Begin to stat uniform infomation.");
            _uniform.stat(reader);
            LOG_NOTICE("Stat over.");

            // preprocess:
            //   - uniform.
            LOG_NOTICE("Begin to preprocess.");
            Instance_t item;
            const char* temp_lr_file = "temp_lr_preprocess.bin";
            FILE* temp_file = fopen(temp_lr_file, "wb");
            reader->reset();
            while (reader->read(&item)) {
                Instance_t out_item;
                _uniform.uniform(&out_item, item);
                out_item.write_binary(temp_file);
            }
            fclose(temp_file);
            LOG_NOTICE("End preprocess.");

            _reader = new BinaryFeatureReader_t(temp_lr_file);
            _reader->reset();

            if (_use_momentum) {
                _velo.set(_theta_num);
            }
        }

    private:
        LearnRateAdjustMethod_t _adjust_method;

        Param_t _theta;
        int     _theta_num;
        MeanStdvar_Uniform _uniform;

        Param_t _best_theta;
        float   _best_loss;
        int     _best_round;

        // optima.
        Param_t _velo;
        float   _momentum_ratio;
        bool    _use_momentum;

        // shrink in N.
        int     _early_stop_N;
        int     _shrink_N;
        int     _shrink_times;
        int     _shrink_limit;
        float   _original_rate;
        size_t  _update_times;
        size_t  *_theta_update_times;
        float   _min_loss_diff;

        // profile timer.
        Timer  _predict_tm;
        Timer  _calc_tm;
        Timer  _uniform_tm;
        Timer  _total_update_tm;

        float predict_no_uniform(const Instance_t& uniformed_item) const {
            return sigmoid( sparse_dot(_theta, uniformed_item.features) );
        }

        /**
         *  input:
         *      target, X.
         *  return: 
         *      Loss.
         *
         *  item is uniformed by preprocesser.
         */
        virtual float _update(Instance_t& item) {
            _total_update_tm.begin();

            /*
            _uniform_tm.begin();
            _uniform.self_uniform(&item);
            _uniform_tm.end();
            */

            _predict_tm.begin();
            float p = predict_no_uniform(item);
            _predict_tm.end();

            _update_times ++;
            float cur_rate = _learn_rate;
            if (_adjust_method == Decay || _adjust_method == FeatureDecay) {
                cur_rate = _learn_rate / sqrt(_update_times);
            } else if (_adjust_method == Constant || _adjust_method == Shrink) {
                cur_rate = _learn_rate; 
            }
            
            // @MAXL
            _calc_tm.begin();
            float desc = (item.label - p);
            if (_use_momentum) {
                _velo.b = _velo.b * (1.0 - _momentum_ratio) + _momentum_ratio * desc;
                _theta.b += _velo.b * cur_rate;

            } else {
                _theta.b = _theta.b + desc * cur_rate;
            }
            for (size_t i=0; i<item.features.size(); ++i) {
                int index = item.features[i].index;
                if (index >= _theta_num) {
                    continue;
                }

                if (_adjust_method == FeatureDecay) {
                    _theta_update_times[index] ++;
                    cur_rate = _learn_rate / sqrt(_theta_update_times[index]);
                }

                float x = item.features[i].value;
                float gradient = desc * x;

                if (_use_momentum) {
                    _velo.w[index] = _velo.w[index] * (1.0 - _momentum_ratio) + _momentum_ratio * gradient;
                    _theta.w[index] += _velo.w[index] * cur_rate;

                } else {
                    _theta.w[index] += gradient * cur_rate;
                }
            }
            _calc_tm.end();
            
            float loss = 0.0;
            /* Entropy Loss. */
            /*
            loss = -(item.label * safe_log(p) + (1-item.label) * safe_log(1-p));
            */
            /* L2 Loss. */
            loss = 0.5 * (item.label - p) * (item.label - p);
            _total_update_tm.end();
            //LOG_NOTICE("loss=%f", loss);
            return loss;
        }

        virtual void _epoch_end() {
            LOG_NOTICE("TimeUsed=%f (uniform=%f pred=%f calc=%f)", 
                    _total_update_tm.cost_time(),
                    _uniform_tm.cost_time(),
                    _predict_tm.cost_time(), 
                    _calc_tm.cost_time());
            _total_update_tm.clear();
            _predict_tm.clear();
            _calc_tm.clear();
            _uniform_tm.clear();

            float expect_loss_inc = _learn_rate * _epoch_loss;
            LOG_NOTICE("Round %d: loss=%.8f best_loss=%.8f cur_rate=%f", 
                    _iter_round, _epoch_loss, _best_loss, _learn_rate);
            LOG_NOTICE("          loss_inc=%.8f min_loss=%f exp_loss=%f",
                    _best_loss - _epoch_loss,
                    _min_loss,
                    expect_loss_inc);

            bool improvment = false;
            bool big_improvement = false;
            int no_progress_in_N = -1;
            if (_epoch_loss < _best_loss - _min_loss_diff) {
                improvment = true;
                if (_epoch_loss < _best_loss - expect_loss_inc) {
                    big_improvement = true;
                }
            } else {
                no_progress_in_N = _iter_round - _best_round;
            }

            // Shrink update.
            if (_adjust_method == Shrink) {
                if (big_improvement) {
                    _best_loss = _epoch_loss;
                    _best_theta = _theta;
                    _best_round = _iter_round;
                    LOG_NOTICE("ShrinkAdjust: accept param.");

                } else {
                    LOG_NOTICE("ShrinkAdjust: REJECT param."); 
                    if (no_progress_in_N >= _shrink_N) {
                        _shrink_times ++;
                        LOG_NOTICE("Shrink! %f -> %f (%d/%d times)", _learn_rate, _learn_rate*0.5, _shrink_times, _shrink_limit);
                        _learn_rate *= 0.5;

                        if (_shrink_times > _shrink_limit) {
                            _force_stop = true;
                        }
                    }
                }
            } else {
                if (improvment) {
                    _best_loss = _epoch_loss;
                    _best_theta = _theta;
                    _best_round = _iter_round;
                } else {
                    if (_early_stop_N>=0 && no_progress_in_N > _early_stop_N) {
                        LOG_NOTICE("NO_PROGRESS_IN_N[%d] > EARLY_STOP_N[%d], stop!!",
                                no_progress_in_N, _early_stop_N);
                        _force_stop = true;
                    }
                }
            }
        }

        virtual void _train_end() {
            LOG_NOTICE("Replace theta with best_param at loss@%f", _best_loss);
            _theta = _best_theta;
        }

        virtual void _train_begin() {
            _update_times = 0;
            _original_rate = _learn_rate;
            _shrink_N = 0;
        }
};

#endif  //__LOGIT_H__

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
