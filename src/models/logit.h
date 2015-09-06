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

enum UniformMethod_t {
    NoUniform = 0,
    PreUniform,
    OnlineUniform
};

enum LearnRateAdjustMethod_t {
    FeatureDecay = 0,
    GradientFeatureDecay,
    Decay,
    Constant,
};

enum RegularizationMethod_t {
    RegNone = 0,
    RegL1,
    RegL2
};

class LogitSolver:
    public Updatable_t
{
    public:
        LogitSolver(size_t theta_num, 
                    const Param_t& cur_param, 
                    UniformMethod_t uniform_method,
                    MeanStdvar_Uniform* uniformer,
                    float learn_rate,
                    LearnRateAdjustMethod_t decay_method,
                    const Param_t& decay,
                    RegularizationMethod_t reg_method,
                    float reg_weight);

        ~LogitSolver();

        float update(Instance_t& item);

    public:
        Param_t _theta;
        int     _theta_num;

        UniformMethod_t     _uniform_method;
        MeanStdvar_Uniform* _uniform;

        float   _learn_rate;
        LearnRateAdjustMethod_t _decay_method;
        Param_t _decay;

        RegularizationMethod_t _reg_method;
        float   _reg_weight;
};

LogitSolver::~LogitSolver() {
}

LogitSolver::LogitSolver(size_t theta_num, 
                const Param_t& cur_param, 
                UniformMethod_t uniform_method,
                MeanStdvar_Uniform* uniformer,
                float learn_rate,
                LearnRateAdjustMethod_t decay_method,
                const Param_t& decay,
                RegularizationMethod_t reg_method,
                float reg_weight):
    _theta(cur_param),
    _theta_num(theta_num),
    _uniform_method(uniform_method),
    _uniform(uniformer),
    _learn_rate(learn_rate),
    _decay_method(decay_method),
    _decay(decay),
    _reg_method(reg_method),
    _reg_weight(reg_weight)
{
}    

float LogitSolver::update(Instance_t& item) {
    if (_uniform_method == OnlineUniform) {
        _uniform->self_uniform(&item);
    }

    float cur_rate = _learn_rate;
    float p = sigmoid( sparse_dot(_theta, item.features) );
    float desc = (item.label - p);
    
    // desc@ MSE-loss:
    //desc *= (1-p) * p;

    // variant-update : x_square.
    /*
    float x_square = 0;
    float h_cst = 100.;
    for (size_t i=0; i<item.features.size(); ++i) {
        x_square += item.features[i].value * item.features[i].value;
    }

    // variant-update for square.
    desc *= (1. - exp(-h_cst * x_square)) / x_square;
    */

    // variant-update for hinge.
    /*
    float fake_y = 2. * (item.label - .5);
    float fake_p = 2. * (p - .5);
    desc = -fake_y * min( 0.f, (1-fake_y*fake_p)/x_square );
    */

    float reg = 0;

    if (_decay_method == FeatureDecay || _decay_method == Decay) {
        _decay.b += 1.0;
        cur_rate = _learn_rate / (1.0 + sqrt(_decay.b));
    } else if (_decay_method == GradientFeatureDecay) {
        _decay.b += desc * desc;
        cur_rate = _learn_rate / (1.0 + sqrt(_decay.b));
    }
    if (_reg_method == RegNone) {
        reg = 0;
    } else if (_reg_method == RegL1) {
        reg = - sgn(_theta.b) * _reg_weight;
    } else if (_reg_method == RegL2) {
        reg = -0.5 * (_theta.b * _reg_weight);
    }
    _theta.b = _theta.b + (desc + reg) * cur_rate;
    for (size_t i=0; i<item.features.size(); ++i) {
        int index = item.features[i].index;
        if (index >= _theta_num) {
            continue;
        }

        float x = item.features[i].value;
        float gradient = desc * x;

        if (_decay_method == FeatureDecay) {
            _decay.w[index] += 1.0;
            cur_rate = _learn_rate / (1.0 + sqrt(_decay.w[index]));
        } else if (_decay_method == GradientFeatureDecay) {
            _decay.w[index] += gradient * gradient;
            cur_rate = _learn_rate / (1.0 + sqrt(_decay.w[index]));
        }

        if (_reg_method == RegNone) {
            reg = 0;
        } else if (_reg_method == RegL1) {
            reg = - sgn(_theta.w[index]) * _reg_weight;
        } else if (_reg_method == RegL2) {
            reg = -0.5 * (_theta.w[index] * _reg_weight);
        }

        _theta.w[index] += (gradient + reg) * cur_rate;
    }

    float loss = 0.0;
    // Loss@MSE
    //loss = 0.5 * (item.label - p) * (item.label - p);
    // Loss@Log
    loss = -(item.label * safe_log(p) + (1-item.label) * safe_log(1-p));
    // Loss@Hinge
    //loss = max(0., 1- 4. * (item.label - .5) * (p - .5));
    return loss;
}

struct UniformerJob_t {
    int job_id;
    IReader_t* reader;
    PCPool_t<Instance_t>* pool;
    MeanStdvar_Uniform* uniformer;
    ThreadData_t<FILE*>* output_fp;
};

void* thread_uniformer(void* c) {
    UniformerJob_t &job = *(UniformerJob_t*)c;
    if (job.reader) {
        LOG_NOTICE("thread[%d] : I am a reader.", job.job_id);
        size_t c = 0;
        while (1) {
            Instance_t* cell = job.pool->begin_put();
            if (!job.reader->read(cell)) {
                job.pool->end_put(false);
                break;
            }
            job.pool->end_put();
            c ++;
        }
        LOG_NOTICE("reader: load over. count=%d", c);
        job.pool->set_putting(false);

    } else {
        LOG_NOTICE("thread[%d] : I am a worker.", job.job_id);
        Instance_t item;
        int count = 0;
        while (job.pool->get(&item)) {
            job.uniformer->self_uniform(&item);
            FILE* out = job.output_fp->borrow();
            item.write_binary(out);
            job.output_fp->give_back();
            count ++;
        }
        LOG_NOTICE("thread[%d] : over. %d processed.", job.job_id, count);
    }
    return NULL;
}


class LogisticRegression_t 
    : public IterModel_t
{
    public:
        LogisticRegression_t(const Config_t& conf, const char* section):
            IterModel_t(conf, section),
            _continuous_training(false),
            _theta_num(0),
            _best_loss(1.0)
        {
            _momentum_ratio = 0.0f;
            _use_momentum = false;
            if (conf.conf_float(section, "momentum_ratio", &_momentum_ratio)) {
                _use_momentum = true;
                LOG_NOTICE("use_momentum:true. momentum_ratio:%f", _momentum_ratio);
            }

            // load learning rate decay method.
            string method;
            method = conf.conf_str_default(section, "learn_rate_adjust_method", "feature_decay");
            const char* method_str[] = {"FeatureDecay", "GradientFeatureDecay", "Decay", "Constant"};
            if (method == "decay") {
                _adjust_method = Decay;
            } else if (method == "constant") {
                _adjust_method = Constant;
            } else if (method == "gradient_feature_decay") {
                _adjust_method = GradientFeatureDecay;
            } else if (method == "feature_decay") {
                _adjust_method = FeatureDecay;
            } else {
                LOG_ERROR("Illegal adjust method: %s", method.c_str());
                _adjust_method = GradientFeatureDecay;
            }
            LOG_NOTICE("LEARNING_RATE_ADJUST_METHOD : %s", method_str[_adjust_method]);

            // load regularization method.
            method = conf.conf_str_default(section, "regularization_method", "none");
            const char* reg_method_str[] = {"none", "l1", "l2"};
            if (method == "none") {
                _reg_method = RegNone;
            } else if (method == "l1") {
                _reg_method = RegL1;
            } else if (method == "l2") {
                _reg_method = RegL2;
            } else {
                LOG_ERROR("Illegal REGULARIZATION_METHOD : %s", method.c_str());
                _reg_method = RegNone;
            }
            LOG_NOTICE("REGULARIZATION_METHOD: %s", reg_method_str[_reg_method]);

            _reg_weight = conf.conf_float_default(section, "regularization_weight", 0.02);
            LOG_NOTICE("regularization_weight: %f", _reg_weight);

            _early_stop_N = conf.conf_int_default(section, "early_stop_n", -1);
            LOG_NOTICE("early_stop_N: %d", _early_stop_N);

            method = conf.conf_str_default(section, "uniform_method", "pre");
            if (method == "none") {
                _uniform_method = NoUniform;
            } else if (method == "pre") {
                _uniform_method = PreUniform;
            } else if (method == "online") {
                _uniform_method = OnlineUniform;
            } else {
                LOG_NOTICE("Illegal uniform method [%s] given", method.c_str());
                _uniform_method = PreUniform;
            }
            const char* uniform_method_str[] = {"NoUniform", "PreUniform", "OnlineUniform"};
            LOG_NOTICE("uniform method: %s", uniform_method_str[_uniform_method]);

            _min_loss_diff = conf.conf_float_default(section, "min_loss_diff", 1e-6);
            LOG_NOTICE("min_loss_diff=%f", _min_loss_diff);

            _use_shrink = conf.conf_int_default(section, "use_shrink", 0);
            _shrink_limit = conf.conf_int_default(section, "shrink_limit", 0);
            LOG_NOTICE("use_shrink:%d shrink_limit:%d", _use_shrink, _shrink_limit);

            _middle_dump_dir = conf.conf_str_default(section, "middle_dump_dir", "");
            _middle_dump_interval = conf.conf_int_default(section, "middle_dump_interval", -1);
            if (_middle_dump_interval>=1) {
                LOG_NOTICE("in turn dump model: output_dir=[%s], interval=%d", _middle_dump_dir.c_str(), _middle_dump_interval);
            }
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
                int ind;
                float value;
                fscanf(stream, "%d:%f\n", &ind, &value);
                _theta.w[ind] = value;
            }
            _continuous_training = true;
        }

        virtual void init(IReader_t* reader) {
            IterModel_t::init(reader);

            if (_continuous_training) {
                LOG_NOTICE("ContinuousTrining: will skip theta initialization and uniform intialization.");

            } else {
                _theta_num = reader->dim();
                LOG_NOTICE("theta_dim = %d", _theta_num);

                _theta.set(_theta_num);
                for (int i=0; i<_theta_num; ++i) {
                    _theta.w[i] = random_05();
                }
                _theta.b = random_05();

                if (_uniform_method != NoUniform) {
                    LOG_NOTICE("Begin to stat uniform infomation.");
                    _uniform.stat(reader);
                    LOG_NOTICE("Stat over.");
                } else {
                    LOG_NOTICE("Skip uniform!");
                }
            } 

            // init theta decay.
            _decay.set(_theta_num);

            // preprocess:
            //   - uniform.
            if (_uniform_method == PreUniform) {
                LOG_NOTICE("Begin to pre-uniform.");
                int thread_num = 11;
                UniformerJob_t jobs[thread_num];
                PCPool_t<Instance_t> *pool = new PCPool_t<Instance_t>(2000000);
                const char* temp_lr_file = "temp_lr_preprocess.bin";
                FILE* output_file = fopen(temp_lr_file, "wb");
                ThreadData_t<FILE*> *out = new ThreadData_t<FILE*>(output_file);
                for (int i=0; i<thread_num; ++i) {
                    jobs[i].reader = NULL;
                    jobs[i].pool = pool;
                    jobs[i].job_id = i;
                    jobs[i].output_fp = out;
                    jobs[i].uniformer = &_uniform;
                } 
                jobs[0].reader = reader;
                reader->reset();

                multi_thread_jobs(thread_uniformer, jobs, thread_num, thread_num);

                LOG_NOTICE("End preprocess.");

                fclose(output_file);
                delete out;
                delete pool;

                _reader = new BinaryReader_t(temp_lr_file);
                _reader->reset();
            }

            if (_use_momentum) {
                _velo.set(_theta_num);
            }
        }

    private:

        bool    _continuous_training;
        Param_t _theta;
        int     _theta_num;

        UniformMethod_t    _uniform_method;
        MeanStdvar_Uniform _uniform;

        Param_t _best_theta;
        float   _best_loss;
        int     _best_round;

        LearnRateAdjustMethod_t _adjust_method;
        Param_t _decay;

        // optima.
        Param_t _velo;
        float   _momentum_ratio;
        bool    _use_momentum;

        int     _early_stop_N;
        float   _original_rate;
        float   _min_loss_diff;

        // shrink in N.
        bool    _use_shrink;
        int     _shrink_limit;
        int     _shrink_times;


        // profile timer.
        Timer  _predict_tm;
        Timer  _calc_tm;
        Timer  _uniform_tm;
        Timer  _total_update_tm;

        // dump model on training process.
        string  _middle_dump_dir;
        int     _middle_dump_interval;

        RegularizationMethod_t  _reg_method;
        float                   _reg_weight;

        float predict_no_uniform(const Instance_t& uniformed_item) const {
            return sigmoid( sparse_dot(_theta, uniformed_item.features) );
        }

        virtual void _epoch_end() {
            LOG_NOTICE("TimeUpdateUsed=%f (uniform=%f pred=%f calc=%f)", 
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
            LOG_NOTICE("          loss_inc=%.8f min_loss=%f exp_loss_inc=%f",
                    _best_loss - _epoch_loss,
                    _min_loss,
                    expect_loss_inc);

            // check improvement type.
            bool improvement = false;
            bool big_improvement = false;
            int no_progress_in_N = -1;
            if (_epoch_loss < _best_loss - _min_loss_diff) {
                improvement = true;
                if (_epoch_loss < _best_loss - expect_loss_inc) {
                    big_improvement = true;
                }
            } else {
                no_progress_in_N = _iter_round - _best_round;
                LOG_NOTICE("No progress in %d iteration(s)", no_progress_in_N)
            }

            // accept or reject the parameters.
            if (improvement) {
                _best_loss = _epoch_loss;
                _best_theta = _theta;
                _best_round = _iter_round;
            } else {
                LOG_NOTICE("Reject theta update.");
                if (_early_stop_N>=0 && no_progress_in_N > _early_stop_N) {
                    LOG_NOTICE("NO_PROGRESS_IN_N[%d] > EARLY_STOP_N[%d], stop!!",
                            no_progress_in_N, _early_stop_N);
                    _force_stop = true;
                }
            }

            // shrink learn_rate.
            if (!improvement && _use_shrink) {
                _shrink_times ++;
                LOG_NOTICE("Shrink! %f -> %f (%d/%d times)", _learn_rate, _learn_rate*0.5, _shrink_times, _shrink_limit);
                _learn_rate *= 0.5;
                if (_shrink_times > _shrink_limit) {
                    _force_stop = true;
                }
            }

            if (_middle_dump_interval >= 1) {
                if (_iter_round % _middle_dump_interval == 0 && _iter_round>0) {
                    char filename_buffer[128];
                    snprintf(filename_buffer, sizeof(filename_buffer), "%s/epoch.%ld.model", _middle_dump_dir.c_str(), _iter_round);
                    FILE* out_fp = fopen(filename_buffer, "w");
                    write_model(out_fp);
                    fclose(out_fp);
                }
            }
        }

        virtual void _train_end() {
            LOG_NOTICE("Replace theta with best_param at loss@%f", _best_loss);
            _theta = _best_theta;
        }

        virtual void _train_begin() {
            _original_rate = _learn_rate;
        }


        Updatable_t* _new_updatable_object() {
            LogitSolver* solver = new LogitSolver(
                    _theta_num, 
                    _theta,
                    _uniform_method,
                    &_uniform,
                    _learn_rate,
                    _adjust_method,
                    _decay,
                    _reg_method,
                    _reg_weight);
            return solver;
        }

        void _join_updatable(Updatable_t** updatable, size_t num) {
            _theta.b = 0;
            for (size_t i=0; i<_theta.sz; ++i) {
                _theta.w[i] = 0;
            }

            for (size_t i=0; i<num; ++i) {
                LogitSolver* solver = (LogitSolver*)updatable[i];
                // inc param. 
                _theta += solver->_theta;

                // inc update_times.
                _decay.b = max(solver->_decay.b, _decay.b);
                for (int i=0; i<_theta_num; ++i) {
                    _decay.w[i] = max(solver->_decay.w[i], _decay.w[i]);
                }
                delete solver;
            }

            _theta.b /= num;
            for (size_t i=0; i<_theta.sz; ++i) {
                _theta.w[i] /= num;
            }
        }
};

#endif  //__LOGIT_H__

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
