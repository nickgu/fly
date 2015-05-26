/**
 * @file iter.h
 * @author nickgu
 * @date 2015/01/16 12:13:41
 * @brief 
 *  
 **/

#ifndef  __ITER_H_
#define  __ITER_H_

#include "../cfg.h"
#include "../fly_core.h"
#include "../fly_measure.h"

class IterModel_t: 
    public FlyModel_t
{
    public:
        IterModel_t (const Config_t& config, const char* section)
        {
            _config = &config;

            _iter_num = config.conf_int_default(section, "iter_num", 200);
            LOG_NOTICE("_iter_num=%d", _iter_num);

            _learn_rate = config.conf_float_default(section, "learn_rate", 0.1);
            LOG_NOTICE("_learn_rate=%f", _learn_rate);

            _min_loss = config.conf_float_default(section, "min_loss", 0.005);
            LOG_NOTICE("_min_loss=%f", _min_loss);

            _final_loss_diff = config.conf_float_default(section, "final_loss_diff", 0.01);
            LOG_NOTICE("_final_loss_diff=%f", _final_loss_diff);
        }

        virtual ~IterModel_t() 
        {}

        virtual void init(FlyReader_t* reader) {
            _reader = reader;
            _reader->reset();
        }

        virtual void train() {
            float last_loss = 1e10;
            _force_stop = false;
            _train_begin();
            for (int i=0; i<_iter_num; ++i) {
                _iter_round = i + 1;
                float loss = _epoch();
                LOG_NOTICE("round %d: loss=%f", i+1, loss);

                if (_force_stop) {
                    LOG_NOTICE("model force stop!");
                    break;
                }
                if (loss < _min_loss) {
                    LOG_NOTICE("min loss reached, break!");
                    break;
                }
                float diff_loss = last_loss - loss;
                if ( diff_loss > -1e-6 && diff_loss < _final_loss_diff ) {
                    LOG_NOTICE("loss delta is too small, break!");
                    break;
                } 
                last_loss = loss;
            }
            _train_end();
        }

    protected:
        int     _iter_num;
        float   _learn_rate;
        float   _min_loss;
        float   _final_loss_diff;
        bool    _force_stop;
        size_t  _iter_round;

        const Config_t*  _config;
        FlyReader_t* _reader;

        /**
         * Interface.
         *  return loss.
         */
        virtual float _update(const Instance_t& item) = 0;

        virtual void _train_begin() {};
        virtual void _train_end() {};
        virtual void _epoch_begin() {};
        virtual void _epoch_end() {};
        float _epoch_loss;

        float _epoch() {
            int last_per = 0;
            float loss = 0;
            _reader->reset();
            _epoch_begin();
            Instance_t item;
            while (_reader->read(&item)) {
                loss += _update(item);

                // output training progress.
                int per = _reader->percentage();
                if (per - last_per >= 5) {
                    fprintf(stderr, "%cIteration:%d%% [%u/%u line(s)]", 
                            13, per, _reader->processed_num(), _reader->size());
                    fflush(stderr);
                    last_per = per;
                }
            }
            fprintf(stderr, "\n");
            loss = loss / _reader->size();
            _epoch_loss = loss;
            _epoch_end();
            loss = _epoch_loss;
            return loss; 
        }
};


#endif  //__ITER_H_

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
