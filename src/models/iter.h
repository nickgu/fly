/**
 * @file iter.h
 * @author nickgu
 * @date 2015/01/16 12:13:41
 * @brief 
 *  
 **/

#ifndef  __ITER_H_
#define  __ITER_H_

#include "cfg.h"
#include "fly_core.h"
#include "fly_measure.h"

void* update_thread(void* c);

class IterModel_t;

// interface of Updatable
class Updatable_t {
    public:
        virtual ~Updatable_t() {};
        virtual float update(Instance_t& item) = 0;
};

struct JobUpdate_t {
    int job_id;
    PCPool_t<Instance_t>* pool;
    FlyReader_t* reader;
    Updatable_t* updatable;
    double total_loss;
};

class IterModel_t: 
    public FlyModel_t
{
    public:
        friend void* update_thread(void*);

        IterModel_t (const Config_t& config, const char* section)
        {
            _config = &config;

            _iter_num = config.conf_int_default(section, "iter_num", 200);
            LOG_NOTICE("_iter_num=%d", _iter_num);

            _learn_rate = config.conf_float_default(section, "learn_rate", 0.1);
            LOG_NOTICE("_learn_rate=%f", _learn_rate);

            _min_loss = config.conf_float_default(section, "min_loss", 0.005);
            LOG_NOTICE("_min_loss=%f", _min_loss);

            _cache_size = config.conf_int_default(section, "cache_size", 2000000);
            LOG_NOTICE("_cache_size=%d", _cache_size);

            _thread_num = config.conf_int_default(section, "thread_num", 11);
            LOG_NOTICE("_thread_num=%d", _thread_num);
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
                _epoch_timer.clear();
                _epoch_timer.begin();
                float loss = _epoch();
                _epoch_timer.end();
                LOG_NOTICE("ROUND[%d] LOSS=[[ %f ]] EPOCH_TIMER=%.3f(s)", i+1, loss, _epoch_timer.cost_time());

                if (_force_stop) {
                    LOG_NOTICE("model force stop!");
                    break;
                }
                if (loss < _min_loss) {
                    LOG_NOTICE("min loss reached, break!");
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
        bool    _force_stop;
        size_t  _iter_round;
        size_t  _cache_size;
        int     _thread_num;

        const Config_t*  _config;
        FlyReader_t* _reader;
        Timer _epoch_timer;

        virtual void _train_begin() {};
        virtual void _train_end() {};
        virtual void _epoch_begin() {};
        virtual void _epoch_end() {};
        float _epoch_loss;

        // factory function.
        virtual Updatable_t* _new_updatable_object() { 
            LOG_ERROR("Directly call Iter::_new_updatable_object()");
            return NULL;
        }
        virtual void _join_updatable(Updatable_t**, size_t num) { /* do nothing. */ }

        float _epoch() {
            PCPool_t<Instance_t>* ppool = new PCPool_t<Instance_t>(_cache_size);
            JobUpdate_t *jobs = new JobUpdate_t[_thread_num];
            Updatable_t** updatables = new Updatable_t*[_thread_num-1];

            for (int i=0; i<_thread_num; ++i) {
                jobs[i].job_id = i;
                if (i == 0) {
                    jobs[i].reader = _reader;
                } else {
                    jobs[i].reader = NULL;
                }

                if (i != 0) {
                    jobs[i].updatable = _new_updatable_object();
                    updatables[i-1] = jobs[i].updatable;
                }
                jobs[i].pool = ppool;
                jobs[i].total_loss = 0;
            }

            _reader->reset();
            _epoch_begin();

            multi_thread_jobs(update_thread, jobs, _thread_num, _thread_num);

            double loss = 0;
            for (int i=1; i<_thread_num; ++i) {
                loss += jobs[i].total_loss;
            }
            _join_updatable(updatables, _thread_num-1);

            LOG_NOTICE("TOTAL_LOSS=%f", loss);
            loss = loss / _reader->size();
            _epoch_loss = loss;
            _epoch_end();
            loss = _epoch_loss;

            delete ppool;
            delete [] updatables;
            delete [] jobs;
            return loss; 
        }
};

void* update_thread(void* c) {
    JobUpdate_t& job = *(JobUpdate_t*)c;
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
        // end putting.
        job.pool->set_putting(false);
    } else {
        LOG_NOTICE("thread[%d] : I am a slave.", job.job_id);
        Instance_t item;
        job.total_loss = 0;
        size_t c = 0;
        while (job.pool->get(&item)) {
            job.total_loss += job.updatable->update(item);
            c ++;
        }
        LOG_NOTICE("update_thread[%d] : process over. %d item(s)", job.job_id, c);
    }
    return NULL;
}



#endif  //__ITER_H_

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
