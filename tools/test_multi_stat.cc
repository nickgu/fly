/***************************************************************************
 * 
 * Copyright (c) 2015 Baidu.com, Inc. All Rights Reserved
 * 
 **************************************************************************/
 
 
 
/**
 * @file test_multi_stat.cc
 * @author gusimiu(com@baidu.com)
 * @date 2015/05/27 18:06:38
 * @brief 
 *  
 **/

#include "fly_core.h"
#include "helper.h"

#include <pthread.h>

/*
 * Producer and Customer Poot
 * Condition: _p_id + 1 != _c_id
 * one empty cell to make validation.
 */
template <typename T>
class PCPool_t {
    public:
        PCPool_t(size_t buffer_size) {
            _buffer_size = buffer_size;
            _buffer = new T[_buffer_size];
            _c_id = 0;
            _p_id = 0;
            _flag_putting = true;
            pthread_spin_init(&_spinlock, 0);
        }
        ~PCPool_t() {
            if (_buffer) {
                delete [] _buffer;
                _buffer_size = 0;
            }
        }

        // Producer put item.
        void put(const T& item) {
            // try util ok.
            while (1) {
                size_t next_id = (_p_id + 1) % _buffer_size;
                if (next_id == _c_id) {
                    // full: need to wait for putting.
                    continue;
                }
                _buffer[_p_id] = item;
                _p_id = next_id;
                // unlock.
                return ;
            }
        }

        // Customer try to get.
        // loop util get.
        // return false if nothing to process forever.
        bool get(T* out_item) {
            // retry until work or full.
            while (1) {
                pthread_spin_lock(&_spinlock);
                if (_c_id == _p_id) { // empty or stop.
                    pthread_spin_unlock(&_spinlock);
                    // need to wait for processing.
                    if (!_flag_putting) {
                        return false;
                    }
                    continue;
                }
                size_t m = _c_id;
                _c_id = (_c_id + 1) % _buffer_size;
                *out_item = _buffer[m];
                // unlock.
                pthread_spin_unlock(&_spinlock);
                return true;
            }
        }

        void set_putting(bool putting) {
            _flag_putting = putting;
        }

    private:
        T*      _buffer;
        size_t  _buffer_size;
        size_t  _c_id;
        size_t  _p_id;
        bool    _flag_putting;

        pthread_spinlock_t _spinlock;
};

struct Context_t {
    int id;
    PCPool_t<Instance_t>* pool;
    const char* filename;
};

void* stat(void* context) {
    Context_t* con = (Context_t*)context;
    int counter = 0;
    if (con->filename) {
        LOG_NOTICE("Thread[%d] is reader.", con->id);
        BinaryFeatureReader_t reader(con->filename);
        Instance_t item;
        while (reader.read(&item)) {
            con->pool->put(item);
            counter ++;
            if (counter % 10000 ==0) {
                LOG_NOTICE("writer.%d c=%d", con->id, counter);
            }
        }   
        con->pool->set_putting(false);
    } else {
        LOG_NOTICE("Thread[%d] is processor.", con->id);
        Instance_t item;
        while (con->pool->get(&item)) {
            counter ++;
        }
        LOG_NOTICE("Thread[%d] counter=%d", con->id, counter);
    }
    return NULL;
}


int main(int argc, const char** argv) {
    PCPool_t<Instance_t> pool(1000000);
    int thread_num = 10;
    Context_t* jobs = new Context_t[thread_num];
    for (int i=0; i<thread_num; ++i) {
        jobs[i].id = i;
        jobs[i].pool = &pool;
        jobs[i].filename = NULL;
    }
    jobs[0].filename = argv[1];
    multi_thread_jobs(stat, jobs, thread_num, thread_num);
}

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
