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

struct Context_t {
    int id;
    PCPool_t<Instance_t>* pool;
    const char* filename;
    
    int dim_count;
    int counter;
};

void* stat(void* context) {
    Context_t* con = (Context_t*)context;
    con->counter = 0;
    con->dim_count = 0;
    if (con->filename) {
        LOG_NOTICE("Thread[%d] is reader.", con->id);
        BinaryFeatureReader_t reader(con->filename);
        while (1) {
            Instance_t *item = con->pool->begin_put();
            if (!reader.read(item)) {
                con->pool->end_put(false);
                break;
            }

            con->pool->end_put();
            con->counter ++;
            if (con->counter % 1000000 ==0) {
                LOG_NOTICE("writer.%d c=%d pool_status=%d,%d", 
                        con->id, 
                        con->counter,
                        con->pool->num_put(),
                        con->pool->num_get());
            }
        }   
        con->pool->set_putting(false);
    } else {
        LOG_NOTICE("Thread[%d] is processor.", con->id);
        Instance_t item;
        while (con->pool->get(&item)) {
            con->counter ++;
            for (size_t i=0; i<item.features.size(); ++i) {
                if ( item.features[i].index >= con->dim_count) {
                    con->dim_count = item.features[i].index + 1;
                }
            }
        }
        LOG_NOTICE("Thread[%d] counter=%d", con->id, con->counter);
    }

    return NULL;
}


int main(int argc, const char** argv) {
    PCPool_t<Instance_t> pool(10000000);
    int thread_num = atoi(argv[2]);
    Context_t* jobs = new Context_t[thread_num];
    LOG_NOTICE("Threadnum=%d", thread_num);
    for (int i=0; i<thread_num; ++i) {
        jobs[i].id = i;
        jobs[i].pool = &pool;
        jobs[i].filename = NULL;
        jobs[i].counter = 0;
        jobs[i].dim_count = 0;
    }
    jobs[0].filename = argv[1];

    Timer tm;
    tm.begin();
    multi_thread_jobs(stat, jobs, thread_num, thread_num);

    int total_item = 0;
    int feature_count = 0;
    for (int i=1; i<thread_num; ++i) {
        total_item += jobs[i].counter;
        feature_count = max(feature_count, jobs[i].dim_count);
    }
    LOG_NOTICE("Final stat: total_item=%d feature_count=%d", total_item, feature_count);
    tm.end();
    LOG_NOTICE("TIME = %.4f", tm.cost_time());
}

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
