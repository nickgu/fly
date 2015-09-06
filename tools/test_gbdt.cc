/**
 * @file fly.cpp
 * @author nickgu
 * @date 2015/01/13 14:55:16
 * @brief 
 *  
 **/

#include <getopt.h>

#include <vector>

#include "helper.h"
#include "cfg.h"
#include "fly_core.h"
#include "fly_measure.h"

#include "all_models.h"

struct TestJob_t {
    int job_id;
    PCPool_t<Instance_t>* pool;
    IReader_t* reader;
    FArray_t<ResultPair_t> ans_list;
    GBDT_t* model;

    ThreadData_t<FILE*>* binary_output;
    bool output_path;
    bool output_mean;
    int base_dim;
    int tree_node_count;
};

void* thread_test(void* c) {
    TestJob_t& job = *(TestJob_t*)c;
    int tree_count = job.model->get_predict_tree_cut();
    int* leaves = new int [tree_count];
    float* means = new float [tree_count];
    float* buffer = new float[tree_count];

    if (job.reader) {
        LOG_NOTICE("test_gbdt: thread[%d] : I am a reader.", job.job_id);
        size_t c = 0;
        while (1) {
            Instance_t* cell = job.pool->begin_put();
            if (!job.reader->read(cell)) {
                job.pool->end_put(false);
                break;
            }
            if (c % 2000000 == 0) {
                LOG_NOTICE("complete %d puts.", c);
            }
            job.pool->end_put();
            c ++;
        }
        LOG_NOTICE("test_gbdt: reader: load over. count=%d", c);
        job.pool->set_putting(false);
    } else {
        LOG_NOTICE("test_gbdt: thread[%d] : I am a worker.", job.job_id);
        job.ans_list.clear();
        Instance_t item(1200);
        while (job.pool->get(&item)) {
            float ans;
            if (job.binary_output) {
                ans = job.model->predict_and_get_leaves(item, leaves, means, buffer);
                // make it sparse.
                for (int i=0; i<tree_count; ++i) {
                    IndValue_t iv; 
                    iv.index = job.base_dim + i*job.tree_node_count + leaves[i];
                    if (job.output_mean) {
                        iv.value = means[i];
                    } else {
                        iv.value = 1.0;
                    }
                    if (!job.output_mean && job.output_path) {
                        int l = leaves[i];
                        while (l) {
                            l = (l-1)/2;
                            IndValue_t temp_iv; 
                            temp_iv.index = job.base_dim + i*job.tree_node_count + l;
                            temp_iv.value = 1.0;
                            LOG_NOTICE("%d:%f", temp_iv.index, temp_iv.value);
                            item.features.push_back(temp_iv);
                        }
                    }
                    item.features.push_back(iv);
                }
                FILE* output_fp = job.binary_output->borrow();
                item.write_binary(output_fp);
                job.binary_output->give_back();
            } else {
                ans = job.model->predict(item);
            }
            job.ans_list.push_back(ResultPair_t(item.label, ans));
        }

        LOG_NOTICE("thread[%d] : over. %d processed.", job.job_id, job.ans_list.size());
    }
    delete [] buffer;
    delete [] leaves;
    delete [] means;
    return NULL;
}

float test_auc(IReader_t* treader, GBDT_t* model, FILE* dump_feature_binary_file, bool output_mean, bool output_path, int thread_num) {
    // 1 reader + N workers.
    thread_num += 1;
    TestJob_t* jobs = new TestJob_t[thread_num];
    PCPool_t<Instance_t> pool(2000000);
    ThreadData_t<FILE*>* output_data = NULL;
    if (dump_feature_binary_file) {
        output_data = new ThreadData_t<FILE*>(dump_feature_binary_file);
    }
    int base_dim = treader->dim();
    int tree_node_count = model->tree_node_count();

    for (int i=0; i<thread_num; ++i) {
        jobs[i].pool = &pool;
        jobs[i].job_id = i;
        jobs[i].reader = NULL;
        jobs[i].model = model;

        jobs[i].binary_output = output_data;
        jobs[i].output_mean = output_mean;
        jobs[i].output_path = output_path;
        jobs[i].base_dim = base_dim;
        jobs[i].tree_node_count = tree_node_count;
    }
    jobs[0].reader = treader;

    multi_thread_jobs(thread_test, jobs, thread_num, thread_num);
    LOG_NOTICE("thread work is over.");
    FArray_t<ResultPair_t> total_list;
    for (int i=0; i<thread_num; ++i) {
        for (size_t j=0; j<jobs[i].ans_list.size(); ++j) {
            total_list.push_back( jobs[i].ans_list[j] );
        }
    }
    LOG_NOTICE("merge ans list over.");
    float auc = calc_auc(total_list.size(), total_list.buffer());
    if (output_data) {
        delete [] output_data;
    }
    return auc;
}

int main(int argc, char** argv) {
    if (argc <= 2) {
        fprintf(stderr, "Usage: %s <model> <test_file> [<tree_interval> <tree_total>] -D[binary_output] -C[tree_cut] [-m] [-p] [-tN default=5]\n\n", argv[0]);
        fprintf(stderr, "  -m : output mean, other wise output 0/1.\n");
        fprintf(stderr, "  -p : if -Doutput_file is set, output the path-info.\n");
        fprintf(stderr, "  -t : thread num.\n");
        return -1;
    }

    int thread_num = 5;
    int tree_cut = 0;
    bool output_mean = false;
    bool output_path = false;
    for (int i=1; i<argc; ++i) {
        if ( strcmp(argv[i], "-m")==0 ) {
            output_mean = true;
            LOG_NOTICE("Output node mean.");
        }
        if ( strcmp(argv[i], "-p")==0 ) {
            output_path = true;
            LOG_NOTICE("Output path-in-tree feature.");
        }
        if ( strstr(argv[i], "-C")!=NULL ) {
            tree_cut = atoi(argv[i]+2);
            LOG_NOTICE("TreeCut: %d", tree_cut);
        }

        if (strstr(argv[i], "-t")!=NULL) {
            thread_num = atoi(argv[i]+2);
            LOG_NOTICE("ThreadNum: %d", thread_num);
        }
    }

    int test_interval = -1;
    int test_count = -1;
    if (argc>=5) {
        test_interval = atoi(argv[3]);
        test_count = atoi(argv[4]);
        LOG_NOTICE("TestInterval: interval=%d total=%d", test_interval, test_count);
    }

    FILE* dump_feature_binary_file = NULL;
    for (int i=1; i<argc; ++i) {
        if ( strstr(argv[i], "-D")!=NULL ) {
            dump_feature_binary_file = fopen(argv[i]+2, "wb");
            if (dump_feature_binary_file == NULL) {
                LOG_ERROR("open feature file [%s] to write failed!", argv[i]+2);
                return -1;
            } else {
                LOG_NOTICE("Dump feature mode. outfile=[%s]", argv[i]+2);
            }
        }
    }


    const char* model_name = argv[1];
    const char* test_file_name = argv[2];

    IReader_t* test_data_reader  = NULL;
    test_data_reader = new TextReader_t(test_file_name);
    Config_t nil_config;
    GBDT_t *model = new GBDT_t(nil_config, "");

    FILE* model_file = fopen(model_name, "r");
    model->read_model(model_file);
    fclose(model_file);
 
    // simple test on training set.
    IReader_t *treader = test_data_reader;
    float auc = 0;
    if (test_interval>0) {
        for (int T=test_interval; T<=test_count; T+=test_interval) {
            model->set_predict_tree_cut(T);
            auc = test_auc(treader, model, dump_feature_binary_file, output_mean, output_path, thread_num);
            LOG_NOTICE("INTERVAL_TEST\t%d\t%.4f", T, auc);
        }
    } else {
        if (tree_cut) {
            model->set_predict_tree_cut(tree_cut);
        }
        auc = test_auc(treader, model, dump_feature_binary_file, output_mean, output_path, thread_num);
        LOG_NOTICE("auc: %.4f", auc);
    }

    if (test_data_reader) {
        delete test_data_reader;
        test_data_reader = NULL;
    }
    delete model;
    return 0;
}

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
