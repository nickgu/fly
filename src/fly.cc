/**
 * @file fly.cpp
 * @author nickgu
 * @date 2015/01/13 14:55:16
 * @brief 
 *  
 **/

#include <getopt.h>

#include "helper.h"
#include "cfg.h"
#include "fly_core.h"
#include "fly_measure.h"

#include "all_models.h"

void test(IReader_t* treader, FlyModel_t* model, int thread_num, FILE* output_file);

void show_help() {
    fprintf(stderr, 
"Usage: \n\
    fly [options] \n\
        -f --file      : input training file. \n\
        -T --test      : test file. \n\
        -b --binary    : if this is set, input will use binary_reader. \n\
        -S --save      : save model to file. this config must combine with -f\n\
        -L --load      : load model from file \n\
        -M --model     : [lr, cglr, mnn, gbdt] is available, default is lr. \n\
        -o --output    : output file, output the predict result of input training data. \n\
        -N --no_index  : text input with no input. default seperator is ' ' (space)\n\
        -c --config    : configs. \n\
                            use -S [default=fly] to config Fly itself. \n\
                            use -s [default=modelname] to config model infomation. \n\
        -s --section   : which section will be used when config is loaded. \n\
        -t --transform : output binary file, if this set, other training will be ignored. \n\
        -d --debug     : debug mode. \n\
        -p --pipes     : thread num (used by test.) \n\
        -H -h --help   : show this help. \n\
\n"); 
}

int main(int argc, char** argv) {
    srand((int)time(0));

    int opt;
    char* opt_string = "dS:L:hHf:M:o:c:s:t:bT:p:N";
    static struct option long_options[] = {
        {"file", required_argument, NULL, 'f'},
        {"load", required_argument, NULL, 'L'},
        {"save", required_argument, NULL, 'S'},
        {"help", no_argument, NULL, 'h'},
        {"model", required_argument, NULL, 'M'},
        {"output", required_argument, NULL, 'o'},
        {"config", required_argument, NULL, 'c'},
        {"section", required_argument, NULL, 's'},
        {"transform", required_argument, NULL, 't'},
        {"binary", required_argument, NULL, 'b'},
        {"test", required_argument, NULL, 'T'},
        {"pipes", required_argument, NULL, 'p'},
        {"debug", required_argument, NULL, 'd'},
        {"no_index", required_argument, NULL, 'N'},
        {0, 0, 0, 0}
    };

    const char* input_file = NULL;
    const char* test_file = NULL;
    const char* output_file = NULL;
    const char* output_binary_file = NULL;
    const char* model_name = "lr";
    const char* model_save_file = NULL;
    const char* model_load_file = NULL;
    bool binary_mode = false;
    Config_t model_config;
    const char* config_section = NULL;
    int thread_num = 5;
    bool no_index = false;
    while ( (opt=getopt_long(argc, argv, opt_string, long_options, NULL))!=-1 ) {
        switch (opt) {
            case 'd':
                LOG_NOTICE("Debug is on.");
                __hidden::set_debug(true);
                break;

            case 'S':
                model_save_file = optarg;
                LOG_NOTICE("save model to [%s]", model_save_file);
                break;

            case 'L':
                model_load_file = optarg;
                LOG_NOTICE("load model from [%s]", model_load_file);
                break;

            case 'f':
                input_file = optarg;
                LOG_NOTICE("Input file: [%s]", input_file);
                break;

            case 'b':
                binary_mode = true;
                LOG_NOTICE("use binary reader to read input.");
                break;

            case 'M':
                model_name = optarg;
                LOG_NOTICE("Model: %s", model_name);
                break;

            case 'o':
                output_file=optarg;
                LOG_NOTICE("output_file: %s", output_file);
                break;

            case 'c':
                model_config.load(optarg);
                LOG_NOTICE("config file load: %s", optarg);
                break;

            case 's':
                config_section = optarg;
                LOG_NOTICE("config section: %s", config_section);
                break;

            case 't':
                output_binary_file = optarg;
                LOG_NOTICE("Transform input file to binary type to [%s]", output_binary_file);
                break;

            case 'T':
                test_file = optarg;
                LOG_NOTICE("test data [%s]", test_file);
                break;

            case 'p':
                thread_num = atoi(optarg);
                LOG_NOTICE("global-pipes: %d", thread_num);
                break;

            case 'N':
                no_index = true;
                LOG_NOTICE("text input with no index.");
                break;

            case 'h':
            case 'H':
                show_help();
                return -1;

            default:
                LOG_ERROR("unrecognized parameter: -%c", opt);
                show_help();
                return -1;
        }
    }

    if (output_binary_file != NULL) {
        BinaryFileIO_t io;
        io.transform(input_file, output_binary_file, no_index, " ");
        LOG_NOTICE("Complete file transform. program exits.");
        return 0;
    }

    if (config_section == NULL) {
        config_section = model_name;
        LOG_NOTICE("use model name as config section [%s]", config_section);
    }


    FlyModel_t *model = NULL;
    if (strcmp(model_name, "lr") == 0) {
        model = new LogisticRegression_t(model_config, config_section);
    } else if (strcmp(model_name, "mnn") == 0) {
        model = new MultiNN_t(model_config, config_section);
    } else if (strcmp(model_name, "gbdt")==0) {
        model = new GBDT_t(model_config, config_section);
    } else if (strcmp(model_name, "meta")==0) {
        model = new MetaModel_t(model_config, config_section);
    } else if (strcmp(model_name, "knn") == 0) {
        model = new KNNModel_t(model_config, config_section);
    } else {
        LOG_NOTICE("bad models.");
        return -1;
    }

    bool test_and_train_is_same = false;
    if (test_file && input_file && strcmp(test_file, input_file)==0) {
        test_and_train_is_same = true;
    }

    IReader_t* train_data_reader = NULL;
    IReader_t* test_data_reader  = NULL;
    if ( binary_mode ) {
        if (input_file) {
            train_data_reader = new BinaryFeatureReader_t(input_file);  
            ((BinaryFeatureReader_t*)train_data_reader)->stat();
        }
        if (test_file) {
            if (test_and_train_is_same) {
                test_data_reader = train_data_reader;
            } else {
                test_data_reader = new BinaryFeatureReader_t(test_file);
            }
        }

    } else {
        if (input_file) {
            train_data_reader = new FeatureReader_t();
            train_data_reader->set(input_file);
            if (no_index) {
                ((FeatureReader_t*)train_data_reader)->set_no_index(true);
            }
        }
        if (test_file) {
            if (test_and_train_is_same) {
                test_data_reader = train_data_reader;
            } else {
                test_data_reader = new FeatureReader_t();
                test_data_reader->set(test_file);
            }
        }
    }

    // load model at first.
    // which can make continuous-trainging.
    if (model_load_file) {
        LOG_NOTICE("Model load from [%s]", model_load_file);
        FILE* model_file = fopen(model_load_file, "r");
        if (!model_file) {
            LOG_ERROR("Cannot open file [%s] to read model.", model_load_file);
            return -1;
        }
        model->read_model(model_file);
        fclose(model_file);
        LOG_NOTICE("Model load completed.");
    }

    if (input_file) {
        train_data_reader->reset();
        Timer timer;

        timer.begin();
        model->init(train_data_reader);
        model->train();
        timer.end();
        LOG_NOTICE("training total time: %.3fs", timer.cost_time());
    }

    if (model_save_file) {
        // dump models.
        LOG_NOTICE("Model save to [%s]", model_save_file);
        FILE* model_file = fopen(model_save_file, "w");
        model->write_model(model_file);
        fclose(model_file);
        LOG_NOTICE("Model save completed.");
    }
 
    // simple test.
    if (test_data_reader != NULL) {
        LOG_NOTICE("Training over. Begin to test!");
        if (test_and_train_is_same) {
            test_data_reader->reset();
        }

        FILE* output_fd = NULL;
        if (output_file) {
            output_fd = fopen(output_file, "w");
            if (!output_fd) {
                LOG_ERROR("Cannot open output file [%s] to write.", output_fd);
                exit(-1);
            }
        }
        test(test_data_reader, model, thread_num, output_fd);
        if (output_fd) {
            fclose(output_fd);
        }

        if (!test_and_train_is_same) {
            delete test_data_reader;
        }
        test_data_reader = NULL;
    }

    if (train_data_reader) {
        delete train_data_reader;
        train_data_reader = NULL;
    }
    delete model;
    return 0;
}

struct TestJob_t {
    int job_id;
    PCPool_t<Instance_t>* pool;
    IReader_t* reader;
    FArray_t<ResultPair_t> ans_list;
    FlyModel_t* model;
    ThreadData_t<FILE*>* output_file;
};

void* thread_test(void* c) {
    TestJob_t& job = *(TestJob_t*)c;
    if (job.reader) {
        LOG_NOTICE("thread[%d] : I am a reader.", job.job_id);
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
        LOG_NOTICE("reader: load over. count=%d", c);
        job.pool->set_putting(false);
    } else {
        LOG_NOTICE("thread[%d] : I am a worker.", job.job_id);
        job.ans_list.clear();
        Instance_t item;
        uint32_t order_id;
        while (job.pool->get(&item, &order_id)) {
            float ans;
            ans = job.model->predict(item);
            job.ans_list.push_back(ResultPair_t(item.label, ans));

            if (job.output_file) {
                FILE* of = job.output_file->borrow();
                fprintf(of, "%f\t%f\t%d\n", item.label, ans, order_id);
                job.output_file->give_back();
            }
        }
        LOG_NOTICE("thread[%d] : over. %d processed.", job.job_id, job.ans_list.size());
    }
    return NULL;
}

void test(IReader_t* treader, FlyModel_t* model, int thread_num, FILE* output_file) {
    // 1 reader + thread_num workers.
    thread_num += 1;
    TestJob_t* jobs = new TestJob_t[thread_num];
    PCPool_t<Instance_t> pool(2000000);
    ThreadData_t<FILE*> shared_output(output_file);
    for (int i=0; i<thread_num; ++i) {
        jobs[i].pool = &pool;
        jobs[i].job_id = i;
        jobs[i].reader = NULL;
        jobs[i].model = model;
        if (output_file) {
            jobs[i].output_file = &shared_output;
        } else {
            jobs[i].output_file = NULL;
        }
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

    ResultPair_t* res_buffer = total_list.buffer();
    size_t c = total_list.size();
    LOG_NOTICE("auc: %.6f", calc_auc(c, res_buffer));
    LOG_NOTICE("logMLE: %.6f", calc_log_mle(c, res_buffer));
    float rmse = calc_rmse(c, res_buffer);
    LOG_NOTICE("rmse: %f, mse=%f", rmse, rmse*rmse);
    LOG_NOTICE("confussion: %s", calc_confussion_matrix(c, res_buffer).str().c_str());

    int error = calc_error(c, res_buffer);
    LOG_NOTICE("error: %d/%d (%.2f%%)", error, c, error*100.0/c);
}


/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
