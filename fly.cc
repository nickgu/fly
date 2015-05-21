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
        -c --config    : configs. \n\
                            use -S [default=fly] to config Fly itself. \n\
                            use -s [default=modelname] to config model infomation. \n\
        -s --section   : which section will be used when config is loaded. \n\
        -t --transform : output binary file, if this set, other training will be ignored. \n\
        -d --debug     : debug mode. \n\
        -H -h --help   : show this help. \n\
\n"); 
}

int main(int argc, char** argv) {
    srand((int)time(0));

    int opt;
    char* opt_string = "dS:L:hHf:M:o:c:s:t:bT:";
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
        {"debug", required_argument, NULL, 'd'},
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
        io.transform(input_file, output_binary_file);
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
    } else if (strcmp(model_name, "cglr") == 0) {
        model = new CG_LogisticRegression_t(model_config, config_section);
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


    FlyReader_t* train_data_reader = NULL;
    FlyReader_t* test_data_reader  = NULL;
    if ( binary_mode ) {
        if (input_file) {
            train_data_reader = new BinaryFeatureReader_t(input_file);  
        }
        if (test_file) {
            test_data_reader = new BinaryFeatureReader_t(test_file);
        }

    } else {
        if (input_file) {
            train_data_reader = new FeatureReader_t();
            train_data_reader->set(input_file);
        }
        if (test_file) {
            test_data_reader = new FeatureReader_t();
            test_data_reader->set(test_file);
        }
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
 
    // simple test.
    FlyReader_t *treader = NULL;
    if (test_data_reader != NULL) {
        LOG_NOTICE("Training over. Begin to test!");
        treader = test_data_reader;
        treader->reset();
        int c = 0;
        Instance_t item;
        ResultPair_t* res_list = new ResultPair_t [treader->size()];
        FILE* out_fd = NULL;
        if (output_file) {
            out_fd = fopen(output_file, "w");
        }
        Timer tm;
        tm.begin();
        while (treader->read(&item)) {
            //item.write(stderr);
            res_list[c].target = item.label;
            res_list[c].output = model->predict(item);
            if (out_fd) {
                fprintf(out_fd, "%f\t%f\n", res_list[c].target, res_list[c].output);
            }
            c++;
        }
        tm.end();
        LOG_NOTICE("performance: %.3f sec, qps=%.2f", tm.cost_time(), c*1.0/tm.cost_time());
        if (out_fd) {
            fclose(out_fd);
        }
        LOG_NOTICE("auc: %.6f", calc_auc(c, res_list));
        LOG_NOTICE("logMLE: %.6f", calc_log_mle(c, res_list));
        float rmse = calc_rmse(c, res_list);
        LOG_NOTICE("rmse: %f, mse=%f", rmse, rmse*rmse);
        LOG_NOTICE("confussion: %s", calc_confussion_matrix(c, res_list).str().c_str());

        int error = calc_error(c, res_list);
        LOG_NOTICE("error: %d/%d (%.2f%%)", error, c, error*100.0/c);

        delete [] res_list;
        delete test_data_reader;
        test_data_reader = NULL;
    }

    if (train_data_reader) {
        delete train_data_reader;
        train_data_reader = NULL;
    }
    delete model;
    return 0;
}

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
