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

float test_auc(FlyReader_t* treader, GBDT_t* model) {
    treader->reset();
    Instance_t item;
    FArray_t<ResultPair_t> ans_list;
    while (treader->read(&item)) {
        float ret = model->predict(item);
        ans_list.push_back(ResultPair_t(item.label, ret));
    }

    float auc = calc_auc(ans_list.size(), ans_list.buffer());
    return auc;
}

int main(int argc, char** argv) {
    if (argc != 3 && argc!=5) {
        fprintf(stderr, "Usage: %s <model> <test_file> [<tree_interval> <tree_total>]\n\n", argv[0]);
        return -1;
    }

    int test_interval = -1;
    int test_count = -1;
    if (argc==5) {
        test_interval = atoi(argv[3]);
        test_count = atoi(argv[4]);
        LOG_NOTICE("TestInterval: interval=%d total=%d", test_interval, test_count);
    }

    const char* model_name = argv[1];
    const char* test_file_name = argv[2];

    FlyReader_t* test_data_reader  = NULL;
    test_data_reader = new BinaryFeatureReader_t(test_file_name);
    Config_t nil_config;
    GBDT_t *model = new GBDT_t(nil_config, "");

    FILE* model_file = fopen(model_name, "r");
    model->read_model(model_file);
    fclose(model_file);
 
    // simple test on training set.
    FlyReader_t *treader = test_data_reader;
    float auc = 0;
    if (test_interval>0) {
        for (int T=test_interval; T<=test_count; T+=test_interval) {
            model->set_predict_tree_cut(T);
            auc = test_auc(treader, model);
            LOG_NOTICE("INTERVAL_TEST\t%d\t%.4f", T, auc);
        }
    } else {
        auc = test_auc(treader, model);
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
