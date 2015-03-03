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

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <model> <test_file>\n\n", argv[0]);
        return -1;
    }

    const char* model_name = argv[1];
    const char* test_file_name = argv[2];

    FlyReader_t* test_data_reader  = NULL;
    test_data_reader = new FeatureReader_t(test_file_name);
    Config_t nil_config;
    GBDT_t *model = new GBDT_t(nil_config, "");

    FILE* model_file = fopen(model_name, "r");
    model->read_model(model_file);
    fclose(model_file);
 
    // simple test on training set.
    FlyReader_t *treader = test_data_reader;
    treader->reset();
    Instance_t item;
    FArray_t<ResultPair_t> ans_list;
    while (treader->read(&item)) {
        float ret = model->predict(item);
        printf("%f\n", ret);

        ans_list.push_back(ResultPair_t(item.label, ret));
    }

    LOG_NOTICE("count: %u", ans_list.size());
    LOG_NOTICE("auc: %.4f", calc_auc(ans_list.size(), ans_list.buffer()));

    if (test_data_reader) {
        delete test_data_reader;
        test_data_reader = NULL;
    }
    delete model;
    return 0;
}

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
