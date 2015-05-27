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

float test_auc(FlyReader_t* treader, GBDT_t* model, FILE* dump_feature_binary_file, bool output_mean, bool output_path) {
    treader->reset();
    Instance_t item;
    FArray_t<ResultPair_t> ans_list;
    int base_dim = treader->dim();
    int tree_node_count = model->tree_node_count();
    int percent = 0;
    int cnt = 0;

    LOG_NOTICE("Transform: base_dim=%d tree_node_count=%d", base_dim, tree_node_count);
    while (treader->read(&item)) {
        std::vector<int> leaves;
        std::vector<float> means;
        float ret;
        float node_sum = 0;
        if (dump_feature_binary_file) {
            ret = model->predict_and_get_leaves(item, &leaves, &means);
            // make it sparse.
            for (size_t i=0; i<leaves.size(); ++i) {
                IndValue_t iv; 
                iv.index = base_dim + i*tree_node_count + leaves[i];
                node_sum += means[i];
                if (output_mean) {
                    iv.value = means[i];
                } else {
                    iv.value = 1.0;
                }
                if (!output_mean && output_path) {
                    int l = leaves[i];
                    while (l) {
                        l = (l-1)/2;
                        IndValue_t temp_iv; 
                        temp_iv.index = base_dim + i*tree_node_count + l;
                        temp_iv.value = 1.0;
                        item.features.push_back(temp_iv);
                    }
                }
                item.features.push_back(iv);
            }
            item.write_binary(dump_feature_binary_file);

        } else {
            ret = model->predict(item);
        }
        ans_list.push_back(ResultPair_t(item.label, ret));

        cnt += 1;
        int p = treader->percentage();
        if (p>percent) {
            percent = p;
            fprintf(stderr, "%cProgress: %d%% (%d/%d)", 13, percent, cnt, treader->size()); 
        }
    }

    float auc = calc_auc(ans_list.size(), ans_list.buffer());
    return auc;
}

int main(int argc, char** argv) {
    if (argc <= 2) {
        fprintf(stderr, "Usage: %s <model> <test_file> [<tree_interval> <tree_total>] -D[binary_output] -C[tree_cut] [-m] [-p]\n\n", argv[0]);
        fprintf(stderr, "  -m : output mean, other wise output 0/1.\n");
        fprintf(stderr, "  -p : if -Doutput_file is set, output the path-info.\n");
        return -1;
    }

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

    FlyReader_t* test_data_reader  = NULL;
    test_data_reader = new FeatureReader_t(test_file_name);
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
            auc = test_auc(treader, model, dump_feature_binary_file, output_mean, output_path);
            LOG_NOTICE("INTERVAL_TEST\t%d\t%.4f", T, auc);
        }
    } else {
        if (tree_cut) {
            model->set_predict_tree_cut(tree_cut);
        }
        auc = test_auc(treader, model, dump_feature_binary_file, output_mean, output_path);
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
