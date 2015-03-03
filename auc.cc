/***************************************************************************
 * 
 * Copyright (c) 2015 Baidu.com, Inc. All Rights Reserved
 * 
 **************************************************************************/
 
 
 
/**
 * @file multi_auc.cc
 * @author gusimiu(com@baidu.com)
 * @date 2015/01/27 17:39:08
 * @brief 
 *  
 **/

#include "fly_core.h"
#include "fly_measure.h"

int main() {
    char line[1024];
    FArray_t<ResultPair_t> res_list;
    while (fgets(line, sizeof(line), stdin)) {
        ResultPair_t res;
        res.target = atof(line);
        for (int i=0; line[i]; ++i) {
            if (line[i] == '\t') {
                res.output = atof(line+i+1);
                break;
            }
        }
        res_list.push_back(res);
    }
    LOG_NOTICE("LOAD_OVER. size=%d", res_list.size());

    float auc = calc_auc(res_list.size(), res_list.buffer());
    fprintf(stdout, "auc:%.3f\n", auc);
    return 0;
}


/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
