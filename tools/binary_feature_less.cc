/***************************************************************************
 * 
 * Copyright (c) 2015 Baidu.com, Inc. All Rights Reserved
 * 
 **************************************************************************/
 
 
 
/**
 * @file binary_feature_less.cc
 * @author gusimiu(com@baidu.com)
 * @date 2015/05/21 12:01:07
 * @brief 
 *  
 **/
#include "fly_core.h"

int main(int argc, const char** argv) {
    BinaryReader_t reader;
    reader.set(argv[1]);
    int d = reader.dim();
    int s = reader.size();
    LOG_NOTICE("num=%d dim=%d", s, d);
    Instance_t item;
    reader.reset();
    while (reader.read(&item)) {
        item.write(stdout);
    }
    return 0;
}

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
