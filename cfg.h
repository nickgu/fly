/***************************************************************************
 * 
 * Copyright (c) 2015 Baidu.com, Inc. All Rights Reserved
 * 
 **************************************************************************/
 
 
 
/**
 * @file cfg.h
 * @author gusimiu(com@baidu.com)
 * @date 2015/01/18 15:09:31
 * @brief 
 *  
 **/

#ifndef  __CFG_H_
#define  __CFG_H_

#include <cstdio>
#include <string>
#include <map>
#include <stdexcept>
using namespace std;

class Config_t {
    public:
        void load(FILE* stream) {
            string current_section;
            char line[8192];
            while (fgets(line, sizeof(line), stream)) {
                const char* key;
                const char* value;
                const char* temp_section;
                _parse_line(line, &key, &value, &temp_section);
                if (key && value) {
                    string ckey = current_section + "/" + string(key);
                    _config[ckey] = value;
                    //LOG_NOTICE("load: %s: %s", ckey.c_str(), value);
                } else if (temp_section) {
                    current_section = temp_section;
                }
            }
        }

        void load(const char* file_name) {
            FILE* fp = fopen(file_name, "r");
            if (!fp) {
                throw std::runtime_error("Cannot open config file [" + string(file_name) + "] to read.");
            }
            load(fp);
            fclose(fp);
        }

        bool conf_int(const char* section, const char* key, int* out) const {
            string ckey = string(section) + "/" + string(key);
            map<string, string>::const_iterator it = _config.find(ckey);
            if (it == _config.end()) {
                return false;
            }
            *out = atoi(it->second.c_str());
            return true;
        }

        bool conf_float(const char* section, const char* key, float* out) const {
            string ckey = string(section) + "/" + string(key);
            map<string, string>::const_iterator it = _config.find(ckey);
            if (it == _config.end()) {
                return false;
            }
            *out = atof(it->second.c_str());
            return true;
        }

        bool conf_str(const char* section, const char* key, string* out) const {
            string ckey = string(section) + "/" + string(key);
            map<string, string>::const_iterator it = _config.find(ckey);
            if (it == _config.end()) {
                return false;
            }
            *out = it->second;
            return true;
        }

        int conf_int_default(const char* section, const char* key, int default_value=-1) const {
            int ret;
            if (conf_int(section, key, &ret)) {
                return ret;
            } else {
                return default_value;
            }
        }

        float conf_float_default(const char* section, const char* key, float default_value=-1.0) const {
            float ret;
            if (conf_float(section, key, &ret)) {
                return ret;
            } else {
                return default_value;
            }
        }

        string conf_str_default(const char* section, const char* key, string default_value) const {
            string ret;
            if (conf_str(section, key, &ret)) {
                return ret;
            } else {
                return default_value;
            }
        }

    private:
        map<string, string> _config;

        void _parse_line(char* line, const char** key, const char** value, const char** section) {
            *key = NULL;
            *value = NULL;
            *section = NULL;

            char* beg = line;
            // eliminate the content after '#'
            for (size_t i=0; beg[i]; ++i) {
                if (beg[i] == '#') {
                    beg[i] = 0;
                    break;
                }
            }

            // strip line.
            size_t len = strlen(line);
            while (len>0) {
                if (line[len-1]=='\n' || line[len-1]=='\r' || line[len-1]==' ' || line[len-1]=='\t') {
                    len --;
                    line[len] = 0;
                } else {
                    break;
                }
            }

            while (*beg) {
                if (*beg==' ' || *beg=='\t') {
                    *beg ++;
                } else {
                    break;
                }
            }

            len = strlen(beg);
            if (*beg == '[' && beg[len-1] == ']') {
                beg[len-1] = 0;
                beg ++;
                *section = beg; 
                return;
            }

            for (size_t i=0; beg[i]; ++i) {
                if (beg[i] == '=') {
                    beg[i] = 0;
                    *key = beg;
                    *value = beg+i+1;
                    return ;
                }
            }
        }
};

#endif  //__CFG_H_

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
