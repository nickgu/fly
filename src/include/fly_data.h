/***************************************************************************
 * 
 * Copyright (c) 2015 Baidu.com, Inc. All Rights Reserved
 * 
 **************************************************************************/
 
 
 
/**
 * @file fly_data.h
 * @author gusimiu(com@baidu.com)
 * @date 2015/07/08 12:13:38
 * @brief 
 *  
 **/

#ifndef  __FLY_DATA_H_
#define  __FLY_DATA_H_

#include <cmath>

#include <vector>
#include <stdexcept>

#include <stdint.h>
#include <cstdlib>
#include <cstring>
using namespace std;
#include <ext/hash_map>
using namespace __gnu_cxx;

#include "helper.h"

/*
 * feature type 
 * and corresponding binary file storage method:
 *      Feature_Normal: sizeof(float)
 *      Feature_Sparse: sizeof(uint32_t),sizeof(float)
 */
enum BinaryFeatureType_t {
    BF_Normal = 0,  // normal feature, float type.
    BF_Sparse,      // index and value.
    BF_Binary,      // {0, 1}
    BF_SparseBinary // {0, 1} and the amount of 1 is very little.
};


#if 0
class FileMeta_t {
    public:
        void write(FILE* stream) const {
            fwrite(&FileMeta_t::HeaderMagic, sizeof(FileMeta_t::HeaderMagic), 1, stream);
            int lemma_num = _kv.size();
            fwrite(&lemma_num, sizeof(lemma_num), stream);

            for (map<string, string>::const_iterator iter = _kv.begin(); iter!=_kv.end(); ++iter) {
                _write_str(stream, iter->first);
                _write_str(stream, iter->second);
            }
        }

        /*
         * return true if reade meta ok.
         */
        bool read(FILE* stream) {
            int header;
            fread(&header, sizeof(header), 1, stream);
            if (header == HeaderMagic) {
                int lemma_num;
                fread(&lemma_num, sizeof(lemma_num), 1, stream);
                for (int i=0; i<lemma_num; ++i) {
                    key = _read_str(stream);
                    value = _read_str(stream);
                    _kv[key] = value;
                }
                return true;
            } else {
                LOG_ERROR("Bad header for filemeta.");
                return false;
            }
        }

        bool get(const string& key, string& out_value) const {
            if (_kv.find(key)!=_kv.end()) {
                out_value = _kv[key];
                return true;
            }
            return false;
        }

        void set(const string& key, const string& value) {
            _kv[key] = value;
        }

    private:
        const static int HeaderMagic = 0xDEADDEEF;
        map<string, string> _kv;

        string _read_str(FILE* stream) {
            int sz;
            fread(sz, sizeof(int), 1, stream);
            char buf = new char[sz + 1];
            fread(buf, 1, sz, stream);
            string s(buf);
            delete [] buf;
            return s;
        }

        void _write_str(FILE* stream, const string& s) {
            int sz = (int)s.length();
            fwrite(sz, sizeof(int), 1, stream);
            fwrite(s.c_str(), 1, s.length(), stream);
        }
};
#endif

struct IndValue_t {
    int index;
    float value;
};

struct Instance_t {
    float label;
    FArray_t<IndValue_t> features;

    Instance_t(int feature_extend_size=32):
        features(feature_extend_size)
    {}

    void write(FILE* stream) const {
        fprintf(stream, "%f", label);
        for (size_t i=0; i<features.size(); ++i) {
            fprintf(stream, " %d:%f", features[i].index, features[i].value);
        }
        fprintf(stream, "\n");
    }

    bool parse_item_no_index(char* line, const char* sep=" ") {
        if (*line == 0) {
            return false;
        }
        vector<string> flds;
        split(line, sep, flds);

        char* end_ptr;
        float f = strtod(flds[0].c_str(), &end_ptr);
        label = int(f+0.5);
        features.clear();
        for (size_t i=1; i<flds.size(); ++i) {
            IndValue_t iv;
            iv.index = i;
            iv.value = strtod(flds[i].c_str(), &end_ptr);
            features.push_back(iv);
        }
        return true;
    }

    bool parse_item(char *line) {
        if (*line == 0) {
            return false;
        }

        char* end_ptr;
        label = strtod(line, &end_ptr);
        if (end_ptr == line) {
            LOG_ERROR("parse_item() failed: label_parse: [%s]", end_ptr);
            return false;
        }
        features.clear();

        size_t begin = 0;
        IndValue_t iv;
        iv.index = -1;
        bool index_illegal = false;
        for (size_t i=0; line[i]; ++i) {
            if (line[i] == ':') {
                iv.index = strtol(line + begin, &end_ptr, 10);
                if (end_ptr == line+begin) {
                    LOG_ERROR("parse_item() failed: index_parse: [%s]", line+begin);
                    return false;
                }

                begin = i+1;
                index_illegal = true;
            }
            if (line[i] == ' ' || line[i] == '\t' || line[i]=='\n' || line[i]=='\r') {
                if (!index_illegal) {
                    begin = i + 1;
                    continue;
                }
                iv.value = strtod(line + begin, &end_ptr);
                if (end_ptr == line+begin || (end_ptr!=NULL && *end_ptr!=' ' && *end_ptr!='\n' && *end_ptr!='\t' && *end_ptr!='\r')) {
                    LOG_ERROR("parse_item() failed: value_parse: index=%d [%s]", iv.index, line+begin);
                    return false;
                }

                //LOG_NOTICE("%d:%f", iv.index, iv.value);
                features.push_back(iv);
                iv.index = -1;
                index_illegal = false;
                begin = i + 1;
            }
        }
        return true;
    }

    /*
     * write to binary file.
     * return offset of file.
     */
    size_t write_binary(FILE* stream) {
        size_t offset = ftell(stream);
        fwrite(&label, 1, sizeof(label), stream);
        features.write(stream);
        return offset;
    }

    void read_binary(FILE* stream) {
        fread(&label, 1, sizeof(label), stream);
        features.read(stream);
        return ;
    }
};


/*
 * Interface of feature file reader.
 */
class IReader_t {
    public:
        virtual size_t size() const = 0;
        virtual size_t processed_num() const = 0;
        virtual size_t dim() const = 0;
        virtual int percentage() const = 0;
        virtual void set(const char* filename) = 0;

        /**
         *  reset the stream.
         */
        virtual void reset() = 0;

        /**
         *  usage:
         *      FeatureReader_t r(stream);
         *      Instance_t item;
         *      while (reader.read(&item)) {
         *          // do what you need to do.
         *      }
         */ 
        virtual bool read(Instance_t* item) = 0;
};

#if 0
class DenseReader_t:
    IReader_t
{
    public:
        virtual size_t size() const {
        }

        virtual size_t processed_num() const {
        }

        virtual size_t dim() const {
        }

        virtual int percentage() const = 0;
        virtual void set(const char* filename) {
        }

        /**
         *  reset the stream.
         */
        virtual void reset() = 0;

        /**
         *  usage:
         *      FeatureReader_t r(stream);
         *      Instance_t item;
         *      while (reader.read(&item)) {
         *          // do what you need to do.
         *      }
         */ 
        virtual bool read(Instance_t* item) = 0;
};
#endif

class BinaryFeatureReader_t 
    : public IReader_t
{
    public:
        BinaryFeatureReader_t(const char* filename=NULL):
            _cur_id(0),
            _size(0),
            _theta_num(0)
        {
            if (filename != NULL) {
                set(filename);
            }
        }

        virtual ~BinaryFeatureReader_t() {
            if (_stream) {
                fclose(_stream);
            }
        }

        virtual size_t size() const { 
            if (!_is_stat) {
                throw std::runtime_error("Access size or dim before stat.");
            }
            return _size; 
        }
        virtual size_t processed_num() const { return _cur_id; }
        virtual size_t dim() const {
            if (!_is_stat) {
                throw std::runtime_error("Access size or dim before stat.");
            }
            return _theta_num;
        }
        virtual int percentage() const {
            return int(_cur_id * 100.0f / _size);
        }

        virtual void set(const char* filename) {
            _is_stat = false;
            LOG_NOTICE("BinaryReader open [%s]", filename);
            _stream = fopen(filename, "rb");
            if (_stream == NULL) {
                throw std::runtime_error(string("Cannot open file : ") + string(filename));
            }

            if (strcmp(filename, "/dev/stdin") != 0) {
                stat();
            } else {
                LOG_NOTICE("Input is /dev/stdin. Streming ignore stat.");
            }
        }

        void stat() {
            if (_is_stat) {
                return ;
            }
            _is_stat = true;
            LOG_NOTICE("DO_STAT ON BINARY FILE.");
            fseek(_stream, 0, SEEK_END);
            size_t total_file_size = ftell(_stream);
            int percentage = 0;

            _cur_id = 0;
            _size = 0;
            _theta_num = 0;
            LOG_NOTICE("Preprocess(Stat theta_num and item_num)..");
            fseek(_stream, 0, SEEK_SET);
            while ( !feof(_stream) ) {
                Instance_t new_item;
                new_item.read_binary(_stream);

                // stat _size and _theta_num
                _size ++;
                for (size_t i=0; i<new_item.features.size(); ++i) {   
                    int idx = new_item.features[i].index;
                    if (idx >= _theta_num) {
                        _theta_num = idx + 1;
                    }
                }

                size_t process_size = ftell(_stream);
                int cur_per = int(process_size * 100.0f / total_file_size);
                if (cur_per > percentage) {
                    percentage = cur_per;
                    fprintf(stderr, "%cPreprocessing complete %d%% [%d/%d mb] [%lu record(s)]", 
                            13, percentage, process_size>>20, total_file_size>>20, _size);
                    fflush(stderr);
                }
            }
            fprintf(stderr, "\n");
            LOG_NOTICE("processed: %llu records. theta_num=%d", _size, _theta_num);
            reset();
        }

        virtual void reset() {
            fseek(_stream, 0, SEEK_SET);
            _cur_id = 0;
        }

        virtual bool read(Instance_t* item) {
            if ( !feof(_stream) ) {
                item->read_binary(_stream);
                _cur_id ++;
                return true;
            } else {
                return false;
            }
        }

    private:
        FILE*   _stream;
        size_t  _cur_id;
        size_t  _size;  // total record num.
        int     _theta_num;
        bool    _is_stat;
};

/**
 *  Feature file reader.
 */
class FeatureReader_t 
    : public IReader_t
{
    public:
        FeatureReader_t(const char* filename=NULL, size_t roll_size=5000):
            _roll_size(roll_size),
            _cur_id(0),
            _theta_num(0),
            _buffer(8192),
            _no_index_format(false)
        {
            if (filename != NULL) {
                set(filename);
            }
        }

        virtual ~FeatureReader_t() {
            if (_stream) {
                fclose(_stream);
            }
        }

        void set_no_index(bool b) {
            _no_index_format = b;
        }

        virtual size_t size() const { return _buffer.size(); }
        virtual size_t processed_num() const { return _cur_id; }
        virtual size_t dim() const {return _theta_num;}
        virtual int percentage() const {
            return int(_cur_id * 100.0f / _buffer.size());
        }

        virtual void set(const char* filename) {
            LOG_NOTICE("TextFeatureReader open [%s]", filename);
            //_preprocess = preprocess;
            _stream = fopen(filename, "r");
            if (_stream == NULL) {
                throw std::runtime_error(string("Cannot open file : ") + string(filename));
            }

            fseek(_stream, 0, SEEK_END);
            size_t total_file_size = ftell(_stream);
            fseek(_stream, 0, SEEK_SET);
            int percentage = 0;

            // load all data into memory.
            _cur_id = 0;
            _buffer.clear();
            _theta_num = 0;
            char line[MaxLineLength];
            size_t c = 0;
            while (fgets(line, sizeof(line), _stream)) {
                Instance_t new_item;
                if (_no_index_format) {
                    new_item.parse_item_no_index(line);
                } else {
                    new_item.parse_item(line);
                }
                _buffer.push_back(new_item);

                for (size_t i=0; i<new_item.features.size(); ++i) {   
                    int idx = new_item.features[i].index;
                    if (idx >= _theta_num) {
                        _theta_num = idx + 1;
                    }
                }

                c++;
                size_t process_size = ftell(_stream);
                int cur_per = int(process_size * 100.0f / total_file_size);
                if (cur_per > percentage) {
                    percentage = cur_per;
                    fprintf(stderr, "%cLoading complete %d%% (%d/%d kb)", 
                            13, percentage, process_size/1024, total_file_size/1024);
                    fflush(stderr);
                }
            }
            fprintf(stderr, "\n");
            LOG_NOTICE("record_num=%d, dim=%d", c, _theta_num);
            fclose(_stream);
            _stream = NULL;
        }

        virtual void reset() {
            //fseek(_stream, 0, SEEK_SET);
            _cur_id = 0;
        }

        virtual bool read(Instance_t* item) {
            if (_cur_id >= _buffer.size()) {
                // read none.
                return false;
            }

            *item = _buffer[_cur_id ++];
            return true;
        }

    private:
        static const size_t MaxLineLength = 40960;

        FILE*   _stream;
        size_t  _roll_size;
        size_t  _cur_id;

        int     _theta_num;
        FArray_t<Instance_t> _buffer;

        bool    _no_index_format;
};


#endif  //__FLY_DATA_H_

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
