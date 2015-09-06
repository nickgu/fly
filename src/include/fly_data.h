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

struct IndValue_t {
    int index;
    float value;

    IndValue_t() {}
    IndValue_t(int ind, float val) :
        index(ind),
        value(val)
    {}
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
 * compact-feature instance.
 * values is feature buffer. and dim info is not maintained in structure.
 * values will not be changed between its lifetime.
 */
struct CompactInstance_t {
    float label;
    float *values;
    size_t dim;

    CompactInstance_t(size_t d=0) {
        label = 0;
        dim = d;
        if (d<=0) {
            values = NULL;
        } else {
            values = (float*)malloc(sizeof(float) * d);
            memset(values, 0, sizeof(float)*d);
        }
    }
    ~CompactInstance_t() {
        if (values!=NULL) {
            free(values);
        }
        values = NULL;
        dim = 0;
    }

    CompactInstance_t(const CompactInstance_t& o) {
        clear();
        label = o.label;
        dim = o.dim;
        if (dim>0) {
            values = (float*)malloc(sizeof(float) * dim);
            memcpy(values, o.values, sizeof(float)*dim);
        }
    }

    const CompactInstance_t& operator = (const CompactInstance_t& o) {
        clear();
        label = o.label;
        dim = o.dim;
        if (dim>0) {
            values = (float*)malloc(sizeof(float) * dim);
            memcpy(values, o.values, sizeof(float)*dim);
        }
        return *this;
    }

    void clear() {
        if (values!=NULL) {
            free(values);
        }
        values = NULL;
        dim = 0;
    }

    void set_dim(size_t d) {
        if (values!=NULL) {
            free(values);
        }
        dim =d;
        if (dim<=0) {
            values = NULL;
        } else {
            values = (float*)malloc(sizeof(float) * dim);
            memset(values, 0, sizeof(float)*dim);
        }
    }

    void convert_to_instance(Instance_t* ins) {
        if (values) {
            ins->label = label;
            ins->features.clear();
            for (size_t i=0; i<dim; ++i) {
                ins->features.push_back(IndValue_t(i, values[i]));
            }
        } else {
            throw std::runtime_error("Want to convert raw CompactInstance_t to Instance_t");
        }
    }

    size_t parse_item(char* line, const char* sep=" ") {
        if (*line == 0) {
            return 0;
        }
        vector<string> flds;
        split(line, sep, flds);

        char* end_ptr;
        float f = strtod(flds[0].c_str(), &end_ptr);
        label = int(f+0.5);
        size_t d = flds.size()-1;
        if (dim>0 && d!=dim) {
            throw std::runtime_error("CompactValue parse failed! values not match in one line.");
        }
        if (values == NULL) {
            set_dim(d);
        }
        for (size_t i=1; i<flds.size(); ++i) {
            size_t ind = i-1;
            values[ind] = strtod(flds[i].c_str(), &end_ptr);
        }
        return dim;
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
         *      TextReader_t r(stream);
         *      Instance_t item;
         *      while (reader.read(&item)) {
         *          // do what you need to do.
         *      }
         */ 
        virtual bool read(Instance_t* item) = 0;
};

class BinaryReader_t 
    : public IReader_t
{
    public:
        BinaryReader_t(const char* filename=NULL);
        virtual ~BinaryReader_t();
        virtual size_t size() const;
        virtual size_t processed_num() const { return _cur_id; }
        virtual size_t dim() const;
        virtual int percentage() const;
        virtual void set(const char* filename);
        void stat();
        virtual void reset();
        virtual bool read(Instance_t* item);

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
class TextReader_t 
    : public IReader_t
{
    public:
        enum TextFeatureMode_t {
            TFM_AutoDetected = 0,
            TFM_IndValue,
            TFM_Values,
            TFM_Ind_1,
        };

        TextReader_t(const char* filename=NULL, size_t roll_size=5000);
        virtual ~TextReader_t();

        virtual size_t processed_num() const { return _cur_id; }
        virtual size_t dim() const {return _theta_num;}

        virtual size_t size() const;
        virtual int percentage() const;

        TextFeatureMode_t feature_mode() const {
            return _feature_mode;
        }
        void set_feature_mode(TextFeatureMode_t mode) {
            _feature_mode = mode;
        }

        virtual void set(const char* filename);
        virtual void reset();
        virtual bool read(Instance_t* item);

        static TextFeatureMode_t auto_detect_mode(const char* line);

    private:
        static const size_t MaxLineLength = 40960;

        FILE*   _stream;
        size_t  _roll_size;
        size_t  _cur_id;

        int     _theta_num;

        /*
         * Two types of instance.
         *  1. ind-value. (Instance_t)
         *  2. value-buffer. (CompactInstance_t)
         */
        TextFeatureMode_t    _feature_mode;
        FArray_t<Instance_t> _buffer;
        FArray_t<CompactInstance_t> _compact_buffer;

        bool __use_buffer() const {
            return _feature_mode != TFM_Values;
        }

};

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
         *      TextReader_t r(stream);
         *      Instance_t item;
         *      while (reader.read(&item)) {
         *          // do what you need to do.
         *      }
         */ 
        virtual bool read(Instance_t* item) = 0;
};
#endif

#endif  //__FLY_DATA_H_

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
