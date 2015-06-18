/**
 * @file read.h
 * @author nickgu
 * @date 2015/01/13 12:15:28
 * @brief 
 *  
 **/

#ifndef  __FLY_CORE_H__
#define  __FLY_CORE_H__

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

typedef hash_map<int, float> ParamDict_t;

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

    bool parse_item(char *line) {
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

class FlyReader_t {
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

class FlyModel_t {
    public:
        virtual float predict(const Instance_t& ins) const = 0;
        virtual void  write_model(FILE* stream) const = 0;
        virtual void  read_model(FILE* stream) = 0;
        virtual void  init(FlyReader_t* reader) = 0;
        virtual void  train() = 0;
};

class BinaryFileIO_t {
    public:
        BinaryFileIO_t() {}
        ~BinaryFileIO_t() {}

        void transform(const char* input_file, const char* output_file) {
            FILE* stream = fopen(input_file, "r");
            FILE* output_stream = fopen(output_file, "wb");
            if (stream == NULL) {
                throw std::runtime_error(string("Cannot open file : ") + string(input_file));
            }
            if (output_stream == NULL) {
                throw std::runtime_error(string("Cannot open file to write : ") + string(output_file));
            }

            fseek(stream, 0, SEEK_END);
            size_t total_file_size = ftell(stream);
            fseek(stream, 0, SEEK_SET);
            int percentage = 0;

            // load all data into memory.
            while (fgets(_line_buffer, sizeof(_line_buffer), stream)) {
                Instance_t new_item;
                new_item.parse_item(_line_buffer);
                new_item.write_binary(output_stream);

                size_t process_size = ftell(stream);
                int cur_per = int(process_size * 100.0f / total_file_size);
                if (cur_per > percentage) {
                    percentage = cur_per;
                    fprintf(stderr, "%cPreprocessing complete %d%% (%d/%d mb)", 
                            13, percentage, process_size>>20, total_file_size>>20);
                    fflush(stderr);
                }
            }
            fprintf(stderr, "\n");

            fclose(output_stream);
            fclose(stream);
        }

    private:
        static const size_t MaxLineLength = 40960;
        char    _line_buffer[MaxLineLength];
};

class BinaryFeatureReader_t 
    : public FlyReader_t
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
    : public FlyReader_t
{
    public:
        FeatureReader_t(const char* filename=NULL, size_t roll_size=5000):
            _roll_size(roll_size),
            _cur_id(0),
            _theta_num(0),
            _buffer(8192)
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
                new_item.parse_item(line);
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
};


#endif  //__FLY_CORE_H_

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
