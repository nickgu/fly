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
#include "fly_data.h"

typedef hash_map<int, float> ParamDict_t;

class FlyModel_t {
    public:
        virtual float predict(const Instance_t& ins) const = 0;
        virtual void  write_model(FILE* stream) const = 0;
        virtual void  read_model(FILE* stream) = 0;
        virtual void  init(IReader_t* reader) = 0;
        virtual void  train() = 0;
};

class BinaryFileIO_t {
    public:
        BinaryFileIO_t() {}
        ~BinaryFileIO_t() {}

        void transform(const char* input_file, const char* output_file, bool no_index_format, const char* seperator) {
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
                bool ret = false;
                if (no_index_format) {
                    ret = new_item.parse_item_no_index(_line_buffer, seperator);
                } else {
                    ret = new_item.parse_item(_line_buffer);
                }
                if (!ret) {
                    continue;
                }
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


#endif  //__FLY_CORE_H_

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
