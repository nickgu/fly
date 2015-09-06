/**
 * @file fly_data.cc
 * @author nickgu
 * @date 2015/09/06 12:05:20
 * @brief 
 **/
#include <fly_data.h>


BinaryReader_t::
BinaryReader_t(const char* filename):
    _cur_id(0),
    _size(0),
    _theta_num(0)
{
    if (filename != NULL) {
        set(filename);
    }
}

BinaryReader_t::
~BinaryReader_t() {
    if (_stream) {
        fclose(_stream);
    }
}

size_t 
BinaryReader_t::size() const { 
    if (!_is_stat) {
        throw std::runtime_error("Access size or dim before stat.");
    }
    return _size; 
}

size_t 
BinaryReader_t::dim() const {
    if (!_is_stat) {
        throw std::runtime_error("Access size or dim before stat.");
    }
    return _theta_num;
}

int 
BinaryReader_t::percentage() const {
    return int(_cur_id * 100.0f / _size);
}

void 
BinaryReader_t::set(const char* filename) {
    _is_stat = false;
    LOG_NOTICE("BinaryReader_t open [%s]", filename);
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

void 
BinaryReader_t::stat() {
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

void 
BinaryReader_t::reset() {
    fseek(_stream, 0, SEEK_SET);
    _cur_id = 0;
}

bool 
BinaryReader_t::read(Instance_t* item) {
    if ( !feof(_stream) ) {
        item->read_binary(_stream);
        _cur_id ++;
        return true;
    } else {
        return false;
    }
}


TextReader_t::
TextReader_t(const char* filename, size_t roll_size):
    _roll_size(roll_size),
    _cur_id(0),
    _theta_num(0),
    _feature_mode(TFM_AutoDetected),
    _buffer(8192),
    _compact_buffer(8192)
{
    if (filename != NULL) {
        set(filename);
    }
}

TextReader_t::~TextReader_t() {}

void 
TextReader_t::set(const char* filename) {
    LOG_NOTICE("TextFeatureReader open [%s] mode=%d", filename, _feature_mode);
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
    _compact_buffer.clear();
    _theta_num = 0;
    char line[MaxLineLength];
    size_t c = 0;

    while (fgets(line, sizeof(line), _stream)) {
        if (_feature_mode == TFM_AutoDetected) {
            _feature_mode = auto_detect_mode(line);
        }

        if (_feature_mode == TFM_IndValue) {
            Instance_t new_item;
            new_item.parse_item(line);
            _buffer.push_back(new_item);
            for (size_t i=0; i<new_item.features.size(); ++i) {   
                int idx = new_item.features[i].index;
                if (idx >= _theta_num) {
                    _theta_num = idx + 1;
                }
            }

        } else if (_feature_mode == TFM_Values) {
            CompactInstance_t compact_item(_theta_num);
            size_t s = compact_item.parse_item(line, " ");
            if (_theta_num==0) {
                _theta_num = s;
                LOG_NOTICE("ModeValues : read first line and get theta_num = %d", _theta_num);
            }
            _compact_buffer.push_back(compact_item);
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

void 
TextReader_t::reset() {
    //fseek(_stream, 0, SEEK_SET);
    _cur_id = 0;
}

int 
TextReader_t::percentage() const { 
    if (__use_buffer()) {
        return int(_cur_id * 100.0f / _buffer.size()); 
    } else {
         return int(_cur_id * 100.0f / _compact_buffer.size()); 
    }
}

size_t 
TextReader_t::size() const { 
    if (__use_buffer()) {
        return _buffer.size(); 
    } else {
        return _compact_buffer.size();
    }
}

bool 
TextReader_t::read(Instance_t* item) {
    if (__use_buffer()) {
        if (_cur_id >= _buffer.size()) {
            // read none.
            return false;
        }

        *item = _buffer[_cur_id ++];
    } else {
        if (_cur_id >= _compact_buffer.size()) {
            // read none.
            return false;
        }

        _compact_buffer[_cur_id ++].convert_to_instance(item);
    }
    return true;
}

TextReader_t::TextFeatureMode_t
TextReader_t::auto_detect_mode(const char* line) {
    /*
     * case:
     *   contain ':'  => IndValue.
     *   contain '.' or 'e' or '-' => float inside => Values.
     *   other case   => Ind_1 
     */
    bool is_indvalue = false;
    bool is_values = false;
    const char*p = line;
    while (*p) {
        if (*p == ':') {
            is_indvalue = true;
        } else if (*p == '.' || *p == 'e' || *p == '-') {
            is_values = true;
        }
        p ++;
    }

    if (is_indvalue) {
        LOG_NOTICE("-->> AutoDetectFeatureMode: IndValue <<---");
        return TFM_IndValue;

    } else if (is_values) {
        LOG_NOTICE("-->> AutoDetectFeatureMode: Values <<---");
        return TFM_Values;

    } else {
        LOG_NOTICE("-->> AutoDetectFeatureMode: Ind_1 <<---");
        return TFM_Ind_1;
    }
}


/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
