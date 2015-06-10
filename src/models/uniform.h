/***************************************************************************
 * 
 * Copyright (c) 2015 Baidu.com, Inc. All Rights Reserved
 * 
 **************************************************************************/
 
 
 
/**
 * @file models/uniform.h
 * @author gusimiu(com@baidu.com)
 * @date 2015/05/20 20:23:34
 * @brief 
 *  
 **/

#ifndef  __UNIFORM_H_
#define  __UNIFORM_H_

class MeanStdvar_Uniform {
    public:
        MeanStdvar_Uniform() :
            _dim_num(0),
            _min(NULL),
            _max(NULL)
        {
        }

        ~MeanStdvar_Uniform() {
            if (_min) {
                delete [] _min;
                _min = NULL;
            }
            if (_max) {
                delete [] _max;
                _max = NULL;
            }
        }

        void stat(FlyReader_t* reader) {
            _dim_num = reader->dim();
            _min = new float[_dim_num];
            _max = new float[_dim_num];
            memset(_min, 0, sizeof(float)*_dim_num);
            memset(_max, 0, sizeof(float)*_dim_num);
            size_t n = 0;

            reader->reset();
            Instance_t item;
            while (reader->read(&item)) {
                n ++;
                if (n % 1000000 == 0) {
                    fprintf(stderr, "%cUniformStat: %d item(s)", 13, n);
                }
                for (size_t i=0; i<item.features.size(); ++i) {
                    size_t fid = item.features[i].index;
                    float value = item.features[i].value;
                    if (fid >= 0 && fid < _dim_num) {
                        _min[fid] = min(_min[fid], value);
                        _max[fid] = max(_max[fid], value);
                    }
                }
            }
            fprintf(stderr, "\n");
        }

        void read(FILE* stream) {
            if (_min) {
                delete [] _min;
                _min = NULL;
            }
            if (_max) {
                delete [] _max;
                _max = NULL;
            }

            fscanf(stream, "%d\n", &_dim_num); 
            _min = new float[_dim_num];
            _max = new float[_dim_num];
            memset(_min, 0, sizeof(float)*_dim_num);
            memset(_max, 0, sizeof(float)*_dim_num);

            for (size_t i=0; i<_dim_num; ++i) {
                fscanf(stream, "%f:%f\n", &_min[i], &_max[i]); 
            }
            //debug();
        }

        void write(FILE* stream) const {
            fprintf(stream, "%d\n", _dim_num); 
            for (size_t i=0; i<_dim_num; ++i) {
                fprintf(stream, "%f:%f\n",_min[i], _max[i]); 
            }
        }

        void uniform(Instance_t* out, const Instance_t& in) const {
            *out = in;
            for (size_t i=0; i<out->features.size(); ++i) {
                IndValue_t& iv = out->features[i];
                if (iv.index < (int)_dim_num) {
                    // min-max.
                    float mn = _min[iv.index];
                    float mx = _max[iv.index];
                    if (mx > mn) {
                        float v = iv.value;
                        if (v>mx) v=mx;
                        else if (v<mn) v=mn; 
                        iv.value = (v - mn) / (mx - mn);
                    } else {
                        iv.value = 0; // never seen this feature.
                    }
                }
            }
        }

        void self_uniform(Instance_t* in_out) const {
            for (size_t i=0; i<in_out->features.size(); ++i) {
                IndValue_t& iv = in_out->features[i];
                if (iv.index < (int)_dim_num) {
                    // min-max.
                    float mn = _min[iv.index];
                    float mx = _max[iv.index];
                    if (mx > mn) {
                        float v = iv.value;
                        if (v>mx) v=mx;
                        else if (v<mn) v=mn; 
                        iv.value = (v - mn) / (mx - mn);
                    } else {
                        iv.value = 0; // never seen this feature.
                    }
                }
            }
        }


        void debug() {
            // debug code.
            for (size_t fid=0; fid<_dim_num; ++fid) {
                LOG_NOTICE("uniform: fid=%d min=%.4f max=%.4f", 
                        fid, _min[fid], _max[fid]);
            }
        }

    private:
        size_t _dim_num;
        float* _min;
        float* _max;
};


#endif  //__UNIFORM_H_

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
