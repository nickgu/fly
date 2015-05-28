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
            _mean(NULL),
            _stdvar(NULL),
            _min(NULL),
            _max(NULL)
        {
        }

        ~MeanStdvar_Uniform() {
            if (_mean) {
                delete [] _mean;
                _mean = NULL;
            }
            if (_stdvar) {
                delete [] _stdvar;
                _stdvar = NULL;
            }
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
            _mean = new float[_dim_num];
            _stdvar = new float[_dim_num];
            _min = new float[_dim_num];
            _max = new float[_dim_num];
            memset(_mean, 0, sizeof(float)*_dim_num);
            memset(_stdvar, 0, sizeof(float)*_dim_num);
            memset(_min, 0, sizeof(float)*_dim_num);
            memset(_max, 0, sizeof(float)*_dim_num);
            size_t n = 0;

            reader->reset();
            Instance_t item;
            while (reader->read(&item)) {
                n ++;
                for (size_t i=0; i<item.features.size(); ++i) {
                    size_t fid = item.features[i].index;
                    float value = item.features[i].value;
                    if (fid >= 0 && fid < _dim_num) {
                        _mean[fid] = _mean[fid] * ((n-1.0)/n) + value / n; 

                        _min[fid] = min(_min[fid], value);
                        _max[fid] = max(_max[fid], value);
                    }
                }
            }

            reader->reset();
            n = 0;
            int perc = 0;
            while (reader->read(&item)) {
                n ++;
                for (size_t i=0; i<item.features.size(); ++i) {
                    size_t fid = item.features[i].index;
                    float value = item.features[i].value;
                    if (fid >= 0 && fid < _dim_num) {
                        value = fabs(value - _mean[fid]);
                        _stdvar[fid] = _stdvar[fid] * ((n-1.0)/n) + value / n; 
                    }
                }

                int cur = reader->percentage();
                if (cur > perc) {
                    perc = cur;
                    fprintf(stderr, "%cProcessed: %d%%", 13, cur);
                }
            }
            fprintf(stderr, "\n");
            //debug();
        }

        void read(FILE* stream) {
            if (_mean) {
                delete [] _mean;
                _mean = NULL;
            }
            if (_stdvar) {
                delete [] _stdvar;
                _stdvar = NULL;
            }
            if (_min) {
                delete [] _min;
                _min = NULL;
            }
            if (_max) {
                delete [] _max;
                _max = NULL;
            }

            fscanf(stream, "%d\n", &_dim_num); 
            _mean = new float[_dim_num];
            _stdvar = new float[_dim_num];
            _min = new float[_dim_num];
            _max = new float[_dim_num];
            memset(_mean, 0, sizeof(float)*_dim_num);
            memset(_stdvar, 0, sizeof(float)*_dim_num);
            memset(_min, 0, sizeof(float)*_dim_num);
            memset(_max, 0, sizeof(float)*_dim_num);

            for (size_t i=0; i<_dim_num; ++i) {
                fscanf(stream, "%f:%f:%f:%f\n", &_mean[i], &_stdvar[i],
                        &_min[i], &_max[i]); 
            }
            //debug();
        }

        void write(FILE* stream) const {
            fprintf(stream, "%d\n", _dim_num); 
            for (size_t i=0; i<_dim_num; ++i) {
                fprintf(stream, "%f:%f:%f:%f\n", _mean[i], _stdvar[i],
                        _min[i], _max[i]); 
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
                    }

                    // avg-stddev.
                    /*
                    iv.value -= _mean[iv.index];
                    if (_stdvar[iv.index]>0) {
                        iv.value /= _stdvar[iv.index];
                    }*/
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
                    }
                }
            }
        }


        void debug() {
            // debug code.
            for (size_t fid=0; fid<_dim_num; ++fid) {
                LOG_NOTICE("stdvar_uniform: fid=%d mean=%.4f stdvar=%.4f min=%.4f max=%.4f", 
                        fid, _mean[fid], _stdvar[fid],
                        _min[fid], _max[fid]);
            }
        }

    private:
        size_t _dim_num;
        float* _mean;
        float* _stdvar;
        float* _min;
        float* _max;
};


#endif  //__UNIFORM_H_

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
