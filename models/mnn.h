/**
 * @file mnn.h
 * @author nickgu
 * @date 2015/01/16 12:05:47
 * @brief 
 *  
 **/

#ifndef  __MNN_H_
#define  __MNN_H_

#include "../fly_math.h"
#include "iter.h"

class MultiNN_t :
    public IterModel_t
{
    public:
        MultiNN_t(const Config_t& conf, const char* section):
            IterModel_t(conf, section)
        {
            _layer_num = conf.conf_int_default(section, "layer_num", 2);
            _layer_width = conf.conf_int_default(section, "layer_width", 4);
            LOG_NOTICE("Layer_num: %d", _layer_num);
            LOG_NOTICE("Layer_width: %d", _layer_width);
        }

        virtual void init(FlyReader_t* reader) {
            // base process.
            IterModel_t::init(reader);
            _init_net(reader->dim(), _layer_num, _layer_width);
            _learn_rate_in_use = _learn_rate;
        }

        virtual float predict(const Instance_t& item) const {
            // layer_1:
            for (size_t i=0; i<_out_num[0]; ++i) {
                _out[0][i] = sigmoid( sparse_dot(_input_num, _theta[0], item.features) + _const[0][i] );
            }
            
            // layer_2~all.
            for (size_t l=1; l<_layer_num; ++l) {
                for (size_t i=0; i<_out_num[l]; ++i) {
                    _out[l][i] = sigmoid( vec_dot(_out_num[l-1], _theta[l], _out[l-1]) + _const[l][i] );
                }
            }
            return _out[_layer_num - 1][0];
        }

        virtual void write_model(FILE* stream) const {
            for (size_t L=0; L<_layer_num; ++L) {
                for (size_t i=0; i<_out_num[L]; ++i) {
                    fprintf(stream, "L:%d,O:%d:\t", L, i);
                    fprintf(stream, "%f\t", _const[L][i]);
                    int in_num = _reader->dim();
                    if (L>0) {
                        in_num = _out_num[L-1];
                    }

                    for (int j=0; j<in_num; ++j) {
                        fprintf(stream, "%f,", _ref_theta(L, i, j) );
                    }
                    fprintf(stream, "\n");
                }
            }
            return ;
        }

        virtual void read_model(FILE* stream) {
            // TODO.
            return ;
        }

    private:
        size_t  _input_num;
        float** _theta;
        float** _const;
        float** _out;
        float** _delta; 
        size_t* _out_num;

        float** _bak_theta;
        float** _bak_const;

        size_t  _layer_num;
        size_t  _layer_width;

        float _current_loss;
        float _learn_rate_in_use;

    private:
        void _train_begin() {
            _current_loss = 1.0;
        }

        void _train_end() {
            _recover_state();
        }

        void _epoch_begin() {
            _save_state();
        }

        void _epoch_end() {
            if (_epoch_loss >= _current_loss) {
                LOG_NOTICE("loss[%f] >= current_loss[%f], lrate=%f RECOVER.", _epoch_loss, _current_loss, _learn_rate_in_use);
                _recover_state();
                _learn_rate_in_use *= 0.5;
                if (_learn_rate_in_use < 1e-6) {
                    _force_stop = true;
                }
            } else {
                LOG_NOTICE("loss[%f] < current_loss[%f], lrate=%f ACCEPTED.", _epoch_loss, _current_loss, _learn_rate_in_use);
                _current_loss = _epoch_loss;
                _learn_rate_in_use = _learn_rate;
            }
        }

        void _save_state() {
            for (size_t i=0; i<_layer_num; ++i) {
                size_t in = _layer_width;
                size_t out = _layer_width;
                if (i == 0) {
                    in = _input_num;
                }
                if (i+1 == _layer_num) {
                    out = 1;
                }

                memcpy(_bak_theta[i], _theta[i], sizeof(float) * in * out);
                memcpy(_bak_const[i], _const[i], sizeof(float) * out);
            }
        }

        void _recover_state() {
            for (size_t i=0; i<_layer_num; ++i) {
                size_t in = _layer_width;
                size_t out = _layer_width;
                if (i == 0) {
                    in = _input_num;
                }
                if (i+1 == _layer_num) {
                    out = 1;
                }

                memcpy(_theta[i], _bak_theta[i], sizeof(float) * in * out);
                memcpy(_const[i], _bak_const[i], sizeof(float) * out);
            }
        }

        void _release() {
            _input_num = 0;
            if (_theta) {
                for (size_t i=0; i<_layer_num; ++i) {
                    delete [] _theta[i];
                    delete [] _bak_theta[i];

                    delete [] _out[i];
                    delete [] _delta[i];

                    delete [] _const[i];
                    delete [] _bak_const[i];
                }
                delete [] _theta;
                delete [] _bak_theta;
            }

            if (_const) {
                delete [] _const;
                delete [] _bak_const;
            }
            if (_delta) {
                delete [] _delta;
            }
            if (_out) {
                delete [] _out;
            }

            _const = NULL;
            _bak_const = NULL;

            _theta = NULL;
            _bak_theta = NULL;

            _out = NULL;
            _delta = NULL;

            if (_out_num) {
                delete [] _out_num;
                _out_num = NULL;
            }
            _layer_num = 0;
        }

        void _init_net(size_t input_num, size_t layer_num, size_t layer_width) {
            _release();

            _layer_num = layer_num;
            _input_num = input_num;
            _out_num = new size_t [_layer_num];
            _theta = new float* [_layer_num];
            _bak_theta = new float* [_layer_num];
            _out   = new float* [_layer_num];
            _delta = new float* [_layer_num];
            _const = new float* [_layer_num];
            _bak_const = new float* [_layer_num];
            for (size_t i=0; i<_layer_num; ++i) {
                size_t in = layer_width;
                size_t out = layer_width;
                if (i == 0) {
                    in = input_num;
                }
                if (i+1 == _layer_num) {
                    out = 1;
                }

                _out_num[i] = out;
                _theta[i] = new float [in * out];
                for (size_t j=0; j<in*out; ++j) {
                    _theta[i][j] = random_05();
                } 
                _bak_theta[i] = new float [in * out];
                _const[i] = new float[out];
                _bak_const[i] = new float[out];
                for (size_t j=0; j<out; ++j) {
                    _const[i][j] = random_05();
                }
                _out[i] = new float[out];
                _delta[i] = new float[out];

                LOG_NOTICE("net_structure: L:%d theta_num:%d const:%d", i, in*out, out);
            }
            _out_num[layer_num-1] = 1;
        }

        float& _ref_theta(size_t layer, size_t out_idx, size_t in_idx) {
            return _theta[layer][in_idx * _out_num[layer] + out_idx];
        }

        const float& _ref_theta(size_t layer, size_t out_idx, size_t in_idx) const {
            return _theta[layer][in_idx * _out_num[layer] + out_idx];
        }

        virtual float _update(const Instance_t& item) {
            // forward predict, record output data.
            float new_p = predict(item);

            /*
            item.write(stderr);
            LOG_NOTICE("target=%f output=%f", item.label, new_p);
            */

            // back-probagation.
            _delta[_layer_num-1][0] = (item.label - new_p);
            for (int L=_layer_num-1; L>=0; --L) {
                // reset previous layer delta.
                if (L>0) {
                    memset(_delta[L-1], 0, sizeof(float) * _out_num[L-1]);
                }

                for (size_t O=0; O<_out_num[L]; ++O) {
                    float desc = _delta[L][O];
                    //LOG_NOTICE("update @L%d,U%d desc=%f, out=%f", L, O, desc, _out[L][O]);
                    _const[L][O] += desc * _learn_rate_in_use;
                    // update const.
                    if (L == 0) {
                        for (size_t i=0; i<item.features.size(); ++i) {
                            float in = item.features[i].value;
                            float gradient = desc * in;
                            _ref_theta(0, O, item.features[i].index) += gradient * _learn_rate_in_use;
                        }
                    } else {
                        for (size_t in_idx=0; in_idx<_out_num[L-1]; ++in_idx) {
                            float in = _out[L-1][in_idx];
                            float gradient = desc * in;

                            float old_theta = _ref_theta(L, O, in_idx);
                            
                            /*
                            LOG_NOTICE("delta [%d][%d] : %f -> %f (%f)", 
                                    L-1, in_idx,
                                    _delta[L-1][in_idx], 
                                    _delta[L-1][in_idx] + desc * _ref_theta(L, O, in_idx), 
                                    desc * _ref_theta(L, O, in_idx));
                                    */

                            _delta[L-1][in_idx] += desc * _ref_theta(L, O, in_idx);
                            _ref_theta(L, O, in_idx) += gradient * _learn_rate_in_use;

                            //LOG_NOTICE(" theta: %f -> %f", old_theta, _ref_theta(L, O, in_idx));
                        }
                    }
                }

                //getchar();
            }

            // calc loss.
            float loss = -(item.label * safe_log(new_p) + (1-item.label) * safe_log(1-new_p));
            return loss;
        }
};

#endif  //__MNN_H_

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */




