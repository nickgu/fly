/**
 * @file meta.h
 * @author nickgu
 * @date 2015/03/03 10:35:36
 * @brief 
 *  
 **/

#ifndef  __META_H_
#define  __META_H_

#include "../fly_core.h"
#include "../fly_math.h"
#include "../fly_measure.h"
#include "../cfg.h"

class MultiClassFakeReader_t 
    : public FlyReader_t 
{
    public:
        MultiClassFakeReader_t(FlyReader_t *reader, int class_id) {
            _reader = reader;
            _class_id = class_id;
        }

        virtual size_t size() const {
            return _reader->size();
        }
        virtual size_t processed_num() const {
            return _reader->processed_num();
        }
        virtual size_t dim() const {
            return _reader->dim();
        }
        virtual int percentage() const {
            return _reader->percentage();
        }
        virtual void set(const char* filename, bool preprocess=false) {
            _reader->set(filename, preprocess);
        }

        /**
         *  reset the stream.
         */
        virtual void reset() {
            _reader->reset();
        }

        /**
         *  usage:
         *      FeatureReader_t r(stream);
         *      Instance_t item;
         *      while (reader.read(&item)) {
         *          // do what you need to do.
         *      }
         */ 
        virtual bool read(Instance_t* item, bool original=false) {
            bool ret = _reader->read(item, original);
            //float old_label = item->label;
            if (fabs(item->label - _class_id) <= 0.5) {
                item->label = 1.0;
            } else {
                item->label = 0.0;
            }
            //LOG_NOTICE("%d) %f %f", _class_id, old_label, item->label);
            return ret;
        }

    private:
        FlyReader_t* _reader;
        int         _class_id;
};

/*
 * multi-class classifier.
 */
class MetaModel_t
    : public FlyModel_t 
{
    public:
        MetaModel_t(const Config_t& conf, const char* section) {
            if ( !conf.conf_int(section, "class_num", &_class_num) ) {
                throw std::runtime_error("Meta model needs config: <class_num>");
            }

            string s;
            string sub_section;
            if (!conf.conf_str(section, "meta_model", &s)) {
                throw std::runtime_error("Meta model needs config: <meta_model>");
            }
            if (!conf.conf_str(section, "meta_section", &sub_section)) {
                throw std::runtime_error("Meta model needs config: <meta_section>");
            }

            _classifiers = new FlyModel_t*[_class_num];
            for (int i=0; i<_class_num; ++i) {
                if (s == "lr") {
                    _classifiers[i] = new LogisticRegression_t(conf, sub_section.c_str());
                } else if (s == "gbdt") {
                    _classifiers[i] = new GBDT_t(conf, sub_section.c_str());
                }
            }
        }

        virtual float predict(const Instance_t& ins) const {
            float best_score = 0;
            int best_class = -1;
            for (int c=0; c<_class_num; ++c) {
                float score = _classifiers[c]->predict(ins);
                //LOG_NOTICE("l%d c%d score=%f", ins.label, c, score);
                if (best_class == -1 || best_score < score) {
                    best_score = score;
                    best_class = c;
                }
            }
            return best_class;
        }

        virtual void init(FlyReader_t* reader) {
            _fake_readers = new MultiClassFakeReader_t* [_class_num];
            for (int i=0; i<_class_num; ++i) {
                _fake_readers[i] = new MultiClassFakeReader_t(reader, i);
                _fake_readers[i]->reset();
                _classifiers[i]->init(_fake_readers[i]);
            }
        }

        virtual void train() {
            for (int i=0; i<_class_num; ++i) {
                _fake_readers[i]->reset();
                _classifiers[i]->train();
            }
        }

        virtual void write_model(FILE* stream) const {
            fwrite(&_class_num, 1, sizeof(int), stream);
            for (int i=0; i<_class_num; ++i) {
                _classifiers[i]->write_model(stream);
            }
        }

        virtual void read_model(FILE* stream) {
            fread(&_class_num, 1, sizeof(int), stream);
            for (int i=0; i<_class_num; ++i) {
                _classifiers[i]->read_model(stream);
            }
        }

    private:
        MultiClassFakeReader_t**    _fake_readers;
        FlyModel_t**                _classifiers;
        int                         _class_num;
};


#endif  //__META_H_

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
