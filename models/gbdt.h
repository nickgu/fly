/**
 * @file models/gbdt.h
 * @author nickgu
 * @date 2015/01/19 11:47:45
 * @brief 
 *  
 **/

#ifndef  __GBDT_H_
#define  __GBDT_H_

#include "../fly_core.h"
#include "../cfg.h"

#include <set>

#define INVALID_SAME_KEY (0xffffffff)

int _L(int x) {return x*2+1;}
int _R(int x) {return x*2+2;}

int sample_const = 0;
int sample_threshold = 256;
#define ITEM_SAMPLE(idx) (int(((idx+137)*(sample_const+1)+79) & 0xff) <= sample_threshold)

struct SortedIndex_t {
    /*
     * if this bit is set:
     *  the value of this index is same with last one.
     */
    uint32_t same:1; 
    uint32_t index:31; 
};

struct TreeNode_t {
    // Decision info.
    int fidx;   // feature index.
    double threshold;  // threshold.
    double mean;     // predict value.

    // Training info.
    float score;    // mse delta.
    int begin;
    int end;
    int cnt;
    int split;
    uint32_t split_id;

    // aid info.
    int grow; // help for O(n) sort..
    uint32_t same_key;
    double sum;
    double square_sum;
    double temp_sum;

    void init(size_t b, size_t e) {
        fidx = -1;
        mean = 0;
        threshold = 0;
        score = 0;

        begin = b;
        end = e;
        cnt = end - begin;
        split = b;
        split_id = 0;

        grow = b;
        same_key = INVALID_SAME_KEY;
        sum = 0;
        square_sum = 0;
    }

    void read(FILE* stream) {
        fread(&fidx, 1, sizeof(fidx), stream);
        fread(&threshold, 1, sizeof(threshold), stream);
        fread(&mean, 1, sizeof(mean), stream);
    }

    void write(FILE* stream) const {
        fwrite(&fidx, 1, sizeof(fidx), stream);
        fwrite(&threshold, 1, sizeof(threshold), stream);
        fwrite(&mean, 1, sizeof(mean), stream);
    }

    bool operator< (const TreeNode_t& o) const {
        return (o.fidx!=-1 && (fidx==-1 || o.score < score));
    }

    string decision_info() const {
        char buf[256];

        float orig_score = 0;
        if (cnt>0) {
            orig_score = (square_sum - mean*mean) / cnt;
        }
        snprintf(buf, sizeof(buf), "{(f_%d<%f (item%d.f_%d)) score:%f=>%f %d=%d+%d)}",
                fidx, threshold, 
                split_id, fidx, 
                orig_score, score, 
                cnt, 
                split - begin,
                end - split);
        return string(buf);
    }
};

struct SmallTreeNode_t {
    // Decision info.
    short fidx;   // feature index.
    float threshold;  // threshold.

    void init(size_t b, size_t e) {
        fidx = -1;
        threshold = 0;
    }

    void copy(const TreeNode_t& o) {
        fidx = o.fidx;
        threshold = o.threshold;
    }
};


struct ItemInfo_t {
    float   residual;
    short   in_which_node;
};

struct Job_LayerFeatureProcess_t {
    bool selected;
    uint32_t item_count;
    int feature_index;
    int beg_node;
    int end_node;
    int all_node_count;

    const SortedIndex_t* finfo;
    const ItemInfo_t* iinfo;
    TreeNode_t* tree;
    uint32_t* calc_last_layer;
};

inline float __mid_rmse_score(float la, int lc, float ra, int rc)
{
    /**
     * la : left average.
     * lc : left count.
     * ra : right average.
     * rc : right count.
     */

    float ret=0;
    if (lc>0) ret += la/lc*la;
    if (rc>0) ret += ra/rc*ra;
    return ret;
}

void* __worker_layer_processor(void* input) {
    Timer ta;

    Job_LayerFeatureProcess_t& job= *(Job_LayerFeatureProcess_t*)input;
    uint32_t* dim_id_sorted = job.calc_last_layer;
    if (!job.selected) {
        pthread_exit(0);
    }

    // reset growth id.
    for (int i=0; i<job.all_node_count; ++i) {
        job.tree[i].grow = job.tree[i].begin;
        job.tree[i].same_key = INVALID_SAME_KEY;
    }
    for (int i=job.beg_node; i<job.end_node; ++i) {
        job.tree[i].temp_sum = 0;
        job.tree[i].cnt = job.tree[i].end - job.tree[i].begin;
        job.tree[i].score = __mid_rmse_score(0, 0, job.tree[i].sum, job.tree[i].cnt);
    }

    ta.begin();
    // critical time demand.
    // make sort step over. in O(n)
    uint32_t same_key = INVALID_SAME_KEY;
    const ItemInfo_t* iinfo = job.iinfo;
    uint32_t item_count = job.item_count;
    for (uint32_t i=0; i<item_count; ++i) {
        SortedIndex_t si = job.finfo[i];

        if (!si.same) { 
            // set same_key as 
            // first index of continuous same value.
            same_key = i;
        }
        
        // sample out not useful data.
        if ( !ITEM_SAMPLE(si.index) ) {
            continue;
        }

        TreeNode_t& nod = job.tree[ job.iinfo[ si.index ].in_which_node ];
        if (!si.same || nod.same_key!=same_key) { 
            nod.same_key = same_key;
            float temp_score = __mid_rmse_score(
                    nod.temp_sum, nod.grow-nod.begin,
                    nod.sum-nod.temp_sum, nod.end-nod.grow);
            if (temp_score > nod.score) {
                nod.fidx = job.feature_index;
                nod.score = temp_score;
                nod.split = nod.grow;
                nod.split_id = si.index;
            }
        }
        dim_id_sorted[ nod.grow++ ] = si.index;
        nod.temp_sum += iinfo[ si.index ].residual;
    }
    for (int n=job.beg_node; n<job.end_node; ++n) {
        TreeNode_t& node = job.tree[n];
        node.end = node.grow;
        node.score = (node.square_sum - node.score) / node.cnt;
    }

    ta.end();
    LOG_DEBUG("Feature %d training time=%.2fs", job.feature_index, ta.cost_time());
    pthread_exit(0);
}

class GBDT_t 
    : public FlyModel_t
{
    struct FeatureInfo_t {
        int index;
        float value;

        bool operator < (const FeatureInfo_t& o) const {
            return value < o.value;
        }
    };

    struct ItemID_ReverseInfo_t {
        uint32_t item_id;
        int tree_id;
        int node_id;

        bool operator < (const ItemID_ReverseInfo_t& o) const {
            return item_id < o.item_id;
        }
    };

    public:
        GBDT_t(const Config_t& config, const char* section):
            _trees(NULL),
            _ffd(NULL),
            _sorted_fields(NULL),
            _predict_buffer(NULL),
            _compact_trees(NULL),
            _mean(NULL),
            _feature_weight(NULL),
            _output_feature_weight(false),
            _predict_tree_cut(-1)
        {
            _sample_feature = config.conf_float_default(section, "sample_feature", 1.0);
            _sample_instance = config.conf_float_default(section, "sample_instance", 1.0);
            LOG_NOTICE("Sample_info: feature=%.2f instance=%.2f", _sample_feature, _sample_instance);

            // directory.
            _feature_load_dir = config.conf_str_default(section, "feature_load_dir", "");
            LOG_NOTICE("Feature_load_dir: [%s]", _feature_load_dir.c_str());

            _tree_count = config.conf_int_default(section, "tree_num", 100);
            _max_layer = config.conf_int_default(section, "layer_num", 5);
            _thread_num = config.conf_int_default(section, "thread_num", 8);
            LOG_NOTICE("thread_num=%d", _thread_num);

            _sr = config.conf_float_default(section, "shrinkage", 0.3);
            LOG_NOTICE("shrinkage=%f", _sr);

            _output_feature_weight = config.conf_int_default(section, "output_feature_weight", 0);
            LOG_NOTICE("output_feature_weight=%d", _output_feature_weight);

            string s = config.conf_str_default(section, "feature_mask", "");
            vector<string> vs;
            split((char*)s.c_str(), ",", vs);
            for (size_t i=0; i<vs.size(); ++i) {
                int f = atoi(vs[i].c_str());
                LOG_NOTICE("mask_feature: %d", f);
                _feature_mask.insert(f);
            }

            _tree_size = 1 << (_max_layer + 2);

            _labels = NULL;
        }

        virtual ~GBDT_t() {
            LOG_NOTICE("Destroy work for GBDT begins.");
            if (_feature_weight) {
                delete [] _feature_weight;
                _feature_weight = NULL;
            }
            if (_trees) {
                for (int i=0; i<_tree_count; ++i) {
                    delete [] _trees[i];
                }
                delete [] _trees;
                _trees = NULL;
            }

            if (_compact_trees) {
                for (int i=0; i<_tree_count; ++i) {
                    delete [] _compact_trees[i];
                }
                delete [] _compact_trees;
                _compact_trees = NULL;
            }

            if (_mean) {
                for (int i=0; i<_tree_count; ++i) {
                    delete [] _mean[i];
                }
                delete [] _mean;
            }

            if (_labels) {
                delete [] _labels;
                _labels = NULL;
            }
            if (_ffd) {
                for (int i=0; i<_dim_count; ++i) {
                    fclose(_ffd[i]);
                    _ffd[i] = NULL;
                }
                delete [] _ffd;
                _ffd = NULL;
            }
            if (_sorted_fields) {
                for (int i=0; i<_dim_count; ++i) {
                    delete [] _sorted_fields[i];
                }
                delete [] _sorted_fields;
                _sorted_fields = NULL;
            }

            if (_predict_buffer) {
                delete [] _predict_buffer;
                _predict_buffer = NULL;
            }
            LOG_NOTICE("Destroy work for GBDT ends");
        }

        void set_predict_tree_cut(int N=-1) {
            _predict_tree_cut = N;
        }

        virtual float predict(const Instance_t& ins) const {
            return predict_and_get_leaves(ins, NULL, NULL);
        }

        virtual float predict_and_get_leaves(const Instance_t& ins, 
                vector<int>* output_leaf_id_in_each_tree,
                vector<float>* output_mean) const 
        {
            float ret = 0.0f;

            memset(_predict_buffer, 0, sizeof(float) * _dim_count);
            for (size_t f=0; f<ins.features.size(); ++f) {
                if (ins.features[f].index < _dim_count) {
                    _predict_buffer[ins.features[f].index] = ins.features[f].value;
                }
            }

            SmallTreeNode_t** end_tree = _compact_trees + _tree_count;
            if (output_leaf_id_in_each_tree!=NULL) {
                output_leaf_id_in_each_tree->clear();
            }
            if (output_mean) {
                output_mean->clear();
            }

            int tc = 0;
            for (SmallTreeNode_t** tree=_compact_trees; tree<end_tree; ++tree) {
                if (_predict_tree_cut>=0 && tc>=_predict_tree_cut) {
                    break;
                }
                tc++;

                int nid = 0;
                while (1) {
                    SmallTreeNode_t* node = (*tree)+nid;
                    if (node->fidx != -1) {
                        nid = _L(nid);
                        if (_predict_buffer[node->fidx] >= node->threshold) {
                            nid ++;
                        }
                    } else {
                        break;
                    }
                }
                if (output_leaf_id_in_each_tree!=NULL) {
                    output_leaf_id_in_each_tree->push_back(nid - (1<<(_max_layer-1)));
                    if (output_mean) {
                        output_mean->push_back(_mean[tree - _compact_trees][nid]);
                    }
                }
                ret += _mean[tree - _compact_trees][nid];
            }
            return ret;
        }


        virtual void write_model(FILE* stream) const {
            fwrite(&_tree_count, 1, sizeof(_tree_count), stream);
            fwrite(&_tree_size, 1, sizeof(_tree_size), stream);
            fwrite(&_sr, 1, sizeof(_sr), stream);

            for (int T=0; T<_tree_count; ++T) {
                for (int i=0; i<_tree_size; ++i) {
                    _trees[T][i].write(stream);
                }
            }
            return ;
        }

        virtual void read_model(FILE* stream) {
            if (stream == NULL) {
                LOG_ERROR("Cannot load model from input stream.");
                return;
            }
            fread(&_tree_count, 1, sizeof(_tree_count), stream);
            fread(&_tree_size, 1, sizeof(_tree_size), stream);
            fread(&_sr, 1, sizeof(_sr), stream);
            LOG_NOTICE("LOADING_INFO: _tree_count=%d _tree_size=%d _sr=%f", _tree_count, _tree_size, _sr);
            _max_layer = 0;
            size_t t =_tree_size;
            while (t) {
                t >>= 1;
                _max_layer += 1;
            }
            _max_layer -= 2 + 1; //trick imp, for former trick imp..
            LOG_NOTICE("LOADING_INFO: infer: tree_layer=%d", _max_layer);

            _dim_count = 0;
            _compact_trees = new SmallTreeNode_t*[_tree_count];
            _mean = new float*[_tree_count];
            TreeNode_t temp_node;
            for (int T=0; T<_tree_count; ++T) {
                _compact_trees[T] = new SmallTreeNode_t[_tree_size];
                _mean[T] = new float[_tree_size];
                for (int i=0; i<_tree_size; ++i) {
                    temp_node.read(stream);
                    _compact_trees[T][i].copy(temp_node);
                    _mean[T][i] = temp_node.mean * _sr;
                    if (_dim_count <= _compact_trees[T][i].fidx) {
                        _dim_count = _compact_trees[T][i].fidx + 1;
                    }
                }
            }
            if (_predict_buffer) {
                delete [] _predict_buffer;
                _predict_buffer = NULL;
            }
            _predict_buffer = new float[_dim_count];
            return ;
        }

        virtual void  init(FlyReader_t* reader) {
            // construct column infomation.
            _reader = reader;
            _item_count = (unsigned)reader->size();

            _dim_count = reader->dim();
            if (_feature_weight) {
                delete [] _feature_weight;
            }
            _feature_weight = new float[_dim_count];
            memset(_feature_weight, 0, sizeof(float)*_dim_count);
            if (_predict_buffer) {
                delete [] _predict_buffer;
                _predict_buffer = NULL;
            }
            _predict_buffer = new float[_dim_count];

            _labels = new ItemInfo_t[_item_count];
            _ffd = new FILE*[_dim_count];
            if (_feature_load_dir != "") { 
                LOG_NOTICE("Load feature from dir..[%s]", _feature_load_dir.c_str());
                for (int fid=0; fid<_dim_count; ++fid) {
                    char buf[32];
                    snprintf(buf, sizeof(buf), "%s/feature.%d", _feature_load_dir.c_str(), fid);
                    _ffd[fid] = fopen(buf, "r");
                    if (_ffd[fid] == NULL) {
                        LOG_ERROR("Cannot open file [%s] to write index info.");
                        throw std::runtime_error(_feature_load_dir);
                    }
                }
                Instance_t item;
                size_t item_id = 0;
                int cur_per = 0;
                while (reader->read(&item/*, true*/)) {
                    _labels[item_id].residual = item.label;
                    item_id ++;

                    int per = reader->percentage();
                    if (per > cur_per) {
                        cur_per = per;
                        fprintf(stderr, "%c%4d%% loaded..", 13, cur_per);
                    }
                }
                fprintf(stderr, "\n");

            } else { // save to _feature_output_dir
                FeatureInfo_t **ptr = new FeatureInfo_t*[_dim_count];
                for (int i=0; i<_dim_count; ++i) {
                    ptr[i] = new FeatureInfo_t[_item_count];
                }
                SortedIndex_t *idx_list = new SortedIndex_t[_item_count];

                Instance_t item;
                size_t item_id = 0;
                int cur_per = 0;
                while (reader->read(&item/*, true*/)) {
                    _labels[item_id].residual = item.label;
                    // some feature may be missing.
                    for (int i=0; i<_dim_count; ++i) {
                        ptr[i][item_id].index = item_id;
                    }
                    for (size_t i=0; i<item.features.size(); ++i) {
                        const IndValue_t& f = item.features[i];
                        ptr[f.index][item_id].value = f.value;
                    }
                    item_id ++;

                    int per = reader->percentage();
                    if (per > cur_per) {
                        cur_per = per;
                        fprintf(stderr, "%c%4d%% loaded..", 13, cur_per);
                    }
                }
                fprintf(stderr, "\n");

                system("rm -rf gbdt_temp");
                system("mkdir gbdt_temp");

                for (int fid=0; fid<_dim_count; ++fid) {
                    LOG_NOTICE("sort dim : %d", fid);
                    sort(ptr[fid], ptr[fid]+_item_count);
                    LOG_NOTICE("sort dim %d over.", fid);

                    for (size_t i=0; i<_item_count; ++i) {
                        idx_list[i].index = ptr[fid][i].index;
                        if (i>0 && ptr[fid][i].value == ptr[fid][i-1].value) {
                            idx_list[i].same = 1;
                        } else {
                            idx_list[i].same = 0;
                        }
                    }

                    char buf[32];
                    snprintf(buf, sizeof(buf), "gbdt_temp/feature.%d", fid);
                    FILE* fout = fopen(buf, "wb");
                    if (!fout) {
                        LOG_ERROR("open temp directory to save field information failed! [%s]", buf);
                        exit(-1);
                    }
                    fwrite(idx_list, _item_count, sizeof(SortedIndex_t), fout);
                    fclose(fout);
                    _ffd[fid] = fopen(buf, "rb");

                    LOG_NOTICE("write feature file over [%s]", buf);
                }

                // free memory.
                delete [] idx_list;
                for (int i=0; i<_dim_count; ++i) {
                    delete [] ptr[i];
                }
                delete [] ptr;
            }
        
            // temp: load all field in memory.
            _sorted_fields = new SortedIndex_t*[_dim_count];
            Timer tm;
            tm.begin();

            for (int i=0; i<_dim_count; ++i) {
                _sorted_fields[i] = new SortedIndex_t[_item_count];
                fseek(_ffd[i], 0, SEEK_SET);
                fread(_sorted_fields[i], _item_count, sizeof(SortedIndex_t), _ffd[i]);
            }

            tm.end();
            LOG_NOTICE("load field time: %.2fs", tm.cost_time());
            return ;
        }

        virtual void train() {
            _trees = new TreeNode_t*[_tree_count];
            for (int i=0; i<_tree_count; ++i) {
                _trees[i] = new TreeNode_t[_tree_size];
            }

            ItemInfo_t* iinfo = new ItemInfo_t[_item_count];  // [item_id] : residual, in_which_node.
            Job_LayerFeatureProcess_t* jobs = new Job_LayerFeatureProcess_t[_dim_count];
            pthread_t* tids = new pthread_t[_dim_count];

            for (int D=0; D<_dim_count; ++D) {
                jobs[D].tree = new TreeNode_t[_tree_size];
                jobs[D].calc_last_layer = new uint32_t[_item_count];
            }
            // initialize target.
            for (size_t i=0; i<_item_count; ++i) {
                iinfo[i].residual = _labels[i].residual;
            }

            for (int T=0; T<_tree_count; ++T) {
                Timer tree_tm, sample_tm;
                tree_tm.begin();
            
                // Initialize.
                TreeNode_t& root = _trees[T][0];
                root.init(0, _item_count);
                int beg_node = 0;
                int end_node = 1;
                int all_node_count = 1;

                sample_tm.begin();
                sample_const = T * 7;
                sample_threshold = int(256 * _sample_instance);
                size_t sample_item_count = 0;
                for (size_t i=0; i<_item_count; ++i) {
                    iinfo[i].in_which_node = 0;
                    if ( ITEM_SAMPLE(i) ) {
                        sample_item_count += 1;
                        root.sum += iinfo[i].residual;
                        root.square_sum += iinfo[i].residual * iinfo[i].residual;
                    }
                }
                sample_tm.end();

                LOG_NOTICE("Begin training tree[%d/%d] : mse=%f (sqsum=%f,sum=%f,cnt=%d(samp=%d)) sample_tm=%.2f", 
                        T, _tree_count, 
                        (root.square_sum - root.sum*root.sum/root.cnt)/root.cnt,
                        root.square_sum, root.sum, root.cnt, sample_item_count, sample_tm.cost_time() );

                for (int i=0; i<_tree_size; ++i) {
                    _trees[T][i].fidx = -1;
                }

                // Each layer
                for (int L=0; L<_max_layer; ++L) {
                    Timer multi_tm, post_tm;

                    multi_tm.begin();
                    int selected_feature_count = 0;
                    for (int D=0; D<_dim_count; ++D) {
                        // sample features.
                        jobs[D].selected = false;
                        
                        if (_feature_mask.find(D)!=_feature_mask.end()) {
                            continue;
                        }

                        if (_sample(_sample_feature) 
                                && selected_feature_count<_dim_count*_sample_feature) 
                        {
                            jobs[D].selected = true;
                            jobs[D].item_count = _item_count;
                            jobs[D].feature_index = D;
                            jobs[D].beg_node = beg_node;
                            jobs[D].end_node = end_node;
                            jobs[D].all_node_count = all_node_count;

                            jobs[D].finfo = _sorted_fields[D];
                            jobs[D].iinfo = iinfo;
                            memcpy(jobs[D].tree, _trees[T], _tree_size * sizeof(TreeNode_t));

                            selected_feature_count ++;
                        }
                    }
                    multi_thread_jobs(__worker_layer_processor, jobs, _dim_count, _thread_num);
                    multi_tm.end();

                    post_tm.begin();
                    // find best node.
                    for (int i=beg_node; i<end_node; ++i) {
                        for (int D=0; D<_dim_count; ++D) {
                            if (jobs[D].selected && _trees[T][i] < jobs[D].tree[i]) {
                                _trees[T][i] = jobs[D].tree[i];
                            }
                        }
                    }
                    for (int n=beg_node; n<end_node; ++n) {
                        TreeNode_t& node = _trees[T][n];
                        if (node.fidx != -1) {
                            // accumlulate the score to the feature weight.
                            _feature_weight[node.fidx] += _trees[T][n].score;

                            _trees[T][_L(n)].init(node.begin, node.split);
                            _trees[T][_R(n)].init(node.split, node.end);
                            double sum = 0;
                            double ssum = 0;
                            int b = _trees[T][_L(n)].begin;
                            int e = _trees[T][_L(n)].end;
                            for (int i=b; i<e; ++i) {
                                uint32_t id = jobs[node.fidx].calc_last_layer[i];
                                float d = iinfo[id].residual;
                                iinfo[id].in_which_node = _L(n);
                                sum += d;
                                ssum += d*d;
                            }
                            _trees[T][_L(n)].sum = sum;
                            _trees[T][_L(n)].square_sum = ssum;
                            sum = ssum = 0;
                            b = _trees[T][_R(n)].begin;
                            e = _trees[T][_R(n)].end;
                            for (int i=b; i<e; ++i) {
                                uint32_t id = jobs[node.fidx].calc_last_layer[i];
                                float d = iinfo[id].residual;
                                iinfo[id].in_which_node = _R(n);
                                sum += d;
                                ssum += d*d;
                            }
                            _trees[T][_R(n)].sum = sum;
                            _trees[T][_R(n)].square_sum = ssum;
                        }
                    }

                    for (int i=beg_node; i<end_node; ++i) {
                        if (_trees[T][i].fidx>=0) {
                            if (all_node_count<=_R(i)) {
                                all_node_count = _R(i)+1;
                            }
                        }
                    }
                    post_tm.end();

                    float total_tm = multi_tm.cost_time() + post_tm.cost_time();
                    LOG_NOTICE("T%d L%d multi=%.2f post=%.2f tm=%.2fs", 
                            T, L, 
                            multi_tm.cost_time(),
                            post_tm.cost_time(),
                            total_tm
                        );

                    // dump feature weight at each layer's ending.
                    if (_output_feature_weight) {
                        for (int i=0; i<_dim_count; ++i) {
                            LOG_NOTICE("FWeight:\tT:%d\tL:%d\tF:%d\t%.5f", T, L, i, _feature_weight[i]);
                        }
                    }

                    // one layer forward.
                    beg_node = end_node;
                    end_node = all_node_count;
                } // layer end.

                Timer tree_finalize_tm;
                tree_finalize_tm.begin();
                for (int i=0; i<_tree_size; ++i) {
                    TreeNode_t & node = _trees[T][i];
                    if (node.cnt>0) {
                        node.mean = node.sum / node.cnt;
                    }
                }

                // update residual.
                for (uint32_t i=0; i<_item_count; ++i) {
                    // sample out.
                    if (!ITEM_SAMPLE(i)) {
                        continue;
                    }
                    float predict_value = _trees[T][iinfo[i].in_which_node].mean;
                    iinfo[i].residual -= _sr * predict_value;
                    //LOG_NOTICE("T=%d i=%d res=%f", T, i, iinfo[i].residual);
                }
                tree_finalize_tm.end();

                tree_tm.end();
                LOG_NOTICE("Training tree[%d] tm=%.2fs (finalize=%.2fs)", 
                        T, tree_tm.cost_time(),
                        tree_finalize_tm.cost_time());
            } // tree end.

            // make-up missing value: threshold and mean.
            vector<ItemID_ReverseInfo_t> reverse_array;
            for (int t=0; t<_tree_count; ++t) {
                for (int i=0; i<_tree_size; ++i) {
                    TreeNode_t & node = _trees[t][i];
                    if (node.fidx>=0) {
                        ItemID_ReverseInfo_t item;
                        item.item_id = node.split_id;
                        item.tree_id = t;
                        item.node_id = i;
                        reverse_array.push_back(item);
                    }
                }
            }

            sort(reverse_array.begin(), reverse_array.end());
            // reading items. merge to node.
            _reader->reset();
            Instance_t item;
            _reader->read(&item/*, true*/);
            uint32_t id = 0;
            for (size_t i=0; i<reverse_array.size(); ++i) {
                const ItemID_ReverseInfo_t& info = reverse_array[i];
                while (id<info.item_id) {
                    if ( !_reader->read(&item/*, true*/) ) {
                        break;
                    }
                    id ++;
                }
                if (id < info.item_id) {
                    LOG_ERROR("reader is ended, but info is still remaining. [item_id:%d]",
                            info.item_id);
                    break;
                }

                TreeNode_t& node = _trees[info.tree_id][info.node_id];
                node.threshold = 0.0;
                for (size_t f=0; f<item.features.size(); ++f) {
                    if (item.features[f].index == node.fidx) {
                        node.threshold = item.features[f].value;
                        break;
                    }
                }
            }

            // copy tree to compact_tree.
            _compact_trees = new SmallTreeNode_t*[_tree_count];
            _mean = new float*[_tree_count];
            for (int T=0; T<_tree_count; ++T) {
                _compact_trees[T] = new SmallTreeNode_t[_tree_size];
                _mean[T] = new float[_tree_size];
                for (int i=0; i<_tree_size; ++i) {
                    _compact_trees[T][i].copy(_trees[T][i]);
                    _mean[T][i] = _trees[T][i].mean * _sr;
                }
            }

            delete [] tids;
            for (int D=0; D<_dim_count; ++D) {
                delete [] jobs[D].tree;
                delete [] jobs[D].calc_last_layer;
            }
            delete [] jobs;
            delete [] iinfo;
        }

        int layer_num() const { return _max_layer; }

    private:
        FlyReader_t* _reader;

        int         _tree_count;
        int         _max_layer;
        string      _feature_load_dir;
        int         _thread_num;
        float       _sr;

        float _sample_feature;
        float _sample_instance;

        int             _tree_size;
        TreeNode_t**    _trees;  // node buffer.

        ItemInfo_t*     _labels;
        FILE**          _ffd;
        SortedIndex_t** _sorted_fields;
        float*          _predict_buffer;

        uint32_t  _item_count;
        int     _dim_count;
        size_t  _maximum_memory;

        SmallTreeNode_t** _compact_trees;
        float**         _mean;

        // debug feature weight.
        float    *_feature_weight;
        bool     _output_feature_weight;
        bool     _feature_begin_at_0;
        std::set<int> _feature_mask;
        int      _predict_tree_cut;

        bool _sample(float ratio) const {
            return ((random()%10000) / 10000.0) <= ratio;
        }
};

#endif  //__GBDT_H_

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
