
#ifndef __HELPER_H__
#define __HELPER_H__




#include <ctime>

#include <vector>
#include <string>
#include <stdexcept>


#include <cstdio>
#include <sys/time.h>

#include <errno.h>
#include <pthread.h>

#include <cstdlib>
#include <cstring>
using namespace std;

#include <linux/unistd.h>
#include <linux/kernel.h>

#include <stdint.h>
#include <signal.h>

#ifndef LOG_DEBUG
#define LOG_DEBUG(format, ...) {if (__hidden::is_debug_on()) fprintf(stderr, "DEBUG: " format "\n", ##__VA_ARGS__);}
#endif 

#ifndef LOG_NOTICE
#define LOG_NOTICE(format, ...) {fprintf(stderr, "NOTICE: " format "\n", ##__VA_ARGS__);}
#endif 

#ifndef LOG_ERROR
#define LOG_ERROR(format, ...) {fprintf(stderr, " ERROR: " format "\n", ##__VA_ARGS__);}
#endif

/*
_syscall1(int, sysinfo, struct sysinfo*, info);
size_t remain_memory() {
    struct sysinfo s_info;
    sysinfo(&s_info);
    return s_info.freeram;
}
*/

namespace __hidden {
    static bool __g_debug_is_on__ = false;

    inline bool is_debug_on() {
        return __g_debug_is_on__;
    }

    inline void set_debug(bool b) {
        __g_debug_is_on__ = b;
    }
};


class Timer {
public:
    Timer() :_sum(0.0) {}

    void begin() {
        gettimeofday(&_begin_tv, NULL);
    }

    void end() {
        gettimeofday(&_end_tv, NULL);
        _sum += (_end_tv.tv_sec - _begin_tv.tv_sec) + (_end_tv.tv_usec - _begin_tv.tv_usec) * 0.000001f;
    }

    /* return unit : seconds.*/
    float cost_time() const {
        return _sum;
    }

    void clear() {
        _sum = 0;
    }

private:
    float   _sum;

    timeval _begin_tv;
    timeval _end_tv;
};

template <typename T>
class FArray_t {
    public:
        FArray_t(size_t extend_num=16):
            _l(NULL),
            _num(0),
            _bnum(0),
            _extend_num(extend_num)
        {}

        FArray_t(const FArray_t& o) {
            _bnum = o._bnum;
            _num = o._num;
            _extend_num = o._extend_num;
            if (_bnum > 0) {
                _l = (T*)malloc(_bnum * sizeof(T));
                //LOG_NOTICE("m: %p [%u]", _l, _bnum);
                memcpy(_l, o._l, _num * sizeof(T));
            }
        }

        FArray_t& operator = (const FArray_t& o) {
            _num = o._num;
            _extend_num = o._extend_num;

            if (_bnum < o._bnum) {
                if (_bnum>0) {
                    free(_l);
                }
                _bnum = o._bnum;
                _l = (T*)malloc(_bnum * sizeof(T));
            }
            memcpy(_l, o._l, _num * sizeof(T));
            return *this;
        }

        ~FArray_t() { _release(); }

        size_t size() const {return _num;}
        T* buffer() const {return _l;}

        void push_back(const T& o) {
            if (_num == _bnum) {
                _extend();
            }
            _l[_num ++] = o;
        }

        void clear() { _num = 0; }

        T& operator [] (size_t idx) {
            if (idx >= _num) {
                throw std::runtime_error("index out of range.");
            }
            return _l[idx];
        }
        T& operator [] (size_t idx) const {
            if (idx >= _num) {
                throw std::runtime_error("index out of range.");
            }
            return _l[idx];
        }

        void read(FILE* stream) {
            fread(&_num, 1, sizeof(_num), stream);
            if (_num > _bnum) {
                if (_bnum > 0) {
                    free(_l);
                }
                _l = (T*)malloc(_num * sizeof(T));
                if (_l == NULL) {
                    throw std::runtime_error("extend buffer for FArray failed!");
                }
                _bnum = _num;
            }
            fread(_l, _num, sizeof(T), stream);
        }

        void write(FILE* stream) {
            fwrite(&_num, 1, sizeof(_num), stream);
            fwrite(_l, _num, sizeof(T), stream);
        }

    private:
        T *_l;
        size_t _num;
        size_t _bnum;
        size_t _extend_num;

        void _extend() {
            _l = (T*)realloc(_l, (_bnum + _extend_num) * sizeof(T) );
            //LOG_NOTICE("r: %p", _l);
            if (_l == NULL) {
                throw std::runtime_error("extend buffer for FArray failed!");
            }
            _bnum += _extend_num;
        }

        void _release() {
            if (_l) {
                free(_l);
            }
            _l = NULL;
            _num = _bnum = 0;
        }
};

inline void split(char* s, const char* token, std::vector<std::string>& out) {
    char* p;
    out.clear();
    char* f = strtok_r(s, token, &p);
    while (f) {
        out.push_back(f);
        f = strtok_r(NULL, token, &p); 
    }
    return ;
}

class Lock_t {
    public:
        Lock_t() {
            pthread_mutex_init(&_lock, 0);
        }
        ~Lock_t() {
            pthread_mutex_destroy(&_lock);
        }

        void lock() {
            pthread_mutex_lock(&_lock);
            return ;
        }

        void unlock() {
            pthread_mutex_unlock(&_lock);
        }

    private:
        pthread_mutex_t _lock;
};

template <typename PtrType_t>
class ThreadData_t {
    public:
        ThreadData_t(PtrType_t a) {
            _data = a;
            pthread_mutex_init(&_lock, 0);
        }

        ThreadData_t() {
            pthread_mutex_init(&_lock, 0);
        }

        PtrType_t borrow() {
            pthread_mutex_lock(&_lock);
            return _data;
        }

        void give_back() {
            pthread_mutex_unlock(&_lock);
        }

    private:
        PtrType_t _data;
        pthread_mutex_t _lock;
};

/*
 * Producer and Customer Poot
 * Condition: 
 *      _p_id + 1 != _c_id
 *      _c_id + 2 != _p_id (..)
 * one empty cell to make validation.
 *  * Single putter and multi-getter.
 */
template <typename T>
class PCPool_t {
    public:
        PCPool_t(size_t buffer_size) {
            _buffer_size = buffer_size;
            _buffer = new T[_buffer_size];
            _c_id = 0;
            _p_id = 0;
            _total_get = 0;
            _total_put = 0;
            _flag_putting = true;
            pthread_spin_init(&_spinlock, 0);
        }
        ~PCPool_t() {
            if (_buffer) {
                delete [] _buffer;
                _buffer_size = 0;
            }
        }

        // Producer put item.
        void put(const T& item) {
            // try util ok.
            while (1) {
                size_t next_id = (_p_id + 1) % _buffer_size;
                if (next_id == _c_id) {
                    // full: need to wait for putting.
                    continue;
                }
                _buffer[_p_id] = item;
                _p_id = next_id;
                _total_put += 1;
                return ;
            }
        }

        // Return ptr for writing object.
        T* begin_put() {
            // try util ok.
            while (1) {
                size_t next_id = (_p_id + 1) % _buffer_size;
                if (next_id == _c_id) {
                    // full: need to wait for putting.
                    //LOG_NOTICE("full: %d,%d (%d)", _p_id, _c_id, _total_put);
                    continue;
                }
                return _buffer + _p_id;
            }
        }

        void end_put(bool put_ok=true) {
            if (put_ok) {
                _p_id = (_p_id + 1) % _buffer_size;;
                _total_put += 1;
            }
        }

        // Customer try to get.
        // loop util get.
        // return false if nothing to process forever.
        bool get(T* out_item, uint32_t* out_order_id = NULL) {
            // retry until work or full.
            while (1) {
                pthread_spin_lock(&_spinlock);
                if (_c_id == _p_id) { // empty or stop.
                    pthread_spin_unlock(&_spinlock);
                    // need to wait for processing.
                    if (!_flag_putting) {
                        //LOG_NOTICE("return false");
                        return false;
                    }
                    continue;
                }
                if (_c_id + 1 == _p_id && _flag_putting) {
                    pthread_spin_unlock(&_spinlock);
                    // need to wait for processing.
                    continue;
                }
                size_t m = _c_id;
                //LOG_NOTICE("get: %d,%d (%d)", _p_id, _c_id, _total_put);
                _c_id = (_c_id + 1) % _buffer_size;
                if (out_order_id) {
                    *out_order_id = _total_get;
                }
                _total_get += 1;
                *out_item = _buffer[m];
                // unlock.
                //LOG_NOTICE("return true: c_id=%d(%d):%p p_id=%d", m, _c_id, *out_item, _p_id);
                pthread_spin_unlock(&_spinlock);
                return true;
            }
        }

        size_t num_get() const { return _total_get; }
        size_t num_put() const { return _total_put; }

        void set_putting(bool putting) {
            _flag_putting = putting;
        }

    private:
        T*      _buffer;
        size_t  _buffer_size;
        volatile size_t  _c_id;
        volatile size_t  _p_id;
        bool    _flag_putting;

        size_t  _total_get;
        size_t  _total_put;

        pthread_spinlock_t _spinlock;
};


template<typename Job_t> 
void multi_thread_jobs(void* (func_t)(void*), Job_t* job_context, size_t job_num, size_t thread_num)
{
    size_t n=0;
    pthread_t* tids = new pthread_t[job_num];
    bool* run = new bool[thread_num];
    memset(run, 0, sizeof(bool)*thread_num);

    while (n<job_num) {
        for (size_t i=0; i<thread_num; ++i) {
            bool empty = false;
            if (!run[i]) {
                empty = true;
            } else {
                // fake kill.
                int kill_ret = pthread_kill(tids[i], 0);
                if (kill_ret == ESRCH) {
                    pthread_join(tids[i], NULL);
                    empty = true;
                }
            }

            if (empty) {
                // thread is over.
                //LOG_NOTICE("job[%d] is started @T%d.", n, i);
                pthread_create(tids+i, NULL, func_t, job_context+n);
                run[i] = true;
                n++;
                break;
            }
        }
    }
   
    // waiting for over.
    for (size_t i=0; i<thread_num; ++i) {
        if (run[i]) {
            pthread_join(tids[i], NULL);
        }
    }
    delete [] run;
    delete [] tids;
}

#endif
