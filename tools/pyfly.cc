

#include "python2.7/Python.h"
#include <cstdio>

#include "gbdt.h" 


static PyObject* predict(PyObject *self, PyObject *args)
{
    Instance_t ins;
    long int handle;
    char* line = NULL;
    int res = PyArg_ParseTuple(args, "ls", &handle, &line);
    if (!res) {
        fprintf(stderr, "predict parse input failed!\n");
        return Py_BuildValue("l", -2);
    }
    ins.parse_item(line);
    GBDT_t* p_model = (GBDT_t*)handle;
    float pres = p_model->predict(ins);
    return Py_BuildValue("f", pres);

}

static PyObject* tree_features(PyObject *self, PyObject *args)
{
    Instance_t ins;
    long int handle;
    char* line = NULL;
    int res = PyArg_ParseTuple(args, "ls", &handle, &line);
    if (!res) {
        fprintf(stderr, "predict parse input failed!\n");
        return Py_BuildValue("l", -2);
    }
    ins.parse_item(line);

    GBDT_t* p_model = (GBDT_t*)handle;

    int predict_tree_count = p_model->get_predict_tree_cut();
    int *leaves = new int[predict_tree_count];
    p_model->predict_and_get_leaves(ins, leaves, NULL);

    PyObject* ans_list = PyList_New(predict_tree_count);
    int tree_node_count = p_model->tree_node_count();;
    for (int i=0; i<predict_tree_count; ++i) {
        int s = i*tree_node_count + leaves[i];
        PyList_SET_ITEM(ans_list, i, PyInt_FromLong(s));
    }
    delete [] leaves;
    return ans_list;
}

static PyObject* load_model_cutted(PyObject *self, PyObject *args)
{
    char* model_file = NULL;
    int tree_cut;
    int res = PyArg_ParseTuple(args, "is", &tree_cut, &model_file);
    if (!res) {
        fprintf(stderr, "parse args failed!\n");
        return Py_BuildValue("l", -2);
    }
    Config_t nil_config;
    GBDT_t* p_model = new GBDT_t(nil_config, "");
    FILE* fstream = fopen(model_file, "r");
    p_model->read_model(fstream);
    fclose(fstream);

    p_model->set_predict_tree_cut(tree_cut);

    long int handle = (long int)p_model;
    return Py_BuildValue("l",handle);
}

static PyObject* load_model(PyObject *self, PyObject *args)
{
    char* model_file = NULL;
    int res = PyArg_ParseTuple(args,"s", &model_file);
    if (!res) {
        fprintf(stderr, "parse args failed!\n");
        return Py_BuildValue("l", -1);
    }
    Config_t nil_config;
    FlyModel_t* p_model = new GBDT_t(nil_config, "");

    FILE* fstream = fopen(model_file, "r");
    p_model->read_model(fstream);
    fclose(fstream);
    long int handle = (long int)p_model;
    return Py_BuildValue("l",handle);
}

static PyObject* release(PyObject *self, PyObject *args)
{
    long int handle;
    int res = PyArg_ParseTuple(args, "l", &handle);
    if (!res) {
        fprintf(stderr, "parse release handle failed!\n");    
    }
    GBDT_t* p_model = (GBDT_t*)handle;
    delete p_model;
    return Py_BuildValue("l", 1);
}


static PyMethodDef gbdtMethods[]={
    {"load",load_model,METH_VARARGS},
    {"load_cut",load_model_cutted,METH_VARARGS},
    {"predict",predict,METH_VARARGS},
    {"tree_features",tree_features,METH_VARARGS},
    {"release_trees",release,METH_VARARGS},
    {NULL,NULL}
};

PyMODINIT_FUNC initGBDT()
{
    Py_InitModule3("GBDT",gbdtMethods,"test_model");
}
