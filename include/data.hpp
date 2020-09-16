#ifndef __DATA_H
#define __DATA_H
#include <vector>
#include "stdint.h"
#include "stdio.h"

using namespace std;
class Data
{
private: 
    vector<uint8_t>* feature_vector; //no class at end
    uint8_t label;
    int enum_label;
    double distance;
public:
    Data();
    ~Data();
    void set_feature_vector(vector<uint8_t>*);
    void append_to_feature_vector(uint8_t);
    void set_label(uint8_t);
    void set_enum_label(int);
    void set_distance(double val);
    
    double get_distance();
    int get_feature_vector_size();
    uint8_t get_label();
    int get_enum_label();

    vector<uint8_t>* get_feature_vector();
};

#endif