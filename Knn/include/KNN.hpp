#ifndef  __KNN_H
#define  __KNN_H

#include <vector>
#include "data.hpp"

using namespace std;
class Knn
{
private:
    int k;
    vector<Data *> *neighbors;
    vector<Data *> *training_data;
    vector<Data *> *test_data;
    vector<Data *> *validation_data;
public:
    Knn(int);
    Knn();
    ~Knn();

    void find_knearest(Data* query_point);
    void set_training_data(vector<Data *> *vect);
    void set_test_data(vector<Data *> *vect);
    void set_validation_data(vector<Data *> *vect);
    void set_k(int val);

    int predict();
    double calculate_distance(Data* query_point, Data* input);
    double validate_performance();
    double test_peformance();
};

#endif 
