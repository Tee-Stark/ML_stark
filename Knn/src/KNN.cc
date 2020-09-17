#include "../include/KNN.hpp"
#include <iostream>
#include <cmath>
#include <limits>
#include <map>
#include "stdint.h"
#include "data_handler.hpp"
#include <iomanip>

Knn::Knn(int val)
{
    k = val;
}
Knn::Knn()
{
    //DEFAULT CONSTRUCTOR
}
Knn::~Knn()
{
    //DESTRUCTOR
}

void Knn::find_knearest(Data* query_point)
{
    neighbors = new  vector<Data *>;
    double min = numeric_limits<double>::max();
    double prev_min = min;
    int index = 0;
    for (int i = 0; i < k; i++)
    {
        if (i == 0)
        {
            for(int j = 0; j < training_data->size(); j++)
            {
                double distance = calculate_distance(query_point, training_data->at(j));
                training_data->at(j)->set_distance(distance);
                if (distance < min)
                {
                    min = distance;
                    index = j;
                } 
            }
            neighbors->push_back(training_data->at(index));
            prev_min = min;
            min = numeric_limits<double>::max();
        } else
        {
            for (int j = 0; j < training_data->size(); j++)
            {
                double distance = training_data->at(j)->get_distance();
                if(distance > prev_min && distance < min)
                {
                    min = distance;
                    index = j;
                }
            }
            neighbors->push_back(training_data->at(index));
            prev_min = min;
            min = numeric_limits<double>::max();
        }
    }
    
}
void Knn::set_training_data(vector<Data *> *vect)
{
    training_data = vect;
}
void Knn::set_test_data(vector<Data *> *vect)
{
    test_data = vect;
}
void Knn::set_validation_data(vector<Data *> *vect)
{
    validation_data = vect;
}
void Knn::set_k(int val)
{
    k = val;
}

int Knn::predict()
{
   map<uint8_t, int> class_freq;
   for (int i = 0; i < neighbors->size(); i++)
   {
       if (class_freq.find(neighbors->at(i)->get_label()) == class_freq.end())
       {
           class_freq[neighbors->at(i)->get_label()] = 1;
       } else
       {
           class_freq[neighbors->at(i)->get_label()]++;
       }
   }
   int best = 0;
   int max = 0;
   for (auto kv : class_freq)
   {
       if (kv.second > max)
       {
           max = kv.second;
           best = kv.first;
       }
   }
   neighbors->clear();
   return best;
}
double Knn::calculate_distance(Data* query_point, Data* input)
{
    double distance = 0.0;
    if(query_point->get_feature_vector_size() != input->get_feature_vector_size())
    {
        cerr << "Error! vector size mismatch." << endl;
        exit(1);
    }
#ifdef EUCLID
    for(unsigned i = 0; i < query_point->get_feature_vector_size(); i++)
    {
        distance += pow(query_point->get_feature_vector()->at(i) - input->get_feature_vector()->at(i), 2);
    }
    distance = sqrt(distance);
#elif defined MANHATTAN
//PUT MANHATTAN IMPLEMENTATION HERE
#endif
    return distance;
}
double Knn::validate_performance()
{
    double current_perf = 0;
    int count = 0;
    int data_index = 0;
    for (Data* query_point : *validation_data)
    {
        find_knearest(query_point);
        int prediction = predict();
        if (prediction == query_point->get_label())
        {
            count++;
        }
        data_index++;
        cout << "Current peformance = "
             << fixed << setprecision(3)
             << ((double)count * 100.0)/(double)data_index << endl; 
    }
    current_perf = ((double)count * 100.0)/(double)validation_data->size();
    cout << "Validation Performance for k = " << k << " = "
         << fixed << setprecision(3) << current_perf
         << endl; 
    return current_perf;
}
double Knn::test_peformance()
{
    double current_perf = 0;
    int count = 0;
    for (Data* query_point : *test_data)
    {
        find_knearest(query_point);
        int prediction = predict();
        if (prediction == query_point->get_label())
        {
            count++;
        }
    }
    current_perf = ((double)count * 100.0)/(double)test_data->size();
    cout << "Tested performance = " 
         << fixed << setprecision(3) << current_perf
         << endl;
    return current_perf;
}

int main()
{
    data_handler* dh = new data_handler();
    dh->read_feature_vector("../train-images.idx3-ubyte");
    dh->read_feature_labels("../train-labels.idx1-ubyte");
    dh->split_data();
    dh->count_classes();

    Knn *knearest = new Knn();
    knearest->set_training_data(dh->get_training_data());
    knearest->set_test_data(dh->get_test_data());
    knearest->set_validation_data(dh->get_validation_data());
    double performance = 0;
    double best_performance = 0;
    int best_k = 1;
    for (int i = 0; i < 4; i++)
    {
        if (i == 1)
        {
            knearest->set_k(i);
            performance = knearest->validate_performance();
            best_performance = performance;
        } else
        {
            knearest->set_k(i);
            performance = knearest->validate_performance();
            if (performance > best_performance)
            {
                best_performance = performance;
                best_k = i;
            }
        }
    }
    knearest->set_k(best_k);
    knearest->test_peformance();
}
