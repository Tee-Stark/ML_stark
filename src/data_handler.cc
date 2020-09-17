
#include "data_handler.hpp"
#include <iostream>


data_handler::data_handler()
{
    data_array = new vector<Data*>;
    training_data = new vector<Data*>;
    test_data = new vector<Data*>;
    validation_data = new vector<Data*>;
}
data_handler::~data_handler()
{
    //Free dynamically allocated memory
}

void data_handler::read_feature_vector(string path)
{
    uint32_t header[4];      //[MAGIC][NUM IMAGE][ROWSIZE][COLSIZE]
    unsigned char bytes[4];
    FILE *f = fopen(path.c_str(), "rb");
    if (f)
    {
        for (int i = 0; i < 4; i++)
        {
            if(fread(bytes, sizeof(bytes), 1, f))
            {
                header[i] = conv_little_endian(bytes);
            }
        }
        cout << "Done getting File Header." << endl;
        uint32_t image_size = header[2] * header[3];
        for(int i = 0; i < header[1]; i++)
        {
            Data *d = new Data();
            d->set_feature_vector(new vector<uint8_t>());
            uint8_t element[1];
            for(int j = 0; j < image_size; j++)
            {
                if(fread(element, 1, 1, f))
                {
                    d->append_to_feature_vector(element[0]);
                } else
                {
                    cerr << "Error reading from file." << endl;
                    exit(1);
                }
            }
            data_array->push_back(d);
        }
        cout << "Successfully read and stored " 
             << data_array->size() << " feature vectors." 
             << endl;
    } else
    {
        cerr << "Could not open file." << endl;
        exit(1);
    }
    
    
}
void data_handler::read_feature_labels(string path)
{
    uint32_t header[2];      //[MAGIC][NUM IMAGES]
    unsigned char bytes[4];
    FILE *f = fopen(path.c_str(), "rb");
    if (f)
    {
        for (int i = 0; i < 2; i++)
        {
            if(fread(bytes, sizeof(bytes), 1, f))
            {
                header[i] = conv_little_endian(bytes);
            }
        }
        cout << "Done getting label file header." << endl;
        for(int i = 0; i < header[1]; i++)
        {
            uint8_t element[1];
            if(fread(element, sizeof(element), 1, f))
            {
                data_array->at(i)->set_label(element[0]);
            } else
            {
                cerr << "Error reading from file." << endl;
                exit(1);
            } 
        }
        cout << "Successfully read and stored labels." << endl;
     } else
    {
        cerr << "Could not open file." << endl;
        exit(1);
    }
}
void data_handler::split_data()
{
    unordered_set<int> used_indexes;
    int train_size = data_array->size() * TRAIN_SET_PERCENT;
    int test_size = data_array->size() * TEST_SET_PERCENT;
    int valid_size = data_array->size() * VALIDATION_PERCENT;

    //TRAINING DATA
    int count = 0;
    while(count < train_size)
    {
        int rand_index = (rand() + rand()) % data_array->size(); //0 % data_array->size() = 1
        if (used_indexes.find(rand_index) == used_indexes.end())
        {
            training_data->push_back(data_array->at(rand_index));
            used_indexes.insert(rand_index);
            count++;
        }    
    }
    
    //TEST DATA
    count = 0;
    while(count < test_size)
    {
        int rand_index = (rand() + rand()) % data_array->size(); //0 % data_array->size() = 1
        if (used_indexes.find(rand_index) == used_indexes.end())
        {
            test_data->push_back(data_array->at(rand_index));
            used_indexes.insert(rand_index);
            count++;
        }    
    }
    //VALIDATION DATA
    count = 0;
    while(count < valid_size)
    {
        int rand_index = (rand() + rand()) % data_array->size(); //0 % data_array->size() = 1
        if (used_indexes.find(rand_index) == used_indexes.end())
        {
            validation_data->push_back(data_array->at(rand_index));
            used_indexes.insert(rand_index);
            count++;
        }    
    }

    cout << "Training data size: " << training_data->size() << endl ;
    cout << "Test data size: " << test_data->size() << endl;
    cout << "Validation data size: " << validation_data->size() << endl;

}
void data_handler::count_classes()
{
    int count = 0;
    for(int i = 0; i < data_array->size(); i++)
    {
        if (class_map.find(data_array->at(i)->get_label()) == class_map.end())
        {
            class_map[data_array->at(i)->get_label()] = count;
            data_array->at(i)->set_enum_label(count);
            count++;
        }
    }
    num_classes = count;
    cout << "Successfully extracted " << num_classes << " Unique classes." << endl;

}

uint32_t data_handler::conv_little_endian(const unsigned char* bytes)
{
    return (uint32_t) ((bytes[0] << 24) | 
                       (bytes[1] << 16) |
                       (bytes[2] << 8)  |
                       (bytes[3]));
}

vector<Data*> *data_handler::get_training_data()
{
    return training_data;
}
vector<Data*> *data_handler::get_test_data()
{
    return test_data;
}
vector<Data*> *data_handler::get_validation_data()
{
    return validation_data;
}

/*
int main()
{
    data_handler *dh = new data_handler();
    dh->read_feature_vector("./train-images.idx3-ubyte");
    dh->read_feature_labels("./train-labels.idx1-ubyte");
    dh->split_data();
    dh->count_classes();
}
*/
