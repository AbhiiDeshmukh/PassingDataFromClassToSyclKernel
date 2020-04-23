#include <CL/sycl.hpp>
#include <iostream>
#include <array>
#include <cstdio>
using namespace std;
using namespace cl::sycl;

#define SIZE 20

struct DeviceData
{
    queue q;

    int* A;
    int* d_A;

    void init()
    {
        q = queue(gpu_selector{});

        A = (int* )malloc(sizeof(int)*SIZE);
        for(int i=0; i<SIZE; ++i)
            A[i] = i+1;

        d_A = (int *)malloc_device(sizeof(int)*SIZE, q.get_device(), q.get_context());
    }

    void test()
    {
        cout << "Before" << "\n";
        for(int i=0; i<SIZE; ++i)
        {
            cout << A[i] << " ";
        }
        cout << "\n";

        q.memcpy(d_A, A, sizeof(int)*SIZE);
        q.parallel_for(range<1>{SIZE}, [=,d_A_local=this->d_A](id<1> i){
                d_A_local[i] += 10;
        });
        q.memcpy(A, d_A, sizeof(int)*SIZE).wait();

        cout << "After" << "\n";
        for(int i=0; i<SIZE; ++i)
            cout << A[i] << " ";
        cout << "\n";
    }
    void free()
    {
            std::free(A);
            sycl::free(d_A, q.get_context());
    }
};

int main()
{
    DeviceData dev;
    dev.init();
    dev.test();
    dev.free();
    return 0;
}
