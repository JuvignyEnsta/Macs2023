#include <iostream>
#include <omp.h>

int main()
{
    omp_display_env(1);
    int nbProcs;
# pragma omp parallel 
    {
        nbProcs = omp_get_num_procs();
        std::cout << "Hello from " << omp_get_thread_num() << std::endl;
    }
    std::cout << "Number of threads by default in parallel context : " << nbProcs << std::endl;
    return EXIT_SUCCESS;
}