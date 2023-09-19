#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <omp.h>

int main()
{
#   pragma omp parallel sections
    {
#     pragma omp section
      {
        std::cout << "Ici c'est le thread " << omp_get_thread_num() << std::endl;
      }
#     pragma omp section
      {
        std::cout << "Et la c'est le thread " << omp_get_thread_num() << std::endl;
      }
#     pragma omp section
      {
        std::cout << "C'est donc le thread " << omp_get_thread_num() << std::endl;
      }
#     pragma omp section
      {
        std::cout << "Section avec le thread " << omp_get_thread_num() << std::endl;
      }
    }

    return EXIT_SUCCESS;
}