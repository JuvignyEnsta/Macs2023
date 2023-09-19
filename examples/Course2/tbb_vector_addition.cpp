#include <tbb/tbb.h>
#include <vector>
#include <utility>
#include <cassert>
#include <iostream> 

std::pair<std::vector<float>,std::vector<float>> 
assembleVectors( std::size_t dim )
{
    assert(dim > 0);
    std::vector<float> u(dim);
    std::vector<float> v(dim);

    tbb::parallel_for( 0, int(dim), [&u, &v] (std::int64_t i) {
        u[i] = -0.49f*i + 1.f;
        v[i] =  0.50f*i - 1.f;
    });

    return {u, v};
}

std::vector<float>
addVectors( std::vector<float> const& u, std::vector<float>& v )
{
    assert(u.size() == v.size());
    std::vector<float> w(u.size());

    tbb::parallel_for( 0, int(u.size()), [&u, &v, &w] (std::int64_t i) {
        w[i] = u[i] + v[i];
    });
    return w;
}

int main(int nargs, char* argv[])
{
    std::size_t N = 360;

    auto [u,v] = assembleVectors(N);

    auto w = addVectors(u,v);

    std::cout << "w = [ ";
    for (auto const& val : w )
        std::cout << val << "\t";
    std::cout << "]" << std::endl;

    return EXIT_SUCCESS;
}