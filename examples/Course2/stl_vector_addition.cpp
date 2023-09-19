#include <vector>
#include <utility>
#include <cassert>
#include <iostream> 
#include <execution>
#include <algorithm> 

std::pair<std::vector<float>,std::vector<float>> 
assembleVectors( std::size_t dim )
{
    assert(dim > 0);
    std::vector<float> u(dim);
    std::vector<float> v(dim);
    std::for_each(std::execution::par_unseq, u.begin(), u.end(), [&u,&v](float const& coef) {
        auto i = std::distance((const float*)u.data(), &coef);
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
    std::for_each(std::execution::par_unseq, u.begin(), u.end(), [&u,&v, &w](float const& coef) {
        auto i = std::distance((const float*)u.data(), &coef);
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