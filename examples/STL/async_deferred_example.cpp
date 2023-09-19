#include <iostream>
#include <future>
#include <vector>
#include <array>
#include <random>
#include <cmath>
#include <functional>
#include <cstdint>

/**
 * @brief std::async using example with lazy evaluation policy to use a task programming paragdim.
 *
 * Some explanations : 
 *     - std::async is a STL function which launches a function with a specific policy execution :
 *            - std::launch::deferred : Lazy execution. The function is executed only when these results must be used;
 *            - std::launch::async    : Asynchronous execution. The function is executed in parallel with the next 
 *                                      instructions after the call of the function by async.
 *            - default mode          : This mode takes both policies above. This means that the function is executed in parallel
 *                                      when the computer has a free computing node to execute the function or if the program needs
 *                                      the data computed by the function.
 *     - std::async returns his data in a structure class called std::future. The std::future class has a get method to retrieve
 *       the data computed by the function called by std::async (and eventually triggs the function execution).
 *     - std::future is not copyable but is movable. This means that a future variable can be only used by one function only
 *       or we have retrieved the data in the caller of the std::async function).
 *     - Anyway, the std::async function returns immediatly !
 *
 * In this example, we called three chained function (each function must have the data computed by the previous function).
 * We print a message before calling each function and a message at the beginning at the function. Look at the message order in
 * the console and think about lazy policy execution relative to the order of the messages !
 *
 */

using Point = std::array<double,3>;

double computeDistances( Point const& t_target, std::future<Point> t_barycenter )
{
    std::cout << "Computing distances" << std::endl;
    auto barycenter = t_barycenter.get();
    Point target2Bary{ barycenter[0] - t_target[0],
                       barycenter[1] - t_target[1],
                       barycenter[2] - t_target[2] };
    return std::sqrt(target2Bary[0]*target2Bary[0] + target2Bary[1]*target2Bary[1] + target2Bary[2]*target2Bary[2]);
}

auto generateMasses( std::int64_t nbMasses ) -> std::vector<Point>
{
    std::cout << "Computing some points in 3D spaces" << std::endl;
    std::vector<Point> masses; masses.reserve(nbMasses);
    std::random_device rd1;
    std::mt19937 rd(rd1());
    std::uniform_real_distribution<double> distrib(-1., 1.);
    for ( std::int64_t iMass = 0; iMass < nbMasses; ++iMass )
    {
        masses.emplace_back(std::array{distrib(rd),distrib(rd),distrib(rd)});
    }
    return masses;
}

Point computeBarycenter( std::future<std::vector<Point>>  t_masses )
{
    std::cout << "Computing barycenter" << std::endl;
    Point bary{0.,0.,0.};
    auto masses  = t_masses.get();
    for ( auto m : masses )
        {
            bary[0] += m[0];
            bary[1] += m[1];
            bary[2] += m[2];
        }
    bary[0] /= masses.size();
    bary[1] /= masses.size();
    bary[2] /= masses.size();
    return bary;
}

int main()
{
    constexpr std::int64_t nbBodies = 10'000'000;
    std::cout << "Call mass computation" << std::endl;
    auto masses = std::async(std::launch::deferred, generateMasses, nbBodies);
    std::cout << "Call barycenter computation" << std::endl;
    auto bary   = std::async(std::launch::deferred, computeBarycenter, std::move(masses));
    std::cout << "Compute distance computation" << std::endl;
    Point target{ 2., 0., 0.};
    auto fdist   = std::async(std::launch::deferred, computeDistances, target, std::move(bary));

    std::cout << "Points distance to the target \n " << fdist.get() << std::endl;
    return EXIT_SUCCESS;
}
