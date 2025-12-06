/*
 * simple Monte Carlo Markov Chain sampler class
 */

#ifndef __whiteice_MCMC_h
#define __whiteice_MCMC_h

#include <vector>
#include <unistd.h>

#include "vertex.h"
#include "matrix.h"
#include "dinrhiw_blas.h"

#include <thread>
#include <mutex>
#include <chrono>
#include <atomic>


namespace whiteice
{
  namespace math
  {
    
    template <typename T = whiteice::math::blas_real<float> >
    class MCMC
    {
    public:
      MCMC();
      virtual ~MCMC();

      // probability functions MCMC sampling from
      // P(q) ~ exp(-U(q)) distribution
      virtual T U(const whiteice::math::vertex<T>& q) const = 0;
      
      
      bool startSampler(whiteice::math::vertex<T>& starting_point);
      bool stopSampler();
      
      unsigned int getSamples(std::vector< whiteice::math::vertex<T> >& samples) const;
      unsigned int getNumberOfSamples() const;

      whiteice::math::vertex<T> getMean() const;
      
      // calculates mean probability log(P) for the latest N samples, 0 = all samples
      T getMeanProbability(unsigned int latestN = 0) const;

    private:

      whiteice::math::vertex<T> starting_point;

      std::vector< math::vertex<T> > samples;
      unsigned int sum_N = 0;
      whiteice::math::vertex<T> sum_mean;

      std::atomic<bool> running;

      mutable std::thread* sampling_thread = nullptr;
      mutable std::mutex solution_mutex, thread_mutex;
      
      void sampler_loop();
    };
    


    extern template class MCMC< whiteice::math::blas_real<float> >;
    extern template class MCMC< whiteice::math::blas_real<double> >;
  };
};


#endif

