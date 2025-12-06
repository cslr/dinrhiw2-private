/*
 * Test MCMC sampler with target gaussian/normal probability distribution
 * 
 */

#ifndef __whiteice_MCMC_gaussian_h
#define __whiteice_MCMC_gaussian_h

#include "MCMC.h"

namespace whiteice
{
  namespace math
  {

    template <typename T = whiteice::math::blas_real<float> >
    class MCMC_gaussian
    {
    public:
      MCMC_gaussian();
      virtual ~MCMC_gaussian();
      
      // probability functions MCMC sampling from
      // P(q) ~ exp(-U(q)) distribution
      virtual T U(const whiteice::math::vertex<T>& q) const;
    };

    
    extern template class MCMC_gaussian< whiteice::math::blas_real<float> >;
    extern template class MCMC_gaussian< whiteice::math::blas_real<double> >;
    
  };
};


#endif
