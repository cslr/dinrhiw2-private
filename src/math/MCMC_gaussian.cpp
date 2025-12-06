
#include "MCMC_gaussian.h"


namespace whiteice
{
  namespace math
  {
    template <typename T>
    MCMC_gaussian<T>::MCMC_gaussian()
    {
    }

    template <typename T>
    MCMC_gaussian<T>::~MCMC_gaussian()
    {
    }

    template <typename T>
    T MCMC_gaussian<T>::U(const whiteice::math::vertex<T>& q) const
    {
      if(q.size() == 0) return T(0.0);
      else return T(0.5)*((q * q)[0]); // log(N(0,I)) and remove scaling because we divide p1/p2
    }
    
    

    template class MCMC_gaussian< whiteice::math::blas_real<float> >;
    template class MCMC_gaussian< whiteice::math::blas_real<double> >;
    
  }
}
