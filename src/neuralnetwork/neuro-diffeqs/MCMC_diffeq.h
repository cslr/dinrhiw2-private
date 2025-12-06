
#ifndef __whiteice_MCMC_diffeq_h
#define __whiteice_MCMC_diffeq_h

#include "MCMC.h"
#include "dataset.h"
#include "nnetwork.h"
#include "vertex.h"
#include <vector>


namespace whiteice
{

  template <typename T = math::blas_real<float> >
  class MCMC_diffeq : public whiteice::math::MCMC<T>
  {
  public:
    
    MCMC_diffeq(const whiteice::nnetwork<T>& net,
		const whiteice::dataset<T>& ds,
		const std::vector<T>& times)
    {
      this->net = net;
      this->ds = ds;
      this->times = times;
    }
  
    
    // probability function of MCMC sampling
    // returns P(q) = exp(-U(q))
    T U(const math::vertex<T>& q) const;
  
  protected:

    whiteice::nnetwork<T> net;
    whiteice::dataset<T> ds;
    std::vector<T> times; // time-steps for datapoints we use
  
  };


extern template class MCMC_diffeq< math::blas_real<float> >;
extern template class MCMC_diffeq< math::blas_real<double> >;
  
};


#endif
