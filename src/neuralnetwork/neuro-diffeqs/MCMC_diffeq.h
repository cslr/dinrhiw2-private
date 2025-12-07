
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

    // sequence length tells how long time periods simulate diff.eq. until reset
    MCMC_diffeq(const whiteice::nnetwork<T>& net,
		const whiteice::dataset<T>& ds,
		const std::vector<T>& times,
		const whiteice::math::vertex<T>& diffeq_starting_point,
		const unsigned int SEQUENCE_LENGTH = 15)
    {
      this->net = net;
      this->ds = ds;
      this->times = times;
      this->diffeq_starting_point = diffeq_starting_point;

      if(SEQUENCE_LENGTH > times.size())
	this->SEQUENCE_LENGTH = times.size();
      else
	this->SEQUENCE_LENGTH = SEQUENCE_LENGTH;
    }
  
    
    // probability function of MCMC sampling
    // returns P(q) = exp(-U(q))
    T U(const math::vertex<T>& q) const;
  
  protected:

    unsigned int SEQUENCE_LENGTH = 15;

    whiteice::nnetwork<T> net;
    whiteice::dataset<T> ds;
    std::vector<T> times; // time-steps for datapoints we use
    whiteice::math::vertex<T> diffeq_starting_point;
  
  };


  extern template class MCMC_diffeq< math::blas_real<float> >;
  extern template class MCMC_diffeq< math::blas_real<double> >;
  
};


#endif
