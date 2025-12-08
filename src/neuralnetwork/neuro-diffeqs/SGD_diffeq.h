/*
 * SGD non-linear diff.eq. neural network optimizer
 *
 */

#ifndef __whiteice_SGD_diffeq_h
#define __whiteice_SGD_diffeq_h

#include "SGD_ESgrad.h"
#include "nnetwork.h"
#include "dataset.h"
#include "vertex.h"


namespace whiteice
{
  
  template <typename T=whiteice::math::blas_real<float> >
  class SGD_diffeq : public whiteice::math::SGD_ESgrad<T>
  {
  public:
    SGD_diffeq(const whiteice::nnetwork<T>& net,
	       const whiteice::dataset<T>& ds,
	       const std::vector<T>& times,
	       const whiteice::math::vertex<T>& diffeq_starting_point,
	       const unsigned int SEQUENCE_LENGTH = 15,
	       bool overfit = false) : whiteice::math::SGD_ESgrad<T>(overfit)
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
 
    virtual ~SGD_diffeq(){ }


    virtual T getError(const whiteice::math::vertex<T>& x) const;

  protected:

    // diff.eq. function: prediction error
    virtual T U(const whiteice::math::vertex<T>& x) const;

    // heuristically improve solution x during SGD optimization
    virtual bool heuristics(whiteice::math::vertex<T>& x) const;
    
  private:

    unsigned int SEQUENCE_LENGTH = 15;
    
    whiteice::nnetwork<T> net;
    whiteice::dataset<T> ds;
    std::vector<T> times; // time-steps for datapoints we use
    whiteice::math::vertex<T> diffeq_starting_point;
    
  };


  extern template class SGD_diffeq< whiteice::math::blas_real<float> >;
  extern template class SGD_diffeq< whiteice::math::blas_real<double> >;
  
};

#endif
