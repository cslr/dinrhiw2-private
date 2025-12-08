/*
 * SGD extension which approximates gradient using
 * OpenAI ES-gradient and Antithetic Sampling
 *
 * User must only provide function U(x), gradient is not needed
 */

#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>

#include "dinrhiw_blas.h"
#include "vertex.h"
#include "RNG.h"
#include "SGD.h"

#include "superresolution.h"

#ifndef __whiteice_SGD_ESgrad_h
#define __whiteice_SGD_ESgrad_h

namespace whiteice
{
  namespace math
  {
    template <typename T=blas_real<float> >
    class SGD_ESgrad : public whiteice::math::SGD<T>
    {
    public:
      // overfit: do not use early stopping via getError() function
      SGD_ESgrad(bool overfit=false);      
      virtual ~SGD_ESgrad();
      
    protected:
      /* gradient function */
      virtual vertex<T> Ugrad(const vertex<T>& x) const;
      
      // user must still define function U(x) and heuristics(x) to
      // possible improve/fix x after Adam optimizer gradient step

    private:

      const unsigned int NUM_SAMPLES = 100; // was: 50
      const T sigma = T(1e-2); // assumes functiona input scaling is 1.0
      
      const whiteice::RNG<T> random;
    };


    
    extern template class SGD_ESgrad< blas_real<float> >;
    extern template class SGD_ESgrad< blas_real<double> >;

    extern template class SGD_ESgrad< superresolution<
      blas_real<float>,
      modular<unsigned int> > >;
    
    extern template class SGD_ESgrad< superresolution<
      blas_real<double>,
      modular<unsigned int> > >;
    
  };
};



#endif
