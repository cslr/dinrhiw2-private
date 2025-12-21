/*
 * NARX - simple neural network time-series x(t) prediction with control parameters p(t)
 * which try to cause changes in x(t)
 *
 * Eq. x(t+1) = neural_network([x(t)...x(t-HISTLEN), p(t)...p(t-HISTLEN)]
 *
 * Minimizes squared mean error and uses ADAM optimizer with neural network gradient
 */

#ifndef __whiteice_NARX_h
#define __whiteice_NARX_h

#include "blade_math.h"
#include "dataset.h"
#include "vertex.h"
#include "matrix.h"
#include "nnetwork.h"
#include "NNGradDescent.h"

#include <vector>
#include <mutex>
#include <thread>


namespace whiteice
{

  template <typename T=whiteice::math::blas_real<float> >
  class NARX
  {
  public:
    NARX();
    virtual ~NARX();
    
    // computes solution directly without thread, modern computers should be fast enough
    bool startOptimization(const whiteice::nnetwork<T>& net,
			   const whiteice::dataset<T>& xdata,
			   const whiteice::dataset<T>& pdata,
			   const unsigned int HISTLEN,
			   const unsigned int FUTURELEN);
    
    bool isRunning() const;

    // reported error = E{|corrent-predicted|}, mean norm of difference
    bool getSolution(whiteice::math::vertex<T>& params, T& solution_error);

    bool stopOptimization();


    // time-series given are from the past to the present and then future (control values p)
    bool predict
    (const std::vector< whiteice::math::vertex<T> >& x, // HISTLEN x(t-HISTLEN-1)..x(t) VECTOR ELEMENTS
     const std::vector< whiteice::math::vertex<T> >& p, // HISTLEN p(t-HISTLEN-1)..p(t) VECTOR ELEMENTS
     const std::vector< whiteice::math::vertex<T> >& p_future, // FUTURELEN p(t+1)..p(t-FUTURELEN) [FUTURELEN-1 elements]
     whiteice::math::vertex<T>& y) const; // predicted vector y=E{x(t+FUTURELEN)}
    
    
  private:

    mutable std::mutex compute_mutex;
    
    whiteice::math::NNGradDescent<T> sgd; // neural network optimizer


    // parameters of the model (updated by calls to functions), used by predict()

    unsigned int HISTLEN = 1;
    unsigned int FUTURELEN = 1;

    mutable whiteice::nnetwork<T> net;
    whiteice::dataset<T> xdata;
    whiteice::dataset<T> pdata;
  };

  
  extern template class NARX< whiteice::math::blas_real<float> >;
  extern template class NARX< whiteice::math::blas_real<double> >;
  
}


#endif
