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

#include <string>
#include <vector>
#include <mutex>
#include <thread>
#include <atomic>


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
			   const std::vector<unsigned int>& pred_indexes,
			   const unsigned int HISTLEN,
			   const unsigned int FUTURELEN,
			   const unsigned int REDUCED_DIM = 100);
    
    bool isRunning() const;

    // reported error = E{|corrent-predicted|}, mean norm of difference
    bool getSolution(whiteice::math::vertex<T>& params, T& solution_error);

    bool stopOptimization(); // this function call MAY block if PCA+ICA preprocessing is still active..


    T getError() const; // uses predict() to calculate expected |error| per dimension


    // time-series given are from the past to the present and then future (control values p)
    bool predict
    (const std::vector< whiteice::math::vertex<T> >& x, // HISTLEN x(t-HISTLEN-1)..x(t) VECTOR ELEMENTS
     const std::vector< whiteice::math::vertex<T> >& p, // HISTLEN p(t-HISTLEN-1)..p(t) VECTOR ELEMENTS
     const std::vector< whiteice::math::vertex<T> >& p_future, // FUTURELEN p(t+1)..p(t-FUTURELEN) [FUTURELEN-1 elements]
     whiteice::math::vertex<T>& y) const; // predicted vector y=E{x(t+FUTURELEN)}


    // saves and loads model to/from disk [saves multiple files]
    bool save(const std::string& filename) const; // FIXME don't save dimension reduction (broken)
    
    bool load(const std::string& filename); // FIXME don't load dimension reduction (broken)
    
    
  private:


    // calculates ICA dimension reduction and starts NNGRradDescent<> optimizer
    void optimize_function();

    std::atomic<bool> optimize_running;
    std::thread* optimize_thread = nullptr;
    

    mutable std::mutex compute_mutex;
    
    whiteice::math::NNGradDescent<T> sgd; // neural network optimizer


    // parameters of the model (updated by calls to functions), used by predict()
    
    mutable std::mutex params_mutex;

    unsigned int HISTLEN = 0;
    unsigned int FUTURELEN = 0;
    unsigned int REDUCED_DIM = 0;


    mutable whiteice::nnetwork<T> net;
    whiteice::dataset<T> xdata;
    whiteice::dataset<T> pdata;
    std::vector<unsigned int> pred_indexes;

    whiteice::math::matrix<T> ICA_reduce;
    whiteice::math::vertex<T> ICA_mean_reduce; // parameters

  };

  
  extern template class NARX< whiteice::math::blas_real<float> >;
  extern template class NARX< whiteice::math::blas_real<double> >;
  
}


#endif
