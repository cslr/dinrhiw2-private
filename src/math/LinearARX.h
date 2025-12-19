/*
 * Simple linear time-series x(t) prediction with control parameters p(t)
 * which try to cause changes in x(t)
 *
 * Eq. x(t+1) = A*[x(t)...x(t-HISTLEN)] + B*[p(t)...p(t-HISTLEN)]
 *
 * Minimizes squared mean error and directly computes optimal solution.
 */

#ifndef __whiteice_LinearARX_h
#define __whiteice_LinearARX_h

#include "blade_math.h"
#include "dataset.h"
#include "vertex.h"
#include "matrix.h"

#include <vector>


namespace whiteice
{
  namespace math
  {

    template <typename T=whiteice::math::blas_real<float> >
    class LinearARX
    {
    public:
      LinearARX();
      virtual ~LinearARX();

      // computes solution directly without thread, modern computers should be fast enough
      bool computeSolution(const whiteice::dataset<T>& xdata,
			   const whiteice::dataset<T>& pdata,
			   const unsigned int HISTLEN);

      bool predict
      (const std::vector< whiteice::math::vertex<T> >& x, // HISTLEN x(t)..x(t-HISTLEN-1) VECTOR ELEMENTS
       const std::vector< whiteice::math::vertex<T> >& p, // HISTLEN p(t)..p(t-HISTLEN-1) VECTOR ELEMENTS
       whiteice::math::vertex<T>& y); // predicted vector y=x(t+1)


      bool randomizeSolutionMatrices();

      
      // reports average norm error of prediction using training dataset [should use never seen data]
      T getError();
      

    private:

      // parameters of the model
      
      whiteice::math::matrix<T> A, B;

      whiteice::dataset<T> xdata;
      whiteice::dataset<T> pdata;
      unsigned int HISTLEN = 0;
      
      
    };



    extern template class LinearARX< blas_real<float> >;
    extern template class LinearARX< blas_real<double> >;

    
  };

  
};


#endif
