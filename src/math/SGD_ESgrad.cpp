
#include "SGD_ESgrad.h"
#include "SGD.h"


namespace whiteice
{
  namespace math
  {
    template <typename T>
    SGD_ESgrad<T>::SGD_ESgrad(bool overfit) : SGD<T>(overfit)
    {
      
    }

    template <typename T>
    SGD_ESgrad<T>::~SGD_ESgrad()
    {
      
    }

    template <typename T>
    vertex<T> SGD_ESgrad<T>::Ugrad(const vertex<T>& x) const
    {
      // estimates gradient
      // gradient estimator calls U(x)
      // 2*NUM_SAMPLES times which may be SLOW
      
      vertex<T> grad = x;
      
      grad.zero();

#pragma omp parallel
      {
	vertex<T> epsilon = x;
	whiteice::math::vertex<T> g = grad;

	epsilon.zero();
	g.zero();

#pragma omp for nowait schedule(auto)
	for(unsigned int n=0;n<NUM_SAMPLES;n++)
	{
	  random.normal(epsilon);
	  
	  auto s_epsilon = epsilon;
	  s_epsilon *= sigma;
	  
	  g += (this->U(x + s_epsilon) - this->U(x - s_epsilon))*epsilon;
	}

#pragma omp critical
	{
	  grad += g;
	}
      }
	
      grad /= T(2.0)*NUM_SAMPLES*sigma;

      return grad;
    }




    template class SGD_ESgrad< blas_real<float> >;
    template class SGD_ESgrad< blas_real<double> >;

    template class SGD_ESgrad< superresolution<
      blas_real<float>,
      modular<unsigned int> > >;
    
    template class SGD_ESgrad< superresolution<
      blas_real<double>,
      modular<unsigned int> > >;
    
  }
}
