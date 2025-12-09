
#include "RungeKutta.h"
#include "blade_math.h"


namespace whiteice
{
  namespace math
  {
    
    template <typename T>
    RungeKutta<T>::RungeKutta(odefunction<T>* f){
      this->f = f;
    }
    
    
    template <typename T>
    RungeKutta<T>::~RungeKutta(){
      
    }
    
    
    template <typename T>
    odefunction<T>* RungeKutta<T>::getFunction() const {
      return f;
    }
    
    
    template <typename T>
    void RungeKutta<T>::setFunction(odefunction<T>* f) {
      this->f = f;
    }
    
    
    template <typename T>
    void RungeKutta<T>::calculate
    (const T t0, const T t_end,
     const whiteice::math::vertex<T>& y0,
     std::vector< whiteice::math::vertex<T> >& points,
     std::vector< T >& times)
    {
      whiteice::math::vertex<T> k[4], tmp;
      whiteice::math::vertex<T> y(y0), yn;
      T t = t0;
      T ttmp;
      
      T h  = T(10e-5);
#if 0
      // for scientific accuracy
      const T e0 = T(1e-8); // (error is kept at 10e-8)
      const T h_min = T(1e-13);
      const T h_max = T(1e-2);
#endif
#if 1
      // for machine learning accuracy O(h^4) = O(0.3^4) = O(0.0081).
      const T e0 = T(1e-8); // was: 1e-8 (error was kept at 1e-2)
      const T h_min = T(0.1); // was 1e-2, 0.1
      const T h_max = T(0.3); // was 1e-1, 0.5
#endif
	
      const T f6 = T(1.0/6.0);
      const T f5 = T(1.0/5.0);
      const T f3 = T(1.0/3.0);
      const T fd = T(1.0/((double)y0.size()));

      while(t < t_end){
	// calculates next point with h and 2 x (h/2)
	// results are compared and the step length h is adjusted
	// when error is too big/small. Result from 2x(h/2) is
	// used because it is always more accurate
	
	
	// calculates the next point with a single step
	{
	  k[0] = h * (f->calculate(odeparam<T>(t, y)) );
	  
	  tmp = y + T(0.5)*k[0];
	  ttmp = t + T(0.5)*h;
	  k[1] = h * (f->calculate(odeparam<T>(ttmp, tmp)));
	  
	  tmp = y + T(0.5)*k[1];
	  ttmp = t + T(0.5)*h;
	  k[2] = h * (f->calculate(odeparam<T>(ttmp, tmp)));
	  
	  tmp = y + k[2];
	  ttmp = t + h;
	  k[3] = h * (f->calculate(odeparam<T>(ttmp, tmp)));
	  
	  yn = y + f6*(k[0]+k[3]) + f3*(k[1]+k[2]);
	}
	
	
	// calculates the next point with two steps
	{
	  T h2 = h*T(0.5);
	  
	  k[0] = h2 * (f->calculate(odeparam<T>(t, y)) );
	  
	  tmp = y + T(0.5)*k[0];
	  ttmp = t + T(0.5)*h2;
	  k[1] = h2 * (f->calculate(odeparam<T>(ttmp, tmp)));
	  
	  tmp = y + T(0.5)*k[1];
	  ttmp = t + T(0.5)*h2;
	  k[2] = h2 * (f->calculate(odeparam<T>(ttmp, tmp)));
	  
	  tmp = y + k[2];
	  ttmp = t + h2;
	  k[3] = h2 * (f->calculate(odeparam<T>(ttmp, tmp)));
	  
	  y += f6*(k[0]+k[3]) + f3*(k[1]+k[2]);
	  
	  ttmp = t + h2;
	  k[0] = h2 * (f->calculate(odeparam<T>(ttmp, y)) );
	  
	  tmp = y + T(0.5)*k[0];
	  ttmp = t + T(1.5)*h2;
	  k[1] = h2 * (f->calculate(odeparam<T>(ttmp, tmp)));
	  
	  tmp = y + T(0.5)*k[1];
	  ttmp = t + T(1.5)*h2;
	  k[2] = h2 * (f->calculate(odeparam<T>(ttmp, tmp)));
	  
	  tmp = y + k[2];
	  ttmp = t + h;
	  k[3] = h2 * (f->calculate(odeparam<T>(ttmp, tmp)));
	  
	  y += f6*(k[0]+k[3]) + f3*(k[1]+k[2]);
	}
	
	
	t += h;
	
	// calculates truncation error and adapts step length h
	
	// dimension:th root of norm
	T e = pow((y - yn).norm(), fd);
	
	// h = (h_new + h_old)/2
	h *= T(0.5)*(pow(e0/e, f5) + T(1.0));
	
	if(h < h_min) h = h_min;
	else if(h > h_max) h = h_max;

	// std::cout << "RK: h = " << h << std::endl;

	// adds stochastic differential equation gaussian noise term to y
	{
	  if(sigma_term > T(0.0)){
	    whiteice::math::vertex<T> noise;
	    noise.resize(y.size());
	    
	    random.normal(noise);
	    noise *= sigma_term*sqrt(h); // should be sqrt(dt), is sqrt(h) correct?

	    y += noise;
	  }
	}

	// handles bad and large values in values
	for(unsigned int d=0;d<y.size();d++){
	  if(y[d] < T(-1e30)) y[d] = T(0.0);
	  else if(y[d] > T(+1e30)) y[d] = T(0.0);
	  if(isnan(y[d])) y[d] = T(0.0);
	  if(isinf(y[d])) y[d] = T(0.0);
	}
	
	points.push_back(y);
	times.push_back(t);
      }
    }
    
    
    
    
    
    //////////////////////////////////////////////////////////////////////
    
    template class RungeKutta< float >;
    template class RungeKutta< double >;
    template class RungeKutta< blas_real<float> >;
    template class RungeKutta< blas_real<double> >;
    //template class RungeKutta< blas_complex<float> >;
    //template class RungeKutta< blas_complex<double> >;
  };
};
