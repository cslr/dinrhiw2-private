/*
 * calculates standard 4th order Runge-Kutta (RK4) integration
 * with adaptive step length
 *
 * Uses HISTORY_LEN older values y[t-1,..t-HISTORY_LEN]
 * in ODE function with linear interpolation
 * 
 * ODE function must take vector parameter with dimension: dim(x)*(HISTORY_LEN+1)
 * 
 * TODO: later make this start computation
 * to the own thread + ETA updates + ability to
 * access partial results + ability stop/continue computations
 * if needed
 */

#ifndef RungeKutta_history_h
#define RungeKutta_history_h

#include "odefunction.h"
#include "vertex.h"
#include "RNG.h"
#include <vector>
#include <map>


namespace whiteice
{
  namespace math
  {
    template <typename T>
      class RungeKutta_history
      {
      public:
	RungeKutta_history(odefunction<T>* f = 0);
	~RungeKutta_history();
	
	odefunction<T>* getFunction() const ;
	void setFunction(odefunction<T>* f) ;

	T getSigma() const { return sigma_term; }
	void setSigma(const T& s){ if(s >= T(0.0)) sigma_term = s; }

	unsigned int getHistoryLength() const { return HISTORY_LEN; }
	void setHistoryLength(unsigned int history_len){ this->HISTORY_LEN = history_len; }
	
	
	// calculates values from the starting point y0
	// with adaptive step length, adds results to the end of vector
	// (h_new = h_old (e0/e)^(1/5)), initial h0 = 10e-4, e0=10e-8
	// errors are absolute
	void calculate(const T t0, const T t_end,
		       const whiteice::math::vertex<T>& y0,
		       std::vector< whiteice::math::vertex<T> >& points,
		       std::vector< T >& times);
	
      private:

	// finds closest point in historical values and linearly intepolates
	// good vector from historical values
	whiteice::math::vertex<T> linearly_interpolate_find
	(const T& t,
	 const std::map<T, whiteice::math::vertex<T> >& map_points,
	 const unsigned int DIM) const;

	// calculates ODE vector parameter from historical values
	bool calculate_parameters
	(const whiteice::math::vertex<T>& y,
	 const T& t,
	 const std::map<T, whiteice::math::vertex<T> >& map_points,
	 const unsigned int HISTORY_LEN,
	 whiteice::math::vertex<T>& ode_y) const;
	
	
	odefunction<T>* f;

	unsigned int HISTORY_LEN = 0;

	// Stochastic diff.eq.
	// supports addition of gaussian noise term with sigma^2 variance
	T sigma_term = T(0.0);

	const whiteice::RNG<T> random;
      };
    
    
    //////////////////////////////////////////////////////////////////////
    
    extern template class RungeKutta_history< blas_real<float> >;
    extern template class RungeKutta_history< blas_real<double> >;
    extern template class RungeKutta_history< blas_complex<float> >;
    extern template class RungeKutta_history< blas_complex<double> >;
    
  };
};



#endif

