/*
 * calculates standard 4th order Runge-Kutta (RK4) integration
 * with adaptive step length with parameters p(t)
 *
 * Uses HISTORY_LEN older values y[t-1,..t-HISTORY_LEN]
 * in ODE function with linear interpolation
 *
 * ODE function takes extra parameters p(t) of stimulation of diff.eq. system which
 * are provided by caller. p values are at specified time p(t0),p(t1),p(t2).. and p(t) value
 * is calculated by finding nearest t value (p(t)) and linearly interpolating between points.
 *
 * ODE function must take vector parameter with dimension: dim(x)*(HISTORY_LEN+1)+dim(p)
 * 
 * TODO: later make this start computation
 * to the own thread + ETA updates + ability to
 * access partial results + ability stop/continue computations
 * if needed
 */

#ifndef RungeKuttaP_history_h
#define RungeKuttaP_history_h

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
      class RungeKuttaP_history
      {
      public:
	RungeKuttaP_history(odefunction<T>* f = 0);
	~RungeKuttaP_history();
	
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
		       // p(t) diff.eq. stimulation parameters
		       const std::vector< whiteice::math::vertex<T> >& parameters,
		       const std::vector< T > & parameter_times,
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
	bool calculate_ode_parameters
	(const whiteice::math::vertex<T>& y,
	 const T& t,
	 const std::map<T, whiteice::math::vertex<T> >& map_points,
	 const unsigned int HISTORY_LEN,
	 const std::map<T, whiteice::math::vertex<T> > map_parameters,
	 whiteice::math::vertex<T>& ode_y) const;
	
	
	odefunction<T>* f;

	unsigned int HISTORY_LEN = 0;

	// Stochastic diff.eq.
	// supports addition of gaussian noise term with sigma^2 variance
	T sigma_term = T(0.0);

	const whiteice::RNG<T> random;
      };
    
    
    //////////////////////////////////////////////////////////////////////
    
    extern template class RungeKuttaP_history< float >;
    extern template class RungeKuttaP_history< double >;
    extern template class RungeKuttaP_history< blas_real<float> >;
    extern template class RungeKuttaP_history< blas_real<double> >;
    //extern template class RungeKutta< blas_complex<float> >;
    //extern template class RungeKutta< blas_complex<double> >;
    
  };
};



#endif

