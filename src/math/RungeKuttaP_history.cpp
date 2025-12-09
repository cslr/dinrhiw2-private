
#include "RungeKuttaP_history.h"
#include "blade_math.h"
#include <map>


namespace whiteice
{
  namespace math
  {
    
    template <typename T>
    RungeKuttaP_history<T>::RungeKuttaP_history(odefunction<T>* f){
      this->f = f;
    }
    
    
    template <typename T>
    RungeKuttaP_history<T>::~RungeKuttaP_history(){
      
    }
    
    
    template <typename T>
    odefunction<T>* RungeKuttaP_history<T>::getFunction() const {
      return f;
    }
    
    
    template <typename T>
    void RungeKuttaP_history<T>::setFunction(odefunction<T>* f) {
      this->f = f;
    }


    template <typename T>
    whiteice::math::vertex<T> RungeKuttaP_history<T>::linearly_interpolate_find
    (const T& t,
     const std::map<T, whiteice::math::vertex<T> >& map_points,
     const unsigned int DIM) const
    {
      if(map_points.size() == 0){
	whiteice::math::vertex<T> v;
	v.resize(DIM);
	v.zero();
	return v;
      }

      auto it = map_points.lower_bound(t);

      if(it == map_points.end()){
	it--;

	if(t < it->first){ // time before first/last element
	  whiteice::math::vertex<T> v;
	  v.resize(DIM);
	  v.zero();
	  return v;
	}
	
	return it->second;
      }
      else if(it == map_points.begin()){

	if(t < it->first){ // time before first element
	  whiteice::math::vertex<T> v;
	  v.resize(DIM);
	  v.zero();
	  return v;
	}
	
	return it->second;
      }
      else{
	auto it_prev = std::prev(it);

	// linearly interpolates between points
	// v = x(tk) + (x(t0)-x(tk))*((t-tk)/(t0-tk))

	whiteice::math::vertex<T> v;
	v = it_prev->second + (it->second - it_prev->second)*((t - it_prev->first)/(it->first - it_prev->first));

	return v;
      }
      
    }


    template <typename T>
    bool RungeKuttaP_history<T>::calculate_ode_parameters
    (const whiteice::math::vertex<T>& y,
     const T& t,
     const std::map<T, whiteice::math::vertex<T> >& map_points,
     const unsigned int HISTORY_LEN,
     const std::map<T, whiteice::math::vertex<T> > map_parameters,
     whiteice::math::vertex<T>& ode_y) const
    {
      if(map_parameters.size() == 0) return false;
      
      std::vector< whiteice::math::vertex<T> > vectors;
      unsigned int dimensions = 0;
      
      vectors.push_back(y);
      dimensions += y.size();

      // adds HISTORY_LEN past vectors (linearly interpolated in time points)
      for(unsigned int h=1;h<=HISTORY_LEN;h++){
	whiteice::math::vertex<T> v = linearly_interpolate_find(t-h, map_points, y.size());
	vectors.push_back(v);
	dimensions += v.size();
      }

      const unsigned int dim_p = map_parameters.begin()->second.size(); // parameter p(t) vector dim(p)
      dimensions += dim_p;

      ode_y.resize(dimensions);

      unsigned int d = 0;
      
      for(unsigned int vi = 0;vi<vectors.size();vi++){
	for(unsigned int i=0;i<vectors[vi].size();i++,d++){
	  ode_y[d] = vectors[vi][i];
	}
      }
      
      // linearly interpolates parameter vector p(t)
      {
	whiteice::math::vertex<T> v = linearly_interpolate_find(t, map_parameters, dim_p);

	for(unsigned int i=0;i<v.size();i++,d++)
	  ode_y[d] = v[i];
      }
      
      return true;
    }
    
    
    template <typename T>
    void RungeKuttaP_history<T>::calculate
    (const T t0, const T t_end,
     const whiteice::math::vertex<T>& y0,
     const std::vector< whiteice::math::vertex<T> >& parameters,
     const std::vector< T > & parameter_times,
     std::vector< whiteice::math::vertex<T> >& points,
     std::vector< T >& times)
    {
      std::map<T, whiteice::math::vertex<T> > map_points;
      std::map<T, whiteice::math::vertex<T> > map_parameters;
      
      if(parameters.size() != parameter_times.size()) return;// ERROR

      for(unsigned int i=0;i<parameters.size();i++){
	map_parameters.insert(std::pair<T, whiteice::math::vertex<T> >
			      (parameter_times[i],
			       parameters[i]));
      }
      
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

	// y with historical values (interpolated) for ODE function
	whiteice::math::vertex<T> ode_y; 
	
	// calculates the next point with a single step
	{
	  calculate_ode_parameters(y, t, map_points, HISTORY_LEN,
				   map_parameters,
				   ode_y);
	  k[0] = h * (f->calculate(odeparam<T>(t, ode_y)) );
	  
	  tmp = y + T(0.5)*k[0];
	  ttmp = t + T(0.5)*h;
	  calculate_ode_parameters(tmp, ttmp, map_points, HISTORY_LEN,
				   map_parameters,
				   ode_y);
	  k[1] = h * (f->calculate(odeparam<T>(ttmp, ode_y)));
	  
	  tmp = y + T(0.5)*k[1];
	  ttmp = t + T(0.5)*h;
	  calculate_ode_parameters(tmp, ttmp, map_points, HISTORY_LEN,
				   map_parameters,
				   ode_y);
	  k[2] = h * (f->calculate(odeparam<T>(ttmp, ode_y)));
	  
	  tmp = y + k[2];
	  ttmp = t + h;
	  calculate_ode_parameters(tmp, ttmp, map_points, HISTORY_LEN,
				   map_parameters,
				   ode_y);
	  k[3] = h * (f->calculate(odeparam<T>(ttmp, ode_y)));
	  
	  yn = y + f6*(k[0]+k[3]) + f3*(k[1]+k[2]);
	}
	
	
	// calculates the next point with two steps
	{
	  T h2 = h*T(0.5);
	  
	  calculate_ode_parameters(y, t, map_points, HISTORY_LEN,
				   map_parameters,
				   ode_y);
	  k[0] = h2 * (f->calculate(odeparam<T>(t, ode_y)) );
	  
	  tmp = y + T(0.5)*k[0];
	  ttmp = t + T(0.5)*h2;
	  calculate_ode_parameters(tmp, ttmp, map_points, HISTORY_LEN,
				   map_parameters,
				   ode_y);
	  k[1] = h2 * (f->calculate(odeparam<T>(ttmp, ode_y)));
	  
	  tmp = y + T(0.5)*k[1];
	  ttmp = t + T(0.5)*h2;
	  calculate_ode_parameters(tmp, ttmp, map_points, HISTORY_LEN,
				   map_parameters,
				   ode_y);
	  k[2] = h2 * (f->calculate(odeparam<T>(ttmp, ode_y)));
	  
	  tmp = y + k[2];
	  ttmp = t + h2;
	  calculate_ode_parameters(tmp, ttmp, map_points, HISTORY_LEN,
				   map_parameters,
				   ode_y);
	  k[3] = h2 * (f->calculate(odeparam<T>(ttmp, ode_y)));
	  
	  y += f6*(k[0]+k[3]) + f3*(k[1]+k[2]);
	  
	  ttmp = t + h2;
	  calculate_ode_parameters(y, ttmp, map_points, HISTORY_LEN,
				   map_parameters,
				   ode_y);
	  k[0] = h2 * (f->calculate(odeparam<T>(ttmp, ode_y)) );
	  
	  tmp = y + T(0.5)*k[0];
	  ttmp = t + T(1.5)*h2;
	  calculate_ode_parameters(tmp, ttmp, map_points, HISTORY_LEN,
				   map_parameters,
				   ode_y);
	  k[1] = h2 * (f->calculate(odeparam<T>(ttmp, ode_y)));
	  
	  tmp = y + T(0.5)*k[1];
	  ttmp = t + T(1.5)*h2;
	  calculate_ode_parameters(tmp, ttmp, map_points, HISTORY_LEN,
				   map_parameters,
				   ode_y);
	  k[2] = h2 * (f->calculate(odeparam<T>(ttmp, ode_y)));
	  
	  tmp = y + k[2];
	  ttmp = t + h;
	  calculate_ode_parameters(tmp, ttmp, map_points, HISTORY_LEN,
				   map_parameters,
				   ode_y);
	  k[3] = h2 * (f->calculate(odeparam<T>(ttmp, ode_y)));
	  
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

	map_points.insert(std::pair<T, whiteice::math::vertex<T> >(t, y));
      }
    }
    
    
    
    
    
    //////////////////////////////////////////////////////////////////////
    
    template class RungeKuttaP_history< float >;
    template class RungeKuttaP_history< double >;
    template class RungeKuttaP_history< blas_real<float> >;
    template class RungeKuttaP_history< blas_real<double> >;
    //template class RungeKuttaP_history< blas_complex<float> >;
    //template class RungeKuttaP_history< blas_complex<double> >;
  };
};
