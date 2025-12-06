
#include "MCMC_diffeq.h"
#include "diffeqs.h"


namespace whiteice
{
  
  // probability function of MCMC sampling
  // returns P(q) = exp(-U(q))
  template <typename T>
  T MCMC_diffeq<T>::U(const math::vertex<T>& q) const
  {
    auto datacp(this->ds);
  
    std::vector< math::vertex<T> > xdata;
  
    // now simulate training datapoints
    simulate_diffeq_model2(this->net,
			   this->starting_point,
			   (times[times.size()-1]-times[0]).c[0],
			   xdata, times);

    datacp.clearData(0);
    datacp.add(0, xdata);

    whiteice::nnetwork<T> net(this->net);
    net.importdata(q);
    
    T E = T(0.0f);
    
    // E = SUM 0.5*e(i)^2
#pragma omp parallel shared(E)
    {
      math::vertex<T> err, tmp;
      T e = T(0.0f);
      
#pragma omp for nowait schedule(auto)
      for(unsigned int i=0;i<datacp.size(0);i++){
	net.calculate(datacp.access(0,i), tmp);
	err = datacp.access(1, i) - tmp;
	e = e  + T(0.5f)*(err*err)[0];
      }
      
#pragma omp critical (mvjrwerfweghx)
      {
	E = E + e;
      }
    }

    E /= datacp.size(0);
    
    return (E);    
  }


  template class MCMC_diffeq< math::blas_real<float> >;
  template class MCMC_diffeq< math::blas_real<double> >;
  
};
