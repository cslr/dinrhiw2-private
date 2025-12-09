
#include "MCMC_diffeq.h"
#include "diffeqs.h"


namespace whiteice
{
  
  // probability function of MCMC sampling
  // returns P(q) = exp(-U(q))
  template <typename T>
  T MCMC_diffeq<T>::U(const math::vertex<T>& q) const
  {
    std::vector< math::vertex<T> > xdata;
    
    whiteice::nnetwork<T> net(this->net);
    bool ok = net.importdata(q);
    assert(ok == true);

    // neural network diff.eq. diverge easily so process in small periods.
    // const unsigned int SEQUENCE_LENGTH = 15;

    const T sigma = T(0.0);
  
    // now simulate training datapoints
    simulate_diffeq_model3(net,
			   this->diffeq_starting_point,
			   (times[times.size()-1]-times[0]).c[0],
			   sigma,
			   xdata, times,
			   ds,
			   SEQUENCE_LENGTH);

    assert(xdata.size() > 0);
    assert(xdata.size() == ds.size(0));
    assert(xdata[0].size() == ds.dimension(0));

    T E = T(0.0f);
    
    // E = SUM 0.5*e(i)^2
#pragma omp parallel shared(E)
    {
      math::vertex<T> err, tmp;
      T e = T(0.0f);
      
#pragma omp for nowait schedule(auto)
      for(unsigned int i=0;i<ds.size(0);i++){
	err = xdata[i] - ds.access(0, i);
	e += T(0.5f)*(err*err)[0];
      }
      
#pragma omp critical (mvjrwerfweghx)
      {
	E = E + e;
      }
    }

    //T mean_error = whiteice::math::sqrt(T(2.0)*E / (ds.size(0)*xdata[0].size()) );    // no scaling

    //printf("MEAN ERROR: %f\n", mean_error.c[0]);
    
    // E /= ds.size(0)*xdata[0].size();    // no scaling
    
    return (E);    
  }


  template class MCMC_diffeq< math::blas_real<float> >;
  template class MCMC_diffeq< math::blas_real<double> >;
  
};
