/*
 * Evolution Strategies neural network diff.eq. solver
 * 
 */

#ifndef __whiteice_ES_diffeq_h
#define __whiteice_ES_diffeq_h

#include "EvolutionStrategies.h"
#include "nnetwork.h"
#include "dataset.h"
#include "vertex.h"

#include "diffeqs.h"


namespace whiteice
{

  template <typename T = math::blas_real<float> >
  class ES_diffeq : public whiteice::EvolutionStrategies<T>
  {
  public:
    ES_diffeq(const whiteice::nnetwork<T>& net,
	      const whiteice::dataset<T>& ds,
	      const std::vector<T>& times,
	      const math::vertex<T>& start_point,	      
	      const unsigned int SEQUENCE_LENGTH) // time-steps for datapoints we use
    {
      this->times = times;
      this->start_point = start_point;
      this->net = net;
      this->ds = ds;

      if(SEQUENCE_LENGTH > times.size())
	this->SEQUENCE_LENGTH = times.size();
      else
	this->SEQUENCE_LENGTH = SEQUENCE_LENGTH;
    }
    
    ~ES_diffeq(){ }

    // reward must be positive number >= 0
    virtual bool estimateReward
    (const math::vertex<T>& x,
     const std::vector< math::vertex<T> >& population,
     T& reward) const
    {
      T rx = T(1.0)/U(x);

#if 0
      T rp = T(0.0);

      for(const auto& p : population){
	rp += T(1.0)/U(p);
      }

      rp /= population.size();

      // reward = rx / mean(population_reward)
      
      reward = rx / rp;
#endif

      reward = rx;

      return true;
    }

    // mean reward against reference good standard solution
    virtual bool estimateMeanRewardReference
    (const math::vertex<T>& x, 
     T& reward) const
    {
      reward = T(1.0)/U(x);

      return true;
    }

    // number of parameters in the model
    virtual unsigned int PARAMETER_DIMENSIONS() const
    {
      return net.exportdatasize();
    }


    // probability function of MCMC sampling
    // returns P(q) = exp(-U(q))
    T U(const math::vertex<T>& q) const
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
			     this->start_point,
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
      
      // T mean_error = whiteice::math::sqrt(T(2.0)*E / (ds.size(0)*xdata[0].size()) );    // no scaling
      
      // printf("U(): MEAN ERROR: %f\n", mean_error.c[0]);
      
      // E /= ds.size(0)*xdata[0].size();    // no scaling
      
      return (E + T(1e-3)); // always positive number (larger means worse)
    }
    
	

  private:

    nnetwork<T> net;
    whiteice::dataset<T> ds;
    
    math::vertex<T> start_point;
    std::vector<T> times; // time-steps for datapoints we use

    unsigned int SEQUENCE_LENGTH = 15;
    
  };
  
};




#endif
