/*
 * simple MCMC sampler
 *
 */

#include "MCMC.h"
#include "RNG.h"


namespace whiteice
{
  namespace math
  {

    template <typename T>
    MCMC<T>::MCMC()
    {
      sum_N = 0;
      sum_mean.zero();

      sampling_thread = nullptr;
      
      running = false;
    }
    
    
    template <typename T>
    MCMC<T>::~MCMC()
    {
      std::lock_guard<std::mutex> lock(thread_mutex);
      // std::lock_guard<std::mutex> lock2(solution_mutex);

      running = false;
    
      if(sampling_thread == nullptr)
	return;

      if(sampling_thread->joinable())
	sampling_thread->join();

      delete sampling_thread;
      sampling_thread = nullptr;
    }
    

    template <typename T>
    bool MCMC<T>::startSampler(whiteice::math::vertex<T>& starting_point)
    {
      std::lock_guard<std::mutex> lock(thread_mutex);
      
      if(running)
	return false; // already running
    
      running = true;

      {
	std::lock_guard<std::mutex> lock2(solution_mutex);
	
	this->starting_point = starting_point;
	
	sum_N = 0;
	sum_mean = starting_point;
	sum_mean.zero();

	samples.clear();
      }
    
      try{
	if(sampling_thread){
	  delete sampling_thread;
	  sampling_thread = nullptr;
	}
	
	sampling_thread = new std::thread(&MCMC<T>::sampler_loop, this);
	
	return true;
      }
      catch(std::exception& e){
	std::cout << "ERROR: unexpected exception: " << e.what() << std::endl;
	running = false;

	if(sampling_thread){
	  // sampling_thread->join();
	  delete sampling_thread;
	}

	sampling_thread = nullptr;

	return false;
      }
    }

    
    template <typename T>
    bool MCMC<T>::stopSampler()
    {
      std::lock_guard<std::mutex> lock(thread_mutex);
      //std::lock_guard<std::mutex> lock2(solution_mutex);
    
      if(!running || sampling_thread == nullptr)
	return false;
    
      running = false;

      if(sampling_thread->joinable())
	sampling_thread->join();

      delete sampling_thread;
      sampling_thread = nullptr;

      return true;
    }

    template <typename T>
    unsigned int MCMC<T>::getSamples(std::vector< whiteice::math::vertex<T> >& samples_) const
    {
      std::lock_guard<std::mutex> lock(solution_mutex);

      samples_ = this->samples;

      return samples_.size();
    }

    template <typename T>
    unsigned int MCMC<T>::getNumberOfSamples() const
    {
      std::lock_guard<std::mutex> lock(solution_mutex);

      return samples.size();
    }
    


    template <typename T>
    whiteice::math::vertex<T> MCMC<T>::getMean() const
    {
      std::lock_guard<std::mutex> lock(solution_mutex);

      if(sum_N == 0) return sum_mean;

      auto m = sum_mean/sum_N;

      return m;
    }
    
    // calculates mean probability log(P) for the latest N samples, 0 = all samples
    template <typename T>
    T MCMC<T>::getMeanProbability(unsigned int latestN) const
    {
      std::lock_guard<std::mutex> lock(solution_mutex);
      
      if(!latestN) latestN = samples.size();
      if(latestN > samples.size()) latestN = samples.size();
      
      T sumLogP = T(0.0f);
      
      for(unsigned int i=samples.size()-latestN;i<samples.size();i++)
	sumLogP -= U(samples[i]);
      
      if(latestN > 0)
	sumLogP /= T((float)latestN);
      
      return sumLogP;
    }
    
    
    template <typename T>
    void MCMC<T>::sampler_loop()
    {
      whiteice::math::vertex<T> q = this->starting_point; // current sample
      whiteice::math::vertex<T> r = this->starting_point; // proposal point
      r.zero();
      
      whiteice::RNG<T> rng;

      while(running){
	
	rng.normal(r); // r ~ N(q, I)
	r += q;

	// p(r)/p(q) = exp(-U(r))/exp(-U(q)) = exp(U(q)-U(r))

	T logP = U(q) - U(r);
	
	if(whiteice::math::log(rng.uniform()+T(1e-20)) < logP){
	  q = r;
	}
	
	{
	  std::lock_guard<std::mutex> lock(solution_mutex);
	  
	  samples.push_back(q);

	  sum_mean += q;
	  sum_N++;
	}
	
      }

    }


      
    template class MCMC< whiteice::math::blas_real<float> >;
    template class MCMC< whiteice::math::blas_real<double> >;
  };
};
