/*
 * Implments Stochasic Gradient Descent absract class to be inherited by a specific optimization class.
 * 
 *
 * TODO: allow implementing class to enable dropout heuristic
 * in neural networks.
 *
 */

#include "SGD.h"
#include "linear_equations.h"
#include <iostream>
#include <list>
#include <functional>

#include <unistd.h>

#ifdef _GLIBCXX_DEBUG

#ifndef _WIN32

#undef __STRICT_ANSI__
#include <float.h>
#include <fenv.h>

#endif

#endif


#ifdef WINOS
#include <windows.h>
#endif


namespace whiteice
{
  namespace math
  {

    template <typename T>
    SGD<T>::SGD(bool overfit)
    {
      thread_running = false;
      sleep_mode = false;
      solution_converged = false;
      optimizer_thread = nullptr;
      this->overfit = overfit;
      this->keepWorse = false;
      this->smart_convergence_check = true;
      this->adaptive_lrate = true;

      // this->use_adam = true; ALWAYS TRUE
    }
    
 
    template <typename T>
    SGD<T>::~SGD()
    {
      thread_mutex.lock();
      
      if(thread_running){
	thread_running = false;
	
	// waits for thread to stop running
	// std::unique_lock<std::mutex> lock(thread_is_running_mutex);
	// thread_is_running_cond.wait_for(lock, std::chrono::milliseconds(1000)); // 1 second
      }
      
      if(optimizer_thread){
	optimizer_thread->join();
	delete optimizer_thread;
      }
      optimizer_thread = nullptr;

      thread_mutex.unlock();
    }
    
    
    template <typename T>
    bool SGD<T>::minimize(vertex<T> x0,
			  const T lrate,
			  const unsigned int MAX_ITERS,
			  const unsigned int MAX_NO_IMPROVE_ITERS)
    {
      thread_mutex.lock();
      
      if(thread_running || optimizer_thread != nullptr){
	thread_mutex.unlock();
	return false;
      }

      if(lrate <= T(0.0f) || MAX_NO_IMPROVE_ITERS <= 0){
	thread_mutex.unlock();
	return false;
      }

      // calculates initial solution
      solution_mutex.lock();
      {
	heuristics(x0);
	
	this->bestx = x0;
	this->besty = this->getError(x0);
	
	iterations  = 0;
      }
      solution_mutex.unlock();

      this->lrate = lrate;
      this->MAX_ITERS = MAX_ITERS; // if MAX_ITERS == zero, don't stop until convergence (no improve)
      this->MAX_NO_IMPROVE_ITERS = MAX_NO_IMPROVE_ITERS;
      
      thread_running = true;
      sleep_mode = false;
      solution_converged = false;
      thread_is_running = 0;
      
      try{
	optimizer_thread =
	  new std::thread(std::bind(&SGD<T>::optimizer_loop,
				    this));
	
	// optimizer_thread->detach();
      }
      catch(std::exception& e){
	thread_running = false;
	thread_mutex.unlock();
	return false;
      }
      
      thread_mutex.unlock();
      
      return true;
    }
    
    
    template <typename T>
    bool SGD<T>::getSolution(vertex<T>& x, T& y, unsigned int& iterations) const
    {
      // gets the best found solution
      solution_mutex.lock();
      {
	x = bestx;
	y = besty;
	iterations = this->iterations;
      }
      solution_mutex.unlock();
      
      return true;
    }


    template <typename T>
    bool SGD<T>::getSolutionStatistics(T& y, unsigned int& iterations) const
    {
      std::lock_guard<std::mutex> lock(solution_mutex);

      y = besty;
      iterations = this->iterations;

      return true;
    }
    
    
    // continues, pauses, stops computation
    template <typename T>
    bool SGD<T>::continueComputation()
    {
      sleep_mutex.lock();
      {
	sleep_mode = false;
      }
      sleep_mutex.unlock();
      
      return true;
    }
    
    
    template <typename T>
    bool SGD<T>::pauseComputation()
    {
      sleep_mutex.lock();
      {
	sleep_mode = true;
      }
      sleep_mutex.unlock();
      
      return true;
    }
    
    
    template <typename T>
    bool SGD<T>::stopComputation()
    {
      // FIXME threading sychnronization code is BROKEN!
      
      thread_mutex.lock();
      
      if(thread_running == false){
	thread_mutex.unlock();
	return false;
      }
      
      thread_running = false;
      
      // waits for thread to stop running
      // std::unique_lock<std::mutex> lock(thread_is_running_mutex);
      // thread_is_running_cond.wait_for(lock, std::chrono::milliseconds(1000)); // 1 sec
      
      if(optimizer_thread){
	optimizer_thread->join();
	delete optimizer_thread;
      }
      optimizer_thread = nullptr;
      
      thread_mutex.unlock();
      
      return true;
    }


    // returns true if solution converged and we cannot
    // find better solution
    template <typename T>
    bool SGD<T>::solutionConverged() const
    {
      return solution_converged;
    }
    
    // returns true if optimization thread is running
    template <typename T>
    bool SGD<T>::isRunning() const
    {
      return thread_running;
    }
    
    
    // limit values to sane interval (no too large values)
    template <typename T>
    bool SGD<T>::box_values(vertex<T>& x) const
    {
      // don't allow values larger than 10^4
      for(unsigned int i=0;i<x.size();i++)
	if(x[i] > T(1e4)) x[i] = T(1e4);
	else if(x[i] < T(-1e4)) x[i] = T(-1e4);

      return true;
    }
    
    
    template <typename T>
    void SGD<T>::optimizer_loop()
    {
#ifdef _GLIBCXX_DEBUG  
#ifndef _WIN32    
      {
	// enables FPU exceptions
	feenableexcept(FE_INVALID | FE_DIVBYZERO);
      }
#endif
#endif

      {
	sched_param sch_params;
	int policy = SCHED_FIFO; // SCHED_RR
	
	pthread_getschedparam(pthread_self(), &policy, &sch_params);
	
#ifdef linux
	policy = SCHED_IDLE; // in linux we can set idle priority
#endif
	sch_params.sched_priority = sched_get_priority_min(policy);
	
	if(pthread_setschedparam(pthread_self(),
				 policy, &sch_params) != 0){
	}
	
#ifdef WINOS
	SetThreadPriority(GetCurrentThread(),
			  THREAD_PRIORITY_IDLE);
#endif
	
      }

      
      thread_is_running_cond.notify_all();
      
      
      if(use_adam){
	
	// Use Adam Optimizer instead which should often give better results..
	
	const T alpha = T(0.001);
	const T beta1 = T(0.9);
	const T beta2 = T(0.999);
	const T epsilon = T(1e-8);
	
	std::list<T> errors; // used for convergence check
	
	vertex<T> x;
	T real_besty;

	{
	  std::lock_guard<std::mutex> lock(solution_mutex);
	  
	  this->iterations = 0;
	  x = bestx;
	  real_besty = besty;
	}

	unsigned int no_improve_iterations = 0;
	
	vertex<T> grad;
	
	vertex<T> m(x.size());
	vertex<T> v(x.size());
	vertex<T> m_hat(x.size());
	vertex<T> v_hat(x.size());

	m.zero();
	v.zero();
	
	
	while((iterations < MAX_ITERS || MAX_ITERS == 0) &&
	      (no_improve_iterations) < MAX_NO_IMPROVE_ITERS &&
	      thread_running)
	{
	  iterations++;
	  
	  grad = Ugrad(x);

	  // #pragma omp parallel for schedule(static)
	  for(unsigned int i=0;i<grad.size();i++){
	    for(unsigned int k = 0;k<m[i].size();k++){
	      m[i][k] = beta1[k] * m[i][k] + (T(1.0) - beta1)[k]*grad[i][k];
	      v[i][k] = beta2[k] * v[i][k] + (T(1.0) - beta2)[k]*grad[i][k]*grad[i][k];
	      
	      m_hat[i][k] = m[i][k] / (T(1.0)[k] - whiteice::math::pow(beta1[0], T(iterations)[0]));
	      v_hat[i][k] = v[i][k] / (T(1.0)[k] - whiteice::math::pow(beta2[0], T(iterations)[0]));
	      
	      x[i][k] -= (alpha[k] / (whiteice::math::sqrt(v_hat[i][k]) + epsilon[k])) * m_hat[i][k];
	    }
	  }

	  heuristics(x);

	  T new_error = getError(x);

	  if(whiteice::math::abs(new_error)[0] >= whiteice::math::abs(real_besty)[0]){
	    no_improve_iterations++;
	  }
	  else{
	    std::lock_guard<std::mutex> lock(solution_mutex);
	    
	    bestx = x;
	    besty = new_error;
	    real_besty = new_error;

	    no_improve_iterations = 0;
	  }

	  
	  if(smart_convergence_check){
	    errors.push_back(whiteice::math::abs(real_besty)); // NOTE: getError() must return >= 0.0 values
	    
	    if(errors.size() >= 50){
	      
	      while(errors.size() > 50)
		errors.pop_front();

	      T m = T(0.0f);
	      T s = T(0.0f);
	      
	      for(const auto& e : errors){
		m += e;
		s += e*whiteice::math::conj(e);
	      }
	      
	      m /= errors.size();
	      s /= errors.size();
	      
	      s -= m*m;
	      s = sqrt(abs(s));
	      
	      T r = T(0.0f);
	      
	      if(whiteice::math::abs(m)[0] > whiteice::math::abs(T(0.0f)[0]))
		r = s/m;
	      
	      if(whiteice::math::abs(r)[0] <= whiteice::math::abs(T(0.005f))[0]){
		// convergence: 0.5% st.dev. when compared to mean.
		solution_converged = true;
		break;
	      }
	      
	    }
	  }

	  while(sleep_mode && thread_running){
	    std::chrono::milliseconds duration(200);
	    std::this_thread::sleep_for(duration);
	  }
	  
	}
	

      
	{
	  solution_converged = true;
	  
	  // std::lock_guard<std::mutex> lock(thread_mutex); // needed or safe??
	  
	  thread_running = false; // very tricky here, writing false => false or true => false SHOULD BE ALWAYS SAFE without locks
	}
	// thread_is_running_cond.notify_all(); // waiters have to use wait_for() [timeout milliseconds] as it IS possible to miss notify_all()
	
	return; // exit
      }
      
      
      {
	solution_converged = true;
	
	// std::lock_guard<std::mutex> lock(thread_mutex); // needed or safe??
	
	thread_running = false; // very tricky here, writing false => false or true => false SHOULD BE ALWAYS SAFE without locks
      }
      // thread_is_running_cond.notify_all(); // waiters have to use wait_for() [timeout milliseconds] as it IS possible to miss notify_all()
    }
    
    
    // explicit template instantations
    
    //template class SGD< float >;
    //template class SGD< double >;
    
    template class SGD< blas_real<float> >;
    template class SGD< blas_real<double> >;

    template class SGD< blas_complex<float> >;
    template class SGD< blas_complex<double> >;

    
    template class SGD< superresolution<
			  blas_real<float>,
			  modular<unsigned int> > >;
    
    template class SGD< superresolution<
			  blas_real<double>,
			  modular<unsigned int> > >;
    
    
  };
};

