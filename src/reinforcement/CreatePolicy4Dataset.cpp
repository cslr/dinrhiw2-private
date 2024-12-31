
#include "CreatePolicy4Dataset.h"

#include <pthread.h>
#include <sched.h>
#include <functional>

#ifdef WINOS
#include <windows.h>
#endif

#include "Log.h"


namespace whiteice
{
  
  // calculates reinforcement learning training dataset from database
  // uses database_lock for synchronization
  template <typename T>
  CreatePolicy4Dataset<T>::CreatePolicy4Dataset(RIFL_abstract4<T> const & rifl_, 
						std::vector< std::vector< rifl4_datapoint<T> > > const & episodes_,
						std::vector<T> const & episodes_weights_,
						std::mutex & database_mutex_,
						whiteice::dataset<T>& data_) : 

    rifl(rifl_),
    episodes(episodes_),
    episodes_weights(episodes_weights_),
    database_mutex(database_mutex_),
    data(data_)
  {
    assert(episodes.size() > 0);
    assert(episodes.size() == episodes_weights.size());
    
    std::lock_guard<std::mutex> lock(thread_mutex);
    
    worker_thread = nullptr;
    running = false;
    completed = false;
  }
  
  
  template <typename T>
  CreatePolicy4Dataset<T>::~CreatePolicy4Dataset()
  {
    std::lock_guard<std::mutex> lk(thread_mutex);
    
    if(running || worker_thread != nullptr){
      running = false;
      if(worker_thread) worker_thread->join();
      delete worker_thread;
      worker_thread = nullptr;
    }
  }

  
  // starts thread that creates NUMDATAPOINTS samples to dataset
  template <typename T>
  bool CreatePolicy4Dataset<T>::start(const unsigned int NUMDATAPOINTS)
  {
    if(NUMDATAPOINTS == 0) return false;

    std::lock_guard<std::mutex> lock(thread_mutex);

    if(running == true || worker_thread != nullptr){
      char buf[256];
      snprintf(buf, 256, "CreatePolicy4Dataset<T>::start() FAILED (%d)",
	       (int)running);
      
      logging.info(buf);
      return false;
    } 

    try{
      NUMDATA = NUMDATAPOINTS;
      data.clear();
      data.createCluster("input-state", rifl.numStates+rifl.RECURRENT_DIMENSIONS);
      data.createCluster("episode-range", 2);
      
      completed = false;
      
      running = true;
      worker_thread = new std::thread(std::bind(&CreatePolicy4Dataset<T>::loop, this));
      
    }
    catch(std::exception&){
      running = false;
      if(worker_thread){ delete worker_thread; worker_thread = nullptr; }
      return false;
    }

    return true;
  }
  
  // returns true when computation is completed
  template <typename T>
  bool CreatePolicy4Dataset<T>::isCompleted() const
  {
    std::lock_guard<std::mutex> lock(thread_mutex);
    return completed;
  }
  
  // returns true if computation is running
  template <typename T>
  bool CreatePolicy4Dataset<T>::isRunning() const
  {
    std::lock_guard<std::mutex> lock(thread_mutex);
    return running;
  }

  template <typename T>
  bool CreatePolicy4Dataset<T>::stop()
  {
    std::lock_guard<std::mutex> lock(thread_mutex);
    
    if(running || worker_thread != nullptr){
      running = false;
      if(worker_thread) worker_thread->join();
      delete worker_thread;
      worker_thread = nullptr;

      return true;
    }
    else return false;
  }
  
  // returns reference to dataset
  // (warning: if calculations are running then dataset can change during use)
  template <typename T>
  whiteice::dataset<T> const & CreatePolicy4Dataset<T>::getDataset() const
  {
    return data;
  }
  
  // worker thread loop
  template <typename T>
  void CreatePolicy4Dataset<T>::loop()
  {
    // set thread priority (non-standard) to low (background thread)
    {
      sched_param sch_params;
      int policy = SCHED_FIFO;
      
      pthread_getschedparam(pthread_self(),
			    &policy, &sch_params);

#ifdef linux
      policy = SCHED_IDLE; // in linux we can set idle priority
#endif
      sch_params.sched_priority = sched_get_priority_min(policy);
      
      if(pthread_setschedparam(pthread_self(),
				 policy, &sch_params) != 0){
	// printf("! SETTING LOW PRIORITY THREAD FAILED\n");
      }
      
#ifdef WINOS
      SetThreadPriority(GetCurrentThread(),
			THREAD_PRIORITY_IDLE);
#endif	
    }

    {
      char buf[256];
      snprintf(buf, 256, "CreatePolicy4Dataset:loop() started: NUMDATA = %d\n", (int)NUMDATA);
      logging.info(buf);
    }
    
    T total_weight = T(0.0f);
    
    {
      for(unsigned int i=0;i<episodes_weights.size();i++){
	total_weight += episodes_weights[i];
      }
      
      assert(total_weight > T(0.0f));
    }
    

    unsigned int numdata = 0;

    while(numdata < NUMDATA){

      {
	std::lock_guard<std::mutex> lock(thread_mutex);
	
	if(running == false) // we don't do anything anymore..
	  break; // exits OpenMP loop
      }

      database_mutex.lock();
      
      const T r = rng.uniform();
      T limit = T(0.0f);
      
      unsigned int index = 0;
      
      while(limit <= T(1.0f) && index < episodes.size()){ // O(n) search, slow..
	limit += episodes_weights[index]/total_weight;
	
	if(r <= limit) break;
	
	index++;
      }
      
      if(index >= episodes.size()) index = episodes.size() - 1;
      
      auto e = episodes[index];
      
      
      //const unsigned int index = rng.rand() % episodes.size();
      //auto e = episodes[index];

      database_mutex.unlock();

      const unsigned int EPISODE_LENGTH = 15;
      
      unsigned int C = e.size()/EPISODE_LENGTH;
      
      if((e.size() % EPISODE_LENGTH) > 0)
	C += 1;

      for(unsigned int c=0;c<C;c++){
	const unsigned int start = data.size(0);

	for(unsigned int i=c*EPISODE_LENGTH;i<(c+1)*EPISODE_LENGTH && i<e.size();i++){
	  const auto datum = e[i];
	  
	  whiteice::math::vertex<T> state_plus_r(rifl.numStates + rifl.RECURRENT_DIMENSIONS);
	  
	  state_plus_r.zero();
	  state_plus_r.write_subvertex(datum.state, 0);
	  state_plus_r.write_subvertex(datum.recurrent, rifl.numStates);
	  
	  data.add(0, state_plus_r);
	}

	const unsigned int end = data.size(0);

	{
	  whiteice::math::vertex<T> range(2);
	  
	  range[0] = start;
	  range[1] = end;
	  
	  data.add(1, range);
	}
	
      }


      numdata += e.size();
    }
    
    
    {
      std::lock_guard<std::mutex> lock(thread_mutex);
      
      if(running == false)
	return; // exit point
    }

#if 0
    // add preprocessing to dataset
    {
      data.preprocess
	(0, whiteice::dataset<T>::dnMeanVarianceNormalization);
    }
#endif

    {
      unsigned int state_dimensions = 0;
      
      {
	database_mutex.lock();

	if(episodes.size() > 0){
	  if(episodes[0].size() > 0){
	    state_dimensions = episodes[0][0].state.size();
	  }
	}
	
	database_mutex.unlock();
      }
    
      char buf[256];
      snprintf(buf, 256, "CreatePolicy4Dataset:loop(): data.size(0) = %d data.dimension(0) = %d dim(state) = %d\n", (int)data.size(0), (int)data.dimension(0), (int)state_dimensions);
      logging.info(buf);
    }
      
    {
      std::lock_guard<std::mutex> lock(thread_mutex);
      completed = true;
      running = false;
    }
    
  }
  

  template class CreatePolicy4Dataset< math::blas_real<float> >;
  template class CreatePolicy4Dataset< math::blas_real<double> >;
};
