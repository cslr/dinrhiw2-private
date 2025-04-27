
#include "CreateRIFL4dataset.h"

#include <pthread.h>
#include <sched.h>

#include <functional>

#ifdef WINOS
#include <windows.h>
#endif

#include "Log.h"

// FIXME?? lagged_Q should be copied to process as it may change while we compute.. (not??)

namespace whiteice
{
  
  // calculates reinforcement learning training dataset from database
  // uses database_lock for synchronization
  template <typename T>
  CreateRIFL4dataset<T>::CreateRIFL4dataset(RIFL_abstract4<T> const & rifl_, 
					    std::vector< rifl4_datapoint<T> > const & database_,
					    std::vector< std::vector< rifl4_datapoint<T> > > const & episodes_,
					    std::vector<T> const & episodes_weights_,
					    std::mutex & database_mutex_,
					    unsigned int const& epoch_) : 
  
    rifl(rifl_), 
    database(database_),
    episodes(episodes_),
    episodes_weights(episodes_weights_),
    database_mutex(database_mutex_),
    epoch(epoch_)
  {
    worker_thread = nullptr;
    running = false;
    completed = false;

    assert(episodes.size() > 0);
    assert(episodes_weights.size() == episodes.size());

    {
      {
	std::lock_guard<std::mutex> lock(rifl.policy_mutex);
	
	this->policy_preprocess = rifl.policy_preprocess;
	this->lagged_policy = rifl.lagged_policy;
      }
      {
	std::lock_guard<std::mutex> lock(rifl.Q_mutex);
	
	this->lagged_Q = rifl.lagged_Q;
	this->lagged_Q2 = rifl.lagged_Q2;
	this->Q_preprocess = rifl.Q_preprocess;
      }
    }

    
  }
  
  
  template <typename T>
  CreateRIFL4dataset<T>::~CreateRIFL4dataset()
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
  bool CreateRIFL4dataset<T>::start(const unsigned int NUMDATAPOINTS, const bool smartEpisodes)
  {
    if(NUMDATAPOINTS == 0) return false;
    if(episodes.size() == 0) return false; // zero episodes mean we cannot do anything.

    std::lock_guard<std::mutex> lock(thread_mutex);

    if(running == true || worker_thread != nullptr)
      return false;

    try{
      NUMDATA = NUMDATAPOINTS;
      this->smartEpisodes = smartEpisodes;

      {
	data.clear();
	data.createCluster("input-state", rifl.numStates + rifl.numActions);
	data.createCluster("output-action", 1);
	
	if(smartEpisodes){
	  data.createCluster("episode-ranges", 2);
	}
      }
      
      completed = false;
      
      running = true;
      worker_thread = new std::thread(std::bind(&CreateRIFL4dataset<T>::loop, this));
      
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
  bool CreateRIFL4dataset<T>::isCompleted() const
  {
    return completed;
  }
  
  // returns true if computation is running
  template <typename T>
  bool CreateRIFL4dataset<T>::isRunning() const
  {
    std::lock_guard<std::mutex> lock(thread_mutex);
    
    return running;
  }

  template <typename T>
  bool CreateRIFL4dataset<T>::stop()
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
  whiteice::dataset<T> const & CreateRIFL4dataset<T>::getDataset() const
  {
    return data;
  }
  
  // worker thread loop
  template <typename T>
  void CreateRIFL4dataset<T>::loop()
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
      logging.info("CreateRIFL4dataset debug, lagged_Q network diagnostics");
      this->lagged_Q.diagnosticsInfo();
      
      logging.info("CreateRIFL4dataset debug, lagged_policy network diagnostics");
      this->lagged_policy.diagnosticsInfo();

      
      logging.info("CreateRIFL4dataset debug, database diagnostics");
      database_mutex.lock();

      T rminvalue = T(0.0);
      T rmaxvalue = T(0.0);
      T aminvalue = T(0.0);
      T amaxvalue = T(0.0);
      T sminvalue = T(0.0);
      T smaxvalue = T(0.0);
      T nminvalue = T(0.0);
      T nmaxvalue = T(0.0);
      

      if(database.size()){
	rminvalue = database[0].reinforcement;
	rmaxvalue = database[0].reinforcement;

	if(database[0].action.size()){
	  aminvalue = database[0].action[0];
	  amaxvalue = database[0].action[0];
	}

	if(database[0].state.size()){
	  sminvalue = database[0].state[0];
	  smaxvalue = database[0].state[0];
	}
	
	if(database[0].newstate.size()){
	  nminvalue = database[0].state[0];
	  nmaxvalue = database[0].state[0];
	}
      }

      for(unsigned int i=0;i<database.size();i++){
	if(rminvalue > database[i].reinforcement)
	  rminvalue = database[i].reinforcement;

	if(rmaxvalue < database[i].reinforcement)
	  rmaxvalue = database[i].reinforcement;

	for(unsigned int j=0;j<database[i].action.size();j++){
	  if(aminvalue > database[i].action[j])
	    aminvalue = database[i].action[j];
	  
	  if(amaxvalue < database[i].action[j])
	    amaxvalue = database[i].action[j];
	}
	
	for(unsigned int j=0;j<database[i].state.size();j++){
	  if(sminvalue > database[i].state[j])
	    sminvalue = database[i].state[j];
	  
	  if(smaxvalue < database[i].state[j])
	    smaxvalue = database[i].state[j];
	}
	
	for(unsigned int j=0;j<database[i].newstate.size();j++){
	  if(nminvalue > database[i].newstate[j])
	    nminvalue = database[i].newstate[j];
	  
	  if(nmaxvalue < database[i].newstate[j])
	    nmaxvalue = database[i].newstate[j];
	}
	
      }

      char buffer[256];

      snprintf(buffer, 256,
	       "database rmin %f rmax %f smin %f smax %f nmin %f nmax %f amin %f amax %f",
	       rminvalue.real(), rmaxvalue.real(), sminvalue.real(), smaxvalue.real(),
	       nminvalue.real(), nmaxvalue.real(), aminvalue.real(), amaxvalue.real());

      logging.info(buffer);
      
      database_mutex.unlock();

      if(smartEpisodes)
	logging.info("CreateRIFL4dataset debug: smart episodes ENABLED");
      else
	logging.info("CreateRIFL4dataset debug: smart episodes DISABLED");
    }


    std::map<T, unsigned int> weights;
    
    {
      T total_weight = T(0.0f);

      for(unsigned int i=0;i<episodes_weights.size();i++){
	total_weight += episodes_weights[i];
      }
      
      assert(total_weight > T(0.0f));

      T mixing_factor = T(0.10f); // 10% goes to generic mixing_factor
      T sump = T(0.0f);

      for(unsigned int i=0;i<episodes_weights.size();i++){
	std::pair<T, unsigned int> p;

	// sump += episodes_weights[i]/total_weight;
	sump += (T(1.0f) - mixing_factor)*(episodes_weights[i]/total_weight) + mixing_factor*(T(1.0f)/T(episodes_weights.size()));

	p.first = sump;
	p.second = i;

	weights.insert(p);
      }
    }
      
      

    
    // used to calculate avg max abs(Q)-value
    // (internal debugging for checking that Q-values are within sane limits)
    std::vector<T> maxvalues;

    if(smartEpisodes){

      unsigned int counter = 0;

      while(counter < NUMDATA){

	if(running == false) // we don't do anything anymore..
	  break; // exits loop
	
	database_mutex.lock();

	const T r = rng.uniform();

	std::vector< rifl4_datapoint<T> > episode;

	auto iter = weights.upper_bound(r);

	if(iter == weights.end()){
	  episode = episodes[episodes.size()-1];
	}
	else{
	  episode = episodes[iter->second];
	}

	//const unsigned int  index = rng.rand() % episodes.size();
	//const auto episode = episodes[index];

	database_mutex.unlock();

	// adds episode start and end in dataset
	{
	  std::lock_guard<std::mutex> lock(database_mutex);
	  
	  const unsigned int START = data.size(0);
	  const unsigned int LENGTH = episode.size();
	  
	  whiteice::math::vertex<T> range;
	  range.resize(2);
	  range[0] = START;
	  range[1] = START+LENGTH;
	  data.add(3, range);
	}

	whiteice::math::vertex<T> recurrent;
		
	recurrent.resize(rifl.RECURRENT_DIMENSIONS);
	recurrent.zero();

#pragma omp parallel for schedule(guided)
	for(unsigned i=0;i<episode.size();i++){

	  {
	    std::lock_guard<std::mutex> lock(thread_mutex);
	    
	    if(running == false) // we don't do anything anymore..
	      continue; // exits OpenMP loop
	  }

	  const rifl4_datapoint<T>& datum = episode[i];
	  
	  whiteice::math::vertex<T> in(rifl.numStates + rifl.numActions);
	  in.zero();
	  in.write_subvertex(datum.state, 0);
	  in.write_subvertex(datum.action, rifl.numStates);
	  
	  whiteice::math::vertex<T> out(1);
	  out.zero();
	  
	  // calculates updated utility value
	  whiteice::math::vertex<T> y(1);
	  y.zero();

	  whiteice::math::vertex<T> y2(1);
	  y2.zero();
	  
	  T maxvalue = T(-INFINITY);
	  
	  {
	    whiteice::math::vertex<T>
	      v(rifl.numActions + rifl.RECURRENT_DIMENSIONS),
	      u(rifl.numActions); // new action..
	    
	    u.zero();
	    v.zero();
	    
	    whiteice::math::vertex<T> start_point(rifl.numStates + rifl.RECURRENT_DIMENSIONS), newstate(rifl.numStates);

	    auto tmp_state = datum.state;

	    if(datum.recurrent.is_zero())
	      recurrent.zero();

	    policy_preprocess.preprocess(0, tmp_state);

	    start_point.write_subvertex(tmp_state, 0);
	    start_point.write_subvertex(recurrent, rifl.numStates);

	    if(lagged_policy.calculate(start_point, v, 1, 0) == false)
	      assert(0);

	    if(v.subvertex(recurrent, rifl.numActions, rifl.RECURRENT_DIMENSIONS) == false)
	      assert(0);

	    whiteice::math::vertex<T> tmp(rifl.numStates + rifl.numActions);
	    tmp.zero();
	    
	    if(tmp.write_subvertex(datum.newstate, 0) == false)
	      assert(0);
	    
	    {
	      whiteice::math::vertex<T> input(rifl.numStates + rifl.RECURRENT_DIMENSIONS);
	      
	      auto tmpinput = datum.newstate;
	      
	      policy_preprocess.preprocess(0, tmpinput);

	      input.write_subvertex(tmpinput, 0);
	      input.write_subvertex(recurrent, rifl.numStates);
	      
	      if(lagged_policy.calculate(input, v, 1, 0) == false)
		assert(0);

	      v.subvertex(u, 0, rifl.numActions);
	      
	      policy_preprocess.invpreprocess(1, u); // does nothing..

#if 0
	      {
		std::string line;
		char buf[256];

		snprintf(buf, 256, "CreateRIFL4dataset: policy's action u =");
		line += buf;

		for(unsigned int k=0;k<u.size();k++){
		  snprintf(buf, 256, " %f", u[k].real());
		  line += buf;
		}

		logging.info(line.c_str());
	      }
#endif

#if 0
	      // add exploration noise..
	      auto noise = u;
	      // Normal EX[n]=0 StDev[n]=1 [OPTMIZE ME: don't create new RNG everytime but use global one]
	      rng.normal(noise);
	      u += T(0.05)*noise;
#endif
	      
	      if(tmp.write_subvertex(u, rifl.numStates) == false) // writes policy's action
		assert(0);
	    }
	    
	    this->Q_preprocess.preprocess(0, tmp);

#if 0
	    {
	      std::string line;
	      char buf[256];
	      
	      snprintf(buf, 256, "CreateRIFL4dataset: Q's [state+action] =");
	      line += buf;
	      
	      for(unsigned int k=0;k<tmp.size();k++){
		snprintf(buf, 256, " %f", tmp[k].real());
		line += buf;
	      }
	      
	      logging.info(line.c_str());
	    }
#endif
	    
	    if(this->lagged_Q.calculate(tmp, y, 1, 0) == false)
	      assert(0);

	    if(this->lagged_Q2.calculate(tmp, y2, 1, 0) == false)
	      assert(0);	    
	    
	    this->Q_preprocess.invpreprocess(1, y);
	    this->Q_preprocess.invpreprocess(1, y2);
	    
	    if(maxvalue < abs(y[0]))
	      maxvalue = abs(y[0]);

	    if(maxvalue < abs(y2[0]))
	      maxvalue = abs(y2[0]);
	    
	    if(epoch >= 2 && datum.lastStep == false){
	      if(y[0] < y2[0]){
		out[0] = rifl.gamma*y[0] + datum.reinforcement;
	      }
	      else{
		out[0] = rifl.gamma*y[0] + datum.reinforcement;
		//out[0] = rifl.gamma*y2[0] + datum.reinforcement;
	      }
	    }
	    else{ // the first iteration of reinforcement learning do not use Q or if this is last step
	      out[0] = datum.reinforcement;
	    }

#if 1
	    {
	      char buf[80];
	      snprintf(buf, 80, "CreateRIFL4dataset: output=%f", out[0].real());
	      logging.info(buf);
	    }
#endif
	    
	  }
	  
#pragma omp critical
	  {
	    std::lock_guard<std::mutex> lock(database_mutex);
	    data.add(0, in);
	    data.add(1, out);

	    counter++;
	    
	    maxvalues.push_back(maxvalue);
	  }
	  
	} // for-loop

      } // while loop (counter)
      
    }
    else{

#pragma omp parallel for schedule(guided)
      for(unsigned int i=0;i<NUMDATA;i++){

	{
	  std::lock_guard<std::mutex> lock(thread_mutex);
	  
	  if(running == false) // we don't do anything anymore..
	    continue; // exits OpenMP loop
	}
	
	database_mutex.lock();
	
	const unsigned int index = rng.rand() % database.size();
	
	const auto datum = database[index];
	
	database_mutex.unlock();
	
	whiteice::math::vertex<T> in(rifl.numStates + rifl.numActions);
	in.zero();
	in.write_subvertex(datum.state, 0);
	in.write_subvertex(datum.action, rifl.numStates);
	
	whiteice::math::vertex<T> out(1);
	out.zero();
	
	// calculates updated utility value
	whiteice::math::vertex<T> y(1);
	y.zero();

	whiteice::math::vertex<T> y2(1);
	y2.zero();
	
	T maxvalue = T(-INFINITY);
	
	{
	  whiteice::math::vertex<T> tmp(rifl.numStates + rifl.numActions);
	  tmp.zero();
	  
	  if(tmp.write_subvertex(datum.newstate, 0) == false)
	    assert(0);
	  
	  {
	    whiteice::math::vertex<T>
	      v(rifl.numActions + rifl.RECURRENT_DIMENSIONS),
	      u(rifl.numActions); // new action..

	    u.zero();
	    v.zero();

	    	    
	    whiteice::math::vertex<T> input(rifl.numStates + rifl.RECURRENT_DIMENSIONS);
	    
	    auto tmpinput = datum.newstate;
	    
	    policy_preprocess.preprocess(0, tmpinput);
	    
	    input.write_subvertex(tmpinput, 0);
	    input.write_subvertex(datum.recurrent_new, rifl.numStates);
	    
	    if(lagged_policy.calculate(input, v, 1, 0) == false)
	      assert(0);
	    
	    v.subvertex(u, 0, rifl.numActions);
	    
	    policy_preprocess.invpreprocess(1, u); // does nothing..


#if 0
	    {
	      std::string line;
	      char buf[256];
	      
	      snprintf(buf, 256, "CreateRIFL4dataset: policy's action u =");
	      line += buf;
	      
	      for(unsigned int k=0;k<u.size();k++){
		snprintf(buf, 256, " %f", u[k].real());
		line += buf;
	      }
	      
	      logging.info(line.c_str());
	    }
#endif
	  
	    // add exploration noise..
#if 0
	    auto noise = u;
	    // Normal EX[n]=0 StDev[n]=1 [OPTMIZE ME: don't create new RNG everytime but use global one]
	    rng.normal(noise);
	    u += T(0.05)*noise;
#endif
	    
	    if(tmp.write_subvertex(u, rifl.numStates) == false) // writes policy's action
	      assert(0);
	  }
	  
	  this->Q_preprocess.preprocess(0, tmp);

#if 0
	  {
	    std::string line;
	    char buf[256];
	    
	    snprintf(buf, 256, "CreateRIFL4dataset: Q's [state+action] tmp =");
	    line += buf;
	    
	    for(unsigned int k=0;k<tmp.size();k++){
	      snprintf(buf, 256, " %f", tmp[k].real());
	      line += buf;
	    }
	    
	    logging.info(line.c_str());
	  }

	  {
	    char buf[80];
	    snprintf(buf, 80, "CreateRIFL4dataset: y.before=%f", y[0].real());
	    logging.info(buf);
	  }
#endif
	  
	  if(this->lagged_Q.calculate(tmp, y, 1, 0) == false)
	    assert(0);

	  if(this->lagged_Q2.calculate(tmp, y2, 1, 0) == false)
	    assert(0);

	  this->Q_preprocess.invpreprocess(1, y);
	  this->Q_preprocess.invpreprocess(1, y2);
	  
	  if(maxvalue < abs(y[0]))
	    maxvalue = abs(y[0]);

	  if(maxvalue < abs(y2[0]))
	    maxvalue = abs(y2[0]);
	  
	  if(epoch >= 2 && datum.lastStep == false){
	    if(y[0] < y2[0]){
	      out[0] = rifl.gamma*y[0] + datum.reinforcement;
	    }
	    else{
	      out[0] = rifl.gamma*y[0] + datum.reinforcement;
	      //out[0] = rifl.gamma*y2[0] + datum.reinforcement;
	    }
	  }
	  else{ // the first iteration of reinforcement learning do not use Q or if this is last step
	    out[0] = datum.reinforcement;
	  }

#if 0
	  {
	    char buf[80];
	    snprintf(buf, 80, "CreateRIFL4dataset: output=%f", out[0].real());
	    logging.info(buf);
	  }
#endif
	  
	}
	
#pragma omp critical
	{
	  data.add(0, in);
	  data.add(1, out);
	  
	  maxvalues.push_back(maxvalue);
	}
	
      }
      
    }

    {
      std::lock_guard<std::mutex> lock(thread_mutex);
      
      if(running == false)
	return; // exit point
    }
    
    // add preprocessing to dataset
#if 0
    {
      data.preprocess(0, whiteice::dataset<T>::dnMeanVarianceNormalization);
    
      // data.preprocess(1, whiteice::dataset<T>::dnMeanVarianceNormalization);
    }
#endif

    
    // for debugging purposes (reports average max Q-value)
    if(maxvalues.size() > 0)
    {
      T sum = T(0.0);
      for(auto& m : maxvalues)
	sum += abs(m);

      sum /= T(maxvalues.size());

      double tmp = 0.0;
      whiteice::math::convert(tmp, sum);

      char buffer[80];
      snprintf(buffer, 80, "CreateRIFL4dataset: avg abs(Q)-value %f",
	       tmp);

      whiteice::logging.info(buffer);
    }

    completed = true;

    {
      std::lock_guard<std::mutex> lock(thread_mutex);
      running = false;
    }
    
  }
  

  template class CreateRIFL4dataset< math::blas_real<float> >;
  template class CreateRIFL4dataset< math::blas_real<double> >;
};
