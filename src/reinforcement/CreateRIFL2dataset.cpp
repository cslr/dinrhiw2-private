
#include "CreateRIFL2dataset.h"

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
  CreateRIFL2dataset<T>::CreateRIFL2dataset(RIFL_abstract2<T> const & rifl_, 
					    std::vector< rifl2_datapoint<T> > const & database_,
					    std::vector< std::vector< rifl2_datapoint<T> > > const & episodes_,
					    std::mutex & database_mutex_,
					    unsigned int const& epoch_) : 
  
    rifl(rifl_), 
    database(database_),
    episodes(episodes_),
    database_mutex(database_mutex_),
    epoch(epoch_)
  {
    worker_thread = nullptr;
    running = false;
    completed = false;

    {
      {
	std::lock_guard<std::mutex> lock(rifl.policy_mutex);
	
	this->policy_preprocess = rifl.policy_preprocess;
	this->lagged_policy = rifl.lagged_policy;
      }
      {
	std::lock_guard<std::mutex> lock(rifl.Q_mutex);
	
	this->lagged_Q = rifl.lagged_Q;
	this->Q_preprocess = rifl.Q_preprocess;
      }
    }

    
  }
  
  
  template <typename T>
  CreateRIFL2dataset<T>::~CreateRIFL2dataset()
  {
    std::lock_guard<std::mutex> lock(stop_mutex);
    
    if(running || worker_thread != nullptr){
      running = false;
      if(worker_thread){
	worker_thread->join();
	delete worker_thread;
      }
      
      worker_thread = nullptr;
    }
  }

  
  // starts thread that creates NUMDATAPOINTS samples to dataset
  template <typename T>
  bool CreateRIFL2dataset<T>::start(const unsigned int NUMDATAPOINTS, const bool smartEpisodes)
  {
    if(NUMDATAPOINTS == 0) return false;

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
      worker_thread = new std::thread(std::bind(&CreateRIFL2dataset<T>::loop, this));
      
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
  bool CreateRIFL2dataset<T>::isCompleted() const
  {
    return completed;
  }
  
  // returns true if computation is running
  template <typename T>
  bool CreateRIFL2dataset<T>::isRunning() const
  {
    std::lock_guard<std::mutex> lock(thread_mutex);
    
    return running;
  }

  template <typename T>
  bool CreateRIFL2dataset<T>::stop()
  {
    std::lock_guard<std::mutex> lock(stop_mutex);
    
    if(running || worker_thread != nullptr){
      running = false;
      if(worker_thread){
	worker_thread->join();
	delete worker_thread;
      }
      worker_thread = nullptr;

      return true;
    }
    else return false;
  }
  
  // returns reference to dataset
  // (warning: if calculations are running then dataset can change during use)
  template <typename T>
  whiteice::dataset<T> const & CreateRIFL2dataset<T>::getDataset() const
  {
    return data;
  }
  
  // worker thread loop
  template <typename T>
  void CreateRIFL2dataset<T>::loop()
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
      logging.info("CreateRIFL2dataset debug, lagged_Q network diagnostics");
      this->lagged_Q[0].diagnosticsInfo();
      
      logging.info("CreateRIFL2dataset debug, lagged_policy network diagnostics");
      this->lagged_policy.diagnosticsInfo();
      
      
      logging.info("CreateRIFL2dataset debug, database diagnostics");
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
	logging.info("CreateRIFL2dataset debug: smart episodes ENABLED");
      else
	logging.info("CreateRIFL2dataset debug: smart episodes DISABLED");
    }
    
    // used to calculate avg max abs(Q)-value
    // (internal debugging for checking that Q-values are within sane limits)
    std::vector<T> maxvalues;

    if(smartEpisodes){
      
      database_mutex.lock();
      
      // weighted sampling that are calculated from |reinforcement(s,a)| and stdev(Qi(s,a))

      std::vector<T> episodes_weights;
      std::vector<T> episodes_qweights;

      std::vector< std::vector<T> > qvalues;

      episodes_weights.resize(episodes.size());
      episodes_qweights.resize(episodes.size());
      qvalues.resize(episodes.size());

#pragma omp parallel for schedule(guided)
      for(unsigned int e=0;e<episodes.size();e++){
	T esum = T(0.0f);
	T emean = T(0.0f);
	T evar  = T(0.0f);

	T qsum = T(0.0f);
	T qmean = T(0.0f);
	T qvar  = T(0.0f);

	qvalues[e].resize(episodes[e].size());

	unsigned int index = 0;

	for(const auto& ei : episodes[e]){
	  const T w = whiteice::math::pow(whiteice::math::abs(ei.reinforcement), T(2.0f));
	  emean += w;
	  evar  += w*w;

	  T qqmean = T(0.0f);
	  T qqvar  = T(0.0f);
	  
	  whiteice::math::vertex<T> tmp(rifl.numStates + rifl.numActions);
	  tmp.zero();
	  
	  if(tmp.write_subvertex(ei.state, 0) == false)
	    assert(0);

	  if(tmp.write_subvertex(ei.action, rifl.numStates) == false)
	    assert(0);

	  this->Q_preprocess.preprocess(0, tmp);
	  
	  for(unsigned int i=0;i<lagged_Q.size();i++){
	    whiteice::math::vertex<T> q;
	    q.resize(1);
	    q.zero();
	    
	    if(this->lagged_Q[i].calculate(tmp, q, 1, 0) == false)
	      assert(0);
	    
	    this->Q_preprocess.invpreprocess(1, q);
	    
	    qqmean += q[0];
	    qqvar += q[0]*q[0];
	  }
	  
	  
	  qqmean /= lagged_Q.size();
	  qqvar  /= lagged_Q.size();
	  
	  qqvar = whiteice::math::sqrt(whiteice::math::abs(qqvar - qqmean*qqmean));

	  qvalues[e][index] = qqvar;
		
	  qmean += qqvar;
	  qvar += qqvar*qqvar;

	  index++;
	}

	if(episodes[e].size()){
	  emean /= episodes[e].size();
	  evar /= episodes[e].size();
	  
	  qmean /= episodes[e].size();
	  qvar /= episodes[e].size();
	}

	const T estdev = whiteice::math::sqrt(whiteice::math::abs(evar - emean*emean)) + T(1e-9f);
	const T qstdev = whiteice::math::sqrt(whiteice::math::abs(qvar - qmean*qmean)) + T(1e-9f);

	index = 0;

	for(const auto& ei : episodes[e]){
	  const T w = whiteice::math::pow(whiteice::math::abs(ei.reinforcement), T(2.0f));
	  esum += T(1.0f)/(T(1.0f) + whiteice::math::exp(-(w-emean)/estdev)); // softmax so outliers dont dominate
	  
	  const T qw = qvalues[e][index];
	  qsum += T(1.0f)/(T(1.0f) + whiteice::math::exp(-(qw-qmean)/qstdev)); // softmax so outliers dont dominate
	  
	  index++;
	}

	if(episodes[e].size()){
	  esum /= episodes[e].size();
	  qsum /= episodes[e].size();
	}

	episodes_weights[e] = esum;
	episodes_qweights[e] = qsum;
      }

      database_mutex.unlock();

      std::map<T, unsigned int> weights;
      
      {
	T total_weight = T(0.0f);
	T total_qweight = T(0.0f);
	
	for(unsigned int i=0;i<episodes_weights.size();i++){
	  total_weight += episodes_weights[i];
	  total_qweight += episodes_qweights[i];
	}

	if(total_weight <= T(0.0f))
	  total_weight = T(1.0f);

	if(total_qweight <= T(0.0f))
	  total_qweight = T(1.0f);
	
	T mixing_factor = T(0.25f); // 75% go to high reinforcement value
	T sump = T(0.0f);

	//if(rifl.use_smart_weights)
	//mixing_factor = T(0.0f);
	
	for(unsigned int i=0;i<episodes_weights.size();i++){
	  std::pair<T, unsigned int> p;
	  
	  // sump += episodes_weights[i]/total_weight;
	  sump +=
	    (T(1.0f) - mixing_factor)*(episodes_weights[i]/total_weight) +
	    mixing_factor*(T(episodes_qweights[i])/total_qweight);
	  
	  p.first = sump;
	  p.second = i;
	  
	  weights.insert(p);
	}
      }
      
      

      unsigned int counter = 0;

      // episode size samples

      while(counter < NUMDATA){

	if(running == false) // we don't do anything anymore..
	  break; // exits loop
	
	database_mutex.lock();

	// const unsigned int  index = rng.rand() % episodes.size();
	// const auto episode = episodes[index];

	const T r = rng.uniform();
	
	auto iter = weights.upper_bound(r);

	unsigned int index = 0;

	if(iter == weights.end()){
	  index = (unsigned int)(episodes.size()-1);
	}
	else{
	  index = iter->second;
	}

	const auto episode = episodes[index];

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


#pragma omp parallel for schedule(guided)
	for(unsigned i=0;i<episode.size();i++){

	  {
	    if(running == false) // we don't do anything anymore..
	      continue; // exits OpenMP loop
	  }

	  const rifl2_datapoint<T>& datum = episode[i];
	  
	  whiteice::math::vertex<T> in(rifl.numStates + rifl.numActions);
	  in.zero();
	  in.write_subvertex(datum.state, 0);
	  in.write_subvertex(datum.action, rifl.numStates);
	  
	  whiteice::math::vertex<T> out(1);
	  out.zero();
	  
	  // calculates updated utility value

	  std::vector< whiteice::math::vertex<T> > y;
	  y.resize(lagged_Q.size());

	  for(unsigned int i=0;i<lagged_Q.size();i++){
	    y[i].resize(1);
	    y[i].zero();
	  }

	  T maxvalue = T(-INFINITY);
	  
	  {
	    whiteice::math::vertex<T> tmp(rifl.numStates + rifl.numActions);
	    tmp.zero();
	    
	    if(tmp.write_subvertex(datum.newstate, 0) == false)
	      assert(0);
	    
	    {
	      whiteice::math::vertex<T> u(rifl.numActions); // new action..
	      u.zero();
	      
	      auto input = datum.newstate;
	      
	      policy_preprocess.preprocess(0, input);
	      
	      if(lagged_policy.calculate(input, u, 1, 0) == false){
		printf("ERROR: lagged_policy input dim: %d, input dim: %d\n",
		       lagged_policy.inputSize(), input.size());
		assert(0);
	      }
	      
	      policy_preprocess.invpreprocess(1, u); // does nothing..

	      {
		std::string line;
		char buf[256];

		snprintf(buf, 256, "CreateRIFL2dataset: policy's action u =");
		line += buf;

		for(unsigned int k=0;k<u.size();k++){
		  snprintf(buf, 256, " %f", u[k].real());
		  line += buf;
		}

		logging.info(line.c_str());
	      }

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

	    {
	      std::string line;
	      char buf[256];
	      
	      snprintf(buf, 256, "CreateRIFL2dataset: Q's [state+action] =");
	      line += buf;
	      
	      for(unsigned int k=0;k<tmp.size();k++){
		snprintf(buf, 256, " %f", tmp[k].real());
		line += buf;
	      }
	      
	      logging.info(line.c_str());
	    }

	    for(unsigned int i=0;i<lagged_Q.size();i++){
	      if(this->lagged_Q[i].calculate(tmp, y[i], 1, 0) == false)
		assert(0);

	      this->Q_preprocess.invpreprocess(1, y[i]);
	      
	      if(maxvalue < abs(y[i][0]))
		maxvalue = abs(y[i][0]);
	    }
	    
	    
	    if(epoch >= 2 && datum.lastStep == false){
	      //auto qmin = y[0][0];
	      //
	      //for(unsigned int i=0;i<y.size();i++){
	      //if(y[i][0] < qmin) qmin = y[i][0];
	      //}

	      auto qmean = T(0.0f);
	      auto q2    = T(0.0f);

	      for(unsigned int i=0;i<y.size();i++){
		qmean += y[i][0];
		q2    += y[i][0]*y[i][0];
	      }

	      qmean /= T(y.size());
	      q2 /= T(y.size());

	      auto qstdev = whiteice::math::sqrt(whiteice::math::abs(q2 - qmean*qmean));
	      
	      out[0] = rifl.gamma*(qmean + T(0.50f)*qstdev) + datum.reinforcement;
	    }
	    else{ // the first iteration of reinforcement learning do not use Q or if this is last step
	      out[0] = datum.reinforcement;
	    }

	    {
	      char buf[80];
	      snprintf(buf, 80, "CreateRIFL2dataset: output=%f", out[0].real());
	      logging.info(buf);
	    }
	    
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

      std::map<T, unsigned int> weights;
      
      {
	database_mutex.lock();

	// weighted sampling that are calculated from |reinforcement(s,a)| and stdev(Qi(s,a))
	
	T total_weight = T(0.0f);
	T mean = T(0.0f);
	T var  = T(0.0f);
		
	T total_qweight = T(0.0f);
	T qmean = T(0.0f);
	T qvar  = T(0.0f);
	
	std::vector<T> qvalues;

	qvalues.resize(database.size());

#pragma omp parallel
	{
	  T pmean = T(0.0f);
	  T pvar = T(0.0f);

	  T pqmean = T(0.0f);
	  T pqvar = T(0.0f);
	  
#pragma omp for schedule(guided)
	  for(unsigned int i=0;i<database.size();i++){
	    const T r = whiteice::math::pow(whiteice::math::abs(database[i].reinforcement), T(2.0f));
	    
	    pmean += r;
	    pvar += r;
	    
	    T qqmean = T(0.0f);
	    T qqvar  = T(0.0f);
	    
	    whiteice::math::vertex<T> tmp(rifl.numStates + rifl.numActions);
	    tmp.zero();
	    
	    if(tmp.write_subvertex(database[i].state, 0) == false)
	      assert(0);
	    
	    if(tmp.write_subvertex(database[i].action, rifl.numStates) == false)
	      assert(0);
	    
	    this->Q_preprocess.preprocess(0, tmp);
	    
	    for(unsigned int i=0;i<lagged_Q.size();i++){
	      whiteice::math::vertex<T> q;
	      q.resize(1);
	      q.zero();
	      
	      if(this->lagged_Q[i].calculate(tmp, q, 1, 0) == false)
		assert(0);
	      
	      this->Q_preprocess.invpreprocess(1, q);
	      
	      qqmean += q[0];
	      qqvar += q[0]*q[0];
	    }
	    
	    qqmean /= lagged_Q.size();
	    qqvar  /= lagged_Q.size();
	    
	    qqvar = whiteice::math::sqrt(whiteice::math::abs(qqvar - qqmean*qqmean));
	    
	    qvalues[i] = qqvar;
	    
	    pqmean += qqvar;
	    pqvar += qqvar*qqvar;
	  }

#pragma omp critical
	  {
	    mean += pmean;
	    var += pvar;

	    qmean += pqmean;
	    qvar += pqvar;
	  }
	}
	
	mean /= database.size();
	var /= database.size();
	
	qmean /= database.size();
	qvar /= database.size();
	
	const T stdev = whiteice::math::sqrt(whiteice::math::abs(var - mean*mean)) + T(1e-9f);
	const T qstdev = whiteice::math::sqrt(whiteice::math::abs(qvar - qmean*qmean)) + T(1e-9f);

	std::vector<T> softmax, qsoftmax;

	for(unsigned int i=0;i<database.size();i++){
	  const T w = whiteice::math::pow(whiteice::math::abs(database[i].reinforcement), T(2.0f));
	  const T s = T(1.0f)/(T(1.0f) + whiteice::math::exp(-(w-mean)/stdev)); // softmax so outliers dont dominate
	  softmax.push_back(s);
	  total_weight += s;
	  
	  const T qw = qvalues[i];
	  const T qs = T(1.0f)/(T(1.0f) + whiteice::math::exp(-(qw-qmean)/qstdev)); // softmax so outliers dont dominate
	  qsoftmax.push_back(qs);
	  total_qweight += qs;
	}
	
	// assert(total_weight > T(0.0f));
	if(total_weight <= T(0.0f))
	  total_weight = 1.0f;

	if(total_qweight <= T(0.0f))
	  total_qweight = 1.0f;
	
	T mixing_factor = T(0.25f); // 75% go to reinforcement values
	T sump = T(0.0f);

	//if(rifl.use_smart_weights)
	//mixing_factor = T(0.0f);
	
	for(unsigned int i=0;i<database.size();i++){
	  std::pair<T, unsigned int> p;
	  
	  // sump += episodes_weights[i]/total_weight;
	  sump +=
	    (T(1.0f) - mixing_factor)*(softmax[i]/total_weight) +
	    mixing_factor*(qsoftmax[i]/total_qweight);
	  
	  p.first = sump;
	  p.second = i;
	  
	  weights.insert(p);
	}
	
	database_mutex.unlock();
      }
      
      

#pragma omp parallel for schedule(guided)
      for(unsigned int i=0;i<NUMDATA;i++){

	{
	  std::lock_guard<std::mutex> lock(thread_mutex);
	  
	  if(running == false) // we don't do anything anymore..
	    continue; // exits OpenMP loop
	}

	
	// const unsigned int index = rng.rand() % database.size();
	// const auto datum = database[index];


	const T r = rng.uniform();
	
	auto iter = weights.upper_bound(r);
	
	unsigned int index = 0;
	
	if(iter != weights.end()){
	  index = iter->second;
	}

	database_mutex.lock();
	
	const auto datum = database[index];
	
	database_mutex.unlock();
	
	
	whiteice::math::vertex<T> in(rifl.numStates + rifl.numActions);
	in.zero();
	in.write_subvertex(datum.state, 0);
	in.write_subvertex(datum.action, rifl.numStates);
	
	whiteice::math::vertex<T> out(1);
	out.zero();
	
	// calculates updated utility value
	
	std::vector< whiteice::math::vertex<T> > y;
	y.resize(lagged_Q.size());
	
	for(unsigned int i=0;i<lagged_Q.size();i++){
	  y[i].resize(1);
	  y[i].zero();
	}
	
	T maxvalue = T(-INFINITY);
	
	{
	  whiteice::math::vertex<T> tmp(rifl.numStates + rifl.numActions);
	  tmp.zero();
	  
	  if(tmp.write_subvertex(datum.newstate, 0) == false)
	    assert(0);
	  
	  {
	    whiteice::math::vertex<T> u(rifl.numActions); // new action..
	    u.zero();
	    
	    auto input = datum.newstate;
	    
	    policy_preprocess.preprocess(0, input);
    
	    if(lagged_policy.calculate(input, u, 1, 0) == false)
	      assert(0);
	    
	    policy_preprocess.invpreprocess(1, u); // does nothing..

	    {
	      std::string line;
	      char buf[256];
	      
	      snprintf(buf, 256, "CreateRIFL2dataset: policy's action u =");
	      line += buf;
	      
	      for(unsigned int k=0;k<u.size();k++){
		snprintf(buf, 256, " %f", u[k].real());
		line += buf;
	      }
	      
	      logging.info(line.c_str());
	    }
	    
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

	  {
	    std::string line;
	    char buf[256];
	    
	    snprintf(buf, 256, "CreateRIFL2dataset: Q's [state+action] tmp =");
	    line += buf;
	    
	    for(unsigned int k=0;k<tmp.size();k++){
	      snprintf(buf, 256, " %f", tmp[k].real());
	      line += buf;
	    }
	    
	    logging.info(line.c_str());
	  }

	  {
	    char buf[80];
	    snprintf(buf, 80, "CreateRIFL2dataset: y.before=%f", y[0][0].real());
	    logging.info(buf);
	  }

	  for(unsigned int i=0;i<lagged_Q.size();i++){
	    
	    if(this->lagged_Q[i].calculate(tmp, y[i], 1, 0) == false)
	      assert(0);

	    this->Q_preprocess.invpreprocess(1, y[i]);
	    
	    if(maxvalue < abs(y[i][0]))
	      maxvalue = abs(y[i][0]);
	  }
	  

	  {
	    char buf[80];
	    snprintf(buf, 80, "CreateRIFL2dataset: y.after=%f", y[0][0].real());
	    logging.info(buf);
	  }

	  
	  if(epoch >= 2 && datum.lastStep == false){
	    auto qmin = y[0][0];

	    for(unsigned int i=0;i<y.size();i++){
	      if(y[i][0] < qmin) qmin = y[i][0];
	    }
	    
	    out[0] = rifl.gamma*qmin + datum.reinforcement;

	  }
	  else{ // the first iteration of reinforcement learning do not use Q or if this is last step
	    out[0] = datum.reinforcement;
	  }

	  {
	    char buf[80];
	    snprintf(buf, 80, "CreateRIFL2dataset: output=%f", out[0].real());
	    logging.info(buf);
	  }
	  
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
    {
      data.preprocess(0, whiteice::dataset<T>::dnMeanVarianceNormalization); // Q input vectors are normalized!
	
      // data.preprocess(1, whiteice::dataset<T>::dnMeanVarianceNormalization);
	
    }

    
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
      snprintf(buffer, 80, "CreateRIFL2dataset: avg abs(Q)-value %f",
	       tmp);

      whiteice::logging.info(buffer);
    }

    completed = true;

    {
      std::lock_guard<std::mutex> lock(thread_mutex);
      running = false;
    }
    
  }
  

  template class CreateRIFL2dataset< math::blas_real<float> >;
  template class CreateRIFL2dataset< math::blas_real<double> >;
};
