
#include "CreatePolicyDataset.h"

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
  CreatePolicyDataset<T>::CreatePolicyDataset(RIFL_abstract2<T> const & rifl_, 
					      std::vector< rifl2_datapoint<T> > const & database_,
					      std::mutex & database_mutex_,
					      whiteice::dataset<T>& data_) : 

    rifl(rifl_),
    database__(database_),
    database_mutex(database_mutex_),    
    data(data_)
  {
    std::lock_guard<std::mutex> lock(thread_mutex);
    
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
  CreatePolicyDataset<T>::~CreatePolicyDataset()
  {
    std::thread* t = nullptr;
    
    {
      std::lock_guard<std::mutex> lk(thread_mutex);
      
      if(running || worker_thread != nullptr){
	running = false;
	t = worker_thread;
	worker_thread = nullptr;
      }
    }
      
    if(t){
      t->join();
      delete t;
    }
  }

  
  // starts thread that creates NUMDATAPOINTS samples to dataset
  template <typename T>
  bool CreatePolicyDataset<T>::start(const unsigned int NUMDATAPOINTS)
  {
    if(NUMDATAPOINTS == 0) return false;

    std::lock_guard<std::mutex> lock(thread_mutex);

    if(running == true || worker_thread != nullptr){
      char buf[256];
      snprintf(buf, 256, "CreatePolicyDataset<T>::start() FAILED (%d)",
	       (int)running);
      
      logging.info(buf);
      return false;
    } 

    try{
      NUMDATA = NUMDATAPOINTS;
      data.clear();
      data.createCluster("input-state", rifl.numStates);
      data.createCluster("importance sampling prob", 1);
      
      completed = false;
      
      running = true;
      worker_thread = new std::thread(std::bind(&CreatePolicyDataset<T>::loop, this));
      
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
  bool CreatePolicyDataset<T>::isCompleted() const
  {
    std::lock_guard<std::mutex> lock(thread_mutex);
    return completed;
  }
  
  // returns true if computation is running
  template <typename T>
  bool CreatePolicyDataset<T>::isRunning() const
  {
    std::lock_guard<std::mutex> lock(thread_mutex);
    return running;
  }

  template <typename T>
  bool CreatePolicyDataset<T>::stop()
  {
    std::thread* t = nullptr;
    bool rv = false;
    
    {
      std::lock_guard<std::mutex> lk(thread_mutex);
      
      if(running || worker_thread != nullptr){
	running = false;
	t = worker_thread;
	worker_thread = nullptr;

	rv = true;
      }
    }
      
    if(t){
      t->join();
      delete t;
    }

    return rv;
  }
  
  // returns reference to dataset
  // (warning: if calculations are running then dataset can change during use)
  template <typename T>
  whiteice::dataset<T> const & CreatePolicyDataset<T>::getDataset() const
  {
    return data;
  }
  
  // worker thread loop
  template <typename T>
  void CreatePolicyDataset<T>::loop()
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
      snprintf(buf, 256, "CreatePolicyDataset:loop() started: NUMDATA = %d\n", (int)NUMDATA);
      logging.info(buf);
    }

    {
      std::lock_guard<std::mutex> lock(database_mutex);
      
      //this->database = database__;
      
      unsigned int DSIZE = 10*NUMDATA;
      
      if(DSIZE >= database__.size()){
	this->database = database__;
      }
      else{
	database.resize(DSIZE);

	for(unsigned int i=0;i<database.size();i++){
	  database[i] = database__[rng.rand() % database__.size()];
	}
      }
      
    }
    

    std::map<double, unsigned int> weights;
    std::vector<double> pweights;

#if 1
    {
      //database_mutex.lock();

      // weighted sampling that are calculated from |reinforcement(s,a)| and stdev(Qi(s,a))--
      
      double total_weight = 0.0;
      T mean = T(0.0f);
      T var  = T(0.0f);

      double total_qweight = 0.0;
      T qmean = T(0.0f);
      T qvar  = T(0.0f);

      std::vector<T> qvalues;
      qvalues.resize(database.size());

      T rmean = T(0.0f);

      for(const auto& ei : database)
	rmean += ei.reinforcement;

      rmean /= database.size();

#pragma omp parallel
      {
	T pmean = T(0.0f);
	T pvar  = T(0.0f);
	
	T pqmean = T(0.0f);
	T pqvar  = T(0.0f);


#pragma omp for nowait
	for(unsigned int i=0;i<database.size();i++){
	  const T w = whiteice::math::pow(whiteice::math::abs(database[i].reinforcement-rmean), T(2.0f));
	  
	  pmean += w;
	  pvar  += w*w;
	  
	  whiteice::math::vertex<T> tmp(rifl.numStates + rifl.numActions);
	  tmp.zero();
	  
	  if(tmp.write_subvertex(database[i].state, 0) == false)
	    assert(0);

	  whiteice::math::vertex<T> tmp2(rifl.numStates), action;
	  tmp2.zero();

	  if(tmp2.write_subvertex(database[i].state, 0) == false)
	    assert(0);

	  policy_preprocess.preprocess(0, tmp2);

	  if(lagged_policy.calculate(tmp2, action, 1, 0) == false)
	    assert(0);

	  policy_preprocess.invpreprocess(1, action);
	  
	  if(tmp.write_subvertex(action, rifl.numStates) == false)
	    assert(0);
	  
	  this->Q_preprocess.preprocess(0, tmp);
	  
	  T qqmean = T(0.0f);
	  T qqvar  = T(0.0f);
	  
	  for(unsigned int k=0;k<lagged_Q.size();k++){
	    whiteice::math::vertex<T> q;
	    q.resize(1);
	    q.zero();
	    
	    if(this->lagged_Q[k].calculate(tmp, q, 1, 0) == false)
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

      const T stdev = whiteice::math::sqrt(whiteice::math::abs(var - mean*mean)) + T(1e-9);
      const T qstdev = whiteice::math::sqrt(whiteice::math::abs(qvar - qmean*qmean)) + T(1e-9);

      const double beta = 1.5; // temperature, more weights to high values [2,4]

      for(unsigned int i=0;i<database.size();i++){       
	const T w = whiteice::math::pow(whiteice::math::abs(database[i].reinforcement-rmean), T(2.0f));

	double z = beta*((double)(w.c[0]-mean.c[0])/stdev.c[0]);

	if(z > 20.0) z = 20.0;
	
	const double s = whiteice::math::exp(z);
	
	//const T s = T(1.0f)/(T(1.0f) + whiteice::math::exp(-beta*(w-mean)/stdev)); // softmax so outliers dont dominate
	total_weight += s;

	const T qw = whiteice::math::pow(whiteice::math::abs(qvalues[i]), T(2.0f));
	//const T qs = T(1.0f)/(T(1.0f) + whiteice::math::exp(-beta*(qw-qmean)/qstdev)); // softmax so outliers dont dominate

	double qz = beta*((double)(qw.c[0]-qmean.c[0])/qstdev.c[0]);

	if(qz > 20.0) qz = 20.0;
	
	const double qs = whiteice::math::exp(qz);
	
	total_qweight += qs;
      }
      
      // assert(total_weight > T(0.0f));
      if(total_weight <= 0.0)
	total_weight = 1.0;

      if(total_qweight <= 0.0)
	total_qweight = 1.0;
      

      const double mixing_factor = 0.50; // was: 0.20: 80% go to high reinforcement values..
      double sump = 0.0;

      //if(rifl.use_smart_weights)
      // mixing_factor = T(0.0f);
      
      for(unsigned int i=0;i<database.size();i++){
	std::pair<double, unsigned int> p;

	// sump += episodes_weights[i]/total_weight;
	const T w = whiteice::math::pow(whiteice::math::abs(database[i].reinforcement-rmean), T(2.0f));
	//const T s = T(1.0f)/(T(1.0f) + whiteice::math::exp(-beta*(w-mean)/stdev)); // softmax so outliers dont dominate

	double z = beta*((double)(w.c[0]-mean.c[0])/stdev.c[0]);

	if(z > 20.0) z = 20.0;
	
	const double s = whiteice::math::exp(z);

	const T qw = whiteice::math::pow(whiteice::math::abs(qvalues[i]), T(2.0f));		
	//const T qs = T(1.0f)/(T(1.0f) + whiteice::math::exp(-beta*(qw-qmean)/qstdev)); // softmax so outliers dont dominate

	double qz = beta*((double)(qw.c[0]-qmean.c[0])/qstdev.c[0]);

	if(qz > 20.0) qz = 20.0;
	
	const double qs = whiteice::math::exp(qz);

	const double pi = (1.0 - mixing_factor)*(s/total_weight) +
	  //(1.0 - mixing_factor)*(1.0/database.size()) +
	  mixing_factor*(qs/total_qweight);
	
	sump += pi;


	p.first = sump;
	p.second = i;

	weights.insert(p);
	pweights.push_back(pi);
      }

      // database_mutex.unlock();
    }
#endif

    

#pragma omp parallel for schedule(guided)
    for(unsigned int i=0;i<NUMDATA;i++){

      {
	if(!running.load(std::memory_order_relaxed)) // we don't do anything anymore..
	  continue; // exits OpenMP loop
      }
      
      // const unsigned int index = rng.rand() % database.size();

      unsigned int index = 0;

      double p = 1.0;

#if 1
      if(rng.rand() &  1){ // 50% of the samples are weighted
      
	const double r = rng.uniformd();
	
	auto iter = weights.upper_bound(r);
	
	if(iter != weights.end()){
	  index = iter->second;
	}

	// p is same for both random selections
	p = 0.5*pweights[index]*((double)weights.size())/((double)NUMDATA) + 0.5*1.0/((double)NUMDATA);
      }
      else
#endif
      {
	index = rng.rand() % database.size();
	
	p = 0.5*pweights[index]*((double)weights.size())/((double)NUMDATA) + 0.5*1.0/((double)NUMDATA);
      }
	  
	
	
      //database_mutex.lock();
      
      const auto datum = database[index];

      whiteice::math::vertex<T> pv;
      pv.resize(1);
      pv[0] = p;
      
      //database_mutex.unlock();
      
      
#pragma omp critical
      {
	data.add(0, datum.state);
	data.add(1, pv); // importance sampling probablities for the case
	

	//// std::cout << "policy dataset: state = " << datum.state << std::endl;
      }
      
    }

    {
      std::lock_guard<std::mutex> lock(thread_mutex);
      
      if(running == false)
	return; // exit point
    }

#if 1
    // add preprocessing to dataset (ENABLED!)
    {
      data.preprocess(0, whiteice::dataset<T>::dnMeanVarianceNormalization);
	
    }
#endif

    {
      unsigned int state_dimensions = 0;
      
      {
	//database_mutex.lock();
	
	if(database.size() > 0){
	  state_dimensions = database[0].state.size();
	}
	
	//database_mutex.unlock();
      }
    
      char buf[256];
      snprintf(buf, 256, "CreatePolicyDataset:loop(): data.size(0) = %d data.dimension(0) = %d dim(state) = %d\n", (int)data.size(0), (int)data.dimension(0), (int)state_dimensions);
      logging.info(buf);
    }
      
    {
      database.clear();
      
      std::lock_guard<std::mutex> lock(thread_mutex);
      completed = true;
      running = false;
    }
    
  }
  

  template class CreatePolicyDataset< math::blas_real<float> >;
  template class CreatePolicyDataset< math::blas_real<double> >;
};
