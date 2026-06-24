// Reinforcement learning using continuous state and continuous actions


#include "RIFL_abstract2.h"

#include "NNGradDescent.h"
#include "PolicyGradAscent.h"

#include "Log.h"
#include "linear_ETA.h"
#include "blade_math.h"

#include <assert.h>
#include <functional>
#include <list>


namespace whiteice
{

  template <typename T>
  RIFL_abstract2<T>::RIFL_abstract2(unsigned int numActions_,
				    unsigned int numStates_,
				    const bool alsoNegativeQValues,
				    const int sequentialRandomMoves_,
				    const unsigned int stateHistoryLen) :
    numActions(numActions_),
    numStates(numStates_*stateHistoryLen),
    STATE_HISTORY_LEN(stateHistoryLen)
  {
    // initializes parameters
    {
      assert(STATE_HISTORY_LEN >= 1);
      
      // zero = learn pure Q(state,action) = x function which action=policy(state) is optimized
      gamma = T(0.95); // how much weight future values Q() have: was 0.95 WAS: 0.80
      SAMPLESIZE = 4000; // dataset size used to learning

      if(sequentialRandomMoves_ >= 1)
	this->sequentialRandomMoves = sequentialRandomMoves_;
      else
	this->sequentialRandomMoves = 1;
      
      {
	std::lock_guard<std::mutex> locke(epsilon_mutex);
	epsilon = T(0.80);
      }

      learningMode = false;
      sleepMode = true;

      {
	std::lock_guard<std::mutex> lockh(has_model_mutex);

	hasModel.resize(NUM_Q_NNETWORKS + 1);

	for(unsigned int i=0;i<(NUM_Q_NNETWORKS + 1);i++)
	  hasModel[i] = 0; // Q-network... + policy-network
      }
	
      latestError = 0.0f;

      assert(numActions > 0);
      assert(numStates > 0);
    }

    
    // initializes neural network architecture and weights randomly
    // neural network is deep 6-layer residual neural network (NOW: 3 layers only)
    {
      std::vector<unsigned int> arch;

      // const unsigned int RELWIDTH = 20; // of the network (20..100)
      
      {
	std::lock_guard<std::mutex> lock(Q_mutex);

	// NOW: 10-layer small width neural network
	arch.push_back(numStates + numActions);
	arch.push_back(50);
	arch.push_back(50);
	arch.push_back(50);
	
	//arch.push_back(50);
	//arch.push_back(50);
	//arch.push_back(50);
	//arch.push_back(50);
	//arch.push_back(50);
	
	//arch.push_back(50);
	//arch.push_back(50);
	//arch.push_back(50);
	//arch.push_back(50);
	//arch.push_back(50);
	//arch.push_back(50);
	//arch.push_back(50);
	//arch.push_back(50);
	//arch.push_back(50);
	//arch.push_back(50);
	arch.push_back(1);
	
	{
	  whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::rectifier);
	  // whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::sigmoid); // tanh, sigmoid, halfLinear

	  nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<T>::pureLinear);
	  /*
	  if(alsoNegativeQValues == false){
	    nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<T>::sigmoid); // ([-1,+1])
	  }
	  else{
	    nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<T>::tanh); // ([0,1])
	  }
	  */
	  
	  // nn.randomize(2, T(0.5)); // was 1.0
	  nn.setResidual(true);
	  nn.setBatchNorm(false); // WAS: Q didn't have batch norm (NNGradDescent iteratively support batchnorm)

	  Q.resize(NUM_Q_NNETWORKS);
	  lagged_Q.resize(NUM_Q_NNETWORKS);

	  for(unsigned int i=0;i<NUM_Q_NNETWORKS;i++){
	    nn.randomize(2, T(0.5)); // was 1.0
	    Q[i].importNetwork(nn);
	    lagged_Q[i].importNetwork(nn);
	  }

	  whiteice::logging.info("RIFL_abstract2: ctor Q diagnostics");
	  for(unsigned int i=0;i<NUM_Q_NNETWORKS;i++){
	    Q[i].diagnosticsInfo();
	    lagged_Q[i].diagnosticsInfo();
	  }

	  Q_preprocess.createCluster("input-state", numStates + numActions);
	  Q_preprocess.createCluster("output-state", 1); // q-value
	}
      }
      
      
      {
	std::lock_guard<std::mutex> lock(policy_mutex);

	// NOW: 10-layer small width neural network
	arch.clear();
	arch.push_back(numStates);
	arch.push_back(50);
	arch.push_back(50);
	arch.push_back(50);
	
	//arch.push_back(50);
	//arch.push_back(50);
	//arch.push_back(50);
	//arch.push_back(50);
	//arch.push_back(50);
	
	//arch.push_back(50);
	//arch.push_back(50);
	//arch.push_back(50);
	//arch.push_back(50);
	//arch.push_back(50);
	//arch.push_back(50);
	//arch.push_back(50);
	//arch.push_back(50);
	//arch.push_back(50);
	//arch.push_back(50);	
	arch.push_back(numActions);

	// policy outputs action is (should be) +[-1,+1]^D vector
	{
	  whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::rectifier);
	  // whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::tanh);
	  // whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::tanh);
	  // whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::sigmoid);

	  // nn.setNonlinearity(0, whiteice::nnetwork<T>::pureLinear);
	  // nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<T>::sigmoid);
	  nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<T>::tanh);
	  
	  nn.randomize(2, T(0.9)); // was 1.0
	  nn.setResidual(true);
	  nn.setBatchNorm(false); // was: true
	  
	  policy.importNetwork(nn);
	  lagged_policy.importNetwork(nn);

	  whiteice::logging.info("RIFL_abstract2: ctor policy diagnostics");
	  lagged_policy.diagnosticsInfo();

	  policy_preprocess.createCluster("input-state", numStates);
	  policy_preprocess.createCluster("output-state", numActions);
	}
      }
      
    }
    
    
    thread_is_running = 0;
    rifl_thread = nullptr;
  }

  
  template <typename T>
  RIFL_abstract2<T>::RIFL_abstract2(unsigned int numActions_,
				    unsigned int numStates_,
				    const bool alsoNegativeQValues,
				    std::vector<unsigned int> Q_arch,
				    std::vector<unsigned int> policy_arch,
				    const int sequentialRandomMoves_,
				    const unsigned int stateHistoryLen) :
    numActions(numActions_),
    numStates(numStates_*stateHistoryLen),
    STATE_HISTORY_LEN(stateHistoryLen)
  {
    // initializes parameters
    {
      // zero = learn pure Q(state,action) = x function which action=policy(state) is optimized
      gamma = T(0.95); // how much weight future values Q() have: WAS: 0.95
      SAMPLESIZE = 4000; // dataset size used to learning

      if(sequentialRandomMoves_ >= 1)
	this->sequentialRandomMoves = sequentialRandomMoves_;
      else
	this->sequentialRandomMoves = 1;
      
      {
	std::lock_guard<std::mutex> locke(epsilon_mutex);
	epsilon = T(0.80);
      }

      learningMode = false;
      sleepMode = true;

      {
	std::lock_guard<std::mutex> lockh(has_model_mutex);
	
	hasModel.resize(NUM_Q_NNETWORKS + 1);

	for(unsigned int i=0;i<(NUM_Q_NNETWORKS + 1);i++)
	  hasModel[i] = 0; // Q-network... + policy-network
	
      }

      latestError = 0.0f;

      assert(numActions > 0);
      assert(numStates > 0);
      

      if(Q_arch.size() < 2){
	Q_arch.resize(2);
      }

      Q_arch[0] = numStates + numActions;
      Q_arch[Q_arch.size()-1] = 1;

      if(policy_arch.size() < 2){
	policy_arch.resize(2);
      }

      policy_arch[0] = numStates;
      policy_arch[policy_arch.size()-1] = numActions;
      
    }

    
    // initializes neural network architecture and weights randomly
    // neural network is deep 6-layer residual neural network (NOW: 3 layers only)
    {
      std::vector<unsigned int> arch;

      // const unsigned int RELWIDTH = 20; // of the network (20..100)
      
      {
	std::lock_guard<std::mutex> lock(Q_mutex);

	arch = Q_arch;

	{
	  whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::rectifier);

	  nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<T>::pureLinear); // [0,+1]

	  /*
	  if(alsoNegativeQValues == false){
	    nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<T>::sigmoid); // [0,+1]
	  }
	  else{
	    nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<T>::tanh); // [-1,+1]
	  }
	  */
	  
	  // nn.randomize(2, T(0.5)); // was 1.0
	  nn.setResidual(true);
	  nn.setBatchNorm(false); // BatchNorm wasn't enabled for Q network

	  Q.resize(NUM_Q_NNETWORKS);
	  lagged_Q.resize(NUM_Q_NNETWORKS);

	  for(unsigned int i=0;i<NUM_Q_NNETWORKS;i++){
	    nn.randomize(2, T(0.5)); // was 1.0
	    Q[i].importNetwork(nn);
	    lagged_Q[i].importNetwork(nn);
	  }

	  whiteice::logging.info("RIFL_abstract2: ctor Q diagnostics");
	  for(unsigned int i=0;i<NUM_Q_NNETWORKS;i++){
	    Q[i].diagnosticsInfo();
	    lagged_Q[i].diagnosticsInfo();
	  }
	  
	  Q_preprocess.createCluster("input-state", numStates + numActions);
	  Q_preprocess.createCluster("output-state", 1); // q-value
	  
	}
      }
      
      
      {
	std::lock_guard<std::mutex> lock(policy_mutex);

	arch.clear();

	arch = policy_arch;

	// policy outputs action is (should be) +[-1,+1]^D vector
	{
	  whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::rectifier);
	  // whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::tanh);
	  // whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::tanh);
	  // whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::sigmoid);
	  nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<T>::tanh);
	  // nn.setNonlinearity(0, whiteice::nnetwork<T>::pureLinear);
	  // nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<T>::sigmoid);
	  
	  nn.randomize(2, T(0.9)); // was 1.0
	  nn.setResidual(true);
	  nn.setBatchNorm(false); // was: true for policy
	  
	  policy.importNetwork(nn);

	  lagged_policy.importNetwork(nn);

	  whiteice::logging.info("RIFL_abstract2: ctor policy diagnostics");
	  lagged_policy.diagnosticsInfo();

	  policy_preprocess.createCluster("input-state", numStates);
	  policy_preprocess.createCluster("output-state", numActions);
	}
      }
      
    }
    
    
    thread_is_running = 0;
    rifl_thread = nullptr;
  }
  

  template <typename T>
  RIFL_abstract2<T>::~RIFL_abstract2() 
  {
    // stops executing thread
    {
      if(thread_is_running <= 0) return;

      std::lock_guard<std::mutex> lock(thread_mutex);

      if(thread_is_running <= 0) return;

      thread_is_running--;

      if(rifl_thread){
	rifl_thread->join();
	delete rifl_thread;
      }

      rifl_thread = nullptr;
    }
  }

  
  // starts Reinforcement Learning thread
  template <typename T>
  bool RIFL_abstract2<T>::start()
  {
    if(thread_is_running != 0) return false;

    std::lock_guard<std::mutex> lock(thread_mutex);

    if(thread_is_running != 0) return false;

    try{
      whiteice::logging.info("RIFL_abstract2: starting main thread");
      
      thread_is_running++;
      rifl_thread = new std::thread(std::bind(&RIFL_abstract2<T>::loop, this));
    }
    catch(std::exception& e){
      thread_is_running = 0;
      rifl_thread = nullptr;

      return false;
    }

    return true;
  }

  
  // stops Reinforcement Learning thread
  template <typename T>
  bool RIFL_abstract2<T>::stop()
  {
    if(thread_is_running <= 0) return false;

    std::lock_guard<std::mutex> lock(thread_mutex);

    if(thread_is_running <= 0) return false;

    thread_is_running--;

    if(rifl_thread){
      rifl_thread->join();
      delete rifl_thread;
    }

    rifl_thread = nullptr;
    return true;
  }

  template <typename T>
  bool RIFL_abstract2<T>::isRunning() const
  {
    return (thread_is_running > 0);
  }

  template <typename T>
  bool RIFL_abstract2<T>::setGamma(T gamma)
  {
    if(gamma <= T(0.0) || gamma >= T(1.0))
      return false;
    
    this->gamma = gamma;
    
    return true;
  }

  template <typename T>
  T RIFL_abstract2<T>::getGamma() const
  {
    return gamma;
  }

  // epsilon E [0,1] percentage of actions are chosen according to model
  //                 1-e percentage of actions are random (exploration)
  template <typename T>
  bool RIFL_abstract2<T>::setEpsilon(T epsilon) 
  {
    std::lock_guard<std::mutex> locke(epsilon_mutex);
    
    if(epsilon < T(0.0) || epsilon > T(1.0)) return false;
    this->epsilon = epsilon;
    
    return true;
  }
  

  template <typename T>
  T RIFL_abstract2<T>::getEpsilon() const 
  {
    std::lock_guard<std::mutex> locke(epsilon_mutex);
    return epsilon;
  }


  template <typename T>
  void RIFL_abstract2<T>::setLearningMode(bool learn) 
  {
    learningMode = learn;
  }

  template <typename T>
  bool RIFL_abstract2<T>::getLearningMode() const 
  {
    return learningMode;
  }

  template <typename T>
  void RIFL_abstract2<T>::setSleepingMode(bool sleep) 
  {
    sleepMode = sleep;
  }

  template <typename T>
  bool RIFL_abstract2<T>::getSleepingMode() const 
  {
    return sleepMode;
  }


  template <typename T>
  void RIFL_abstract2<T>::setHasModel(unsigned int hasModel) 
  {
    std::lock_guard<std::mutex> lockh(has_model_mutex);

    for(unsigned int i=0;i<(NUM_Q_NNETWORKS+1);i++)
      this->hasModel[i] = hasModel;
  }

  template <typename T>
  unsigned int RIFL_abstract2<T>::getHasModel() 
  {
    std::lock_guard<std::mutex> lockh(has_model_mutex);

    unsigned int min = hasModel[0];

    for(unsigned int i=1;i<(NUM_Q_NNETWORKS+1);i++)
      if(min > hasModel[i]) min = hasModel[i];
    
    return min;
  }


  template <typename T>
  float RIFL_abstract2<T>::getLatestEpisodeError() const
  {
    return latestError;
  }

  template <typename T>
  unsigned int RIFL_abstract2<T>::getDatabaseSize() const
  {
    std::lock_guard<std::mutex> lock(database_mutex);
    
    return database.size();
  }


  // how many percent smaller is reinforcement value with random actions vs policy actions
  template <typename T>
  bool RIFL_abstract2<T>::clearStatistics()
  {
    std::lock_guard<std::mutex> lock(reinforcements_mutex);

    reinforcements.clear();
    reinforcements_random.clear();

    distances.clear();
    distances_random.clear();

    return true; 
  }
  

  // how many percent smaller is reinforcement value with random actions vs policy actions
  template <typename T>
  bool RIFL_abstract2<T>::executionStatistics(T& percent_change,
					      T& distances_percent_change,
					      T& average_change,
					      T& linear_curve_distance_percent_change,
					      const bool rescale_to_min_value,
					      const bool use_only_most_recent,
					      unsigned int history_size) const
  {
    if(history_size <= 0) history_size = 1000;
    
    std::lock_guard<std::mutex> lock(reinforcements_mutex);
    
    percent_change = T(0.0f);
    average_change = T(0.0f);
    linear_curve_distance_percent_change = T(0.0f);
    
    if(reinforcements.size() <= 10 || reinforcements_random.size() <= 10)
      return false;

    {
      std::lock_guard<std::mutex> locke(epsilon_mutex);
      if(epsilon == T(0.0f) || epsilon == T(1.0f)) return false;
    }

        // reinforcement
    {
      T mean = T(0.0), stdev = T(0.0);
      T mean_random = T(0.0), stdev_random = T(0.0);
      
      if(use_only_most_recent == false){
	for(const auto& r : reinforcements){
	  mean += r;
	  stdev += r*r;
	}
	
	mean /= reinforcements.size();
	stdev /= reinforcements.size();
	
	stdev -= mean*mean;
	if(stdev < T(0.0))
	  stdev = T(0.0);
	
	stdev = sqrt(stdev/reinforcements.size()); // mean's stdev
      }
      else{
	int SAMPLES = 1;
	{
	  std::lock_guard<std::mutex> locke(epsilon_mutex);
	  SAMPLES = (int)round(history_size*epsilon.c[0]);
	}
	
	if(SAMPLES <= 0) SAMPLES = 1;
	
	int start = reinforcements.size() - SAMPLES;
	int end = reinforcements.size();
	
	if(start <= 0) start = 0;
	
	for(int i=start;i<end;i++){
	  const auto& r = reinforcements[i];
	  
	  mean += r;
	  stdev += r*r;
	}
	
	mean /= (end-start);
	stdev /= (end-start);
	
	stdev -= mean*mean;
	if(stdev < T(0.0))
	  stdev = T(0.0);
	
	stdev = sqrt(stdev/(end-start)); // mean's stdev
      }
      
      
      T min_random = T(0.0);
      
      if(use_only_most_recent == false){
	min_random = reinforcements_random[0];
	
	for(const auto& r : reinforcements_random){
	  mean_random += r;
	  stdev_random += r*r;
	  if(r < min_random) min_random = r;
	}
	
	mean_random /= reinforcements_random.size();
	stdev_random /= reinforcements_random.size();
	
	stdev_random -= mean_random*mean_random;
	if(stdev_random < T(0.0))
	  stdev_random = T(0.0);
	
	stdev_random = sqrt(stdev_random/reinforcements_random.size()); // mean's stdev
      }
      else{
	int SAMPLES = 1;
	
	{
	  std::lock_guard<std::mutex> locke(epsilon_mutex);
	  SAMPLES = (int)round(history_size*(1.0 - epsilon.c[0]));
	}
	
	if(SAMPLES <= 0) SAMPLES = 1;
	
	int start = reinforcements_random.size()-SAMPLES;
	int end = reinforcements_random.size();
	
	if(start <= 0) start = 0;
	
	min_random = reinforcements_random[start];
	
	for(int i=start;i<end;i++){
	  const auto& r = reinforcements_random[i];
	  mean_random += r;
	  stdev_random += r*r;
	  if(r < min_random) min_random = r;
	}
	
	mean_random /= (end-start);
	stdev_random /= (end-start);
	
	stdev_random -= mean_random*mean_random;
	if(stdev_random < T(0.0))
	  stdev_random = T(0.0);
	
	stdev_random = sqrt(stdev_random/(end-start)); // mean's stdev
      }
      
      if(rescale_to_min_value == false)
	min_random = 0.0;
      
      // p% = (mean - mean_random)/mean
      // min(p%) = (mean-stdev -(mean_random+stdev_random))/(mean+stdev)
      
      // if(mean+stdev <= T(0.0)) return false;
      
      if((mean_random-min_random) <= T(0.0)) return false;
      
      // percent_change = (mean-stdev - (mean_random+stdev_random))/(mean+stdev);

      percent_change = T(100.0)*(mean - mean_random)/(mean_random - min_random); // percentages
      
      average_change = mean;
    }

    // distances
    {
      T mean = T(0.0), stdev = T(0.0);
      T mean_random = T(0.0), stdev_random = T(0.0);
      
      if(use_only_most_recent == false){

	std::vector< whiteice::math::vertex<> > x, y;
	unsigned int index = 0;
	
	for(const auto& r : distances){
	  mean += r;
	  stdev += r*r;

	  whiteice::math::vertex<> v;
	  v.resize(1);
	  
	  v[0] = (float)(index+1);
	  x.push_back(v);

	  v[0] = r;
	  y.push_back(v);

	  index++;
	}
	
	mean /= distances.size();
	stdev /= distances.size();
	
	stdev -= mean*mean;
	if(stdev < T(0.0))
	  stdev = T(0.0);
	
	stdev = sqrt(stdev/distances.size()); // mean's stdev
	
	whiteice::math::matrix<> A;
	whiteice::math::vertex<> b;
	whiteice::math::blas_real<float> error;
	
	if(whiteice::math::linear_optimization<>(x, y, A, b, error) == false)
	  return false;
	
	// 1d model is y = a*x + b
	
	// IMPORTANT: assumes distance is between [0,1]
	
	float y0 = A(0,0).c[0]*x[0][0].c[0] + b[0].c[0];
	float y1 = A(0,0).c[0]*x[x.size()-1][0].c[0] + b[0].c[0];
	
	if(y0 < 0.0f) y0 = 0.0f;
	else if(y0 >= 1.0f){
	  linear_curve_distance_percent_change = 0.0f;
	  // return false; // => zero division
	}
	else{
	  
	  if(y1 < 0.0f) y1 = 0.0f;
	  else if(y1 >= 1.0f) y1 = 1.0f;
	  
	  const float p = 100.0f*(y0-y1)/(1.0f - y0);
	  
	  linear_curve_distance_percent_change = p;
	  //percent_measure_growth = p;
	}
      }
      else{
	int SAMPLES = 1;
	{
	  std::lock_guard<std::mutex> locke(epsilon_mutex);
	  SAMPLES = (int)round(history_size*epsilon.c[0]);
	}
	
	if(SAMPLES <= 0) SAMPLES = 1;
	
	int start = distances.size() - SAMPLES;
	int end = distances.size();
	
	if(start <= 0) start = 0;

	std::vector< whiteice::math::vertex<> > x, y;
	
	for(int i=start;i<end;i++){
	  const auto& r = distances[i];

	  whiteice::math::vertex<> v;
	  v.resize(1);
	  
	  v[0] = (float)(i+1);
	  x.push_back(v);

	  v[0] = r;
	  y.push_back(v);
	  
	  mean += r;
	  stdev += r*r;
	}
	
	mean /= (end-start);
	stdev /= (end-start);
	
	stdev -= mean*mean;
	if(stdev < T(0.0))
	  stdev = T(0.0);
	
	stdev = sqrt(stdev/(end-start)); // mean's stdev

	whiteice::math::matrix<> A;
	whiteice::math::vertex<> b;
	whiteice::math::blas_real<float> error;
	
	if(whiteice::math::linear_optimization<>(x, y, A, b, error) == false)
	  return false;
	
	// 1d model is y = a*x + b

	// IMPORTANT: assumes distance is between [0,1]
	
	float y0 = A(0,0).c[0]*x[0][0].c[0] + b[0].c[0];
	float y1 = A(0,0).c[0]*x[x.size()-1][0].c[0] + b[0].c[0];
	
	if(y0 < 0.0f) y0 = 0.0f;
	else if(y0 >= 1.0f){
	  linear_curve_distance_percent_change = 0.0f;
	  // return false; // => zero division
	}
	else{
	  
	  if(y1 < 0.0f) y1 = 0.0f;
	  else if(y1 >= 1.0f) y1 = 1.0f;
	  
	  const float p = 100.0f*(y0-y1)/(1.0f - y0);
	  
	  linear_curve_distance_percent_change = p;
	  //percent_measure_growth = p;
	}
	
      }
      
      
      T min_random = T(0.0);
      
      if(use_only_most_recent == false){
	min_random = distances_random[0];
	
	for(const auto& r : distances_random){
	  mean_random += r;
	  stdev_random += r*r;
	  if(r < min_random) min_random = r;
	}
	
	mean_random /= distances_random.size();
	stdev_random /= distances_random.size();
	
	stdev_random -= mean_random*mean_random;
	if(stdev_random < T(0.0))
	  stdev_random = T(0.0);
	
	stdev_random = sqrt(stdev_random/reinforcements_random.size()); // mean's stdev
      }
      else{
	int SAMPLES = 1;
	
	{
	  std::lock_guard<std::mutex> locke(epsilon_mutex);
	  SAMPLES = (int)round(history_size*(1.0 - epsilon.c[0]));
	}
	
	if(SAMPLES <= 0) SAMPLES = 1;
	
	int start = distances_random.size()-SAMPLES;
	int end = distances_random.size();
	
	if(start <= 0) start = 0;
	
	min_random = distances_random[start];
	
	for(int i=start;i<end;i++){
	  const auto& r = distances_random[i];
	  mean_random += r;
	  stdev_random += r*r;
	  if(r < min_random) min_random = r;
	}
	
	mean_random /= (end-start);
	stdev_random /= (end-start);
	
	stdev_random -= mean_random*mean_random;
	if(stdev_random < T(0.0))
	  stdev_random = T(0.0);
	
	stdev_random = sqrt(stdev_random/(end-start)); // mean's stdev
      }
      
      if(rescale_to_min_value == false)
	min_random = 0.0;
      
      // p% = (mean - mean_random)/mean
      // min(p%) = (mean-stdev -(mean_random+stdev_random))/(mean+stdev)
      
      // if(mean+stdev <= T(0.0)) return false;
      
      if((mean_random-min_random) <= T(0.0)) return false;
      
      // percent_change = (mean-stdev - (mean_random+stdev_random))/(mean+stdev);

      distances_percent_change = T(100.0)*(mean_random - mean)/(mean_random - min_random); // percentages
    }

    return true;
  }

  
  // saves learnt Reinforcement Learning Model to file
  template <typename T>
  bool RIFL_abstract2<T>::save(const std::string& filename) const
  {
    char buffer[256];
    
    {
      std::lock_guard<std::mutex> lock1(Q_mutex);
      std::lock_guard<std::mutex> lock2(policy_mutex);

      for(unsigned int i=0;i<NUM_Q_NNETWORKS;i++){
	snprintf(buffer, 256, "%s-q%d", filename.c_str(), i);
	
	if(Q[i].save(buffer) == false){
	  logging.error("RIFL_abstract2::save() saving Q failed");
	  return false;
	}

	snprintf(buffer, 256, "%s-lagged-q%d", filename.c_str(), i);
	
	if(lagged_Q[i].save(buffer) == false){
	  logging.error("RIFL_abstract2::save() saving lagged-q failed");
	  return false;
	}
      }
            
      snprintf(buffer, 256, "%s-policy", filename.c_str());
      if(policy.save(buffer) == false){
	logging.error("RIFL_abstract2::save() saving policy failed");
	return false;
      }
      
      snprintf(buffer, 256, "%s-lagged-policy", filename.c_str());
      if(lagged_policy.save(buffer) == false){
	logging.error("RIFL_abstract2::save() saving lagged-policy failed");
	return false;
      }
      
      snprintf(buffer, 256, "%s-q-preprocess", filename.c_str());    
      if(Q_preprocess.save(buffer) == false){
	logging.error("RIFL_abstract2::save() saving q-preprocess failed");
	return false;
      }
      
      snprintf(buffer, 256, "%s-policy-preprocess", filename.c_str());
      if(policy_preprocess.save(buffer) == false){
	logging.error("RIFL_abstract2::save() saving policy-preprocess failed");
	return false;
      }
    }

    {
      snprintf(buffer, 256, "%s-hasmodel", filename.c_str());

      whiteice::dataset<T> db;

      db.createCluster("has_model", NUM_Q_NNETWORKS+1);

      whiteice::math::vertex<T> v;
      v.resize(NUM_Q_NNETWORKS+1);
      v.zero();

      std::lock_guard<std::mutex> lockh(has_model_mutex);

      for(unsigned int i=0;i<(NUM_Q_NNETWORKS+1);i++)
	v[i] = T(hasModel[i]);
      

      if(db.add(0, v) == false){
	logging.error("RIFL_abstract2::save(): saving hasModel data failed.");
	return false;
      }

      if(db.save(buffer) == false){
	logging.error("RIFL_abstract2::save(): saving hasModel dataset file failed.");
	return false;
      }
    }

    {
      std::lock_guard<std::mutex> lock(reinforcements_mutex);
      
      snprintf(buffer, 256, "%s-measurements", filename.c_str());

      whiteice::dataset<T> db;

      db.createCluster("measurements", reinforcements.size());
      db.createCluster("measurements_random", reinforcements_random.size());
      db.createCluster("distances", distances.size());
      db.createCluster("distances_random", distances_random.size());

      whiteice::math::vertex<T> v;
      v.resize(reinforcements.size());
      v.zero();

      for(unsigned int i=0;i<reinforcements.size();i++)
	v[i] = reinforcements[i];

      if(db.add(0, v) == false){
	logging.error("RIFL_abstract2::save(): saving measurements data failed (1).");
	return false;
      }

      v.resize(reinforcements_random.size());
      v.zero();

      for(unsigned int i=0;i<reinforcements_random.size();i++)
	v[i] = reinforcements_random[i];

      if(db.add(1, v) == false){
	logging.error("RIFL_abstract2::save(): saving measurements data failed (2).");
	return false;
      }

      v.resize(distances.size());
      v.zero();

      for(unsigned int i=0;i<distances.size();i++)
	v[i] = distances[i];

      if(db.add(2, v) == false){
	logging.error("RIFL_abstract4::save(): saving measurements data failed (3).");
	return false;
      }

      v.resize(distances_random.size());
      v.zero();

      for(unsigned int i=0;i<distances_random.size();i++)
	v[i] = distances_random[i];

      if(db.add(3, v) == false){
	logging.error("RIFL_abstract4::save(): saving measurements data failed (4).");
	return false;
      }

      if(db.save(buffer) == false){
	logging.error("RIFL_abstract2::save(): saving measurements data failed.");
	return false;
      }
    }

    {
      snprintf(buffer, 256, "%s-database", filename.c_str());

      whiteice::dataset<T> db;

      std::lock_guard<std::mutex> lock1(database_mutex);

      if(database.size() > 0)
	db.createCluster("state", database[0].state.size());
      else
	db.createCluster("state", 1);

      if(database.size() > 0)
	db.createCluster("newstate", database[0].newstate.size());
      else
	db.createCluster("newstate", 1);

      if(database.size() > 0)
	db.createCluster("action", database[0].action.size());
      else
	db.createCluster("action", 1);

      if(database.size() > 0)
	db.createCluster("reinforcement", 1);
      else
	db.createCluster("reinforcement", 1);

      if(database.size() > 0)
	db.createCluster("last_step", 1);
      else
	db.createCluster("last_step", 1);

      if(database.size() > 0)
	db.createCluster("random", 1);
      else
	db.createCluster("random", 1);

      if(database.size() > 0)
	db.createCluster("reinforcement_pure", 1);
      else
	db.createCluster("reinforcement_pure", 1);

      if(database.size() > 0)
	db.createCluster("distance", 1);
      else
	db.createCluster("distance", 1);
      

      for(unsigned int i=0;i<database.size();i++){
	db.add(0, database[i].state);
	db.add(1, database[i].newstate);
	db.add(2, database[i].action);

	whiteice::math::vertex<T> v;
	v.resize(1);
	v[0] = database[i].reinforcement;
	db.add(3, v);

	if(database[i].lastStep)
	  v[0] = T(1.0f);
	else
	  v[0] = T(0.0f);

	db.add(4, v);

	if(database[i].random)
	  v[0] = T(1.0f);
	else
	  v[0] = T(0.0f);

	db.add(5, v);

	v[0] = database[i].reinforcement_pure;
	db.add(6, v);

	v[0] = database[i].distance;
	db.add(7, v);
      }

      if(db.save(buffer) == false){
	logging.error("RIFL_abstract2::save() saving database failed");
	return false;
      }
    }


    {
      snprintf(buffer, 256, "%s-episodes", filename.c_str());

      whiteice::dataset<T> db;

      std::lock_guard<std::mutex> lock1(database_mutex);

      if(database.size() > 0)
	db.createCluster("state", database[0].state.size());
      else
	db.createCluster("state", 1);

      if(database.size() > 0)
	db.createCluster("newstate", database[0].newstate.size());
      else
	db.createCluster("newstate", 1);

      if(database.size() > 0)
	db.createCluster("action", database[0].action.size());
      else
	db.createCluster("action", 1);

      if(database.size() > 0)
	db.createCluster("reinforcement", 1);
      else
	db.createCluster("reinforcement", 1);

      if(database.size() > 0)
	db.createCluster("last_step", 1);
      else
	db.createCluster("last_step", 1);

      if(database.size() > 0)
	db.createCluster("random", 1);
      else
	db.createCluster("random", 1);

      if(database.size() > 0)
	db.createCluster("reinforcement_pure", 1);
      else
	db.createCluster("reinforcement_pure", 1);

      if(database.size() > 0)
	db.createCluster("distance", 1);
      else
	db.createCluster("distance", 1);

      db.createCluster("episodes-range", 2);
      

      for(unsigned int e=0;e<episodes.size();e++){
	const unsigned int start = db.size(0);
	
	for(unsigned int i=0;i<episodes[e].size();i++){
	  db.add(0, episodes[e][i].state);
	  db.add(1, episodes[e][i].newstate);
	  db.add(2, episodes[e][i].action);
	  
	  whiteice::math::vertex<T> v;
	  v.resize(1);
	  
	  v[0] = episodes[e][i].reinforcement;
	  
	  db.add(3, v);
	  
	  if(episodes[e][i].lastStep)
	    v[0] = T(1.0f);
	  else
	    v[0] = T(0.0f);
	  
	  db.add(4, v);

	  if(episodes[e][i].random)
	    v[0] = T(1.0f);
	  else
	    v[0] = T(0.0f);
	  
	  db.add(5, v);
	  
	  v[0] = episodes[e][i].reinforcement_pure;
	  
	  db.add(6, v);

	  v[0] = episodes[e][i].distance;
	  
	  db.add(7, v);
	}

	const unsigned int end = db.size(0);

	whiteice::math::vertex<T> v;
	v.resize(2);

	v[0] = T(start);
	v[1] = T(end);

	db.add(8, v);
      }


      if(db.save(buffer) == false){
	logging.error("RIFL_abstract2::save() saving episodes failed");
	return false;
      }
    }
    

    return true;
  }

  
  // loads learnt Reinforcement Learning Model from file
  template <typename T>
  bool RIFL_abstract2<T>::load(const std::string& filename)
  {
    char buffer[256];

    Q_mutex.lock();
    policy_mutex.lock();
    has_model_mutex.lock();
    database_mutex.lock();
    reinforcements_mutex.lock();

    auto Q_load = Q;
    auto lagged_Q_load = lagged_Q;
    
    auto policy_load = policy;
    auto lagged_policy_load = lagged_policy;
    auto Q_preprocess_load = Q_preprocess;
    auto policy_preprocess_load = policy_preprocess;
    auto hasModel_load = hasModel;
    auto database_load = database;
    auto reinforcements_load = reinforcements;
    auto reinforcements_random_load = reinforcements_random;
    auto distances_load = distances;
    auto distances_random_load = distances_random;
    auto episodes_load = episodes;

    reinforcements_mutex.unlock();
    database_mutex.unlock();
    has_model_mutex.unlock();
    policy_mutex.unlock();
    Q_mutex.unlock();
       
    {
      Q_load.resize(NUM_Q_NNETWORKS);
      lagged_Q_load.resize(NUM_Q_NNETWORKS);
      
      for(unsigned int i=0;i<NUM_Q_NNETWORKS;i++){

	snprintf(buffer, 256, "%s-q%d", filename.c_str(), i);
	
	if(Q_load[i].load(buffer) == false){
	  logging.error("RIFL_abstract2::load() loading Q failed");
	  return false;
	}
	
	snprintf(buffer, 256, "%s-lagged-q%d", filename.c_str(), i);
	
	if(lagged_Q_load[i].load(buffer) == false){
	  logging.error("RIFL_abstract2::load() loading lagged-q failed");
	  return false;
	}
      }
      
      snprintf(buffer, 256, "%s-policy", filename.c_str());
      if(policy_load.load(buffer) == false){
	logging.error("RIFL_abstract2::load() loading policy failed");
	return false;
      }
      
      snprintf(buffer, 256, "%s-lagged-policy", filename.c_str());
      if(lagged_policy_load.load(buffer) == false){
	logging.error("RIFL_abstract2::load() loading lagged-policy failed");
	return false;
      }
      
      snprintf(buffer, 256, "%s-q-preprocess", filename.c_str());    
      if(Q_preprocess_load.load(buffer) == false){
	logging.error("RIFL_abstract2::load() loading q_preprocess failed");
	return false;
      }
      
      snprintf(buffer, 256, "%s-policy-preprocess", filename.c_str());
      if(policy_preprocess_load.load(buffer) == false){
	logging.error("RIFL_abstract2::load() loading policy_preprocess failed");
	return false;
      }
    }

    {
      snprintf(buffer, 256, "%s-hasmodel", filename.c_str());
      
      whiteice::dataset<T> db;

      if(db.load(buffer) == false){
	logging.error("RIFL_abstract2::load() loading hasModel dataset file failed");
	return false;
      }

      if(db.size(0) != 1 && db.dimension(0) != (NUM_Q_NNETWORKS+1)){
	logging.error("RIFL_abstract2::load() loading hasModel dataset file failed (2)");
	return false;
      }

      whiteice::math::vertex<T> v;
      v.resize(NUM_Q_NNETWORKS+1);
      v.zero();

      v = db.access(0,0);

      hasModel_load.resize(NUM_Q_NNETWORKS+1);

      for(unsigned int i=0;i<(NUM_Q_NNETWORKS+1);i++){
	hasModel_load[i] = (int)v[i].c[0];
      }
    }

    {
      std::lock_guard<std::mutex> lock(reinforcements_mutex);
      
      snprintf(buffer, 256, "%s-measurements", filename.c_str());

      whiteice::dataset<T> db;

      if(db.load(buffer) == false){
	logging.error("RIFL_abstract2::load(): loading measurements data failed.");
	return false;
      }

      if(db.getNumberOfClusters() != 4){
	logging.error("RIFL_abstract2::load(): loading measurements data failed (2).");
	return false;
      }

      if(db.size(0) != 1 || db.size(1) != 1  || db.size(2) != 1 || db.size(3) != 1){
	logging.error("RIFL_abstract2::load(): loading measurements data failed (3).");
	return false;
      }

      whiteice::math::vertex<T> v;
      v.resize(db.dimension(0));
      v = db.access(0,0);
      
      reinforcements_load.resize(v.size());
      
      for(unsigned int i=0;i<reinforcements_load.size();i++)
	reinforcements_load[i] = v[i];

      v.resize(db.dimension(1));
      v = db.access(1,0); 
      reinforcements_random_load.resize(v.size());

      for(unsigned int i=0;i<reinforcements_random_load.size();i++)
	reinforcements_random_load[i] = v[i];
      
      v.resize(db.dimension(2));
      v = db.access(2,0); 
      distances_load.resize(v.size());

      for(unsigned int i=0;i<distances_load.size();i++)
	distances_load[i] = v[i];

      v.resize(db.dimension(3));
      v = db.access(3,0);
      distances_random_load.resize(v.size());

      for(unsigned int i=0;i<distances_random_load.size();i++)
	distances_random_load[i] = v[i];
    }

    {
      snprintf(buffer, 256, "%s-database", filename.c_str());
      
      whiteice::dataset<T> db;

      if(db.load(buffer) == false){
	char buf[128];
	snprintf(buf, 128, "RIFL_abstract2::load(\"%s\") loading dataset FAILED", buffer);
	logging.error(buf);
	return false;
      }

      if(db.getNumberOfClusters() != 8){
	logging.error("RIFL_abstract2::load() database wrong number of clusters");
	return false;
      }

      if(db.dimension(0) != db.dimension(1) || db.dimension(3) != 1 || db.dimension(4) != 1){
	char buf[128];
	snprintf(buf, 128, "RIFL_abstract2::load() database wrong dimensions %d %d %d %d %d",
		 db.dimension(0), db.dimension(1), db.dimension(3), db.dimension(3),
		 db.dimension(4));
	logging.error(buf);
	return false;
      }

      if(db.dimension(0) != this->numStates){
	char buf[128];
	snprintf(buf, 128, "RIFL_abstract2::load() database wrong dimensions %d %d (2)",
		 db.dimension(0), this->numStates);
	logging.error(buf);
	return false;
      }
      
      if(db.dimension(2) != this->numActions){
	char buf[128];
	snprintf(buf, 128, "RIFL_abstract2::load() database wrong dimensions %d %d (3)",
		 db.dimension(2), this->numActions);
	logging.error(buf);

	return false;
      }

      if(db.size(0) != db.size(1) || db.size(1) != db.size(2) || db.size(2) != db.size(3) ||
	 db.size(3) != db.size(4)){

	char buf[128];
	snprintf(buf, 128, "RIFL_abstract2::load() database wrong size %d %d %d %d %d",
		 db.size(0), db.size(1), db.size(2), db.size(3), db.size(4)); 
	logging.error(buf);
	
	return false;
      }
					     
      
      
      database_load.clear();
      
      whiteice::rifl2_datapoint<T> p;
      whiteice::math::vertex<T> v;

      for(unsigned int i=0;i<db.size(0);i++){
	p.state = db.access(0, i);
	p.newstate = db.access(1, i);
	p.action = db.access(2, i);
	v = db.access(3, i);
	p.reinforcement = v[0];
	v = db.access(4, i);
	if(v[0] > T(0.5)) p.lastStep = true;
	else p.lastStep = false;

	v = db.access(5, i);
	if(v[0] > T(0.5)) p.random = true;
	else p.random = false;
	
	v = db.access(6, i);
	p.reinforcement_pure = v[0];

	v = db.access(7, i);
	p.distance = v[0];
	
	database_load.push_back(p);
      }
      
    }


    {
      snprintf(buffer, 256, "%s-episodes", filename.c_str());
      
      whiteice::dataset<T> db;

      if(db.load(buffer) == false){
	char buf[1024];
	snprintf(buf, 1024, "RIFL_abstract2::load(\"%s\") loading episodes dataset FAILED", buffer);
	logging.error(buf);
	return false;
      }

      if(db.getNumberOfClusters() != 9){
	logging.error("RIFL_abstract2::load() episodes database wrong number of clusters");
	return false;
      }

      if(db.dimension(0) != db.dimension(1) ||
	 db.dimension(3) != 1 || db.dimension(4) != 1 || db.dimension(8) != 2){
	char buf[128];
	snprintf(buf, 128, "RIFL_abstract2::load() database wrong dimensions %d %d %d %d %d %d",
		 db.dimension(0), db.dimension(1), db.dimension(3), db.dimension(3),
		 db.dimension(4), db.dimension(8));
	logging.error(buf);
	return false;
      }

      if(db.dimension(0) != this->numStates){
	char buf[128];
	snprintf(buf, 128, "RIFL_abstract2::load() database wrong dimensions %d %d (2)",
		 db.dimension(0), this->numStates);
	logging.error(buf);
	return false;
      }
      
      if(db.dimension(2) != this->numActions){
	char buf[128];
	snprintf(buf, 128, "RIFL_abstract2::load() database wrong dimensions %d %d (3)",
		 db.dimension(2), this->numActions);
	logging.error(buf);

	return false;
      }

      if(db.dimension(8) != 2){
	char buf[128];
	snprintf(buf, 128, "RIFL_abstract2::load() database wrong dimensions %d %d (4)",
		 db.dimension(5), 2);
	logging.error(buf);

	return false;	
      }

      if(db.size(0) != db.size(1) || db.size(1) != db.size(2) || db.size(2) != db.size(3) ||
	 db.size(3) != db.size(4)){

	char buf[128];
	snprintf(buf, 128, "RIFL_abstract2::load() database wrong size %d %d %d %d %d",
		 db.size(0), db.size(1), db.size(2), db.size(3), db.size(4));
	logging.error(buf);
	
	return false;
      }
      
      
      episodes_load.clear();

      for(unsigned int e=0;e<db.size(8);e++){

	std::vector< whiteice::rifl2_datapoint<T> > epi;
	
	whiteice::rifl2_datapoint<T> p;
	whiteice::math::vertex<T> v;

	v = db.access(8, e);

	unsigned int START = 0;
	unsigned int END = 0;

	whiteice::math::convert(START, v[0]);
	whiteice::math::convert(END, v[1]);

	assert(START < db.size(0));
	assert(END <= db.size(0));

	for(unsigned int i=START;i<END;i++){
	  p.state = db.access(0, i);
	  p.newstate = db.access(1, i);
	  p.action = db.access(2, i);
	  
	  v = db.access(3, i);
	  p.reinforcement = v[0];
	  
	  v = db.access(4, i);
	  if(v[0] > T(0.5)) p.lastStep = true;
	  else p.lastStep = false;

	  v = db.access(5, i);
	  if(v[0] > T(0.5)) p.random = true;
	  else p.random = false;

	  v = db.access(6, i);
	  p.reinforcement_pure = v[0];

	  v = db.access(7, i);
	  p.distance = v[0];
	  
	  epi.push_back(p);
	}

	episodes_load.push_back(epi);	
      }
      
    }
    
    
    {
      std::lock_guard<std::mutex> lock1(Q_mutex);
      std::lock_guard<std::mutex> lock2(policy_mutex);
      std::lock_guard<std::mutex> lockh(has_model_mutex);
      std::lock_guard<std::mutex> lockd(database_mutex);
      std::lock_guard<std::mutex> lockr(reinforcements_mutex);
      
      Q = Q_load;
      lagged_Q = lagged_Q_load;
      policy = policy_load;
      lagged_policy = lagged_policy_load;
      Q_preprocess = Q_preprocess_load;
      policy_preprocess = policy_preprocess_load;
      hasModel = hasModel_load;
      database = database_load;
      reinforcements = reinforcements_load;
      reinforcements_random = reinforcements_random_load;
      distances = distances_load;
      distances_random = distances_random_load;
      episodes = episodes_load;
    }
    
    return true;
  }


  template <typename T>
  void RIFL_abstract2<T>::onehot_prob_select(const whiteice::math::vertex<T>& action,
					     whiteice::math::vertex<T>& new_action,
					     const T temperature)
  {
    assert(action.size() > 0);
    
    unsigned long ACTION = 0;

    T psum = T(0.0f);
    std::vector<T> p;
    
    for(unsigned int i=0;i<action.size();i++){
      auto value = action[i];
      
      if(value < T(-6.0f)) value = T(-6.0f);
      else if(value > T(+6.0f)) value = T(+6.0f);
      
      auto q = exp(value/temperature);
      psum += q;
      p.push_back(q);
    }
    
    for(unsigned int i=0;i<p.size();i++)
      p[i] /= psum;
    
    psum = T(0.0f);
    for(unsigned int i=0;i<p.size();i++){
      auto more = p[i];
      p[i] += psum;
      psum += more;
    }
    
    T r = rng.uniform();
    
    unsigned long index = 0;
    
    while(r > p[index]){
      index++;
      if(index >= p.size()){
	index = p.size()-1;
	break;
      }
    }
    
    ACTION = index;

    // std::cout << "action = " << action << " => SELECT ACTION: " << ACTION << std::endl;

    new_action.resize(action.size());
    new_action.zero();

#if 1
    for(unsigned int i=0;i<new_action.size();i++){
      new_action[ACTION] = T(-1.0f);
    }
#endif
    
    new_action[ACTION] = T(1.0f);
  }

  
  template <typename T>
  void RIFL_abstract2<T>::loop()
  {
    // number of iteratios to use per epoch for optimization
    // const unsigned int Q_OPTIMIZE_ITERATIONS = 100; // 40, was 1 (dont work), 5, 10, WAS: 5000
    // const unsigned int P_OPTIMIZE_ITERATIONS = 100; // 10, was 1 (dont work), 5, 10, WAS: 1000

    // number of iteratios to use per epoch for optimization
    const unsigned int Q_OPTIMIZE_ITERATIONS_FIRST = 100; // WAS: 1000,20
    const unsigned int P_OPTIMIZE_ITERATIONS_FIRST = 100; // WAS: 100,20

    const unsigned int Q_OPTIMIZE_ITERATIONS = 20; // 3; // WAS: 500
    const unsigned int P_OPTIMIZE_ITERATIONS = 20; // 3; // WAS: 100
    
    // tau = 1.0 => no lagged neural networks [don't work]
    // const T tau = T(0.001); // lagged Q and policy network [keeps tau%=1% of the new weights [was: 0.001, 0.05, 1.0*]
    // const T tau_policy = T(0.005); // was: 1.0*
    
    // std::vector< std::vector< rifl2_datapoint<T> > > episodes;
    std::vector< rifl2_datapoint<T> > episode;

    FILE* episodesFile = fopen("episodes-result.txt", "w");    

    bool endFlag = false; // did the simulation end during this time step?

    std::vector< whiteice::dataset<T> > data;
    std::vector< whiteice::CreateRIFL2dataset<T>* > dataset_thread;
    std::vector< whiteice::math::NNGradDescent<T> > grad;

    data.resize(NUM_Q_NNETWORKS);
    dataset_thread.resize(NUM_Q_NNETWORKS);
    grad.resize(NUM_Q_NNETWORKS);

    for(unsigned int i=0;i<NUM_Q_NNETWORKS;i++)
      dataset_thread[i] = NULL;
    
    
    //whiteice::CreateRIFL2dataset<T>* dataset_thread = nullptr;
    //whiteice::CreateRIFL2dataset<T>* dataset_q2_thread = nullptr;
    //whiteice::math::NNGradDescent<T> grad; // Q(state,action) model optimizer
    //whiteice::math::NNGradDescent<T> q2grad; // Q2(state,action) model optimizer
    
    // deep pretraining using stacked RBMs
    // (requires sigmoidal nnetwork and training
    //  policy nnetwork (calculating gradients) dont work with sigmoid)
    const bool deep = false;
    whiteice::dataset<T> data2;
    whiteice::CreatePolicyDataset<T>* dataset2_thread = nullptr;
    whiteice::PolicyGradAscent<T> grad2(deep);   // policy(state)=action model optimizer

    std::vector< whiteice::linear_ETA<double> > eta; // estimates how long single epoch of optimization takes
    whiteice::linear_ETA<double> eta2; // estimates how long single epoch of optimization takes

    eta.resize(NUM_Q_NNETWORKS);
    
    std::vector<unsigned int> epoch;

    epoch.resize(NUM_Q_NNETWORKS + 1);

    for(unsigned int i=0;i<(NUM_Q_NNETWORKS + 1);i++)
      epoch[i] = 0;

    std::vector<int> grad_iterations;

    grad_iterations.resize(NUM_Q_NNETWORKS + 1);

    for(unsigned int i=0;i<(NUM_Q_NNETWORKS + 1);i++)
      grad_iterations[i] = -1;
    

    const unsigned long DATASIZE = 1000000; // was: 100K / 1M history of samples
    // assumes each episode length is 100 so this is ~ equal to 1.000.000 samples
    const unsigned long EPISODES_MAX_SIZE = 100000;
    // const unsigned long MINIMUM_EPISODE_SIZE = 15;// was: 25
    // const unsigned long MINIMUM_DATASIZE = 500; // samples required to start learning, was:1000
    // const unsigned long SAMPLESIZE = 4000; // number of samples used in learning, was: 5000,*2000*,1000,4000,8000
    unsigned long database_counter = 0;
    unsigned long episodes_counter = 0;

    int random_counter = 0; // how many times to do random action

    latestError = 0.0f;
    
    bool firstTime = true;
    whiteice::math::vertex<T> state(numStates);
    whiteice::math::vertex<T> action(numActions);

    state.zero();

    std::vector< whiteice::math::vertex<T> > state_history;

    state_history.resize(STATE_HISTORY_LEN);

    for(unsigned int i=0;i<STATE_HISTORY_LEN;i++){
      state_history[i].resize(state.size()/STATE_HISTORY_LEN);
      state_history[i].zero();
    }


    {
      std::lock_guard<std::mutex> lockr(reinforcements_mutex);
      reinforcements.clear();
      reinforcements_random.clear();

      distances.clear();
      distances_random.clear();
    }

    after_effects_buffer.clear();

    bool random = false;

    // to properly handle cases where performAction() fails
    // [don't getState() or use policy network again] until performAction() is successful
    unsigned int performActionFailed = 0; 

    whiteice::nnetwork<T> nn;

    unsigned long long counter = 0; // N:th iteration
    
    whiteice::logging.info("RIFL_abstract2: starting optimization loop");

    {
      std::lock_guard<std::mutex> lock1(Q_mutex), lock2(policy_mutex);
      
      whiteice::logging.info("RIFL_abstract2: initial Q diagnostics");
      lagged_Q[0].diagnosticsInfo();
      
      whiteice::logging.info("RIFL_abstract2: initial policy diagnostics");
      lagged_policy.diagnosticsInfo();
    }

    // timing
    auto start = std::chrono::high_resolution_clock::now();

    
    while(thread_is_running > 0){

      
      if(learningMode == false){

	for(unsigned int i=0;i<NUM_Q_NNETWORKS;i++){
	  if(dataset_thread[i]){
	    delete dataset_thread[i];
	    dataset_thread[i] = nullptr;
	  }
	    
	  grad[i].stopComputation();
	  grad[i].reset();
	}

	if(dataset2_thread){
	  delete dataset2_thread;
	  dataset2_thread = nullptr;
	}
	
	grad2.stopComputation();
	grad2.reset();
      }

      if(LOOP_UPDATE_HZ > 0){
	// const float LOOP_UPDATE_HZ = 50.0f; // 50 Hz update frequency (1000/50 = 20ms update speed)
	
	auto elapsed = std::chrono::high_resolution_clock::now() - start;
	
	const long long microseconds_elapsed =
	  std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	
	const long long microseconds_sleep_time = 1000000.0/LOOP_UPDATE_HZ;
	
	if(microseconds_elapsed > microseconds_sleep_time){
	  char buf[256];

	  const float hz = (float)LOOP_UPDATE_HZ;

	  snprintf(buf, 256, "RIFL_abstract2::loop(): Warning sampling out of sync! (%f secs off) [update %f Hz]",
		   (microseconds_sleep_time - microseconds_elapsed)/1000000.0, hz);
	  
	  whiteice::logging.warn(buf);
	}
	else{
	  std::this_thread::sleep_for
	    (std::chrono::microseconds(microseconds_sleep_time - microseconds_elapsed));
	  
	  // sleep 1000000us/100 Hz = 10ms between updates => 100 Hz polling/sampling interval
	  // std::this_thread::sleep_for(std::chrono::milliseconds(1000/SAMPLING_HZ));
	}
	
	start = std::chrono::high_resolution_clock::now();
      }

      counter++;
      
      if(sleepMode == true){
	std::this_thread::sleep_for(std::chrono::milliseconds(50));
	continue; // we do not do anything and only sleep
      }


      // 1. gets current state
      if(performActionFailed == 0){
	auto oldstate = state;

	whiteice::math::vertex<T> s;
      
	if(getState(s) == false){
	  state = oldstate;
	  if(firstTime) continue;

	  whiteice::logging.error("ERROR: RIFL_abstact2::getState() FAILED.");
	}
	else{ // got new state
	  state_history.push_back(s);
	  state_history.erase(state_history.begin()); // state_history.pop_front();

	  unsigned int index = 0;

	  for(unsigned int j=0;j<state_history.size();j++){
	    for(unsigned int i=0;i<state_history[j].size();i++,index++){
	      state[index] = state_history[state_history.size()-1-j][i];
	    }
	  }
	}

	firstTime = false;
      }

      // 2. selects action using policy
      // (+ random selection if there is no model or in
      //    1-epsilon probability)
      
      
      if(performActionFailed == 0){
	std::lock_guard<std::mutex> lock(policy_mutex);

	whiteice::math::vertex<T> u;

	auto input = state;
	policy_preprocess.preprocess(0, input);

	if(lagged_policy.calculate(input, u, 1, 0) == true){
	  if(u.size() != numActions){
	    u.resize(numActions);
	    for(unsigned int i=0;i<numActions;i++){
	      u[i] = T(0.0);
	    }

	    random = true;
	  }
	  else{
	    policy_preprocess.invpreprocess(1, u);
	    random = false;
	  }
	}
	else{
	  u.resize(numActions);
	  for(unsigned int i=0;i<numActions;i++){
	    u[i] = T(0.0);
	  }

	  random = true;
	}

	// it is assumed that action data should have zero mean and is roughly
	// normally distributed (with StDev[n] = 1) so data is close to zero

	// FIXME add better random normally distributed noise (exploration)
	{
	  std::lock_guard<std::mutex> locke(epsilon_mutex);
	  
	  if(rng.uniform() < ((T(1.0f) - epsilon)/sequentialRandomMoves) && random_counter <= 0){ 
	    random_counter = sequentialRandomMoves;
	  }

	  bool no_model = false;

	  {
	    std::lock_guard<std::mutex> lockh(has_model_mutex);
	    
	    for(unsigned int i=0;i<hasModel.size();i++)
	      if(hasModel[i] == 0) no_model = true;
	  }

	  if(random_counter > 0 && no_model == false) // 1-epsilon % are chosen randomly [requires models]
#if 0
	  {
	    auto noise = u;
	    
	    rng.normal(noise); // Normal E[n]=0 StDev[n]=1
	    
	    u = u + T(0.666f)*noise; // was 0.1, 0.3, 0.6, *1.0

	    for(unsigned int j=0;j<u.size();j++){ // action is [-1,1]^D valued vector
	      if(u[j] < T(-1.0f)) u[j] = T(-1.0f);
	      else if(u[j] > T(1.0f)) u[j] = T(1.0f);
	    }

	    random = true;
	  }
#endif
#if 1
	  {
	    std::vector< whiteice::math::vertex<T> > actions;
	    std::vector< T > stdev, absw;

	    actions.resize(10); // was: 5 random samples from which to choose

	    for(unsigned int i=0;i<actions.size();i++){
	      auto noise = u;
	      
	      rng.normal(noise); // Normal E[n]=0 StDev[n]=1
	      
	      actions[i] = u + T(0.666f)*noise; // was 0.1, 0.3, 0.6, *1.0
	      
	      for(unsigned int j=0;j<actions[i].size();j++){ // action is [-1,1]^D valued vector
		if(actions[i][j] < T(-1.0f)) actions[i][j] = T(-1.0f);
		else if(actions[i][j] > T(1.0f)) actions[i][j] = T(1.0f);
	      }
	      
	      
	      T mean = T(0.0f);
	      T absmean = T(0.0f);
	      T var  = T(0.0f); 

	      for(unsigned int k=0;k<lagged_Q.size();k++){
		whiteice::math::vertex<T> y;
		y.resize(1);
		y.zero();

		whiteice::math::vertex<T> tmp(numStates + numActions);
		tmp.zero();

		tmp.write_subvertex(state, 0);
		tmp.write_subvertex(actions[i], numStates);

		data[k].preprocess(0, tmp);
		lagged_Q[k].calculate(tmp, y, 1, 0);
		data[k].invpreprocess(1, y);

		mean += y[0];
		absmean += whiteice::math::abs(y[0]);
		var  += y[0]*y[0];
	      }

	      mean /= lagged_Q.size();
	      absmean /= lagged_Q.size();
	      var  /= lagged_Q.size();
	      
	      var = whiteice::math::sqrt(whiteice::math::abs(var - mean*mean));

	      stdev.push_back(var); // this actions st.dev in Q with current state
	      absw.push_back(absmean);
	    }
	    
	    std::map<T, unsigned int> weights;
	    
	    T total_weights = T(0.0f);
	    T total_weights2 = T(0.0f);

	    for(unsigned int k=0;k<stdev.size();k++){
	      total_weights += stdev[k];
	      total_weights2 += absw[k];
	    }

	    if(total_weights <= T(0.0f))
	      total_weights = T(1.0f);

	    if(total_weights2 <= T(0.0f))
	      total_weights2 = T(1.0f);

	    T sump = T(0.0f);
	    const T mixing_factor = T(1.0f); // WAS: 60% stdev weight, 40% absolute value weight, now: 100% stdev
	    const T epsilon = T(1e-6);
	    
	    for(unsigned int k=0;k<stdev.size();k++){
	      std::pair<T, unsigned int> p;

	      sump += mixing_factor*((stdev[k]+epsilon)/(total_weights + T(stdev.size())*epsilon)) +
		(T(1.0f)-mixing_factor)*((absw[k]+epsilon)/(total_weights2 + T(absw.size())*epsilon));

	      p.first = sump;
	      p.second = k;

	      weights.insert(p);
	    }

	    // now we have weights, pick weighted random sample

	    const T r = rng.uniform();

	    auto iter = weights.upper_bound(r);

	    unsigned int index = 0;

	    if(iter != weights.end())
	      index = iter->second;

	    u = actions[index];
	    
	    random = true;
	  }
#endif

#if 0
	  {
	    auto noise = u, u2 = u;
	    rng.normal(noise);
	    u2 += T(0.01)*noise;

	    T mean1 = T(0.0f), mean2 = T(0.0f);
	    T var1  = T(0.0f), var2 = T(0.0f);

	    for(unsigned int k=0;k<lagged_Q.size();k++){
	      whiteice::math::vertex<T> y;
	      y.resize(1);
	      y.zero();
	      
	      whiteice::math::vertex<T> tmp(numStates + numActions);
	      
	      tmp.zero();	      
	      tmp.write_subvertex(state, 0);
	      tmp.write_subvertex(u, numStates);
	      
	      data[k].preprocess(0, tmp);
	      lagged_Q[k].calculate(tmp, y, 1, 0);
	      data[k].invpreprocess(1, y);
	      
	      mean1 += y[0];
	      var1  += y[0]*y[0];

	      tmp.zero();	      
	      tmp.write_subvertex(state, 0);
	      tmp.write_subvertex(u2, numStates);
	      
	      data[k].preprocess(0, tmp);
	      lagged_Q[k].calculate(tmp, y, 1, 0);
	      data[k].invpreprocess(1, y);
	      
	      mean2 += y[0];
	      var2  += y[0]*y[0];
	    }
	    
	    mean1 /= lagged_Q.size();
	    mean2 /= lagged_Q.size();
	    var1  /= lagged_Q.size();
	    var2  /= lagged_Q.size();
	    
	    var1 = whiteice::math::sqrt(whiteice::math::abs(var1 - mean1*mean1));
	    var2 = whiteice::math::sqrt(whiteice::math::abs(var2 - mean2*mean2));

	    const T epsilon = T(1e-5);

	    const T uncertainty = (var1+var2)/(mean1+epsilon);

	    const T sigma_min = T(0.025);
	    const T sigma_max = T(0.666);
	    
	    const T sigma = sigma_min + (sigma_max - sigma_min)*uncertainty;

	    noise = u;
	    rng.normal(noise); // Normal EX[n]=0 StDev[n]=1
	    u += sigma*noise; // was: sigma = 0.025

	    for(unsigned int i=0;i<u.size();i++){ // action is [-1,1]^D valued vector
	      if(u[i] < T(-1.0f)) u[i] = T(-1.0f);
	      else if(u[i] > T(1.0f)) u[i] = T(1.0f);
	    }

	    random = true;
	  }
#endif
	  else{ // just adds some random noise to action based on Q-value [mini-exploration]
#if 1
	    auto noise = u;
	    rng.normal(noise); // Normal EX[n]=0 StDev[n]=1
	    u += T(0.025)*noise; // was: sigma = 0.025
#endif
	    
#if 0
	    auto noise = u, u2 = u;
	    rng.normal(noise);
	    u2 += T(0.01)*noise;

	    T mean1 = T(0.0f), mean2 = T(0.0f);
	    T var1  = T(0.0f), var2 = T(0.0f);

	    for(unsigned int k=0;k<lagged_Q.size();k++){
	      whiteice::math::vertex<T> y;
	      y.resize(1);
	      y.zero();
	      
	      whiteice::math::vertex<T> tmp(numStates + numActions);
	      
	      tmp.zero();	      
	      tmp.write_subvertex(state, 0);
	      tmp.write_subvertex(u, numStates);
	      
	      data[k].preprocess(0, tmp);
	      lagged_Q[k].calculate(tmp, y, 1, 0);
	      data[k].invpreprocess(1, y);
	      
	      mean1 += y[0];
	      var1  += y[0]*y[0];

	      tmp.zero();	      
	      tmp.write_subvertex(state, 0);
	      tmp.write_subvertex(u2, numStates);
	      
	      data[k].preprocess(0, tmp);
	      lagged_Q[k].calculate(tmp, y, 1, 0);
	      data[k].invpreprocess(1, y);
	      
	      mean2 += y[0];
	      var2  += y[0]*y[0];
	    }
	    
	    mean1 /= lagged_Q.size();
	    mean2 /= lagged_Q.size();
	    var1  /= lagged_Q.size();
	    var2  /= lagged_Q.size();
	    
	    var1 = whiteice::math::sqrt(whiteice::math::abs(var1 - mean1*mean1));
	    var2 = whiteice::math::sqrt(whiteice::math::abs(var2 - mean2*mean2));

	    const T epsilon = T(1e-5);

	    const T uncertainty = (var1+var2)/(mean1+epsilon);

	    const T sigma_min = T(0.025);
	    const T sigma_max = T(0.200);
	    
	    const T sigma = sigma_min + (sigma_max - sigma_min)*uncertainty;

	    noise = u;
	    rng.normal(noise); // Normal EX[n]=0 StDev[n]=1
	    u += sigma*noise; // was: sigma = 0.025

	    for(unsigned int i=0;i<u.size();i++){ // action is [-1,1]^D valued vector
	      if(u[i] < T(-1.0f)) u[i] = T(-1.0f);
	      else if(u[i] > T(1.0f)) u[i] = T(1.0f);
	    }
#endif
	    
	  }
	}

	// if there's no model then make random selection (normally distributed)
	{
	  std::lock_guard<std::mutex> lockh(has_model_mutex);
	  
	  bool no_model = false;

	  for(unsigned int i=0;i<hasModel.size();i++)
	    if(hasModel[i] == 0) no_model = true;
	  
	  if(no_model){
	    //auto noise = u;		    
	    //rng.normal(noise); // Normal E[n]=0 StDev[n]=1
	    //u += T(1.00f)*noise; // was 0.1, 0.3*, 0.6

	    rng.uniform(u); // [0,1] valued actions!
	    
	    for(unsigned int i=0;i<u.size();i++)
	      u[i] = T(2.0f)*u[i] - T(1.0f); // [-1,+1] range

	    for(unsigned int i=0;i<u.size();i++){ // action is [-1,1]^D valued vector
	      if(u[i] < T(-1.0f)) u[i] = T(-1.0f);
	      else if(u[i] > T(1.0f)) u[i] = T(1.0f);
	    }
	    
	    // rng.uniform(u);
	    
	    random = true;
	  }
	}
	
	action = u;

	if(oneHotEncodedAction){
	  whiteice::math::vertex<T> new_action;
	  
	  const T temperature = T(0.10f);
	  
	  // maps probabilistic vector values to a single value
	  onehot_prob_select(action, new_action, temperature);
	  
	  action = new_action;
	}
      }

      
      
      //std::cout << "action = " << action << " ";
      //std::cout << "random = " << random << std::endl;

      whiteice::math::vertex<T> newstate, ns;
      T reinforcement = T(0.0);
      T distance = T(0.0f);

      // 3. perform action and get newstate and reinforcement
      {
	
	if(performAction(action, ns, reinforcement, distance, endFlag) == false){
	  //std::cout << "ERROR: RIFL_abstract2::performAction() FAILED." << std::endl;
	  whiteice::logging.error("ERROR: RIFL_abstact::performAction() FAILED.");
	  performActionFailed++;
	  goto optimization_step;
	}
	else{
	  performActionFailed = 0;

	  // construct newstate from ns and history_state

	  auto history = state_history;

	  history.push_back(ns);
	  history.erase(history.begin()); //  history.pop_front();

	  unsigned int index = 0;

	  newstate.resize(numStates);

	  for(unsigned int j=0;j<history.size();j++){
	    for(unsigned int i=0;i<history[j].size();i++,index++){
	      newstate[index] = history[history.size()-1-j][i];
	    }
	  }
	  

	  // did actual random action so reduce random_counter by one
	  random_counter--;
	  if(random_counter <= 0) random_counter = 0;
	}
	
      }

      // 4. update after effects_buffer
      if(AFTER_EFFECT_DELAY_MS == 0){ // after effect is disabled
	if(performActionFailed == 0){ // successful action
	  auto now = std::chrono::high_resolution_clock::now();	
	  const unsigned long long now_ms = 
	    std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

	  struct rifl2_datapoint<T> datum;
	  
	  datum.state = state;
	  datum.action = action;
	  datum.newstate = newstate;
	  datum.distance = distance;
	  datum.random = random;
	  datum.reinforcement_pure = reinforcement; // without after effects
	  datum.reinforcement = reinforcement; // with after effects
	  datum.lastStep = endFlag;

	  after_effects_buffer.insert(std::pair<unsigned long long, rifl2_datapoint<T> >(now_ms, datum));
	}
      }
      else{
	if(performActionFailed == 0){ // successful action
	  auto now = std::chrono::high_resolution_clock::now();	
	  const unsigned long long now_ms = 
	    std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() + AFTER_EFFECT_DELAY_MS;

	  struct rifl2_datapoint<T> datum;
	  
	  datum.state = state;
	  datum.action = action;
	  datum.newstate = newstate;
	  datum.distance = distance;
	  datum.random = random;
	  datum.reinforcement_pure = reinforcement; // without after effects
	  datum.reinforcement = T(0.0f); // with after effects [calculated later]
	  datum.lastStep = endFlag;

	  after_effects_buffer.insert(std::pair<unsigned long long, rifl2_datapoint<T> >(now_ms, datum));
	}
	
      }

      // 5. updates database (of actions and responses)
      {
	auto now = std::chrono::high_resolution_clock::now();	
	const unsigned long long now_ms = 
	  std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
	
	const auto last_iter = after_effects_buffer.upper_bound(now_ms);
	
	for(auto it=after_effects_buffer.begin();it != last_iter; it++){

	  struct rifl2_datapoint<T> datum = it->second;
	  whiteice::math::vertex<T> after_state;
	  
	  if(getState(after_state)){
	    // measure after effect
	    if(AFTER_EFFECT_DELAY_MS){
	      const T r = getReinforcement(datum.state, after_state);
	      const T w1 = T(0.50f)/T(1.50f);
	      const T w2 = T(1.00f)/T(1.50f);
	      
	      datum.reinforcement = w1*datum.reinforcement_pure + w2*r;
	    }
	    else{ // after effects is disabled
	      datum.reinforcement = datum.reinforcement_pure;
	    }
	      
	    
	  }
	  else{
	    datum.reinforcement = datum.reinforcement_pure;
	  }

	  
	  {
	    std::lock_guard<std::mutex> lockr(reinforcements_mutex);
	    
	    if(datum.random){
	      reinforcements_random.push_back(datum.reinforcement);
	      distances_random.push_back(datum.distance);
	    }
	    else{
	      reinforcements.push_back(datum.reinforcement);
	      distances.push_back(datum.distance);
	    }
	  }
	  
	  // for synchronizing access to database datastructure
	  // (also used by CreateRIFL2dataset class/thread)
	  std::lock_guard<std::mutex> lock(database_mutex);
	  
	  episode.push_back(datum);
	  
	  if(datum.lastStep){
	    
	    T total_reward = T(0.0f);
	    
	    for(const auto& e : episode)
	      total_reward += e.reinforcement_pure;
	    
	    total_reward /= T(episode.size());
	    
	    {
	      char buffer[80];
	      
	      std::lock_guard<std::mutex> lockh(has_model_mutex);
	      
	      snprintf(buffer, 80, "Episode %d avg reward: %f (%d moves) [%d %d models, policy %d]",
		       (int)episodes_counter, total_reward.c[0], (int)episode.size(),
		       hasModel[0], hasModel[1], hasModel[NUM_Q_NNETWORKS]);
	      
	      whiteice::logging.info(buffer);
	    }
	    
	    
	    fprintf(episodesFile, "%f\n", total_reward.c[0]);
	    fflush(episodesFile);
	    
	    latestError = (float)total_reward.c[0];

	    { 
	      if(episodes.size() >= EPISODES_MAX_SIZE){
		const unsigned long index = (episodes_counter % EPISODES_MAX_SIZE);
		episodes[index] = episode;
	      }
	      else{
		episodes.push_back(episode);
	      }
	      
	    }
	    
	    episode.clear();
	    episodes_counter++;
	  }

	  if(database_counter >= DATASIZE)
	    database_counter = database_counter % database.size();
	  
	  {
	    if(database.size() >= DATASIZE){
	      const unsigned int index = rng.rand() % database.size();
	      
	      database[index] = datum;
	    }
	    else{
	      database.push_back(datum);
	    }
	    
	  }

	  database_counter++;
	  
	}

	// removes processed values from after_effects_buffer

	after_effects_buffer.erase(after_effects_buffer.begin(), last_iter);
      }
      
      
    optimization_step:

      if(learningMode == false){

	for(unsigned int i=0;i<dataset_thread.size();i++){
	  if(dataset_thread[i]){
	    delete dataset_thread[i];
	    dataset_thread[i] = nullptr;
	  }

	  grad[i].stopComputation();
	  grad[i].reset();
	}

	if(dataset2_thread){
	  delete dataset2_thread;
	  dataset2_thread = nullptr;
	}

	grad2.stopComputation();
	grad2.reset();

	
	continue; // we do not do learning
      }


      /*
      printf("DATABASE SIZE: %d / %d. EPISODES SIZE: %d / %d\n",
	     (int)database.size(), (int)MINIMUM_DATASIZE,
	     (int)episodes.size(), (int)MINIMUM_EPISODE_SIZE);
      */
      
      // 6. update/optimize Q(state, action) network
      // activates batch learning if it is not running
      if(database.size() >= MINIMUM_DATASIZE)
      {
	unsigned int q_index = NUM_Q_NNETWORKS, max_q = epoch[0];

	for(unsigned int i=0;i<NUM_Q_NNETWORKS;i++){
	  if(epoch[i] > max_q) max_q = epoch[i];
	}

	  
	for(unsigned int i=0;i<NUM_Q_NNETWORKS;i++){
	  if(epoch[i] < max_q){
	    q_index = i;
	    break;
	  }
	}

	if(q_index == NUM_Q_NNETWORKS && epoch[NUM_Q_NNETWORKS] < max_q){
	  goto q_optimization_done;
	}
	else if(q_index == NUM_Q_NNETWORKS){
	  q_index = 0; // starts new round of optimizations from Q[0] nnetwork
	}

	
	T error;
	unsigned int iters;

	if(grad[q_index].isRunning() == false){
	  
	  if(grad[q_index].getSolutionStatistics(error, iters) == false){
	    // grad is reset()ed having no solution anymore (read once it) 
	  }
	  else{
	    // gradient have stopped running
	    
	    if(dataset_thread[q_index] == nullptr){
	      
	      char buffer[128];
	      double tmp = 0.0;
	      whiteice::math::convert(tmp, error);
	      snprintf(buffer, 128,
		       "RIFL_abstract2: new optimized Q-model (%f error, %d iters, epoch %d)",
		       tmp, iters, epoch[q_index]);
	      whiteice::logging.info(buffer);
	      
	      {
		logging.info("========> Q RESULT LOADING");
		
		if(grad[q_index].getSolution(nn) == false) assert(0);
		
		std::lock_guard<std::mutex> lock(Q_mutex);
		Q[q_index].importNetwork(nn);
		
		Q_preprocess = data[q_index];
		
		Q_preprocess.clearData(0);
		Q_preprocess.clearData(1);
#if 1
		whiteice::nnetwork<T> nn2;
		std::vector< math::vertex<T> > lagged_weights;
		std::vector< math::vertex<T> > lagged_bndata;
		
		if(lagged_Q[q_index].getBatchNorm()){
		  if(lagged_Q[q_index].exportSamples(nn2, lagged_weights, lagged_bndata, 1) == false)
		    assert(0);
		}
		else{
		  if(lagged_Q[q_index].exportSamples(nn2, lagged_weights, 1) == false)
		    assert(0);
		}

		if(lagged_weights.size() > 0){

		  math::vertex<T> weights;
		  math::vertex<T> bndata;
		  
		  if(nn.exportdata(weights) == false) assert(0);
		  if(nn.getBatchNorm()) if(nn.exportBNdata(bndata) == false) assert(0);

		  {
		    std::lock_guard<std::mutex> lockh(has_model_mutex);
		    
		    if(hasModel[q_index] == 0){
		      // don't lag results with the first update
		      lagged_weights[0] = weights;
		      if(nn.getBatchNorm()) lagged_bndata[0] = bndata;
		    }
		  }
		  
		  lagged_weights[0] = tau*weights + (T(1.0)-tau)*lagged_weights[0];
		  if(nn.getBatchNorm()) lagged_bndata[0]  = tau*bndata  + (T(1.0)-tau)*lagged_bndata[0];
		  
		  if(nn2.importdata(lagged_weights[0]) == false) assert(0);
		  if(nn2.getBatchNorm()) if(nn2.importBNdata(lagged_bndata[0]) == false) assert(0);
		  if(lagged_Q[q_index].importNetwork(nn2) == false) assert(0);
		}
		else{
		  logging.info("lagged_Q updated: NO LAG");
		  
		  lagged_Q[q_index].importNetwork(nn); 
		}
#endif
		
		whiteice::logging.info("RIFL_abstract2: new Q diagnostics");
		lagged_Q[q_index].diagnosticsInfo();
		whiteice::logging.info("RIFL_abstract2: new Q-model imported");
	      }
	      
	      grad[q_index].reset(); // resets gradient to empty gradient descent
	      
	      epoch[q_index]++;
	      
	      {
		std::lock_guard<std::mutex> lockh(has_model_mutex);
		hasModel[q_index]++;
	      }
	    }
	  }

	  
	  if(epoch[q_index] > epoch[q_index+1])
	    goto q_optimization_done;

	  
	  if(dataset_thread[q_index] == nullptr){
	    
	    {
	      std::lock_guard<std::mutex> lock(database_mutex);
	      std::lock_guard<std::mutex> lockh(has_model_mutex);
	      
	      data[q_index].clear();
	      //data.createCluster("input-state", numStates + numActions);
	      //data.createCluster("output-qvalue", 1);

	      if(dataset_thread[q_index]){
		dataset_thread[q_index]->stop();
		delete dataset_thread[q_index];
		dataset_thread[q_index] = nullptr;
	      }
	      
	      
	      dataset_thread[q_index] = new CreateRIFL2dataset<T>(*this,
								  database,
								  episodes,
								  database_mutex,
								  hasModel[q_index]);
	    }
	    
	    dataset_thread[q_index]->start(SAMPLESIZE, useEpisodes);
	    
	    whiteice::logging.info("RIFL_abstract2: new dataset_thread started (Q)");
	    
	    continue;
	    
	  }
	  else{
	    if(dataset_thread[q_index]->isCompleted() != true){
	      continue; // we havent computed proper dataset yet..
	    }
	    else{
	      data[q_index] = dataset_thread[q_index]->getDataset();
	    }
	  }
	  
	  if(dataset_thread[q_index]){
	    whiteice::logging.info("RIFL_abstract2: dataset_thread finished (Q)");
	    dataset_thread[q_index]->stop();
	    delete dataset_thread[q_index];
	    dataset_thread[q_index] = nullptr;
	  }


	  // fetch NN parameters from model
	  whiteice::nnetwork<T> qnn;
	  
	  {
	    std::vector< math::vertex<T> > weights;
	    std::vector< math::vertex<T> > bndatas;
	    
	    std::lock_guard<std::mutex> lock(Q_mutex);
	    
	    if(Q[q_index].getBatchNorm()){
	      if(Q[q_index].exportSamples(qnn, weights, bndatas, 1) == false){ // was: lagged_Q
		assert(0);
	      }
	    }
	    else{
	      if(Q[q_index].exportSamples(qnn, weights, 1) == false){ // was: lagged_Q
		assert(0);
	      }
	    }

	    if(weights.size() <= 0)
	      assert(0);

	    if(qnn.importdata(weights[0]) == false){
	      assert(0);
	    }

	    if(qnn.getBatchNorm()){
	      if(qnn.importBNdata(bndatas[0]) == false){
		assert(0);
	      }
	    }
	  }
	  
	  const bool dropout = false;
	  const bool useInitialNN = true; // WAS: start from scratch everytime
	  
	  grad[q_index].setRegularizer(T(0.0f)); // DISABLE REGULARIZER FOR Q-NETWORK (was: 0.001f)
	  grad[q_index].setNormalizeError(false); // calculate real error values	  
	  
	  {
	    std::lock_guard<std::mutex> lockh(has_model_mutex);
	    
	    if(hasModel[q_index] >= 1){
	      eta[q_index].start(0.0, Q_OPTIMIZE_ITERATIONS);
	      
	      grad[q_index].setUseMinibatch(false);
	      grad[q_index].setSGD(T(-1.0f)); // disable stochastic gradient descent
	      
	      if(grad[q_index].startOptimize(data[q_index], qnn, 1, Q_OPTIMIZE_ITERATIONS,
				    dropout, useInitialNN) == true)
		logging.info("========> Q OPTIMIZATION STARTED");
	      else
		logging.info("========> Q OPTIMIZATION STARTED FAILED");
	    }
	    else{
	      eta[q_index].start(0.0, Q_OPTIMIZE_ITERATIONS_FIRST);
	      
	      grad[q_index].setUseMinibatch(false);
	      grad[q_index].setSGD(T(-1.0f)); // disable stochastic gradient descent
	      
	      if(grad[q_index].startOptimize(data[q_index], qnn, 1, Q_OPTIMIZE_ITERATIONS_FIRST, dropout, useInitialNN) == true)
		logging.info("========> Q OPTIMIZATION STARTED");
	      else
		logging.info("========> Q OPTIMIZATION STARTED FAILED");
	    }
	  }
	  
	  grad_iterations[q_index] = -1;
	  // old_grad_iterations = -1;
	}
	else{
	  T error = T(0.0);
	  unsigned int iters = 0;

	  if(grad[q_index].getSolutionStatistics(error, iters)){
	    if(((signed int)iters) > grad_iterations[q_index]){
	      {
		std::lock_guard<std::mutex> lockh(has_model_mutex);
		
		char buffer[128];
		
		eta[q_index].update(iters);
		
		double e;
		whiteice::math::convert(e, error);
		
		snprintf(buffer, 128,
			 "RIFL_abstract2: Q-optimizer epoch %d iter %d error %.12f hasmodel %d [ETA %.2f mins]",
			 epoch[q_index], iters, e, hasModel[q_index], eta[q_index].estimate()/60.0);
		
		whiteice::logging.info(buffer);
	      }
		
	      grad_iterations[q_index] = (int)iters;
	    }
	  }
	  else{
	    char buffer[80];
	    snprintf(buffer, 80,
		     "RIFL_abstract2: epoch %d grad.getSolution() FAILED",
		     epoch[q_index]);
	    
	    whiteice::logging.error(buffer);
	  }
	}
      }
      
    q_optimization_done:
      
      
      // 6. update/optimize policy(state) network
      // activates batch learning if it is not running
      
      if(database.size() >= MINIMUM_DATASIZE)
      {
	
	// skip if other optimization step is behind us
	// we only start calculating policy after Q() has been optimized..
	//if(epoch[1] > epoch[0] || epoch[0] == 0)
	//  goto policy_optimization_done;

	bool any_q = false;

	for(unsigned int i=0;i<NUM_Q_NNETWORKS;i++)
	  if(epoch[i] == 0) any_q = true;

	unsigned int qs_not_ready = 0;

	for(unsigned int i=0;i<NUM_Q_NNETWORKS;i++){
	  if(epoch[i] <= epoch[NUM_Q_NNETWORKS])
	    qs_not_ready++;
	}

	if(any_q || qs_not_ready > 0)
	  goto policy_optimization_done;

	
	whiteice::nnetwork<T> nn;
	T meanq;
	unsigned int iters;

	
	if(grad2.isRunning() == false){


	  if(grad2.getSolutionStatistics(meanq, iters) == false){
	  }
	  else{
	    // gradient has stopped running
	    
	    if(dataset2_thread == nullptr){

	      {
		logging.info("========> POLICY RESULT LOADING");
		
		std::lock_guard<std::mutex> lock(policy_mutex);
		
		if(grad2.getSolution(nn) == false) assert(0);
		if(grad2.getDataset(this->policy_preprocess) == false) assert(0);
		
		char buffer[128];
		double tmp = 0.0;
		whiteice::math::convert(tmp, meanq);
		snprintf(buffer, 128,
			 "RIFL_abstract2: new optimized policy-model (%f mean-q, %d iters, epoch %d)",
			 tmp, iters, epoch[2]);
		whiteice::logging.info(buffer);
		
		
		policy.importNetwork(nn);

		policy_preprocess.clearData(0);
		policy_preprocess.clearData(1);
		
#if 1
		whiteice::nnetwork<T> nn2;
		std::vector< math::vertex<T> > lagged_weights;
		std::vector< math::vertex<T> > lagged_bndatas;
		
		if(lagged_policy.getBatchNorm()){
		  if(lagged_policy.exportSamples(nn2, lagged_weights, lagged_bndatas, 1) == false){
		    assert(0);
		  }
		}
		else{
		  if(lagged_policy.exportSamples(nn2, lagged_weights, 1) == false){
		    assert(0);
		  }
		}

		math::vertex<T> weights;
		math::vertex<T> bndata;
		
		nn.exportdata(weights);
		if(nn.getBatchNorm()) nn.exportBNdata(bndata);

		{
		  std::lock_guard<std::mutex> lockh(has_model_mutex);
		  
		  if(hasModel[NUM_Q_NNETWORKS] == 0){
		    // don't lag results with the first update
		    lagged_weights[0] = weights;
		    if(nn.getBatchNorm()) lagged_bndatas[0] = bndata;
		  }
		}
		  
		if(lagged_weights.size() > 0){

		  if(weights.size() == lagged_weights[0].size()){

		    logging.info("lagged_policy updated");
				 
		    lagged_weights[0] = tau_policy*weights + (T(1.0)-tau_policy)*lagged_weights[0];
		    if(nn.getBatchNorm()) lagged_bndatas[0] = tau_policy*bndata  + (T(1.0)-tau_policy)*lagged_bndatas[0];
		    
		    nn2.importdata(lagged_weights[0]);
		    if(nn2.getBatchNorm()) nn2.importBNdata(lagged_bndatas[0]);
		    
		    lagged_policy.importNetwork(nn2);
		  }
		  else{
		    logging.info("lagged_policy updated: NO LAG");
		    
		    nn2 = nn;
		    lagged_policy.importNetwork(nn2);
		  }
		}
		else{
		  logging.info("lagged_policy updated: NO LAG");
		  
		  nn2 = nn;
		  lagged_policy.importNetwork(nn2);
		}
#endif
		
		whiteice::logging.info("RIFL_abstract2: new policy diagnostics");
		lagged_policy.diagnosticsInfo();
		whiteice::logging.info("RIFL_abstract2: new policy-model imported");
	      }

	      grad2.reset();

	      epoch[NUM_Q_NNETWORKS]++;

	      {
		std::lock_guard<std::mutex> lockh(has_model_mutex);
		
		hasModel[NUM_Q_NNETWORKS]++;
	      }
	    }
	    
	  }

	  
	  bool any_q = false;
	  
	  for(unsigned int i=0;i<NUM_Q_NNETWORKS;i++)
	    if(epoch[i] == 0) any_q = true;
	  
	  if(any_q || epoch[NUM_Q_NNETWORKS-1] <= epoch[NUM_Q_NNETWORKS])
	    goto policy_optimization_done;
	
	  
	  // const unsigned int BATCHSIZE = database.size(); // was 1000
	  // const unsigned int BATCHSIZE = 1000; // was 128

	  if(dataset2_thread == nullptr){
	    data2.clear();
	    data2.createCluster("input-state", numStates);

	    dataset2_thread = new CreatePolicyDataset<T>(*this,
							 database,
							 database_mutex,
							 data2);
	    dataset2_thread->start(SAMPLESIZE);

	    whiteice::logging.info("RIFL_abstract2: new dataset2_thread started (policy)");

	    continue;
	  }
	  else{
	    if(dataset2_thread->isCompleted() == false)
	      continue; // we havent computed proper dataset yet..
	  }

	  if(dataset2_thread){
	    whiteice::logging.info("RIFL_abstract2: dataset2_thread finished (policy)");
	    dataset2_thread->stop();
	  }
	  
	  
	  // fetch NN parameters from model
	  {
	    whiteice::nnetwork<T> nn;
	    std::vector< whiteice::nnetwork<T> > q_nn;
	    std::vector< whiteice::dataset<T> > Q_preprocess_copy;

	    q_nn.resize(NUM_Q_NNETWORKS);
	    Q_preprocess_copy.resize(NUM_Q_NNETWORKS);

	    for(unsigned int k=0;k<q_nn.size();k++){
	      std::lock_guard<std::mutex> lock(Q_mutex);
	      std::vector< math::vertex<T> > weights;
	      std::vector< math::vertex<T> > bndatas;
	      
	      if(Q[k].getBatchNorm()){
		if(Q[k].exportSamples(q_nn[k], weights, bndatas, 1) == false){ // was: lagged_Q
		  assert(0);
		}
	      }
	      else{
		if(Q[k].exportSamples(q_nn[k], weights, 1) == false){ // was: lagged_Q
		  assert(0);
		}
	      }
	      
	      assert(weights.size() > 0);
	      
	      if(q_nn[k].importdata(weights[0]) == false){
		assert(0);
	      }

	      if(q_nn[k].getBatchNorm()){
		if(q_nn[k].importBNdata(bndatas[0]) == false){
		  assert(0);
		}
	      }

	      Q_preprocess_copy[k] = data[k];
	    }

	    {
	      std::vector< math::vertex<T> > weights;
	      std::vector< math::vertex<T> > bndatas;
	      
	      std::lock_guard<std::mutex> lock(policy_mutex);

	      if(policy.getBatchNorm()){
		if(policy.exportSamples(nn, weights, bndatas, 1) == false){ // was: lagged_policy
		  assert(0);
		}
	      }
	      else{
		if(policy.exportSamples(nn, weights, 1) == false){ // was: lagged_policy
		  assert(0);
		}
	      }
		      
	      
	      assert(weights.size() > 0);
	      
	      if(nn.importdata(weights[0]) == false){
		assert(0);
	      }

	      if(nn.getBatchNorm()){
		if(nn.importBNdata(bndatas[0]) == false){
		  assert(0);
		}
	      }
	    }

	    const bool dropout = false;
	    const bool useInitialNN = true; // WAS: start from scratch everytime
	    const bool alwaysUpdateSolution = false;
	    

	    {
	      std::lock_guard<std::mutex> lockh(has_model_mutex);
	      
	      if(hasModel[NUM_Q_NNETWORKS] >= 1){
		eta2.start(0.0, P_OPTIMIZE_ITERATIONS);
		
		grad2.setUseMinibatch(false);
		grad2.setSGD(T(-1.0)); // what is correct learning rate???
		
		if(grad2.startOptimize(&data2, q_nn, Q_preprocess_copy, nn, 1,
				       P_OPTIMIZE_ITERATIONS,
				       dropout, useInitialNN, alwaysUpdateSolution) == true){
		  logging.info("========> POLICY OPTIMIZATION STARTED (1)");
		}
		else{
		  logging.info("========> POLICY OPTIMIZATION START FAILED (1)");
		}
	      }
	      else{
		eta2.start(0.0, P_OPTIMIZE_ITERATIONS_FIRST);
		
		grad2.setUseMinibatch(false);
		grad2.setSGD(T(-1.0)); // what is correct learning rate???
		
		if(grad2.startOptimize(&data2, q_nn, Q_preprocess_copy, nn, 1,
				       P_OPTIMIZE_ITERATIONS_FIRST,
				       dropout, useInitialNN, alwaysUpdateSolution) == true){
		  logging.info("========> POLICY OPTIMIZATION STARTED (2)");
		}
		else{
		  logging.info("========> POLICY OPTIMIZATION START FAILED (2)");
		}
	      }
	    }


	    grad_iterations[NUM_Q_NNETWORKS] = -1;
	    // old_grad2_iterations = -1;
	    
	    if(dataset2_thread) delete dataset2_thread;
	    dataset2_thread = nullptr;
	  }
	  
	}
	else{
	  
	  if(grad2.getSolutionStatistics(meanq, iters)){
	    if(((signed int)iters) > grad_iterations[NUM_Q_NNETWORKS]){
	      char buffer[128];
	      
	      double v;
	      whiteice::math::convert(v, meanq);

	      eta2.update(iters);

	      {
		std::lock_guard<std::mutex> lockh(has_model_mutex);
		snprintf(buffer, 128,
			 "RIFL_abstract2: grad2 policy-optimizer epoch %d hasmodel %d iter %d mean q-value %.12f [ETA %.2f mins]",
			 epoch[NUM_Q_NNETWORKS], hasModel[NUM_Q_NNETWORKS], iters, v, eta2.estimate()/60.0);
	      }
	      
	      whiteice::logging.info(buffer);

	      grad_iterations[NUM_Q_NNETWORKS] = (int)iters;
	    }
	  }
	  else{
	    whiteice::logging.error("grad2.getSolutionStatistics() FAILED.");
	  }
	}
      }
      
    policy_optimization_done:
      
      (1 == 1); // dummy [work-around bug/feature goto requiring expression]
      
    }

    for(unsigned int i=0;i<NUM_Q_NNETWORKS;i++){
      if(dataset_thread[i]){
	delete dataset_thread[i];
	dataset_thread[i] = nullptr;
      }

      grad[i].stopComputation();
    }
    
    
    grad2.stopComputation();
    
    if(dataset2_thread){
      delete dataset2_thread;
      dataset2_thread = nullptr;
    }

    if(episodesFile) fclose(episodesFile);
    episodesFile = NULL;
    
  }

  template class RIFL_abstract2< math::blas_real<float> >;
  template class RIFL_abstract2< math::blas_real<double> >;
  
};
