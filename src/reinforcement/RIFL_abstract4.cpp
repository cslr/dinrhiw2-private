// *RECURRENT* Reinforcement learning using continuous state and continuous actions
//
// enabled batch norm for neural networks now, fix problems it cause until it works,
// sets mean and variance of layer outputs to (mean=0,stdev=1)
//
// "double Q" implementation is not enabled in the code because it seem to cause problems.
//
// now: activates layernorm which is not properly supported by nnetwork.load() or nnetwork.save() or bayesian_nnetwork etc.

#include "RIFL_abstract4.h"

#include "NNGradDescent.h"
#include "Policy4GradAscent.h"

#include "Log.h"
#include "linear_ETA.h"
#include "blade_math.h"

#include <assert.h>
#include <functional>
#include <list>


namespace whiteice
{

  template <typename T>
  RIFL_abstract4<T>::RIFL_abstract4(unsigned int numActions_,
				    unsigned int numStates_,
				    const bool alsoNegativeQValues,
				    const int sequentialRandomMoves_,
				    const unsigned int RECURRENT_DIMENSIONS_) :
    numActions(numActions_),
    numStates(numStates_),
    RECURRENT_DIMENSIONS(RECURRENT_DIMENSIONS_),
    negativeQ(alsoNegativeQValues)
  {
    // initializes parameters
    {
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
	
	hasModel.resize(3);
	hasModel[0] = 0; // Q-network
	hasModel[1] = 0; // Q2-network
	hasModel[2] = 0; // policy-network
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
	  //if(alsoNegativeQValues == false){
	  //  nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<T>::sigmoid); // ([0,+1])
	  //}
	  //else{
	  //  nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<T>::tanh); // ([-1,1])
	  //}
	  
	  nn.randomize(2, T(0.5)); // was 1.0
	  nn.setResidual(true);
	  nn.setBatchNorm(false); // NOW: Q-also has batch norm enabled
	  nn.setLayerNorm(false);
	  
	  Q.importNetwork(nn);
	  lagged_Q.importNetwork(nn);

	  nn.randomize(2, T(0.5)); // was 1.0
	  Q2.importNetwork(nn);
	  lagged_Q2.importNetwork(nn);

	  whiteice::logging.info("RIFL_abstract4: ctor Q diagnostics");
	  lagged_Q.diagnosticsInfo();

	  Q_preprocess.createCluster("input-state", numStates + numActions);
	  Q_preprocess.createCluster("output-state", 1); // q-value
	}
      }
      
      
      {
	std::lock_guard<std::mutex> lock(policy_mutex);

	// NOW: 10-layer small width neural network
	arch.clear();
	arch.push_back(numStates + RECURRENT_DIMENSIONS);
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
	arch.push_back(numActions + RECURRENT_DIMENSIONS);

	// policy outputs action is (should be) +[0,+1]^D vector
	{
	  whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::rectifier);
	  // whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::tanh);
	  // whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::tanh);
	  // whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::sigmoid);

	  // nn.setNonlinearity(0, whiteice::nnetwork<T>::pureLinear);
	  // nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<T>::sigmoid); // [0,+1] values as actions and recurrent
	  nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<T>::tanh); // [-1,+1] values as actions and recurrent
	  //nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<T>::pureLinear);
	  
	  nn.randomize(2, T(0.9)); // was 1.0
	  nn.setResidual(true);
	  nn.setBatchNorm(false);
	  nn.setLayerNorm(false);

	  policy.importNetwork(nn);
	  lagged_policy.importNetwork(nn);

	  whiteice::logging.info("RIFL_abstract4: ctor policy diagnostics");
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
  RIFL_abstract4<T>::RIFL_abstract4(unsigned int numActions_,
				    unsigned int numStates_,
				    const bool alsoNegativeQValues,
				    std::vector<unsigned int> Q_arch,
				    std::vector<unsigned int> policy_arch,
				    const int sequentialRandomMoves_,
				    const unsigned int RECURRENT_DIMENSIONS_) :
    numActions(numActions_),
    numStates(numStates_),
    RECURRENT_DIMENSIONS(RECURRENT_DIMENSIONS_)
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
	
	hasModel.resize(3);
	hasModel[0] = 0; // Q-network
	hasModel[1] = 0; // Q2-network
	hasModel[2] = 0; // policy-network
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

      policy_arch[0] = numStates + RECURRENT_DIMENSIONS;
      policy_arch[policy_arch.size()-1] = numActions + RECURRENT_DIMENSIONS;
      
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
	  // whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::sigmoid); // tanh, sigmoid, halfLinear
	  nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<T>::pureLinear);
	  //if(alsoNegativeQValues == false){
	  //  nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<T>::sigmoid); // [0,+1]
	  //}
	  //else{
	  //  nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<T>::tanh); // [-1,+1]
	  //}
	  
	  nn.randomize(2, T(0.5)); // was 1.0
	  nn.setResidual(true);
	  nn.setBatchNorm(false); // NOW: Q-also have batch norm enabled! (=> needs many iterations to stabilize)
	  nn.setLayerNorm(false);
	  
	  Q.importNetwork(nn);
	  lagged_Q.importNetwork(nn);

	  nn.randomize(2, T(0.5)); // was 1.0
	  Q2.importNetwork(nn);
	  lagged_Q2.importNetwork(nn);

	  whiteice::logging.info("RIFL_abstract4: ctor Q diagnostics");
	  lagged_Q.diagnosticsInfo();

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
	  // nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<T>::tanh);
	  // nn.setNonlinearity(0, whiteice::nnetwork<T>::pureLinear);
	  // nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<T>::sigmoid); // policy action is [0,1]-valued value
	  nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<T>::tanh); // policy action is [-1,1]-valued value
	  
	  nn.randomize(2, T(0.9)); // was 1.0
	  nn.setResidual(true);
	  nn.setBatchNorm(false);
	  nn.setLayerNorm(false);
	  
	  policy.importNetwork(nn);
	  lagged_policy.importNetwork(nn);

	  whiteice::logging.info("RIFL_abstract4: ctor policy diagnostics");
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
  RIFL_abstract4<T>::~RIFL_abstract4() 
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
  bool RIFL_abstract4<T>::start()
  {
    if(thread_is_running != 0) return false;

    std::lock_guard<std::mutex> lock(thread_mutex);

    if(thread_is_running != 0) return false;

    try{
      whiteice::logging.info("RIFL_abstract4: starting main thread");

      thread_is_running++;
      rifl_thread = new std::thread(std::bind(&RIFL_abstract4<T>::loop, this));
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
  bool RIFL_abstract4<T>::stop()
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
  bool RIFL_abstract4<T>::isRunning() const
  {
    std::lock_guard<std::mutex> lock(thread_mutex);
    
    return (thread_is_running > 0);
  }

  template <typename T>
  bool RIFL_abstract4<T>::setGamma(T gamma)
  {
    if(gamma <= T(0.0) || gamma >= T(1.0))
      return false;
    
    this->gamma = gamma;
    
    return true;
  }

  template <typename T>
  T RIFL_abstract4<T>::getGamma() const
  {
    return gamma;
  }

  // epsilon E [0,1] percentage of actions are chosen according to model
  //                 1-e percentage of actions are random (exploration)
  template <typename T>
  bool RIFL_abstract4<T>::setEpsilon(T epsilon) 
  {
    std::lock_guard<std::mutex> locke(epsilon_mutex);
    
    if(epsilon < T(0.0) || epsilon > T(1.0)) return false;
    this->epsilon = epsilon;
    
    return true;
  }
  

  template <typename T>
  T RIFL_abstract4<T>::getEpsilon() const 
  {
    std::lock_guard<std::mutex> locke(epsilon_mutex);
    return epsilon;
  }


  template <typename T>
  void RIFL_abstract4<T>::setLearningMode(bool learn) 
  {
    learningMode = learn;
  }

  template <typename T>
  bool RIFL_abstract4<T>::getLearningMode() const 
  {
    return learningMode;
  }

  template <typename T>
  void RIFL_abstract4<T>::setSleepingMode(bool sleep) 
  {
    sleepMode = sleep;
  }

  template <typename T>
  bool RIFL_abstract4<T>::getSleepingMode() const 
  {
    return sleepMode;
  }


  template <typename T>
  void RIFL_abstract4<T>::setHasModel(unsigned int hasModel) 
  {
    std::lock_guard<std::mutex> lockh(has_model_mutex);
    
    this->hasModel[0] = hasModel;
    this->hasModel[1] = hasModel;
    this->hasModel[2] = hasModel;
  }

  template <typename T>
  unsigned int RIFL_abstract4<T>::getHasModel() 
  {
    std::lock_guard<std::mutex> lockh(has_model_mutex);
    
    unsigned int min = hasModel[0];

    if(min > hasModel[1]) min = hasModel[1];
    if(min > hasModel[2]) min = hasModel[2];

    return min;
  }


  template <typename T>
  float RIFL_abstract4<T>::getLatestEpisodeError() const
  {
    return latestError;
  }

  template <typename T>
  unsigned int RIFL_abstract4<T>::getDatabaseSize() const
  {
    std::lock_guard<std::mutex> lock(database_mutex);
    
    return database.size();
  }


  // how many percent smaller is reinforcement value with random actions vs policy actions
  template <typename T>
  bool RIFL_abstract4<T>::clearStatistics()
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
  bool RIFL_abstract4<T>::executionStatistics(T& percent_change,
					      T& distances_percent_change,
					      const bool rescale_to_min_value,
					      const bool use_only_most_recent,
					      unsigned int history_size) const
  {
    if(history_size <= 0) history_size = 1000;

    std::lock_guard<std::mutex> lock(reinforcements_mutex);
    
    percent_change = T(0.0f);
    
    if(reinforcements.size() <= 10 || reinforcements_random.size() <= 10)
      return false;

    if(distances.size() <= 10 || distances_random.size() <= 10)
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
    }

    // distances
    {
      T mean = T(0.0), stdev = T(0.0);
      T mean_random = T(0.0), stdev_random = T(0.0);
      
      if(use_only_most_recent == false){
	for(const auto& r : distances){
	  mean += r;
	  stdev += r*r;
	}
	
	mean /= distances.size();
	stdev /= distances.size();
	
	stdev -= mean*mean;
	if(stdev < T(0.0))
	  stdev = T(0.0);
	
	stdev = sqrt(stdev/distances.size()); // mean's stdev
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
	
	for(int i=start;i<end;i++){
	  const auto& r = distances[i];
	  
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
  bool RIFL_abstract4<T>::save(const std::string& filename) const
  {
    {
      std::lock_guard<std::mutex> lock(database_mutex);

      if(database.size() == 0 || episodes.size() == 0)
	return false;
    }

    
    char buffer[256];
    
    {
      std::lock_guard<std::mutex> lock1(Q_mutex);
      std::lock_guard<std::mutex> lock2(policy_mutex);
      
      snprintf(buffer, 256, "%s-q", filename.c_str());    
      if(Q.save(buffer) == false){
	logging.error("RIFL_abstract4::save() saving Q failed");
	return false;
      }

      snprintf(buffer, 256, "%s-q2", filename.c_str());    
      if(Q2.save(buffer) == false){
	logging.error("RIFL_abstract4::save() saving Q2 failed");
	return false;
      }
      
      snprintf(buffer, 256, "%s-policy", filename.c_str());
      if(policy.save(buffer) == false){
	logging.error("RIFL_abstract4::save() saving policy failed");
	return false;
      }
      
      snprintf(buffer, 256, "%s-lagged-q", filename.c_str());    
      if(lagged_Q.save(buffer) == false){
	logging.error("RIFL_abstract4::save() saving lagged-q failed");
	return false;
      }

      snprintf(buffer, 256, "%s-lagged-q2", filename.c_str());    
      if(lagged_Q2.save(buffer) == false){
	logging.error("RIFL_abstract4::save() saving lagged-q2 failed");
	return false;
      }
      
      snprintf(buffer, 256, "%s-lagged-policy", filename.c_str());
      if(lagged_policy.save(buffer) == false){
	logging.error("RIFL_abstract4::save() saving lagged-policy failed");
	return false;
      }
      
      snprintf(buffer, 256, "%s-q-preprocess", filename.c_str());    
      if(Q_preprocess.save(buffer) == false){
	logging.error("RIFL_abstract4::save() saving q-preprocess failed");
	return false;
      }
      
      snprintf(buffer, 256, "%s-policy-preprocess", filename.c_str());
      if(policy_preprocess.save(buffer) == false){
	logging.error("RIFL_abstract4::save() saving policy-preprocess failed");
	return false;
      }
    }

    {
      snprintf(buffer, 256, "%s-hasmodel", filename.c_str());

      whiteice::dataset<T> db;

      db.createCluster("has_model", 3);

      whiteice::math::vertex<T> v;
      v.resize(3);
      v.zero();

      std::lock_guard<std::mutex> lockh(has_model_mutex);
      
      if(hasModel.size() == 3){
	v[0] = T(hasModel[0]);
	v[1] = T(hasModel[1]);
	v[2] = T(hasModel[2]);
      }

      if(db.add(0, v) == false){
	logging.error("RIFL_abstract4::save(): saving hasModel data failed.");
	return false;
      }

      if(db.save(buffer) == false){
	logging.error("RIFL_abstract4::save(): saving hasModel dataset file failed.");
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
	logging.error("RIFL_abstract4::save(): saving measurements data failed (1).");
	return false;
      }

      v.resize(reinforcements_random.size());
      v.zero();

      for(unsigned int i=0;i<reinforcements_random.size();i++)
	v[i] = reinforcements_random[i];

      if(db.add(1, v) == false){
	logging.error("RIFL_abstract4::save(): saving measurements data failed (2).");
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
	logging.error("RIFL_abstract4::save(): saving measurements data failed.");
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
      	db.createCluster("recurrent", database[0].recurrent.size());
      else
	db.createCluster("recurrent", 1);

      if(database.size() > 0)
	db.createCluster("recurrent_new", database[0].recurrent_new.size());
      else
	db.createCluster("recurrent_new", 1);

      if(database.size() > 0)
	db.createCluster("action", database[0].action.size());
      else
	db.createCluster("action", 1);

      if(database.size() > 0)
	db.createCluster("random", 1);
      else
	db.createCluster("random", 1);

      if(database.size() > 0)
	db.createCluster("reinforcement", 1);
      else
	db.createCluster("reinforcement", 1);

      if(database.size() > 0)
	db.createCluster("last_step", 1);
      else
	db.createCluster("last_step", 1);

      for(unsigned int i=0;i<database.size();i++){
	db.add(0, database[i].state);
	db.add(1, database[i].newstate);
	db.add(2, database[i].recurrent);
	db.add(3, database[i].recurrent_new);
	db.add(4, database[i].action);

	whiteice::math::vertex<T> v;
	v.resize(1);

	if(database[i].random)
	  v[0] = T(1.0f);
	else
	  v[0] = T(0.0f);

	db.add(5,v);
	
	v[0] = database[i].reinforcement;

	db.add(6, v);

	if(database[i].lastStep)
	  v[0] = T(1.0f);
	else
	  v[0] = T(0.0f);

	db.add(7, v);
      }

      if(db.save(buffer) == false){
	logging.error("RIFL_abstract4::save() saving database failed");
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
      	db.createCluster("recurrent", database[0].recurrent.size());
      else
	db.createCluster("recurrent", 1);

      if(database.size() > 0)
	db.createCluster("recurrent_new", database[0].recurrent_new.size());
      else
	db.createCluster("recurrent_new", 1);

      if(database.size() > 0)
	db.createCluster("action", database[0].action.size());
      else
	db.createCluster("action", 1);

      if(database.size() > 0)
	db.createCluster("random", 1);
      else
	db.createCluster("random", 1);

      if(database.size() > 0)
	db.createCluster("reinforcement", 1);
      else
	db.createCluster("reinforcement", 1);

      if(database.size() > 0)
	db.createCluster("last_step", 1);
      else
	db.createCluster("last_step", 1);

      db.createCluster("episodes-range", 2);

      db.createCluster("episodes-weights", 1);
      

      for(unsigned int e=0;e<episodes.size() && e<episodes_weights.size();e++){

	const unsigned int start = db.size(0);

	
	for(unsigned int i=0;i<episodes[e].size();i++){
	  db.add(0, episodes[e][i].state);
	  db.add(1, episodes[e][i].newstate);
	  db.add(2, episodes[e][i].recurrent);
	  db.add(3, episodes[e][i].recurrent_new);
	  db.add(4, episodes[e][i].action);
	  
	  whiteice::math::vertex<T> v;
	  v.resize(1);
	  
	  if(episodes[e][i].random)
	    v[0] = T(1.0f);
	  else
	    v[0] = T(0.0f);
	  
	  db.add(5,v);
	  
	  v[0] = episodes[e][i].reinforcement;
	  
	  db.add(6, v);
	  
	  if(episodes[e][i].lastStep)
	    v[0] = T(1.0f);
	  else
	    v[0] = T(0.0f);
	  
	  db.add(7, v);
	}

	const unsigned int end = db.size(0);

	whiteice::math::vertex<T> v;
	v.resize(2);

	v[0] = T(start);
	v[1] = T(end);

	db.add(8, v);

	
	v.resize(1);
	
	v[0] = episodes_weights[e];

	db.add(9, v);
      }


      if(db.save(buffer) == false){
	logging.error("RIFL_abstract4::save() saving episodes failed");
	return false;
      }
    }


    {
      logging.info("RIFL_abstract4::save() successfully saved data.");

      {
	std::lock_guard<std::mutex> lock2(policy_mutex);
	logging.info("RIFL_abstract4::save(): saved lagged_policy");
	lagged_policy.diagnosticsInfo();
      }

      {
	std::lock_guard<std::mutex> lock2(Q_mutex);
	logging.info("RIFL_abstract4::save(): saved lagged_Q");
	lagged_Q.diagnosticsInfo();
      }
    }
    
    return true;
  }

  
  // loads learnt Reinforcement Learning Model from file
  template <typename T>
  bool RIFL_abstract4<T>::load(const std::string& filename)
  {
    char buffer[256];

    Q_mutex.lock();
    policy_mutex.lock();
    has_model_mutex.lock();
    database_mutex.lock();
    reinforcements_mutex.lock();

    auto Q_load = Q;
    auto Q2_load = Q2;
    auto policy_load = policy;
    auto lagged_Q_load = lagged_Q;
    auto lagged_Q2_load = lagged_Q2;
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
    auto episodes_weights_load = episodes_weights;

    reinforcements_mutex.unlock();
    database_mutex.unlock();
    has_model_mutex.unlock();
    policy_mutex.unlock();
    Q_mutex.unlock();
    
    
    {
      snprintf(buffer, 256, "%s-q", filename.c_str());    
      if(Q_load.load(buffer) == false){
	logging.error("RIFL_abstract4::load() loading Q failed");
	return false;
      }

      snprintf(buffer, 256, "%s-q2", filename.c_str());    
      if(Q2_load.load(buffer) == false){
	logging.error("RIFL_abstract4::load() loading Q2 failed");
	return false;
      }
      
      snprintf(buffer, 256, "%s-policy", filename.c_str());
      if(policy_load.load(buffer) == false){
	logging.error("RIFL_abstract4::load() loading policy failed");
	return false;
      }
      
      snprintf(buffer, 256, "%s-lagged-q", filename.c_str());    
      if(lagged_Q_load.load(buffer) == false){
	logging.error("RIFL_abstract4::load() loading lagged-q failed");
	return false;
      }

      snprintf(buffer, 256, "%s-lagged-q2", filename.c_str());    
      if(lagged_Q2_load.load(buffer) == false){
	logging.error("RIFL_abstract4::load() loading lagged-q2 failed");
	return false;
      }
      
      snprintf(buffer, 256, "%s-lagged-policy", filename.c_str());
      if(lagged_policy_load.load(buffer) == false){
	logging.error("RIFL_abstract4::load() loading lagged-policy failed");
	return false;
      }
      
      snprintf(buffer, 256, "%s-q-preprocess", filename.c_str());    
      if(Q_preprocess_load.load(buffer) == false){
	logging.error("RIFL_abstract4::load() loading q_preprocess failed");
	return false;
      }
      
      snprintf(buffer, 256, "%s-policy-preprocess", filename.c_str());
      if(policy_preprocess_load.load(buffer) == false){
	logging.error("RIFL_abstract4::load() loading policy_preprocess failed");
	return false;
      }
    }

    {
      snprintf(buffer, 256, "%s-hasmodel", filename.c_str());
      
      whiteice::dataset<T> db;

      if(db.load(buffer) == false){
	logging.error("RIFL_abstract4::load() loading hasModel dataset file failed");
	return false;
      }

      if(db.size(0) != 1 && db.dimension(0) != 3){
	logging.error("RIFL_abstract4::load() loading hasModel dataset file failed (2)");
	return false;
      }

      whiteice::math::vertex<T> v;
      v.resize(3);
      v.zero();

      v = db.access(0,0);

      if(v.size() == 3){
	hasModel_load.resize(3);
	hasModel_load[0] = (int)v[0].c[0];
	hasModel_load[1] = (int)v[1].c[0];
	hasModel_load[2] = (int)v[2].c[0];
      }
      else{
	logging.error("RIFL_abstract4::load() loading hasModel dataset file failed (3)");
	return false;
      }
    }

    {
      std::lock_guard<std::mutex> lock(reinforcements_mutex);
      
      snprintf(buffer, 256, "%s-measurements", filename.c_str());

      whiteice::dataset<T> db;

      if(db.load(buffer) == false){
	logging.error("RIFL_abstract4::load(): loading measurements data failed.");
	return false;
      }

      if(db.getNumberOfClusters() != 4){
	logging.error("RIFL_abstract4::load(): loading measurements data failed (2).");
	return false;
      }

      if(db.size(0) != 1 || db.size(1) != 1  || db.size(2) != 1 || db.size(3) != 1){
	logging.error("RIFL_abstract4::load(): loading measurements data failed (3).");
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
      v = db.access(1,0); 
      distances_load.resize(v.size());

      for(unsigned int i=0;i<distances_load.size();i++)
	distances_load[i] = v[i];

      v.resize(db.dimension(3));
      v = db.access(1,0); 
      distances_random_load.resize(v.size());

      for(unsigned int i=0;i<distances_random_load.size();i++)
	distances_random_load[i] = v[i];

    }

    {
      snprintf(buffer, 256, "%s-database", filename.c_str());
      
      whiteice::dataset<T> db;

      if(db.load(buffer) == false){
	char buf[1024];
	snprintf(buf, 1024, "RIFL_abstract4::load(\"%s\") loading dataset FAILED", buffer);
	logging.error(buf);
	return false;
      }

      if(db.getNumberOfClusters() != 8){
	logging.error("RIFL_abstract4::load() database wrong number of clusters");
	return false;
      }

      if(db.dimension(0) != db.dimension(1) ||
	 db.dimension(2) != db.dimension(3) || 
	 db.dimension(5) != 1 || db.dimension(6) != 1 || db.dimension(7) != 1){
	char buf[128];
	snprintf(buf, 128, "RIFL_abstract4::load() database wrong dimensions %d %d %d %d %d %d %d %d",
		 db.dimension(0), db.dimension(1), db.dimension(3), db.dimension(3),
		 db.dimension(4), db.dimension(5), db.dimension(6), db.dimension(7));
	logging.error(buf);
	return false;
      }

      if(db.dimension(0) != this->numStates){
	char buf[128];
	snprintf(buf, 128, "RIFL_abstract4::load() database wrong dimensions %d %d (2)",
		 db.dimension(0), this->numStates);
	logging.error(buf);
	return false;
      }
      
      if(db.dimension(4) != this->numActions){
	char buf[128];
	snprintf(buf, 128, "RIFL_abstract4::load() database wrong dimensions %d %d (3)",
		 db.dimension(2), this->numActions);
	logging.error(buf);

	return false;
      }

      if(db.size(0) != db.size(1) || db.size(1) != db.size(2) || db.size(2) != db.size(3) ||
	 db.size(3) != db.size(4) || db.size(4) != db.size(5) || db.size(5) != db.size(6) ||
	 db.size(6) != db.size(7)){

	char buf[128];
	snprintf(buf, 128, "RIFL_abstract4::load() database wrong size %d %d %d %d %d %d %d %d",
		 db.size(0), db.size(1), db.size(2), db.size(3), db.size(4), db.size(5), db.size(6), db.size(7)); 
	logging.error(buf);
	
	return false;
      }
					     
      
      
      database_load.clear();
      
      whiteice::rifl4_datapoint<T> p;
      whiteice::math::vertex<T> v;

      for(unsigned int i=0;i<db.size(0);i++){
	p.state = db.access(0, i);
	p.newstate = db.access(1, i);
	p.recurrent = db.access(2, i);
	p.recurrent_new = db.access(3, i);
	p.action = db.access(4, i);

	v = db.access(5, i);
	if(v[0] > T(0.5f)) p.random = true;
	else p.random = false;
	
	v = db.access(6, i);
	p.reinforcement = v[0];
	
	v = db.access(7, i);
	if(v[0] > T(0.5)) p.lastStep = true;
	else p.lastStep = false;
	
	database_load.push_back(p);
      }
      
    }

    {
      snprintf(buffer, 256, "%s-episodes", filename.c_str());
      
      whiteice::dataset<T> db;

      if(db.load(buffer) == false){
	char buf[1024];
	snprintf(buf, 1024, "RIFL_abstract4::load(\"%s\") loading episodes dataset FAILED", buffer);
	logging.error(buf);
	return false;
      }

      if(db.getNumberOfClusters() != 10){
	logging.error("RIFL_abstract4::load() episodes database wrong number of clusters");
	return false;
      }

      if(db.dimension(0) != db.dimension(1) ||
	 db.dimension(2) != db.dimension(3) || 
	 db.dimension(5) != 1 || db.dimension(6) != 1 || db.dimension(7) != 1){
	char buf[128];
	snprintf(buf, 128, "RIFL_abstract4::load() database wrong dimensions %d %d %d %d %d %d %d %d",
		 db.dimension(0), db.dimension(1), db.dimension(3), db.dimension(3),
		 db.dimension(4), db.dimension(5), db.dimension(6), db.dimension(7));
	logging.error(buf);
	return false;
      }

      if(db.dimension(0) != this->numStates){
	char buf[128];
	snprintf(buf, 128, "RIFL_abstract4::load() database wrong dimensions %d %d (2)",
		 db.dimension(0), this->numStates);
	logging.error(buf);
	return false;
      }
      
      if(db.dimension(4) != this->numActions){
	char buf[128];
	snprintf(buf, 128, "RIFL_abstract4::load() database wrong dimensions %d %d (3)",
		 db.dimension(2), this->numActions);
	logging.error(buf);

	return false;
      }

      if(db.dimension(8) != 2){
	char buf[128];
	snprintf(buf, 128, "RIFL_abstract4::load() database wrong dimensions %d %d (4)",
		 db.dimension(8), 2);
	logging.error(buf);

	return false;	
      }

      if(db.dimension(9) != 1){
	char buf[128];
	snprintf(buf, 128, "RIFL_abstract4::load() database wrong dimensions %d %d (5)",
		 db.dimension(9), 1);
	logging.error(buf);

	return false; 	
      }

      if(db.size(0) != db.size(1) || db.size(1) != db.size(2) || db.size(2) != db.size(3) ||
	 db.size(3) != db.size(4) || db.size(4) != db.size(5) || db.size(5) != db.size(6) ||
	 db.size(6) != db.size(7) || db.size(8) != db.size(9)){

	char buf[128];
	snprintf(buf, 128, "RIFL_abstract4::load() database wrong size %d %d %d %d %d %d %d %d %d %d",
		 db.size(0), db.size(1), db.size(2), db.size(3), db.size(4), db.size(5), db.size(6), db.size(7),
		 db.size(8), db.size(9));
	logging.error(buf);
	
	return false;
      }
					     
      
      
      episodes_load.clear();
      episodes_weights_load.clear();

      for(unsigned int e=0;e<db.size(8);e++){

	std::vector< whiteice::rifl4_datapoint<T> > epi;
	
	whiteice::rifl4_datapoint<T> p;
	whiteice::math::vertex<T> v;

	v = db.access(8, e);

	unsigned int START = 0;
	unsigned int END = 0;

	whiteice::math::convert(START, v[0]);
	whiteice::math::convert(END, v[1]);

	assert(START < db.size(0));
	assert(END <= db.size(0));
	assert(START <= END);

	T weight = T(0.0f);

	v = db.access(9, e);
	
	weight = v[0];

	for(unsigned int i=START;i<END;i++){
	  p.state = db.access(0, i);
	  p.newstate = db.access(1, i);
	  p.recurrent = db.access(2, i);
	  p.recurrent_new = db.access(3, i);
	  p.action = db.access(4, i);
	  
	  v = db.access(5, i);
	  if(v[0] > T(0.5f)) p.random = true;
	  else p.random = false;
	  
	  v = db.access(6, i);
	  p.reinforcement = v[0];
	  
	  v = db.access(7, i);
	  if(v[0] > T(0.5)) p.lastStep = true;
	  else p.lastStep = false;
	  
	  epi.push_back(p);
	}

	episodes_load.push_back(epi);
	episodes_weights_load.push_back(weight);
      }
      
    }
    
    
    {
      std::lock_guard<std::mutex> lock1(Q_mutex);
      std::lock_guard<std::mutex> lock2(policy_mutex);
      std::lock_guard<std::mutex> lockh(has_model_mutex);
      std::lock_guard<std::mutex> lockd(database_mutex);
      std::lock_guard<std::mutex> lockr(reinforcements_mutex);
      
      Q = Q_load;
      Q2 = Q2_load;
      policy = policy_load;
      lagged_Q = lagged_Q_load;
      lagged_Q2 = lagged_Q2_load;
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
      episodes_weights = episodes_weights_load;
    }

    {
      logging.info("RIFL_abstract4::load() successfully loaded data.");

      {
	std::lock_guard<std::mutex> lock2(policy_mutex);
	logging.info("RIFL_abstract4::load(): loaded lagged_policy: ");
	lagged_policy.diagnosticsInfo();
      }

      {
	std::lock_guard<std::mutex> lock2(Q_mutex);
	logging.info("RIFL_abstract4::load(): loaded lagged_Q");
	lagged_Q.diagnosticsInfo();
      }
    }
    
    return true;
  }


  template <typename T>
  void RIFL_abstract4<T>::onehot_prob_select(const whiteice::math::vertex<T>& action,
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
  void RIFL_abstract4<T>::loop()
  {
    // number of iteratios to use per epoch for optimization
    const unsigned int Q_OPTIMIZE_ITERATIONS_FIRST = 20; // WAS: 1000, 5, 1000, 50
    const unsigned int P_OPTIMIZE_ITERATIONS_FIRST = 20; // WAS: 100, 5, 50, 50

    const unsigned int Q_OPTIMIZE_ITERATIONS = 20; // 3; // WAS: 1000, 5, 1000
    const unsigned int P_OPTIMIZE_ITERATIONS = 20; // 3; // WAS: 50, 5, 50
    
    // tau = 1.0 => no lagged neural networks [don't work]
    // const T tau = T(0.001); // lagged Q and policy network [keeps tau%=1% of the new weights [was: 0.001, 0.05, 1.0*]
    // const T tau_policy = T(0.005); // was: 1.0*
    
    std::vector< rifl4_datapoint<T> > episode;

    FILE* episodesFile = fopen("episodes-result.txt", "w");    

    bool endFlag = false; // did the simulation end during this time step?
    
    whiteice::dataset<T> data;
    whiteice::CreateRIFL4dataset<T>* dataset_thread = nullptr;
    whiteice::math::NNGradDescent<T> grad; // Q(state,action) model optimizer
    
    // deep pretraining using stacked RBMs
    // (requires sigmoidal nnetwork and training
    //  policy nnetwork (calculating gradients) dont work with sigmoid)
    const bool deep = false;
    whiteice::dataset<T> data2;
    whiteice::CreatePolicy4Dataset<T>* dataset2_thread = nullptr;
    whiteice::CreateRIFL4dataset<T>* dataset_q2_thread = nullptr;
    whiteice::Policy4GradAscent<T> grad2(*this, deep);   // policy(state)=action model optimizer
    whiteice::math::NNGradDescent<T> q2grad; // Q2(state, action) model optimizer

    whiteice::linear_ETA<double> eta, eta2; // estimates how long single epoch of optimization takes
    
    std::vector<unsigned int> epoch;

    epoch.resize(3);
    epoch[0] = 0;
    epoch[1] = 0;
    epoch[2] = 0;

    int old_grad_iterations = -1;
    int old_grad_q2_iterations = -1;
    int old_grad2_iterations = -1;

    const unsigned long DATASIZE = 1000000; // 1M history of samples
    // assumes each episode length is 100 so this is ~ equal to 1.000.000 samples
    const unsigned long EPISODES_MAX_SIZE = 100000; // 100.000 episodes
    const unsigned long MINIMUM_EPISODE_SIZE = 15; // was: 25
    const unsigned long MINIMUM_DATASIZE = 500; // samples required to start learning, was:500
    // const unsigned long SAMPLESIZE = 4000; // number of samples used in learning, was: 2000, 3500, 5000=(10bins^3variables*5samples = 5000)
    unsigned long database_counter = 0;
    unsigned long episodes_counter = 0;

    int random_counter = 0; // how many times to do random action

    latestError = 0.0f;
    
    bool firstTime = true;
    whiteice::math::vertex<T> state;
    whiteice::math::vertex<T> action(numActions);
    whiteice::math::vertex<T> recurrent(RECURRENT_DIMENSIONS);
    whiteice::math::vertex<T> recurrent_new(RECURRENT_DIMENSIONS);
    
    recurrent.zero();
    recurrent_new.zero();

    {
      std::lock_guard<std::mutex> lockr(reinforcements_mutex);
      reinforcements.clear();
      reinforcements_random.clear();
      distances.clear();
      distances_random.clear();
    }

    bool random = false;

    // to properly handle cases where performAction() fails
    // [don't getState() or use policy network again] until performAction() is successful
    unsigned int performActionFailed = 0; 

    // whiteice::nnetwork<T> nn;

    unsigned long counter = 0; // N:th iteration
    
    whiteice::logging.info("RIFL_abstract4: starting optimization loop");

    {
      std::lock_guard<std::mutex> lock1(Q_mutex), lock2(policy_mutex);
      
      whiteice::logging.info("RIFL_abstract4: initial Q diagnostics");
      lagged_Q.diagnosticsInfo();
      
      whiteice::logging.info("RIFL_abstract4: initial policy diagnostics");
      lagged_policy.diagnosticsInfo();
    }

    
    // timing
    auto start = std::chrono::high_resolution_clock::now();

    
    while(thread_is_running > 0){

      if(learningMode == false){
	if(dataset_thread){
	  delete dataset_thread;
	  dataset_thread = nullptr;
	}

	if(dataset_q2_thread){
	  delete dataset_q2_thread;
	  dataset_q2_thread = nullptr;
	}
	
	if(dataset2_thread){
	  delete dataset2_thread;
	  dataset2_thread = nullptr;
	}

	grad.stopComputation();
	grad.reset();

	grad2.stopComputation();
	grad2.reset();
      }

      
      if(LOOP_UPDATE_HZ > 0){
	// const float LOOP_UPDATE_HZ = 50.0f; // 50 Hz update frequency (1000/50 = 20ms update speed)
	
	auto elapsed = std::chrono::high_resolution_clock::now() - start;
	
	const long long microseconds_elapsed =
	  std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	
	const long long microseconds_sleep_time = 1000000/LOOP_UPDATE_HZ;
	
	if(microseconds_elapsed > microseconds_sleep_time){
	  whiteice::logging.warn("RIFL_abstract4::loop(): Warning sampling out of sync!");
	}
	else{
	  std::this_thread::sleep_for
	    (std::chrono::microseconds(microseconds_sleep_time - microseconds_elapsed));
	  
	  // sleep 1000000us/100 Hz = 10ms between updates => 100 Hz polling/sampling interval
	  // std::this_thread::sleep_for(std::chrono::milliseconds(1000/SAMPLING_HZ));
	}
	
	start = std::chrono::high_resolution_clock::now();
      }

      
      if(sleepMode == true){
	std::this_thread::sleep_for(std::chrono::milliseconds(50));
	continue; // we do not do anything and only sleep
      }

      
      counter++;
      
      // 1. gets current state
      if(performActionFailed == 0){
	auto oldstate = state;
      
	if(getState(state) == false){
	  state = oldstate;
	  if(firstTime) continue;

	  whiteice::logging.error("ERROR: RIFL_abstact2::getState() FAILED.");
	}

	firstTime = false;
      }


      // 2. selects action using policy
      // (+ random selection if there is no model or in
      //    1-epsilon probability)
      
            
      if(performActionFailed == 0){
	std::lock_guard<std::mutex> lock(policy_mutex);

	whiteice::math::vertex<T> tmpout, u;

	whiteice::math::vertex<T> tmpstate = state;
	policy_preprocess.preprocess(0, tmpstate);

	whiteice::math::vertex<T> input;
	input.resize(numStates + RECURRENT_DIMENSIONS);
	input.write_subvertex(tmpstate, 0);
	input.write_subvertex(recurrent, numStates);
	

	if(lagged_policy.calculate(input, tmpout, 1, 0) == true){
	  if(tmpout.size() != numActions+RECURRENT_DIMENSIONS){
	    u.resize(numActions);
	    for(unsigned int i=0;i<numActions;i++){
	      u[i] = T(0.0);
	    }

	    recurrent.zero();
	    recurrent_new.zero();

	    random = true;
	  }
	  else{
	    whiteice::math::vertex<T> temp;
	    temp.resize(numActions);
	    tmpout.subvertex(temp, 0, numActions);
	    tmpout.subvertex(recurrent_new, numActions, RECURRENT_DIMENSIONS);
	    
	    policy_preprocess.invpreprocess(1, temp);
	    u = temp;

	    for(unsigned int i=0;i<u.size();i++){ // action is [0,1]^D valued vector
	      if(u[i] < T(-1.0f)) u[i] = T(-1.0f);
	      else if(u[i] > T(1.0f)) u[i] = T(1.0f);
	    }
	    
	    random = false;
	  }
	}
	else{
	  u.resize(numActions);
	  for(unsigned int i=0;i<numActions;i++){
	    u[i] = T(0.0);
	  }

	  // recurrent.zero();
	  recurrent_new.zero();

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

	  if(random_counter > 0){ // 1-epsilon % are chosen randomly

	    auto noise = u;
	    
	    rng.normal(noise); // Normal E[n]=0 StDev[n]=1

	    u += T(1.00f)*noise; // was 0.1, 0.3*, 0.6

	    for(unsigned int i=0;i<u.size();i++){ // action is [0,1]^D valued vector
	      if(u[i] < T(-1.0f)) u[i] = T(-1.0f);
	      else if(u[i] > T(1.0f)) u[i] = T(1.0f);
	    }

#if 0
	    rng.uniform(u); // [0,1] valued actions!

	    for(unsigned int i=0;i<u.size();i++)
	      u[i] = T(2.0f)*u[i] - T(1.0f); // [-1,+1]
#endif
	    //std::cout << "noise = " << noise << std::endl;
	    //std::cout << "action = " << u << std::endl;	    
	    //std::cout << "random = " << random_counter << std::endl;

	    recurrent.zero();
	    //recurrent_new.zero();
	    
	    random = true;
	  }
	  else{ // just adds random noise to action [mini-exploration]
	    auto noise = u;
	    rng.normal(noise); // Normal EX[n]=0 StDev[n]=1
	    u += T(0.025)*noise;

	    for(unsigned int i=0;i<u.size();i++){ // action is [-1,1]^D valued vector
	      if(u[i] < T(-1.0f)) u[i] = T(-1.0f);
	      else if(u[i] > T(1.0f)) u[i] = T(1.0f);
	    }
	  }
	  
	}

	// if there's no model then make random selection (normally distributed)
	{
	  std::lock_guard<std::mutex> lockh(has_model_mutex);
	  
	  if(hasModel[0] == 0 || hasModel[1] == 0 || hasModel[2] == 0){
	    recurrent.zero();
	    // recurrent_new.zero();

	    
	    auto noise = u;
	    
	    rng.normal(noise); // Normal E[n]=0 StDev[n]=1
	    
	    u += T(1.00f)*noise; // was 0.1, 0.3*, 0.6
	    
	    for(unsigned int i=0;i<u.size();i++){ // action is [0,1]^D valued vector
	      if(u[i] < T(-1.0f)) u[i] = T(-1.0f);
	      else if(u[i] > T(1.0f)) u[i] = T(1.0f);
	    }
	    
#if 0
	    rng.uniform(u); // [0,1] valued actions!

	    for(unsigned int i=0;i<u.size();i++)
	      u[i] = T(2.0f)*u[i] - T(1.0f); // [-1,+1]
#endif
	    
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
      
      whiteice::math::vertex<T> newstate;
      T reinforcement = T(0.0);
      T distance = T(0.0);

      // 3. perform action and get newstate and reinforcement
      {
	
	if(performAction(action, newstate, reinforcement, distance, endFlag) == false){
	  //std::cout << "ERROR: RIFL_abstract4::performAction() FAILED." << std::endl;
	  // whiteice::logging.error("ERROR: RIFL_abstact::performAction() FAILED.");
	  performActionFailed++;
	  goto optimization_step;
	}
	else{
	  performActionFailed = 0;

	  // did actual random action so reduce random_counter by one
	  random_counter--;
	  if(random_counter <= 0) random_counter = 0;
	}
	
      }

      
      // 4. updates database (of actions and responses)
      if(performActionFailed == 0){ // successful action..
	struct rifl4_datapoint<T> datum;

	datum.state = state;
	datum.action = action;
	datum.newstate = newstate;
	datum.reinforcement = reinforcement;
	datum.recurrent = recurrent;
	datum.recurrent_new = recurrent_new;
	datum.random = random;
	datum.lastStep = endFlag;

	// update recurrent components of policy network here..
	{
	  recurrent = recurrent_new;
	  recurrent_new.zero();
	}
	
	{
	  std::lock_guard<std::mutex> lockr(reinforcements_mutex);

	  if(random){
	    reinforcements_random.push_back(reinforcement);
	    distances_random.push_back(distance);
	  }
	  else{
	    reinforcements.push_back(reinforcement);
	    distances.push_back(distance);
	  }
	}

	// for synchronizing access to database datastructure
	// (also used by CreateRIFL2dataset class/thread)
	std::lock_guard<std::mutex> lock(database_mutex);

	episode.push_back(datum);

	if(datum.lastStep){

	  T total_reward = T(0.0f);

	  for(const auto& e : episode)
	    total_reward += e.reinforcement;
	  
	  if(episode.size())
	    total_reward /= T(episode.size());

	  T episode_weight = T(0.0f);

	  for(const auto& e : episode){
	    if(negativeQ){
	      episode_weight += whiteice::math::abs((T(1.0f) + e.reinforcement)/T(2.0f)); // values are between [-1,+1] => [0,1]
	    }
	    else{
	      episode_weight += whiteice::math::abs(e.reinforcement);
	    }
	      
	  }
	  
	  if(episode.size())
	    episode_weight /= T(episode.size());

	  if(use_smart_weights == false) // if we don't use weighting, give each episode an equal weight..
	    episode_weight = T(1.0f);

	  {
	    char buffer[80];

	    std::lock_guard<std::mutex> lockh(has_model_mutex);
	    
	    snprintf(buffer, 80, "Episode %d avg reward: %f (%d moves) [%d %d models]",
		     (int)episodes_counter, total_reward.c[0], (int)episode.size(),
		     hasModel[0], hasModel[1]);

	    whiteice::logging.info(buffer);
	  }


	  fprintf(episodesFile, "%f\n", total_reward.c[0]);
	  fflush(episodesFile);

	  latestError = (float)total_reward.c[0];

	  {
	    
	    if(episodes.size() >= EPISODES_MAX_SIZE){
	      const unsigned long index = (episodes_counter % EPISODES_MAX_SIZE);
	      episodes[index] = episode;
	      episodes_weights[index] = episode_weight;
	    }
	    else{
	      episodes.push_back(episode);
	      episodes_weights.push_back(episode_weight);
	    }
	    
	  }

	  episode.clear();
	  episodes_counter++;
	}

	if(database_counter >= DATASIZE)
	  database_counter = database_counter % database.size();

	{
	  if(database.size() >= DATASIZE){
	    const unsigned int index = database_counter % database.size();

	    database[index] = datum;
	  }
	  else{
	    database.push_back(datum);
	  }
	  
	}

	database_counter++;
      }

    optimization_step:
      
      if(learningMode == false){

	if(dataset_thread){
	  delete dataset_thread;
	  dataset_thread = nullptr;
	}

	if(dataset_q2_thread){
	  delete dataset_q2_thread;
	  dataset_q2_thread = nullptr;
	}
	
	if(dataset2_thread){
	  delete dataset2_thread;
	  dataset2_thread = nullptr;
	}

	grad.stopComputation();
	grad.reset();

	grad2.stopComputation();
	grad2.reset();
	
	continue; // we do not do learning
      }
      
      // 5. update/optimize Q(state, action) network
      // activates batch learning if it is not running
      if(database.size() >= MINIMUM_DATASIZE)
      {
	
	if(epoch[0] > epoch[1])
	  goto q2_optimization;
	
	// skip if other optimization step (policy network)
	// is behind us
	if(epoch[1] > epoch[2])
	  goto q_optimization_done;
	
	
	T error;
	unsigned int iters;
	
	
	if(grad.isRunning() == false){

	  if(grad.getSolutionStatistics(error, iters) == false){
	    // grad is reset()ed having no solution anymore (read once it) 
	  }
	  else{
	    // gradient have stopped running

	    if(dataset_thread == nullptr){

	      char buffer[128];
	      double tmp = 0.0;
	      whiteice::math::convert(tmp, error);
	      snprintf(buffer, 128,
		       "RIFL_abstract4: new optimized Q-model (%f error, %d iters, epoch %d)",
		       tmp, iters, epoch[0]);
	      whiteice::logging.info(buffer);

	      whiteice::nnetwork<T> nn;
	      
	      {
		logging.info("========> Q RESULT LOADING");
		
		if(grad.getSolution(nn) == false) assert(0);
		
		std::lock_guard<std::mutex> lock(Q_mutex);
		Q.importNetwork(nn);

		Q_preprocess = data;
		
		Q_preprocess.clearData(0);
		Q_preprocess.clearData(1);
		
#if 1
		whiteice::nnetwork<T> nn2;
		std::vector< math::vertex<T> > lagged_weights;
		std::vector< math::vertex<T> > lagged_bndata;

		if(lagged_Q.getBatchNorm()){
		  if(lagged_Q.exportSamples(nn2, lagged_weights, lagged_bndata, 1) == false)
		    assert(0);
		}
		else{
		  if(lagged_Q.exportSamples(nn2, lagged_weights, 1) == false)
		    assert(0);
		}

		if(lagged_weights.size() > 0){

		  math::vertex<T> weights;
		  math::vertex<T> bndata;
		  
		  if(nn.exportdata(weights) == false) assert(0);
		  if(nn.getBatchNorm()) if(nn.exportBNdata(bndata) == false) assert(0);

		  {
		    std::lock_guard<std::mutex> lockh(has_model_mutex);
		    
		    if(hasModel[1] == 0){
		      // don't lag results with the first update
		      lagged_weights[0] = weights;
		      if(nn.getBatchNorm()) lagged_bndata[0] = bndata;
		    }
		  }
		  
		  lagged_weights[0] = tau*weights + (T(1.0)-tau)*lagged_weights[0];
		  if(nn.getBatchNorm()) lagged_bndata[0]  = tau*bndata  + (T(1.0)-tau)*lagged_bndata[0];

		  // 
		  //auto part1 = tau*weights; // THIS DOES NOT WORK PROPERLY (BUG!)
		  //auto part2 = (T(1.0)-tau)*lagged_weights[0];
		  //
		  //lagged_weights[0] = part1 + part2;
		  //
		  
		  if(nn.importdata(lagged_weights[0]) == false) assert(0);
		  if(nn.getBatchNorm()) if(nn.importBNdata(lagged_bndata[0]) == false) assert(0);
		  if(lagged_Q.importNetwork(nn) == false) assert(0);
		}
		else{
		  logging.info("lagged_Q updated: NO LAG");
		  
		  lagged_Q.importNetwork(nn); 
		}
#endif
		
		whiteice::logging.info("RIFL_abstract4: new Q diagnostics");
		lagged_Q.diagnosticsInfo();
		whiteice::logging.info("RIFL_abstract4: new Q-model imported");
	      }

	      grad.reset(); // resets gradient to empty gradient descent

	      epoch[0]++;
	      
	      {
		std::lock_guard<std::mutex> lockh(has_model_mutex);
		hasModel[0]++;
	      }
	    }
	  }


	  // skip if other optimization step (policy network)
	  // is behind us
	  if(epoch[0] > epoch[1])
	    goto q_optimization_done;

	  
	  // const unsigned int NUMSAMPLES = database.size(); // was 1000
	  // const unsigned int NUMSAMPLES = 2000; // was 1000, 128
	  
	  
	  if(dataset_thread == nullptr){

	    {
	      std::lock_guard<std::mutex> lock(database_mutex);
	      std::lock_guard<std::mutex> lockh(has_model_mutex);
	      
	      data.clear();
	      //data.createCluster("input-state", numStates + numActions);
	      //data.createCluster("output-qvalue", 1);
	      
	      dataset_thread = new CreateRIFL4dataset<T>(*this,
							 database,
							 episodes,
							 episodes_weights,
							 database_mutex,
							 hasModel[0]);
	    }
	    
	    dataset_thread->start(SAMPLESIZE, useEpisodes);
	      
	    whiteice::logging.info("RIFL_abstract4: new dataset_thread started (Q)");
	    
	    continue;
      
	  }
	  else{
	    if(dataset_thread->isCompleted() != true){
	      continue; // we havent computed proper dataset yet..
	    }
	    else{
	      data = dataset_thread->getDataset();
	    }
	  }
	  
	  if(dataset_thread){
	    whiteice::logging.info("RIFL_abstract4: dataset_thread finished (Q)");
	    dataset_thread->stop();
	    delete dataset_thread;
	    dataset_thread = nullptr;
	  }


	  // fetch NN parameters from model
	  whiteice::nnetwork<T> qnn;
	  
	  {
	    std::vector< math::vertex<T> > weights;
	    std::vector< math::vertex<T> > bndatas;
	    
	    std::lock_guard<std::mutex> lock(Q_mutex);

	    if(Q.getBatchNorm()){
	      if(Q.exportSamples(qnn, weights, bndatas, 1) == false){ // was: lagged_Q
		assert(0);
	      }
	    }
	    else{
	      if(Q.exportSamples(qnn, weights, 1) == false){ // was: lagged_Q
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
	  
	  grad.setRegularizer(T(0.0f)); // DISABLE REGULARIZER FOR Q-NETWORK (was: 0.001f)
	  grad.setNormalizeError(false); // calculate real error values	  
	  
	  {
	    std::lock_guard<std::mutex> lockh(has_model_mutex);
	    
	    if(hasModel[0] >= 1){
	      eta.start(0.0, Q_OPTIMIZE_ITERATIONS);
	      
	      grad.setUseMinibatch(false);
	      grad.setSGD(T(-1.0f)); // disable stochastic gradient descent
	      
	      if(grad.startOptimize(data, qnn, 1, Q_OPTIMIZE_ITERATIONS,
				    dropout, useInitialNN) == true)
		logging.info("========> Q OPTIMIZATION STARTED");
	      else
		logging.info("========> Q OPTIMIZATION STARTED FAILED");
	    }
	    else{
	      eta.start(0.0, Q_OPTIMIZE_ITERATIONS_FIRST);
	      
	      grad.setUseMinibatch(false);
	      grad.setSGD(T(-1.0f)); // disable stochastic gradient descent
	      
	      if(grad.startOptimize(data, qnn, 1, Q_OPTIMIZE_ITERATIONS_FIRST, dropout, useInitialNN) == true)
		logging.info("========> Q OPTIMIZATION STARTED");
	      else
		logging.info("========> Q OPTIMIZATION STARTED FAILED");
	    }
	  }
	  

	  old_grad_iterations = -1;
	}
	else{
	  T error = T(0.0);
	  unsigned int iters = 0;

	  if(grad.getSolutionStatistics(error, iters)){
	    if(((signed int)iters) > old_grad_iterations){
	      {
		std::lock_guard<std::mutex> lockh(has_model_mutex);
		
		char buffer[128];
		
		eta.update(iters);
		
		double e;
		whiteice::math::convert(e, error);
		
		snprintf(buffer, 128,
			 "RIFL_abstract4: Q-optimizer epoch %d iter %d error %.12f hasmodel %d [ETA %.2f mins]",
			 epoch[0], iters, e, hasModel[0], eta.estimate()/60.0);
		
		whiteice::logging.info(buffer);
	      }
		
	      old_grad_iterations = (int)iters;
	    }
	  }
	  else{
	    char buffer[80];
	    snprintf(buffer, 80,
		     "RIFL_abstract4: epoch %d grad.getSolution() FAILED",
		     epoch[0]);
	    
	    whiteice::logging.error(buffer);
	  }
	}
      }

      
    q2_optimization:

      // 5. update/optimize Q2(state, action) network
      // activates batch learning if it is not running
      if(database.size() >= MINIMUM_DATASIZE)
      {

	// skip if other optimization step (policy network)
	// is behind us
	if(epoch[1] > epoch[2] || epoch[0] == epoch[1])
	  goto q_optimization_done;
	
	
	T error;
	unsigned int iters;
	
	
	if(q2grad.isRunning() == false){

	  if(q2grad.getSolutionStatistics(error, iters) == false){
	    // grad is reset()ed having no solution anymore (read once it) 
	  }
	  else{
	    // gradient have stopped running

	    if(dataset_q2_thread == nullptr){

	      char buffer[128];
	      double tmp = 0.0;
	      whiteice::math::convert(tmp, error);
	      snprintf(buffer, 128,
		       "RIFL_abstract4: new optimized Q2-model (%f error, %d iters, epoch %d)",
		       tmp, iters, epoch[1]);
	      whiteice::logging.info(buffer);

	      whiteice::nnetwork<T> nn;
	      
	      {
		logging.info("========> Q RESULT LOADING");
		
		if(q2grad.getSolution(nn) == false) assert(0);
		
		std::lock_guard<std::mutex> lock(Q_mutex);
		Q2.importNetwork(nn);

		Q_preprocess = data;

		Q_preprocess.clearData(0);
		Q_preprocess.clearData(1);

#if 1
		whiteice::nnetwork<T> nn2;
		std::vector< math::vertex<T> > lagged_weights;
		std::vector< math::vertex<T> > lagged_bndata;

		if(lagged_Q2.getBatchNorm()){
		  if(lagged_Q2.exportSamples(nn2, lagged_weights, lagged_bndata, 1) == false)
		    assert(0);
		}
		else{
		  if(lagged_Q2.exportSamples(nn2, lagged_weights, 1) == false)
		    assert(0);
		}

		if(lagged_weights.size() > 0){

		  math::vertex<T> weights;
		  math::vertex<T> bndata;
		  
		  if(nn.exportdata(weights) == false) assert(0);
		  if(nn.getBatchNorm()) if(nn.exportBNdata(bndata) == false) assert(0);

		  {
		    std::lock_guard<std::mutex> lockh(has_model_mutex);
		    
		    if(hasModel[1] == 0){
		      // don't lag results with the first update
		      lagged_weights[0] = weights;
		      if(nn.getBatchNorm()) lagged_bndata[0] = bndata;
		    }
		  }
		  
		  lagged_weights[0] = tau*weights + (T(1.0)-tau)*lagged_weights[0];
		  if(nn.getBatchNorm()) lagged_bndata[0]  = tau*bndata  + (T(1.0)-tau)*lagged_bndata[0];

		  
		  if(nn.importdata(lagged_weights[0]) == false) assert(0);
		  if(nn.getBatchNorm()) if(nn.importBNdata(lagged_bndata[0]) == false) assert(0);
		  if(lagged_Q2.importNetwork(nn) == false) assert(0);
		}
		else{
		  logging.info("lagged_Q2 updated: NO LAG");
		  
		  lagged_Q2.importNetwork(nn); 
		}
#endif
		
		whiteice::logging.info("RIFL_abstract4: new Q diagnostics");
		lagged_Q2.diagnosticsInfo();
		whiteice::logging.info("RIFL_abstract4: new Q-model imported");
	      }

	      q2grad.reset(); // resets gradient to empty gradient descent

	      epoch[1]++;
	      
	      {
		std::lock_guard<std::mutex> lockh(has_model_mutex);
		hasModel[1]++;
	      }
	    }
	  }


	  // skip if other optimization step (policy network)
	  // is behind us
	  if(epoch[1] > epoch[2])
	    goto q_optimization_done;

	  
	  // const unsigned int NUMSAMPLES = database.size(); // was 1000
	  // const unsigned int NUMSAMPLES = 2000; // was 1000, 128
	  
#if 0
	  if(dataset_q2_thread == nullptr){

	    {
	      std::lock_guard<std::mutex> lock(database_mutex);
	      std::lock_guard<std::mutex> lockh(has_model_mutex);
	      
	      data.clear();
	      //data.createCluster("input-state", numStates + numActions);
	      //data.createCluster("output-qvalue", 1);
	      
	      dataset_q2_thread = new CreateRIFL4dataset<T>(*this,
							    database,
							    episodes,
							    episodes_weights,
							    database_mutex,
							    hasModel[1]);
	    }
	    
	    dataset_q2_thread->start(SAMPLESIZE, useEpisodes);
	    
	    whiteice::logging.info("RIFL_abstract4: new dataset_thread started (Q)");
	    
	    continue;
      
	  }
	  else{
	    if(dataset_q2_thread->isCompleted() != true){
	      continue; // we havent computed proper dataset yet..
	    }
	    else{
	      data = dataset_q2_thread->getDataset();
	    }
	  }
#endif
	  
	  if(dataset_q2_thread){
	    whiteice::logging.info("RIFL_abstract4: dataset_q2_thread finished (Q)");
	    dataset_q2_thread->stop();
	    delete dataset_q2_thread;
	    dataset_q2_thread = nullptr;
	  }


	  // fetch NN parameters from model
	  whiteice::nnetwork<T> qnn;
	  
	  {
	    std::vector< math::vertex<T> > weights;
	    std::vector< math::vertex<T> > bndatas;
	    
	    std::lock_guard<std::mutex> lock(Q_mutex);

	    if(Q2.getBatchNorm()){
	      if(Q2.exportSamples(qnn, weights, bndatas, 1) == false){ // was: lagged_Q
		assert(0);
	      }
	    }
	    else{
	      if(Q2.exportSamples(qnn, weights, 1) == false){ // was: lagged_Q
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
	  
	  q2grad.setRegularizer(T(0.0f)); // DISABLE REGULARIZER FOR Q-NETWORK (was: 0.001f)
	  q2grad.setNormalizeError(false); // calculate real error values	  
	  
	  {
	    std::lock_guard<std::mutex> lockh(has_model_mutex);
	    
	    if(hasModel[1] >= 1){
	      eta.start(0.0, Q_OPTIMIZE_ITERATIONS);
	      
	      q2grad.setUseMinibatch(false);
	      q2grad.setSGD(T(-1.0f)); // disable stochastic gradient descent
	      
	      if(q2grad.startOptimize(data, qnn, 1, Q_OPTIMIZE_ITERATIONS,
				      dropout, useInitialNN) == true)
		logging.info("========> Q OPTIMIZATION STARTED");
	      else
		logging.info("========> Q OPTIMIZATION STARTED FAILED");
	    }
	    else{
	      eta.start(0.0, Q_OPTIMIZE_ITERATIONS_FIRST);
	      
	      q2grad.setUseMinibatch(false);
	      q2grad.setSGD(T(-1.0f)); // disable stochastic gradient descent
	      
	      if(q2grad.startOptimize(data, qnn, 1, Q_OPTIMIZE_ITERATIONS_FIRST, dropout, useInitialNN) == true)
		logging.info("========> Q OPTIMIZATION STARTED");
	      else
		logging.info("========> Q OPTIMIZATION STARTED FAILED");
	    }
	  }
	  

	  old_grad_q2_iterations = -1;
	}
	else{
	  T error = T(0.0);
	  unsigned int iters = 0;

	  if(q2grad.getSolutionStatistics(error, iters)){
	    if(((signed int)iters) > old_grad_q2_iterations){
	      {
		std::lock_guard<std::mutex> lockh(has_model_mutex);
		
		char buffer[128];
		
		eta.update(iters);
		
		double e;
		whiteice::math::convert(e, error);
		
		snprintf(buffer, 128,
			 "RIFL_abstract4: Q2-optimizer epoch %d iter %d error %.12f hasmodel %d [ETA %.2f mins]",
			 epoch[1], iters, e, hasModel[1], eta.estimate()/60.0);
		
		whiteice::logging.info(buffer);
	      }
		
	      old_grad_q2_iterations = (int)iters;
	    }
	  }
	  else{
	    char buffer[80];
	    snprintf(buffer, 80,
		     "RIFL_abstract4: epoch %d q2grad.getSolution() FAILED",
		     epoch[1]);
	    
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
	if(epoch[0] == 0  || epoch[1] == 0 || epoch[1] <= epoch[2])
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
			 "RIFL_abstract4: new optimized policy-model (%f mean-q, %d iters, epoch %d)",
			 tmp, iters, epoch[1]);
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

		  lagged_weights.resize(1);
		  
		  if(hasModel[2] == 0){
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
		    
		    nn.importdata(lagged_weights[0]);
		    if(nn.getBatchNorm()) nn.importBNdata(lagged_bndatas[0]);
		    
		    lagged_policy.importNetwork(nn);
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
		
		whiteice::logging.info("RIFL_abstract4: new policy diagnostics");
		lagged_policy.diagnosticsInfo();
		whiteice::logging.info("RIFL_abstract4: new policy-model imported");
	      }

	      grad2.reset();

	      epoch[2]++;

	      {
		std::lock_guard<std::mutex> lockh(has_model_mutex);
		
		hasModel[2]++;
	      }
	    }
	    
	  }

	  
	  // skip if other optimization step is behind us
	  // we only start calculating policy after Q() has been optimized..
	  if(epoch[2] >= epoch[1] || epoch[1] == 0 || epoch[0] == 0)
	    goto policy_optimization_done;
	  
	  
	  // const unsigned int BATCHSIZE = database.size(); // was 1000
	  // const unsigned int BATCHSIZE = 1000; // was 128

	  if(dataset2_thread == nullptr){
	    data2.clear();
	    data2.createCluster("input-state", numStates+RECURRENT_DIMENSIONS);
	    data2.createCluster("ranges", 2);
	    
	    dataset2_thread = new CreatePolicy4Dataset<T>(*this,
							  episodes,
							  episodes_weights,
							  database_mutex,
							  data2);
	    dataset2_thread->start(SAMPLESIZE);

	    whiteice::logging.info("RIFL_abstract4: new dataset2_thread started (policy)");

	    continue;
	  }
	  else{
	    if(dataset2_thread->isCompleted() == false)
	      continue; // we havent computed proper dataset yet..
	  }

	  if(dataset2_thread){
	    if(dataset2_thread->isCompleted() == true){
	      whiteice::logging.info("RIFL_abstract4: dataset2_thread finished (policy)");
	      dataset2_thread->stop();
	      delete dataset2_thread;
	      dataset2_thread = nullptr;
	    }
	  }
	  
	  
	  // fetch NN parameters from model and start optimization
	  {
	    whiteice::nnetwork<T> q_nn, nn;
	    whiteice::dataset<T> Q_preprocess_copy;

	    {
	      std::lock_guard<std::mutex> lock(Q_mutex);
	      std::vector< math::vertex<T> > weights;
	      std::vector< math::vertex<T> > bndatas;

	      if(Q.getBatchNorm()){
		if(Q.exportSamples(q_nn, weights, bndatas, 1) == false){ // was: lagged_Q
		  assert(0);
		}
	      }
	      else{
		if(Q.exportSamples(q_nn, weights, 1) == false){ // was: lagged_Q
		  assert(0);
		}
	      }
	      
	      assert(weights.size() > 0);
	      
	      if(q_nn.importdata(weights[0]) == false){
		assert(0);
	      }

	      if(q_nn.getBatchNorm()){
		if(q_nn.importBNdata(bndatas[0]) == false){
		  assert(0);
		}
	      }

	      Q_preprocess_copy = Q_preprocess;
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

	      logging.info("policy loading diagnostics");
	      policy.diagnosticsInfo();
	      nn.diagnosticsInfo();
	    }
	    
	    const bool dropout = false;
	    const bool useInitialNN = true; // WAS: start from scratch everytime
	    const bool alwaysUpdateSolution = false;
	    

	    {
	      std::lock_guard<std::mutex> lockh(has_model_mutex);
	      
	      if(hasModel[2] >= 1){
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

	    
	    

	    
	    old_grad2_iterations = -1;
	    
	    if(dataset2_thread) delete dataset2_thread;
	    dataset2_thread = nullptr;
	  }
	  
	}
	else{
	  
	  if(grad2.getSolutionStatistics(meanq, iters)){
	    if(((signed int)iters) > old_grad2_iterations){
	      char buffer[128];
	      
	      double v;
	      whiteice::math::convert(v, meanq);

	      eta2.update(iters);

	      {
		std::lock_guard<std::mutex> lockh(has_model_mutex);
		snprintf(buffer, 128,
			 "RIFL_abstract4: grad2 policy-optimizer epoch %d hasmodel %d iter %d mean q-value %.12f [ETA %.2f mins]",
			 epoch[2], hasModel[2], iters, v, eta2.estimate()/60.0);
	      }
	      
	      whiteice::logging.info(buffer);

	      old_grad2_iterations = (int)iters;
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

    grad.stopComputation();
    grad2.stopComputation();

    if(episodesFile) fclose(episodesFile);
    episodesFile = NULL;

    if(dataset_thread){
      delete dataset_thread;
      dataset_thread = nullptr;
    }

    if(dataset2_thread){
      delete dataset2_thread;
      dataset2_thread = nullptr;
    }
    
  }

  template class RIFL_abstract4< math::blas_real<float> >;
  template class RIFL_abstract4< math::blas_real<double> >;
  
};
