/*
 * NEW:
 *
 * *RECURRENT* reinforcement learning for continuous
 * actions and continuous states
 *
 * Care must be taken to handle random actions properly in each episode's stream of datapoints.
 * 
 * OLD:
 * Reinforcement learning for 
 * continuous actions and continuous states
 * uses neural network (nnetwork) to learn 
 * utility (Q) and policy functions
 * 
 * Implementation is mostly based on the following paper:
 * 
 * Continuous Control With Deep Reinforcement Learning.
 * Timothy P. Lillicrap*, Jonathan J. Hunt*, 
 * Alexander Pritzel, Nicolas Heess, Tom Erez, 
 * Yuval Tassa, David Silver & Daan Wierstra
 * Google DeepMind, London, UK.
 * Conference paper at ICLR 2016
 *
 */

#ifndef whiteice_RIFL_abstract4_h
#define whiteice_RIFL_abstract4_h

#include <string>
#include <mutex>
#include <thread>
#include <vector>
#include <atomic>

#include "dinrhiw_blas.h"
#include "vertex.h"
#include "bayesian_nnetwork.h"
#include "RNG.h"
#include "dataset.h"


namespace whiteice
{
  template <typename T>
  class CreateRIFL4dataset;
  
  template <typename T>
  class CreatePolicy4Dataset;

  template <typename T>
  class Policy4GradAscent;
  
  
  template <typename T>
  struct rifl4_datapoint
  {
    whiteice::math::vertex<T> state, newstate;
    whiteice::math::vertex<T> recurrent, recurrent_new; // recurrent dimensions
    whiteice::math::vertex<T> action;

    T distance;
    
    T reinforcement, reinforcement_pure; // pure is R without after effect

    bool random;   // true of episode's step was random (important!)
    bool lastStep; // true if was the last step of the simulation
  };
  

  template <typename T = math::blas_real<float> >
  class RIFL_abstract4
  {
  public:
    
    // parameters are dimensions of vectors dimActions and dimStates: R^d
    RIFL_abstract4(unsigned int numActions, unsigned int numStates,
		   const bool alsoNegativeQValues = false,
		   const int sequentialRandomMoves = 1,
		   const unsigned int RECURRENT_DIMENSIONS = 5);
    
    RIFL_abstract4(unsigned int numActions, unsigned int numStates,
		   const bool alsoNegativeQValues, 
		   std::vector<unsigned int> Q_arch,
		   std::vector<unsigned int> policy_arch,
		   const int sequentialRandomMoves = 1,
		   const unsigned int RECURRENT_DIMENSIONS = 5);

    ~RIFL_abstract4() ;

    // starts Reinforcement Learning thread
    bool start();

    // stops Reinforcement Learning thread
    bool stop();
    
    bool isRunning() const;

    // gamma value [0,1], how much predicted future rewards are calculated into Q-value
    bool setGamma(T gamma);

    T getGamma() const;

    // epsilon E [0,1] percentage of actions are chosen according to model
    //                 1-e percentage of actions are random (exploration)
    bool setEpsilon(T epsilon) ;

    T getEpsilon() const ;

    bool setQTau(const T tau){
      if(tau <= T(0.0) || tau >= T(1.0)) return false;
      this->tau = tau;
      return true;
    }
    
    T getQTau() const { return this->tau; }

    bool setPolicyTau(const T tau){
      if(tau <= T(0.0) || tau >= T(1.0)) return false;
      this->tau_policy = tau;
      return true;
    }

    T getPolicyTau() const { return this->tau_policy; }

    /*
     * sets/gets learning mode (default: on)
     * (do we do just control or also try to learn from data)
     */
    void setLearningMode(bool learn) ;
    bool getLearningMode() const ;

    /*
     * sets/gets sleeping mode (default: on)
     * if we sleep RIFL is inactive and does nothing [no learning, no actions]
     */
    void setSleepingMode(bool sleep) ;
    bool getSleepingMode() const ;

    /*
     * hasModel flag means we have a proper model
     * (from optimization or from load)
     *
     * as long as we don't have a proper model
     * we make random actions (initially) 
     */
    void setHasModel(unsigned int hasModel) ;
    unsigned int getHasModel() ;

    float getLatestEpisodeError() const;

    unsigned int getDatabaseSize() const;

    unsigned int getNumActions() const { return numActions; }
    unsigned int getNumStates() const { return numStates; }

    // tells policy() returns value one hot encoded unscaled probability values (log(p_i))
    // from which proper one-hot action vector is sampled.
    void setOneHotAction(bool isOneHotAction){ oneHotEncodedAction = isOneHotAction; }
    bool getOneHotAction() const{ return oneHotEncodedAction; }

    // do we sample episodes and not samples, needed for recurrent neural network learning
    // FIXME: don't happen properly now
    void setSmartEpisodes(bool use_episodes){ useEpisodes = use_episodes; }
    bool getSmartEpisodes() const{ return useEpisodes; }
    
    void setLoopUpdateFrequency(const float update_hz = 0.0f){
      LOOP_UPDATE_HZ = update_hz;

      if(LOOP_UPDATE_HZ < 0.0f) LOOP_UPDATE_HZ = 0.0f;
    }

    float getLoopUpdateFrequency() const{
      return LOOP_UPDATE_HZ;
    }


    void setReinforcementWeighting(bool use_weighting = true){
      use_smart_weights = use_weighting; // does we use weighted sampling using reinforcement-values in RIFL4?
    }

    bool getReinforcementWeighting() const{
      return use_smart_weights;
    }

    int getLearningDatasetSize() const{
      return SAMPLESIZE;
    }

    bool setLearningDatasetSize(unsigned int size){
      if(size <= 0) return false;

      SAMPLESIZE = size;

      return true;
    }

    unsigned long long getAfterEffectsDelayMS() const{
      return AFTER_EFFECT_DELAY_MS;
    }

    void setAfterEffectsDelayMS(unsigned long long delay_ms = 1500){
      AFTER_EFFECT_DELAY_MS = delay_ms; // zero disables after effect
    }
    

    // clear reinforcement statistics
    bool clearStatistics();

    // how many percent smaller is reinforcement value with random actions vs policy actions
    bool executionStatistics(T& percent_change, T& distances_percent_change,
			     const bool rescale_to_min_value = false,
			     const bool use_only_most_recent = false,
			     unsigned int history_size = 0) const;
    
    // saves learnt Reinforcement Learning Model to file
    bool save(const std::string& filename) const;
    
    // loads learnt Reinforcement Learning Model from file
    bool load(const std::string& filename);

  protected:
    
    const unsigned int numActions, numStates; // dimensions of R^d vectors
    const unsigned int RECURRENT_DIMENSIONS;
    
    virtual bool getState(whiteice::math::vertex<T>& state) = 0;

    // action vector is [0,1]^d (output of sigmoid non-linearity)
    virtual bool performAction(const whiteice::math::vertex<T>& action,
			       whiteice::math::vertex<T>& newstate,
			       T& reinforcement,
			       T& distance,
			       bool& endFlag) = 0;

    // return re-inforcement value for (pre_state and after_state)
    // this is used by after effect code, implement this
    // if AFTER_EFFECT_DELAY_MS is set to be non-zero (after effects enabled)
    virtual T getReinforcement(const whiteice::math::vertex<T>& pre_state,
			       const whiteice::math::vertex<T>& after_state){
      return T(0.0f);
    }
    

    void onehot_prob_select(const whiteice::math::vertex<T>& action,
			    whiteice::math::vertex<T>& new_action,
			    const T temperature = T(1.0f));

    // reinforcement Q model: Q(state, action) ~ discounted future cost
    whiteice::bayesian_nnetwork<T> Q, lagged_Q, Q2, lagged_Q2;
    whiteice::dataset<T> Q_preprocess;
    mutable std::mutex Q_mutex;

    // f(state) = action
    whiteice::bayesian_nnetwork<T> policy, lagged_policy;;
    whiteice::dataset<T> policy_preprocess;
    mutable std::mutex policy_mutex;

    T tau = T(0.001), tau_policy = T(0.005);


    // zero means disabled as the default
    unsigned long long AFTER_EFFECT_DELAY_MS = 0;
    
    std::multimap<unsigned long long, rifl4_datapoint<T> > after_effects_buffer;    

    // database
    std::vector< rifl4_datapoint<T> > database;
    std::vector< std::vector< rifl4_datapoint<T> > > episodes;
    std::vector<T> episodes_weights;

    std::atomic<bool> use_smart_weights; // does we use weighted sampling in RIFL4?
    
    mutable std::mutex database_mutex;

    mutable std::mutex reinforcements_mutex;
    std::vector<T> reinforcements;
    std::vector<T> reinforcements_random;

    std::vector<T> distances;
    std::vector<T> distances_random;

    
    std::vector<unsigned int> hasModel;
    mutable std::mutex has_model_mutex;
    
    std::atomic<float> latestError;
    std::atomic<bool> learningMode, sleepMode;

    std::atomic<bool> negativeQ; // is negative Q values enabled q E [-1,+1] instead of [0,+1]

    std::atomic<unsigned int> SAMPLESIZE;
    
    T epsilon;
    mutable std::mutex epsilon_mutex;
    
    int sequentialRandomMoves = 1; // make 1 random moves in the row when selecting action randomly (can be increased in ctor)
    
    T gamma = T(0.90);
    bool oneHotEncodedAction = false;
    bool useEpisodes = true;
    
    class whiteice::RNG<T> rng;
    
    std::atomic<int> thread_is_running;
    std::thread* rifl_thread;
    mutable std::mutex thread_mutex;

    std::atomic<float> LOOP_UPDATE_HZ;
    
    void loop();
    
    // friend thread class to do heavy computations in background
    // out of main loop 
    friend class CreateRIFL4dataset<T>;

    friend class CreatePolicy4Dataset<T>;

    friend class Policy4GradAscent<T>;
  };



  extern template class RIFL_abstract4< math::blas_real<float> >;
  extern template class RIFL_abstract4< math::blas_real<double> >;
};

#include "CreateRIFL4dataset.h"
#include "CreatePolicy4Dataset.h"
#include "Policy4GradAscent.h"

#endif
