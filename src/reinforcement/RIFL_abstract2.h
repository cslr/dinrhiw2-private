/*
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
 * NOTE 2021: Added L2 regularization 0.02*0.5*||w||^2 term 
 *            to optimization of Q and policy so that 
 *            neural network weights cannot explode.
 *
 */

#ifndef whiteice_RIFL_abstract2_h
#define whiteice_RIFL_abstract2_h

#include <string>
#include <mutex>
#include <thread>
#include <vector>
#include <atomic>
#include <map>

#include "dinrhiw_blas.h"
#include "vertex.h"
#include "bayesian_nnetwork.h"
#include "RNG.h"
#include "dataset.h"


namespace whiteice
{
  template <typename T>
    class CreateRIFL2dataset;

  template <typename T>
    class CreatePolicyDataset;
  
  template <typename T>
  struct rifl2_datapoint
  {
    whiteice::math::vertex<T> state, newstate;
    whiteice::math::vertex<T> action;

    T distance;
    
    T reinforcement,
      reinforcement_pure; // pure is reinforcement wihtout after effect,
                          // if after effect is disabled
                          // reinforcement == reinforcement_pure

    unsigned long long t; // measurement time in milliseconds

    bool random;   // true if action was random
    bool lastStep; // true if was the last step of the simulation
  };
  

  template <typename T = math::blas_real<float> >
  class RIFL_abstract2
  {
  public:
    
    // parameters are dimensions of vectors dimActions and dimStates: R^d
    RIFL_abstract2(unsigned int numActions, unsigned int numStates,
		   const bool alsoNegativeQValues = false,
		   const int sequentialRandomMoves = 1,
		   const unsigned int stateHistoryLen = 1);
    
    RIFL_abstract2(unsigned int numActions, unsigned int numStates,
		   const bool alsoNegativeQValues, 
		   std::vector<unsigned int> Q_arch,
		   std::vector<unsigned int> policy_arch,
		   const int sequentialRandomMoves = 1,
		   const unsigned int stateHistoryLen = 1);

    ~RIFL_abstract2() ;

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

    int getLearningDatasetSize() const{
      return SAMPLESIZE;
    }

    bool setLearningDatasetSize(unsigned int size){
      if(size <= 0) return false;
      
      SAMPLESIZE = size;
      
      return true;
    }

    int getMinimumDataSize(){
      return MINIMUM_DATASIZE;
    }

    bool setMinimumDataSize(unsigned int size){
      if(size <= 0) return false;

      MINIMUM_DATASIZE = size;

      return true;
    }

    unsigned long long getAfterEffectsDelayMS() const{
      return AFTER_EFFECT_DELAY_MS;
    }

    void setAfterEffectsDelayMS(unsigned long long delay_ms = 1500){
      AFTER_EFFECT_DELAY_MS = delay_ms; // zero disables after effect
    }

    unsigned long long getHistoryRemoveTimeMS() const{
      return HISTORY_REMOVE_TIME_MS;
    }

    void setHistoryRemoveTimeMS(unsigned long long remove_time_ms = 0){
      HISTORY_REMOVE_TIME_MS = remove_time_ms;
    }

    bool getSaveAndReturnToBestEpisode() const {
      return saveBestEpisode;
    }

    // remembers the best episode and returns to it if no progress
    void setSaveAndReturnToBestEpisode(bool save){
      saveBestEpisode = save;
    }

    // clear reinforcement statistics
    bool clearStatistics();

    // how many percent smaller is reinforcement value with random actions vs policy actions
    bool executionStatistics(T& percent_change, T& distances_percent_change,
			     T& average_change,
			     T& linear_curve_distance_percent_change,
			     const bool rescale_to_min_value = false,
			     const bool use_only_most_recent = false,
			     unsigned int history_size = 0) const;
    
    // saves learnt Reinforcement Learning Model to file
    bool save(const std::string& filename) const;
    
    // loads learnt Reinforcement Learning Model from file
    bool load(const std::string& filename);

  protected:
    
    const unsigned int numActions, numStates; // dimensions of R^d vectors
    
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
    // whiteice::bayesian_nnetwork<T> Q, lagged_Q, Q2, lagged_Q2;
    const unsigned int NUM_Q_NNETWORKS = 10; // was: 5 (dont work??), 10 (works)
    std::vector< whiteice::bayesian_nnetwork<T> > Q, lagged_Q;
    whiteice::dataset<T> Q_preprocess;
    mutable std::mutex Q_mutex;

    // f(state) = action
    whiteice::bayesian_nnetwork<T> policy, lagged_policy;;
    whiteice::dataset<T> policy_preprocess;
    mutable std::mutex policy_mutex;

    // remembers the best episode and returns to it if no progress
    std::atomic<bool> saveBestEpisode = false;    

    T tau = T(0.001), tau_policy = T(0.005);

    // zero means disabled as the default
    unsigned long long AFTER_EFFECT_DELAY_MS = 0;

    std::multimap<unsigned long long, rifl2_datapoint<T> > after_effects_buffer;

    // database
    std::vector< rifl2_datapoint<T> > database;
    std::vector< std::vector< rifl2_datapoint<T> > > episodes;
    std::vector<T> episodes_score;

    // time points when measurement was made
    std::map<unsigned long long, unsigned int> database_t;
    std::map<unsigned long long, unsigned int> episodes_t;

    // how many milliseconds has to pass before element is removed from replay buffer
    // zero means disabled as the default
    unsigned long long HISTORY_REMOVE_TIME_MS = 0;
    
    
    mutable std::mutex database_mutex;

    mutable std::mutex reinforcements_mutex;
    std::vector<T> reinforcements;
    std::vector<T> reinforcements_random;

    std::vector<T> distances;
    std::vector<T> distances_random;

    
    std::vector<unsigned int> hasModel;
    mutable std::mutex has_model_mutex;

    std::atomic<bool> use_smart_weights; // does we use weighted sampling in RIFL2?
    
    std::atomic<float> latestError;
    std::atomic<bool> learningMode, sleepMode;

    const unsigned int STATE_HISTORY_LEN = 1;

    std::atomic<unsigned int> SAMPLESIZE;

    std::atomic<unsigned int> MINIMUM_DATASIZE;
    
    T epsilon;
    mutable std::mutex epsilon_mutex;
    
    int sequentialRandomMoves = 1; // make 1 random moves in the row when selecting action randomly (can be increased in ctor)
    
    T gamma;
    bool oneHotEncodedAction = false;
    bool useEpisodes = false;
    
    class whiteice::RNG<T> rng;
    
    std::atomic<int> thread_is_running;
    std::thread* rifl_thread;
    std::mutex thread_mutex;

    std::atomic<float> LOOP_UPDATE_HZ;
    
    void loop();
    
    // friend thread class to do heavy computations in background
    // out of main loop 
    friend class CreateRIFL2dataset<T>;

    friend class CreatePolicyDataset<T>;
    
    };



  extern template class RIFL_abstract2< math::blas_real<float> >;
  extern template class RIFL_abstract2< math::blas_real<double> >;
};

#include "CreateRIFL2dataset.h"
#include "CreatePolicyDataset.h"

#endif
