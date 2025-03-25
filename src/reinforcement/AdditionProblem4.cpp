

#include "AdditionProblem4.h"
#include "Log.h"

#include <stdio.h>
#include <math.h>

#include <chrono>
#include <functional>

#include <unistd.h>


using namespace std::chrono_literals;


namespace whiteice
{

  template <typename T>
  AdditionProblem4<T>::AdditionProblem4() :
    RIFL_abstract4<T>(3, 3, true, {50,50,50}, {50,50,50})
  {
    this->setOneHotAction(false);
    this->setSmartEpisodes(true);
    
    {
      reset();
      resetLastStep = false;
    }
    
    running = true;
  }


  template <typename T>
  AdditionProblem4<T>::~AdditionProblem4()
  {
    running = false;
  }

  
  template <typename T>
  void AdditionProblem4<T>::reset()
  {
    whiteice::RNG<T> random;

    state.resize(3);

    random.normal(state);
    state /= state.norm();

    state = T(0.25)*state;

    iteration = 0;
    
    // reset in this time step
    resetLastStep = true;
  }
  
  
  template <typename T>
  bool AdditionProblem4<T>::getState(whiteice::math::vertex<T>& state_)
  {
    // whiteice::logging.info("AdditionProblem4: entering getState()");

    state_ = this->state;
    
    // whiteice::logging.info("AdditionProblem4: exiting getState()");
    
    return true;
  }

  
  
  template <typename T>
  bool AdditionProblem4<T>::performAction(const whiteice::math::vertex<T>& action,
					  whiteice::math::vertex<T>& newstate,
					  T& reinforcement,
					  T& distance,
					  bool& endFlag)
  {
    // whiteice::logging.info("AdditionProblem4: entering performAction()");
    
    
    assert(action.size() == 3);
    
    {
      iteration++;

      T small_value = T(2.0);

      if(iteration > 100){
	newstate = state + action;
	reinforcement = T(1.0)/(small_value+newstate.norm());
	// reinforcement = (T(10.0)-newstate.norm())/T(100.0f);
	// reinforcement = T(5.0)-newstate[0];
	// reinforcement = state.norm()-newstate.norm(); [don't work]
	if(reinforcement < T(0.0)) reinforcement = T(0.0);
	if(reinforcement > T(1.0)) reinforcement = T(1.0);
	//std::cout << "ITER " << iteration << " REINFORCEMENT = " << reinforcement << std::endl;

	reset();
	iteration = 0;
	newstate = state;
	
	endFlag = true;

	state = newstate;
	
	return true;
      }
      else{
	newstate = state + action;
	reinforcement = T(1.0)/(small_value+newstate.norm());
	// reinforcement = (T(10.0)-newstate.norm())/T(100.0f);
	// reinforcement = T(5.0)-newstate[0];
	// reinforcement = state.norm()-newstate.norm(); [don't work]
	if(reinforcement < T(0.0)) reinforcement = T(0.0);
	if(reinforcement > T(1.0)) reinforcement = T(1.0);
	//std::cout << "ITER " << iteration << " REINFORCEMENT = " << reinforcement << std::endl;
	endFlag = false;

	state = newstate;

	return true;
      }
    }

    distance = reinforcement;
    
    return true;
  }
 
   
  
  template class AdditionProblem4< math::blas_real<float> >;
  template class AdditionProblem4< math::blas_real<double> >;
};
