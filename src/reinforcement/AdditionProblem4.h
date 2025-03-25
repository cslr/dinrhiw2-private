/*
 * Simple addition test problem with recurrent neural network..
 * 
 */

#ifndef __whiteice__addition_problem4_h
#define __whiteice__addition_problem4_h

#include "RIFL_abstract4.h"

#include <condition_variable>


namespace whiteice
{

  template <typename T>
  class AdditionProblem4 : public RIFL_abstract4<T>
  {
  public:
    AdditionProblem4();
    ~AdditionProblem4();

    bool additionIsRunning(){ return this->running; }
      
  protected:
    
    virtual bool getState(whiteice::math::vertex<T>& state);
    
    virtual bool performAction(const whiteice::math::vertex<T>& action,
			       whiteice::math::vertex<T>& newstate,
			       T& reinforcement,
			       T& distance,
			       bool& endFlag);
    
  protected:

    // resets rotation variables to random values
    void reset();

    whiteice::math::vertex<T> state;
    
    bool resetLastStep;
    int iteration;

    volatile bool running;
  };

  
  extern template class AdditionProblem4< math::blas_real<float> >;
  extern template class AdditionProblem4< math::blas_real<double> >;
};


#endif
