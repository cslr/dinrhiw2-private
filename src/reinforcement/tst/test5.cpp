/*
 * Testsuite 5
 *
 * testing Addition problem with RIFL4 (recurrent reinforcement learning)
 */


#include "AdditionProblem4.h"


int main(void)
{
  printf("*RECURRENT* CONTINUOUS REINFORCEMENT LEARNING TESTCASE 5. (Addition problem)\n");
  fflush(stdout);

  use_gpu_sync = false; 

  srand(time(0));

  whiteice::logging.setOutputFile("debug.log");
  
  whiteice::AdditionProblem4< whiteice::math::blas_real<double> > system;

  
  system.setEpsilon(0.75); // 25% of control choices are random
  system.setSleepingMode(false);
  system.setLearningMode(true);
  //system.setVerbose(true);
  
  system.start();
  
  sleep(1);

  while(true){

    sleep(1);
  }

  
}
