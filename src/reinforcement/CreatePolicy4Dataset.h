// helper class for RIFL4_abstract to generate
// (in the background) dataset for training reinforcement model

#ifndef whiteice_CreatePolicy4Dataset_h
#define whiteice_CreatePolicy4Dataset_h

#include <thread>
#include <mutex>
#include <vector>

#include "dataset.h"
#include "dinrhiw_blas.h"
#include "RIFL_abstract4.h"

#include "RNG.h"

namespace whiteice
{

  template <typename T = math::blas_real<float> >
    class CreatePolicy4Dataset
    {
    public:
      
      // calculates reinforcement learning training dataset from database
      // using database_lock
      CreatePolicy4Dataset(RIFL_abstract4<T> const & rifl, 
			   std::vector< std::vector< rifl4_datapoint<T> > > const & episodes,
			   std::vector<T> const & episodes_weights,
			   std::mutex & database_mutex,
			   whiteice::dataset<T>& data);
      
      virtual ~CreatePolicy4Dataset();
      
      // starts thread that creates NUMDATAPOINTS samples to dataset
      bool start(const unsigned int NUMDATAPOINTS);
      
      // returns true when computation is completed
      bool isCompleted() const;
      
      // returns true if computation is running
      bool isRunning() const;
    
      bool stop();
      
      // returns reference to dataset
      // (warning: if calculations are running then dataset can change during use)
      whiteice::dataset<T> const & getDataset() const;
      
    private:
      
      RIFL_abstract4<T> const & rifl;
      
      std::vector< std::vector< rifl4_datapoint<T> > > const & episodes;
      std::vector<T> const episodes_weights;
      std::mutex & database_mutex;
      
      //whiteice::RNG<T> rng;
      
      unsigned int NUMDATA; // number of datapoints to create
      whiteice::dataset<T>& data;
      bool completed;
      
      std::thread* worker_thread;
      mutable std::mutex   thread_mutex;
      bool running;
      
      // worker thread loop
      void loop();
      
    };


  extern template class CreatePolicy4Dataset< math::blas_real<float> >;
  extern template class CreatePolicy4Dataset< math::blas_real<double> >;
};

#endif
