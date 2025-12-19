
#ifndef __43_diffeqs_h
#define __43_diffeqs_h

#include "vertex.h"
#include "matrix.h"
#include "nnetwork.h"
#include "bayesian_nnetwork.h"
#include "linear_ETA.h"


#include <vector>
#include <map>


using namespace whiteice;

bool create_random_diffeq_model(whiteice::nnetwork<>& diffeq, const unsigned int DIMENSIONS);

template <typename T = math::blas_real<float> > 
  bool simulate_diffeq_model(const whiteice::nnetwork<T>& diffeq,
			     const whiteice::math::vertex<T>& start,
			     const float TIME_LENGTH,
			     const T& sigma,
			     std::vector< whiteice::math::vertex<T> >& data,
			     std::vector<T>& times,
			     const std::vector< whiteice::math::vertex<T> >& parameters,
			     const std::vector<T>& parameter_times,
			     const unsigned int HISTORY_LEN = 0);

// fits simulated data points to correct_times values
template <typename T = math::blas_real<float> > 
  bool simulate_diffeq_model2(const whiteice::nnetwork<T>& diffeq,
			      const whiteice::math::vertex<T>& start,
			      const float TIME_LENGTH,
			      const T& sigma,
			      std::vector< whiteice::math::vertex<T> >& data,
			      const std::vector< whiteice::math::vertex<T> >& parameters,
			      const std::vector<T>& correct_times,
			      const unsigned int HISTORY_LEN = 0);


// simulate diff.eq. model in SEQUENCE_LENGTH long periods in parallel (OpenMP)
template <typename T = math::blas_real<float> > 
bool simulate_diffeq_model3(const whiteice::nnetwork<T>& diffeq,
			    const whiteice::math::vertex<T>& start,
			    const float TIME_LENGTH,
			    const T& sigma,
			    std::vector< whiteice::math::vertex<T> >& data,
			    const std::vector< whiteice::math::vertex<T> >& parameters,
			    const std::vector<T>& correct_times,
			    const whiteice::dataset<T>& correct_data,
			    const unsigned int SEQUENCE_LENGTH,
			    const unsigned int HISTORY_LEN = 0);


template <typename T = math::blas_real<float> > 
bool simulate_diffeq_model_nn_gradient(const whiteice::nnetwork<T>& diffeq,
				       const whiteice::math::vertex<T>& start,
				       const std::vector< whiteice::math::vertex<T> >& xdata,
				       const std::vector< whiteice::math::vertex<T> >& deltas,
				       const std::map<T, unsigned int>& delta_times,
				       std::vector< whiteice::math::vertex<T> >& data,
				       std::vector<T>& times);

// assumes times are are ordered from smallest to biggest
template <typename T = math::blas_real<float> >
bool simulate_diffeq_model_nn_gradient2(const whiteice::nnetwork<T>& diffeq,
					const whiteice::math::vertex<T>& start,
					const std::vector< whiteice::math::vertex<T> >& xdata,
					const std::vector< whiteice::math::vertex<T> >& deltas,
					const std::map<T, unsigned int>& delta_times,
					std::vector< whiteice::math::vertex<T> >& data,
					const std::vector<T>& correct_times);


// uses hamiltonian monte carlo to fit diffeq parameters to (data, times)
// Samples HMC_SAMPLES samples and selects the best parameter w solution from sampled values (max probability)
template <typename T = math::blas_real<float> > 
bool fit_diffeq_to_data_hmc(whiteice::nnetwork<T>& diffeq,
			    const std::vector< whiteice::math::vertex<T> >& data,
			    const std::vector< T >& times,
			    const whiteice::math::vertex<T>& start_point,
			    const unsigned int HMC_SAMPLES);

template <typename T = math::blas_real<float> > 
bool fit_diffeq_to_data_hmc2(whiteice::bayesian_nnetwork<T>& diffeq,
			     //whiteice::nnetwork<T>& diffeq,
			     const std::vector< whiteice::math::vertex<T> >& data,
			     const std::vector<T>& times,
			     const unsigned int HMC_SAMPLES);


// template <typename T = math::blas_real<float> >

extern template bool simulate_diffeq_model< math::blas_real<float> >
(const whiteice::nnetwork< math::blas_real<float> >& diffeq,
 const whiteice::math::vertex< math::blas_real<float> >& start,
 const float TIME_LENGTH,
 const math::blas_real<float>& sigma,
 std::vector< whiteice::math::vertex< math::blas_real<float> > >& data,
 std::vector< math::blas_real<float> >& times,
 const std::vector< whiteice::math::vertex< math::blas_real<float> > >& parameters,
 const std::vector< math::blas_real<float> >& parameter_times,
 const unsigned int HISTORY_LEN);


extern template bool simulate_diffeq_model< math::blas_real<double> >
(const whiteice::nnetwork< math::blas_real<double> >& diffeq,
 const whiteice::math::vertex< math::blas_real<double> >& start,
 const float TIME_LENGTH,
 const math::blas_real<double>& sigma,
 std::vector< whiteice::math::vertex< math::blas_real<double> > >& data,
 std::vector< math::blas_real<double> >& times,
 const std::vector< whiteice::math::vertex< math::blas_real<double> > >& parameters,
 const std::vector< math::blas_real<double> >& parameter_times,
 const unsigned int HISTORY_LEN);


extern template bool simulate_diffeq_model< math::blas_complex<float> >
(const whiteice::nnetwork< math::blas_complex<float> >& diffeq,
 const whiteice::math::vertex< math::blas_complex<float> >& start,
 const float TIME_LENGTH,
 const math::blas_complex<float>& sigma,
 std::vector< whiteice::math::vertex< math::blas_complex<float> > >& data,
 std::vector< math::blas_complex<float> >& times,
 const std::vector< whiteice::math::vertex< math::blas_complex<float> > >& parameters,
 const std::vector< math::blas_complex<float> >& parameter_times,
 const unsigned int HISTORY_LEN);


extern template bool simulate_diffeq_model< math::blas_complex<double> >
(const whiteice::nnetwork< math::blas_complex<double> >& diffeq,
 const whiteice::math::vertex< math::blas_complex<double> >& start,
 const float TIME_LENGTH,
 const math::blas_complex<double>& sigma,
 std::vector< whiteice::math::vertex< math::blas_complex<double> > >& data,
 std::vector< math::blas_complex<double> >& times,
 const std::vector< whiteice::math::vertex< math::blas_complex<double> > >& parameters,
 const std::vector< math::blas_complex<double> >& parameter_times,
 const unsigned int HISTORY_LEN);




// fits simulated data points to correct_times values
// template <typename T = math::blas_real<float> >

extern template bool simulate_diffeq_model2< math::blas_real<float> >
(const whiteice::nnetwork< math::blas_real<float> >& diffeq,
 const whiteice::math::vertex< math::blas_real<float> >& start,
 const float TIME_LENGTH,
 const math::blas_real<float>& sigma,
 std::vector< whiteice::math::vertex< math::blas_real<float> > >& data,
 const std::vector< whiteice::math::vertex< math::blas_real<float> > >& parameters,
 const std::vector< math::blas_real<float> >& correct_times,
 const unsigned int HISTORY_LEN);

extern template bool simulate_diffeq_model2< math::blas_real<double> >
(const whiteice::nnetwork< math::blas_real<double> >& diffeq,
 const whiteice::math::vertex< math::blas_real<double> >& start,
 const float TIME_LENGTH,
 const math::blas_real<double>& sigma,
 std::vector< whiteice::math::vertex< math::blas_real<double> > >& data,
 const std::vector< whiteice::math::vertex< math::blas_real<double> > >& parameters,
 const std::vector< math::blas_real<double> >& correct_times,
 const unsigned int HISTORY_LEN);


extern template bool simulate_diffeq_model2< math::blas_complex<float> >
(const whiteice::nnetwork< math::blas_complex<float> >& diffeq,
 const whiteice::math::vertex< math::blas_complex<float> >& start,
 const float TIME_LENGTH,
 const math::blas_complex<float>& sigma,
 std::vector< whiteice::math::vertex< math::blas_complex<float> > >& data,
 const std::vector< whiteice::math::vertex< math::blas_complex<float> > >& parameters,
 const std::vector< math::blas_complex<float> >& correct_times,
 const unsigned int HISTORY_LEN);

extern template bool simulate_diffeq_model2< math::blas_complex<double> >
(const whiteice::nnetwork< math::blas_complex<double> >& diffeq,
 const whiteice::math::vertex< math::blas_complex<double> >& start,
 const float TIME_LENGTH,
 const math::blas_complex<double>& sigma,
 std::vector< whiteice::math::vertex< math::blas_complex<double> > >& data,
 const std::vector< whiteice::math::vertex< math::blas_complex<double> > >& parameters,
 const std::vector< math::blas_complex<double> >& correct_times,
 const unsigned int HISTORY_LEN);




extern template bool simulate_diffeq_model3< math::blas_real<float> >
(const whiteice::nnetwork< math::blas_real<float> >& diffeq,
 const whiteice::math::vertex< math::blas_real<float> >& start,
 const float TIME_LENGTH,
 const math::blas_real<float>& sigma,
 std::vector< whiteice::math::vertex< math::blas_real<float> > >& data,
 const std::vector< whiteice::math::vertex< math::blas_real<float> > >& parameters,
 const std::vector< math::blas_real<float> >& correct_times,
 const whiteice::dataset< math::blas_real<float> >& correct_data,
 const unsigned int SEQUENCE_LENGTH,
 const unsigned int HISTORY_LEN);

extern template bool simulate_diffeq_model3< math::blas_real<double> >
(const whiteice::nnetwork< math::blas_real<double> >& diffeq,
 const whiteice::math::vertex< math::blas_real<double> >& start,
 const float TIME_LENGTH,
 const math::blas_real<double>& sigma,
 std::vector< whiteice::math::vertex< math::blas_real<double> > >& data,
 const std::vector< whiteice::math::vertex< math::blas_real<double> > >& parameters,
 const std::vector< math::blas_real<double> >& correct_times,
 const whiteice::dataset< math::blas_real<double> >& correct_data,
 const unsigned int SEQUENCE_LENGTH,
 const unsigned int HISTORY_LEN);



extern template bool simulate_diffeq_model3< math::blas_complex<float> >
(const whiteice::nnetwork< math::blas_complex<float> >& diffeq,
 const whiteice::math::vertex< math::blas_complex<float> >& start,
 const float TIME_LENGTH,
 const math::blas_complex<float>& sigma,
 std::vector< whiteice::math::vertex< math::blas_complex<float> > >& data,
 const std::vector< whiteice::math::vertex< math::blas_complex<float> > >& parameters,
 const std::vector< math::blas_complex<float> >& correct_times,
 const whiteice::dataset< math::blas_complex<float> >& correct_data,
 const unsigned int SEQUENCE_LENGTH,
 const unsigned int HISTORY_LEN);

extern template bool simulate_diffeq_model3< math::blas_complex<double> >
(const whiteice::nnetwork< math::blas_complex<double> >& diffeq,
 const whiteice::math::vertex< math::blas_complex<double> >& start,
 const float TIME_LENGTH,
 const math::blas_complex<double>& sigma,
 std::vector< whiteice::math::vertex< math::blas_complex<double> > >& data,
 const std::vector< whiteice::math::vertex< math::blas_complex<double> > >& parameters,
 const std::vector< math::blas_complex<double> >& correct_times,
 const whiteice::dataset< math::blas_complex<double> >& correct_data,
 const unsigned int SEQUENCE_LENGTH,
 const unsigned int HISTORY_LEN);


//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////


extern template bool simulate_diffeq_model_nn_gradient
(const whiteice::nnetwork< math::blas_real<float> >& diffeq,
 const whiteice::math::vertex< math::blas_real<float> >& start,
 const std::vector< whiteice::math::vertex< math::blas_real<float> > >& xdata,
 const std::vector< whiteice::math::vertex< math::blas_real<float> > >& deltas,
 const std::map< math::blas_real<float>, unsigned int>& delta_times,
 std::vector< whiteice::math::vertex< math::blas_real<float> > >& data,
 std::vector< math::blas_real<float> >& times);

extern template bool simulate_diffeq_model_nn_gradient
(const whiteice::nnetwork< math::blas_real<double> >& diffeq,
 const whiteice::math::vertex< math::blas_real<double> >& start,
 const std::vector< whiteice::math::vertex< math::blas_real<double> > >& xdata,
 const std::vector< whiteice::math::vertex< math::blas_real<double> > >& deltas,
 const std::map< math::blas_real<double>, unsigned int>& delta_times,
 std::vector< whiteice::math::vertex< math::blas_real<double> > >& data,
 std::vector< math::blas_real<double> >& times);


// assumes times are are ordered from smallest to biggest
extern template bool simulate_diffeq_model_nn_gradient2
(const whiteice::nnetwork< math::blas_real<float> >& diffeq,
 const whiteice::math::vertex< math::blas_real<float> >& start,
 const std::vector< whiteice::math::vertex< math::blas_real<float> > >& xdata,
 const std::vector< whiteice::math::vertex< math::blas_real<float> > >& deltas,
 const std::map< math::blas_real<float>, unsigned int>& delta_times,
 std::vector< whiteice::math::vertex< math::blas_real<float> > >& data,
 const std::vector< math::blas_real<float> >& correct_times);

extern template bool simulate_diffeq_model_nn_gradient2
(const whiteice::nnetwork< math::blas_real<double> >& diffeq,
 const whiteice::math::vertex< math::blas_real<double> >& start,
 const std::vector< whiteice::math::vertex< math::blas_real<double> > >& xdata,
 const std::vector< whiteice::math::vertex< math::blas_real<double> > >& deltas,
 const std::map< math::blas_real<double>, unsigned int>& delta_times,
 std::vector< whiteice::math::vertex< math::blas_real<double> > >& data,
 const std::vector< math::blas_real<double> >& correct_times);



extern template bool fit_diffeq_to_data_hmc< math::blas_real<float> >
(whiteice::nnetwork< math::blas_real<float> >& diffeq,
 const std::vector< whiteice::math::vertex< math::blas_real<float> > >& data,
 const std::vector< math::blas_real<float> >& times,
 const whiteice::math::vertex< math::blas_real<float> >& start_point,
 const unsigned int HMC_SAMPLES);


extern template bool fit_diffeq_to_data_hmc< math::blas_real<double> >
(whiteice::nnetwork< math::blas_real<double> >& diffeq,
 const std::vector< whiteice::math::vertex< math::blas_real<double> > >& data,
 const std::vector< math::blas_real<double> >& times,
 const whiteice::math::vertex< math::blas_real<double> >& start_point,
 const unsigned int HMC_SAMPLES);


extern template bool fit_diffeq_to_data_hmc2< math::blas_real<float> >
(whiteice::bayesian_nnetwork< math::blas_real<float> >& diffeq,
 //whiteice::nnetwork< math::blas_real<float> >& diffeq,
 const std::vector< whiteice::math::vertex< math::blas_real<float> > >& data,
 const std::vector< math::blas_real<float> >& times,
 const unsigned int HMC_SAMPLES);

extern template bool fit_diffeq_to_data_hmc2< math::blas_real<double> >
(whiteice::bayesian_nnetwork< math::blas_real<double> >& diffeq,
 const std::vector< whiteice::math::vertex< math::blas_real<double> > >& data,
 const std::vector< math::blas_real<double> >& times,
 const unsigned int HMC_SAMPLES);
    



#endif
