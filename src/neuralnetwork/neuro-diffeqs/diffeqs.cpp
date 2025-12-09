

#include "diffeqs.h"
#include "HMC_diffeq.h"

#include "DiffEQ_HMC.h"

#include "vertex.h"
#include "matrix.h"
#include "nnetwork.h"
#include "bayesian_nnetwork.h"
#include "RungeKutta.h"
#include "RungeKutta_history.h"
#include "RungeKuttaP_history.h" // with control parameters p(t) to affect diff.eq. system

#include <vector>


using namespace whiteice;


bool create_random_diffeq_model(whiteice::nnetwork<>& diffeq,
				const unsigned int DIMENSIONS)
{
  if(DIMENSIONS <= 0) return false;

  std::vector<unsigned int> arch; // 11 layers, NOW: 2 layers

  const unsigned int width = 20;

  arch.push_back(DIMENSIONS);
  arch.push_back(width);
  /*
  arch.push_back(width);
  arch.push_back(width);
  arch.push_back(width);
  arch.push_back(width);
  arch.push_back(width);
  arch.push_back(width);
  arch.push_back(width);
  arch.push_back(width);
  arch.push_back(width);
  */
  arch.push_back(DIMENSIONS); 
  if(diffeq.setArchitecture(arch) == false) return false;

  diffeq.randomize();
  diffeq.setResidual(true);
  // diffeq.setDropOut(1.00f);

  return true;
}


template <typename T>
class nnet_ode : public whiteice::math::odefunction<T>
{
public:

  nnet_ode(const whiteice::nnetwork<T>& nnet)
  {
    this->nnet = nnet;
  }

  // returns number of input dimensions
  unsigned int dimensions() const PURE_FUNCTION
  {
    return nnet.getInputs(0);
  }

  

  // calculates value of function
  whiteice::math::vertex<T> operator()
  (const whiteice::math::odeparam<T>& x)
    const PURE_FUNCTION
  {
    whiteice::math::vertex<T> output;

    if(nnet.calculate(x.y, output) == false) return output;

    return output;
  }
  
  // calculates value
  whiteice::math::vertex<T> calculate
  (const whiteice::math::odeparam<T>& x)
    const PURE_FUNCTION
  {
    whiteice::math::vertex<T> output;

    if(nnet.calculate(x.y, output) == false) return output;
    
    return output;
  }
  
  // calculates value 
  // (optimized version, this is faster because output value isn't copied)
  void calculate
  (const whiteice::math::odeparam<T>& x,
   whiteice::math::vertex<T>& y) const
  {
    whiteice::math::vertex<T> output;
    
    nnet.calculate(x.y, output);
    
    y = output;
  }
  
  // creates copy of object
  function< whiteice::math::odeparam<T>,
	    whiteice::math::vertex<T> >* clone() const
  {
    return new nnet_ode<T>(nnet);
  }

private:

  whiteice::nnetwork<T> nnet; 
};


template <typename T>
bool simulate_diffeq_model(const whiteice::nnetwork<T>& diffeq,
			   const whiteice::math::vertex<T>& start,
			   const float TIME_LENGTH,
			   const T& sigma,
			   std::vector< whiteice::math::vertex<T> >& data,
			   std::vector<T>& times,
			   const std::vector< whiteice::math::vertex<T> >& parameters,
			   const std::vector<T>& parameter_times,
			   const unsigned int HISTORY_LEN)
{
  if((start.size()*(1+HISTORY_LEN)) != diffeq.getInputs(0)) return false;
  if(start.size() != diffeq.getNeurons(diffeq.getLayers()-1)) return false;
  if(TIME_LENGTH < 0.0f) return false;
  if(TIME_LENGTH > 1e6f) return false; // too long simulation length [sanity check]
  
  data.clear();
  times.clear();

  // now uses Runge-Kutta to simulate/integrate diff.eq. dx/dt = diffeq(x)

  const float start_time = 0.0f;
  
  nnet_ode< T > ode(diffeq);

  // assumes that neural network input dimensions have proper dimensions
    
  assert(diffeq.input_size() == (start.size()*(1+HISTORY_LEN)));

  bool hasParams = false;

  if(parameters.size() > 0){
    if(parameters[0].size() > 0){
      hasParams = true;
    }
  }

  if(hasParams == false){
  
    whiteice::math::RungeKutta_history< T > rk(&ode);
    
    rk.setHistoryLength(HISTORY_LEN);
    rk.setSigma(sigma);
    
    rk.calculate(start_time, TIME_LENGTH,
		 start,
		 data,
		 times);

  }
  else{
    
    whiteice::math::RungeKuttaP_history< T > rk(&ode);
    
    rk.setHistoryLength(HISTORY_LEN);
    rk.setSigma(sigma);
    
    rk.calculate(start_time, TIME_LENGTH,
		 start,
		 parameters,
		 parameter_times,
		 data,
		 times);
    
  }

  

  if(data.size() > 0 && times.size() > 0 && data.size() == times.size())
    return true;
  else
    return false;
}


// assumes times are are ordered from smallest to biggest
template <typename T>
bool simulate_diffeq_model2(const whiteice::nnetwork<T>& diffeq,
			    const whiteice::math::vertex<T>& start,
			    const float TIME_LENGTH,
			    const T& sigma,			    
			    std::vector< whiteice::math::vertex<T> >& data,
			    const std::vector< whiteice::math::vertex<T> >& parameters,
			    const std::vector<T>& correct_times,
			    const unsigned int HISTORY_LEN)
{
  std::vector< whiteice::math::vertex<T> > data2;
  std::vector< T > times;

  data.clear();
  
  if(simulate_diffeq_model(diffeq, start, TIME_LENGTH,
			   sigma, data2, times,
			   parameters,
			   correct_times,
			   HISTORY_LEN) == false)
    return false;

  unsigned int kbest = 0;
  
  for(unsigned int i=0;i<correct_times.size();i++){
    
    T best_error = (float)INFINITY;

    // O(n) search, should be converted to O(log(n)) search by using search trees
    for(unsigned int k=kbest;k<times.size();k++){
      auto error = whiteice::math::abs(times[k]-correct_times[i]);
      if(error <= best_error){
	kbest = k;
	best_error = error;
      }
      // else break;
    }

    data.push_back(data2[kbest]);
  }

  return true; 
}


//////////////////////////////////////////////////////////////////////


// simulate diff.eq. model in SEQUENCE_LENGTH long periods in parallel (OpenMP)
template <typename T> 
bool simulate_diffeq_model3(const whiteice::nnetwork<T>& diffeq,
			    const whiteice::math::vertex<T>& start,
			    const float TIME_LENGTH,
			    const T& sigma,
			    std::vector< whiteice::math::vertex<T> >& data,
			    const std::vector< whiteice::math::vertex<T> >& parameters,
			    const std::vector<T>& correct_times,
			    const whiteice::dataset<T>& correct_data,
			    const unsigned int SEQUENCE_LENGTH,
			    const unsigned int HISTORY_LEN)
{
  if(SEQUENCE_LENGTH <= 2) return false;
  if(correct_times.size() != correct_data.size(0)) return false;
  
  const unsigned N = correct_data.size(0)/SEQUENCE_LENGTH;
  const unsigned DROPPED = correct_data.size(0) - N*SEQUENCE_LENGTH;
  const float NTIME_LENGTH = ((correct_data.size(0)-DROPPED)/((float)correct_data.size(0)))*TIME_LENGTH;

  data.resize(correct_times.size());
  bool global_ok = true;

  //printf("A\n");

#pragma omp parallel
  {
    std::map<unsigned int, std::vector< whiteice::math::vertex<T> > > results;
    bool ok = true;
    
#pragma omp for nowait
    for(unsigned int n=0;n<N;n++){

      if(ok == false) continue;
      
      std::vector< whiteice::math::vertex<T> > ndata;
      whiteice::math::vertex<T> nstart_point;
      std::vector<T> ncorrect_times;
      std::vector< whiteice::math::vertex<T> > nparams;
      
      const unsigned int nstart = n*SEQUENCE_LENGTH;
      const unsigned int nend = (n+1)*SEQUENCE_LENGTH;

      //printf("B: %d\n", n);
      
      nstart_point = correct_data[nstart];
      
      for(unsigned int ni=nstart;ni<nend;ni++){
	nparams.push_back(parameters[ni]);
	ncorrect_times.push_back(correct_times[ni]);
      }

      //printf("C: %d\n", n);
      
      if(simulate_diffeq_model2(diffeq, nstart_point, NTIME_LENGTH/N, sigma,
				ndata, nparams, ncorrect_times, HISTORY_LEN) == false){
	ok = false;
	continue;
      }
      else{
	//printf("D: %d\n", n);

	results.insert(std::pair< unsigned int,
		       std::vector< whiteice::math::vertex<T> > >(n, ndata));
      }
    }

#pragma omp critical
    {
      if(ok){
	for(const auto& r : results){

	  //printf("AA: %d\n", r.first);
	  
	  for(unsigned int i=0;i<r.second.size();i++){
	    data[r.first*SEQUENCE_LENGTH+i] = r.second[i];
	  }
	}
      }

      if(ok == false) global_ok = false;
    }

    //printf("BB\n");
  }

  // simulates DROPPED elements at the end
  if(DROPPED > 0 && global_ok){
    //printf("BBB: %d %d %d\n", N, SEQUENCE_LENGTH, correct_data.size(0));
    
    whiteice::math::vertex<T> nstart_point = correct_data[N*SEQUENCE_LENGTH];
    std::vector< whiteice::math::vertex<T> > ndata;
    std::vector<T> ncorrect_times;
    std::vector< whiteice::math::vertex<T> > nparams;
      
    const unsigned int nstart = N*SEQUENCE_LENGTH;
    const unsigned int nend = correct_data.size(0);

    //printf("CC\n");
      
    for(unsigned int ni=nstart;ni<nend;ni++){
      nparams.push_back(parameters[ni]);
      ncorrect_times.push_back(correct_times[ni]);
    }

    //printf("DD\n");
        
    if(simulate_diffeq_model2(diffeq, nstart_point, (TIME_LENGTH - NTIME_LENGTH), sigma,
			      ndata, nparams, ncorrect_times, HISTORY_LEN) == false){
      return false;
    }
    else{
      //printf("EE\n");
      
      for(unsigned int i=0;i<ndata.size();i++){
	data[N*SEQUENCE_LENGTH+i] = ndata[i];
      }
    }
  }

  //printf("FF\n");
  
  return global_ok;
}


//////////////////////////////////////////////////////////////////////


template <typename T>
class nnet_gradient_ode : public whiteice::math::odefunction<T>
{public:

  nnet_gradient_ode(const whiteice::nnetwork<T>& nnet_,
		    const std::vector< whiteice::math::vertex<T> >& xdata_,
		    const std::vector< whiteice::math::vertex<T> >& deltas_,
		    //const std::vector<T>& delta_times_,
		    const std::map<T, unsigned int>& delta_times_) :
    nnet(nnet_), xdata(xdata_), deltas(deltas_), delta_times(delta_times_)
  {
    assert(xdata.size() == deltas.size());
    assert(delta_times.size() == xdata.size()); 
  }

  // returns number of input dimensions
  unsigned int dimensions() const PURE_FUNCTION
  {
    return nnet.gradient_size();
  }

  

  // calculates value of function
  whiteice::math::vertex<T> operator()
  (const whiteice::math::odeparam<T>& x)
    const PURE_FUNCTION
  {
    //std::cout << "xdata: " << xdata.size() << std::endl;
    //std::cout << "deltas: " << deltas.size() << std::endl;
    //std::cout << "delta_times: " << delta_times.size() << std::endl;
    
    whiteice::math::vertex<T> output;

    // calculates delta term by linear interpolation

    whiteice::math::vertex<T> delta; // (nnet.output_size());

    auto iter = delta_times.upper_bound(x.t);
    if(iter != delta_times.begin()) iter--;
    unsigned int index = iter->second;
    auto iter2 = iter;
    if(iter != delta_times.end()) iter2++;
    
    if(index == delta_times.size()) index--;
    
    
    if(iter2 != delta_times.end() && (index+1)<deltas.size()){
      auto coef = (x.t - iter->first)/(iter2->first - iter->first);
      delta = deltas[index] + coef*(deltas[index+1] - deltas[index]);
    }
    else delta = deltas[delta_times.size()-1];

    math::vertex<T> xvalue;

    if(iter2 != delta_times.end() && (index+1)<xdata.size()){
      auto coef = (x.t - iter->first)/(iter2->first - iter->first);
      xvalue = xdata[index] + coef*(xdata[index+1] - xdata[index]);
    }
    else xvalue = xdata[delta_times.size()-1];

    
    // now we have delta and x terms, calculate MSE term (no jacobian explicitely computed)
    // we return delta(t)^T*Jacobian(f(x(t), w))

    std::vector< math::vertex<T> > bpdata;
    std::vector< math::vertex<T> > lndata;

    if(nnet.calculate(xvalue, output, bpdata, lndata) == false)
      assert(0);
    
    if(nnet.mse_gradient(delta, bpdata, lndata, output) == false)
      assert(0);

    return output;
  }
  
  // calculates value
  whiteice::math::vertex<T> calculate
  (const whiteice::math::odeparam<T>& x)
    const PURE_FUNCTION
  {
    return (*this)(x);
  }
  
  // calculates value 
  // (optimized version, this is faster because output value isn't copied)
  void calculate
  (const whiteice::math::odeparam<T>& x,
   whiteice::math::vertex<T>& y) const
  {
    y = (*this)(x);
  }
  
  // creates copy of object
  function< whiteice::math::odeparam<T>,
	    whiteice::math::vertex<T> >* clone() const
  {
    return new nnet_gradient_ode<T>(nnet, xdata, deltas, delta_times);
  }

private:

  const whiteice::nnetwork<T>& nnet;
  const std::vector< whiteice::math::vertex<T> >& xdata;
  const std::vector< whiteice::math::vertex<T> >& deltas;
  const std::map<T, unsigned int> delta_times;
  //const std::vector<T>& delta_times;
};


template <typename T>
bool simulate_diffeq_model_nn_gradient(const whiteice::nnetwork<T>& diffeq,
				       const whiteice::math::vertex<T>& start,
				       const std::vector< whiteice::math::vertex<T> >& xdata,
				       const std::vector< whiteice::math::vertex<T> >& deltas,
				       const std::map<T, unsigned int>& delta_times,
				       //const std::vector<T>& delta_times,
				       std::vector< whiteice::math::vertex<T> >& data,
				       std::vector<T>& times)
{
  if(start.size() != diffeq.gradient_size()){
    printf("start size=%d, diffeq.gradient_size()=%d\n",
	   (int)start.size(), (int)diffeq.gradient_size());
    
    assert(0);
    
    return false;
  }

  if(deltas.size() != delta_times.size()) return false;
  if(deltas.size() <= 0) return false;
  

  data.clear();
  times.clear();

  // now uses Runge-Kutta to simulate/integrate diff.eq. d(delta(t)^T*grad(x))/dt = delta(t)^T*grad(diffeq(x))

  float START_TIME = 0.0f;
  float TIME_LENGTH = 0.0f;

  

  whiteice::math::convert(START_TIME, delta_times.begin()->first);
  whiteice::math::convert(TIME_LENGTH, (delta_times.rbegin())->first);

  if(TIME_LENGTH < 0.0f){
    assert(0);
    return false;
  }
  
  if(TIME_LENGTH > 1e6f){
    assert(0);
    return false; // too long simulation length [sanity check]
  }

  //std::cout << "START_TIME: " << START_TIME << std::endl;
  //std::cout << "TIME_LENGTH: " << TIME_LENGTH << std::endl;

  
  nnet_gradient_ode< T > ode(diffeq, xdata, deltas, delta_times);

  whiteice::math::RungeKutta< T > rk(&ode);

  //std::cout << "RungeKutta START" << std::endl;

  rk.calculate(START_TIME, TIME_LENGTH,
	       start,
	       data,
	       times);

  //std::cout << "RungeKutta END" << std::endl;

  if(data.size() > 0 && times.size() > 0 && data.size() == times.size()){
    return true;
  }
  else{
    std::cout << "data.size(): " << data.size() << " times.size():  " << times.size() << std::endl;
    assert(0);
    return false;
  }
}


// assumes times are are ordered from smallest to biggest
template <typename T>
bool simulate_diffeq_model_nn_gradient2(const whiteice::nnetwork<T>& diffeq,
					const whiteice::math::vertex<T>& start,
					const std::vector< whiteice::math::vertex<T> >& xdata,
					const std::vector< whiteice::math::vertex<T> >& deltas,
					const std::map<T, unsigned int>& delta_times,
					// const std::vector<T>& delta_times,
					std::vector< whiteice::math::vertex<T> >& data,
					const std::vector<T>& correct_times)
{
  std::vector< whiteice::math::vertex<T> > data2;
  std::vector< T > times;

  //std::cout << "gradient2: simulate start" << std::endl;
  
  if(simulate_diffeq_model_nn_gradient(diffeq, start, xdata, deltas, delta_times, data2, times) == false)
    return false;

  //std::cout << "gradient2: simulate ok" << std::endl;
  
  data.clear();

  unsigned int kbest = 0;
  
  for(unsigned int i=0;i<correct_times.size();i++){
    
    T best_error = (float)INFINITY;

    // O(n) search, should be converted to O(log(n)) search by using binary trees
    for(unsigned int k=kbest;k<times.size();k++){
      auto error = whiteice::math::abs(times[k]-correct_times[i]);
      if(error <= best_error){
	kbest = k;
	best_error = error;
      }
      //else break;
    }

    data.push_back(data2[kbest]);
  }

  return true; 
}




// uses hamiltonian monte carlo sampler (HMC) to fit diffeq parameters to (data, times)
// Samples HMC_SAMPLES samples and selects the best parameter w solution from sampled values (max probability)
// assumes time starts from zero.
template <typename T>
bool fit_diffeq_to_data_hmc(whiteice::nnetwork<T>& diffeq,
			    const std::vector< whiteice::math::vertex<T> >& data,
			    const std::vector<T>& times,
			    const whiteice::math::vertex<T>& start_point,
			    const unsigned int HMC_SAMPLES)
{
  if(data.size() != times.size()) return false;
  if(HMC_SAMPLES <= 1) return false;
  if(diffeq.getInputs(0) != diffeq.getNeurons(diffeq.getLayers()-1)) return false;
  if(start_point.size() != diffeq.getInputs(0)) return false;
  if(data.size() <= 5) return false; // must have some data


  // TEST: samples initial x observations with given neural network to fit to data times t_i time values
  std::vector< whiteice::math::vertex<T> > xdata;

  auto delta_time = (times[times.size()-1] - times[0]).c[0];

  const T sigma = T(0.0);
  const std::vector< whiteice::math::vertex<T> > parameters; // empty parameter for no control parameters
  
  if(simulate_diffeq_model2(diffeq,
			    start_point,
			    delta_time,
			    sigma,
			    xdata,
			    parameters,
			    times) == false){
    printf("CHECK POINT 2\n");
    return false;
  }

  // setup HMC sampler and samples target number of points

  // TODO: extend HMC to be HMC_diffeq and calculate squared error term using diffeq simulation to get output values.. + insert correct times to HMC for sampler + starting point
  whiteice::dataset<T> ds;

  // create dataset
  ds.createCluster("input from diff.eq. 't-1'", diffeq.getInputs(0));
  ds.createCluster("correct output (to diff.eq.) 't'", diffeq.getInputs(0));

  for(unsigned int i=0;i<xdata.size();i++){
    ds.add(0, xdata[i]);
    ds.add(1, data[i]);
  }

  // whiteice::HMC<T> hmc(diffeq, ds);
  HMC_diffeq<T> hmc(diffeq, ds, start_point, times, true); // DON'T USE ADAPTIVE STEP LENGTH!

  whiteice::linear_ETA<double> eta;

  eta.start(0.0, (double)HMC_SAMPLES);

  hmc.startSampler();

  while(hmc.getNumberOfSamples() <= HMC_SAMPLES){
    sleep(1);
    auto error = hmc.getMeanError(10);

    eta.update((double)hmc.getNumberOfSamples());

    std::cout << "HMC sampler error: " << error << ". ";
    std::cout << "HMC sampler samples (0 means no samples yet): " << hmc.getNumberOfSamples() << ". ";
    std::cout << "ETA: " << eta.estimate()/60.0f << " minute(s)." <<  std::endl;
  }

  hmc.stopSampler();

  auto wbest = hmc.getMean(); // FIXME: select minimum error weight

  if(diffeq.importdata(wbest) == false) return false;

  return true;
}


// uses hamiltonian monte carlo sampler (HMC) to fit diffeq parameters to (data, times)
// Samples HMC_SAMPLES samples and selects the best parameter w solution from sampled values (max probability)
// assumes time starts from zero.
template <typename T>
bool fit_diffeq_to_data_hmc2(whiteice::bayesian_nnetwork<T>& bdiffeq,
			     const std::vector< whiteice::math::vertex<T> >& data,
			     const std::vector<T>& times,
			     const unsigned int HMC_SAMPLES)
{
  if(data.size() != times.size()) return false;
  if(HMC_SAMPLES <= 1) return false;
  if(bdiffeq.getNetwork().getInputs(0) !=
     bdiffeq.getNetwork().getNeurons(bdiffeq.getNetwork().getLayers()-1)) return false;
  if(data.size() <= 5) return false; // must have some data
  if(data[0].size() != bdiffeq.getNetwork().getInputs(0)) return false;

  // construct dataset for HMC which used for training
  // we use samples from time-series data (data, times)
  
  whiteice::dataset<T> ds;

  // create dataset time-series data points
  const unsigned int TIME_SERIES_LENGTH = 10;
  
  ds.createCluster("time-series for differential equations", TIME_SERIES_LENGTH*data[0].size());
  
  // creates dataset
  for(unsigned int i=0;i<data.size();i+=TIME_SERIES_LENGTH){
    
    math::vertex<T> in;
    in.resize(ds.dimension(0));
    
    for(unsigned int j=0;j<TIME_SERIES_LENGTH;j++){
      in.write_subvertex(data[i+j], j*data[0].size());
    }

    ds.add(0, in);
  }

  // creates times for dataset
  auto delta_time = (times[times.size()-1] - times[0]).c[0];
  auto delta_t = delta_time/times.size();

  std::vector<T> dstimes;
  //std::map<T, unsigned int>& dstimes;
  
  for(unsigned int i=0;i<TIME_SERIES_LENGTH;i++){
    //dstimes.insert(std::pair(i*delta_T, i));
    dstimes.push_back(i*delta_t);
  }

  //HMC_diffeq<T> hmc(diffeq, ds, start_point, times, true);

  whiteice::nnetwork<T> diffeq = bdiffeq.getNetwork();
  
  whiteice::DiffEq_HMC<T> hmc(diffeq, ds, dstimes, true, true); // use adaptive step length

  // TEST THIS: ENABLE!!
  // hmc.setTemperature(0.001); // no wandering around in problem surface

  whiteice::linear_ETA<double> eta;

  eta.start(0.0, (double)HMC_SAMPLES);

  hmc.startSampler();

  while(hmc.getNumberOfSamples() <= HMC_SAMPLES){
    sleep(1); // was: 5  
    auto error = hmc.getMeanError(1)/data.size(); // was: 10

    eta.update((double)hmc.getNumberOfSamples());

    std::cout << "HMC sampler error: " << error << ". ";
    std::cout << "HMC sampler samples (0 means no samples yet): " << hmc.getNumberOfSamples() << ". ";
    std::cout << "ETA: " << eta.estimate()/60.0f << " minute(s)." <<  std::endl;
  }

  hmc.stopSampler();

  std::vector< math::vertex<T> > samples;

  hmc.getSamples(samples);

  bdiffeq.importSamples(diffeq, samples);

#if 0
  math::vertex<T> wbest;
  bool first = true;

  for(unsigned int i=samples.size()/2;i<samples.size();i++){
    if(first) wbest = samples[i];
    else wbest += samples[i];
  }

  wbest /= (samples.size()/2);

  // auto wbest = hmc.getMean(); // FIXME: select minimum error weight

  if(diffeq.importdata(wbest) == false) return false;
#endif

  return true;
}



template bool simulate_diffeq_model< math::blas_real<float> >
(const whiteice::nnetwork< math::blas_real<float> >& diffeq,
 const whiteice::math::vertex< math::blas_real<float> >& start,
 const float TIME_LENGTH,
 const math::blas_real<float>& sigma,
 std::vector< whiteice::math::vertex< math::blas_real<float> > >& data,
 std::vector< whiteice::math::blas_real<float> >& times,
 const std::vector< whiteice::math::vertex< math::blas_real<float> > >& parameters,
 const std::vector< math::blas_real<float> >& parameter_times,
 const unsigned int HISTORY_LEN);


template bool simulate_diffeq_model< math::blas_real<double> >
(const whiteice::nnetwork< math::blas_real<double> >& diffeq,
 const whiteice::math::vertex< math::blas_real<double> >& start,
 const float TIME_LENGTH,
 const math::blas_real<double>& sigma,
 std::vector< whiteice::math::vertex< math::blas_real<double> > >& data,
 std::vector< whiteice::math::blas_real<double> >& times,
 const std::vector< whiteice::math::vertex< math::blas_real<double> > >& parameters,
 const std::vector< math::blas_real<double> >& parameter_times,
 const unsigned int HISTORY_LEN);


// fits simulated data points to correct_times values
// template <typename T = math::blas_real<float> >
template bool simulate_diffeq_model2< math::blas_real<float> >
(const whiteice::nnetwork< math::blas_real<float> >& diffeq,
 const whiteice::math::vertex< math::blas_real<float> >& start,
 const float TIME_LENGTH,
 const math::blas_real<float>& sigma,
 std::vector< whiteice::math::vertex< math::blas_real<float> > >& data,
 const std::vector< whiteice::math::vertex< math::blas_real<float> > >& parameters,
 const std::vector< math::blas_real<float> >& correct_times,
 const unsigned int HISTORY_LEN);

template bool simulate_diffeq_model2< math::blas_real<double> >
(const whiteice::nnetwork< math::blas_real<double> >& diffeq,
 const whiteice::math::vertex< math::blas_real<double> >& start,
 const float TIME_LENGTH,
 const math::blas_real<double>& sigma,
 std::vector< whiteice::math::vertex< math::blas_real<double> > >& data,
 const std::vector< whiteice::math::vertex< math::blas_real<double> > >& parameters,
 const std::vector< math::blas_real<double> >& correct_times,
 const unsigned int HISTORY_LEN);


template bool simulate_diffeq_model3< math::blas_real<float> >
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

template bool simulate_diffeq_model3< math::blas_real<double> >
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



template bool simulate_diffeq_model_nn_gradient
(const whiteice::nnetwork< math::blas_real<float> >& diffeq,
 const whiteice::math::vertex< math::blas_real<float> >& start,
 const std::vector< whiteice::math::vertex< math::blas_real<float> > >& xdata,
 const std::vector< whiteice::math::vertex< math::blas_real<float> > >& deltas,
 const std::map< math::blas_real<float>, unsigned int>& delta_times,
 std::vector< whiteice::math::vertex< math::blas_real<float> > >& data,
 std::vector< math::blas_real<float> >& times);

template bool simulate_diffeq_model_nn_gradient
(const whiteice::nnetwork< math::blas_real<double> >& diffeq,
 const whiteice::math::vertex< math::blas_real<double> >& start,
 const std::vector< whiteice::math::vertex< math::blas_real<double> > >& xdata,
 const std::vector< whiteice::math::vertex< math::blas_real<double> > >& deltas,
 const std::map< math::blas_real<double>, unsigned int>& delta_times,
 std::vector< whiteice::math::vertex< math::blas_real<double> > >& data,
 std::vector< math::blas_real<double> >& times);


// assumes times are are ordered from smallest to biggest
template bool simulate_diffeq_model_nn_gradient2
(const whiteice::nnetwork< math::blas_real<float> >& diffeq,
 const whiteice::math::vertex< math::blas_real<float> >& start,
 const std::vector< whiteice::math::vertex< math::blas_real<float> > >& xdata,
 const std::vector< whiteice::math::vertex< math::blas_real<float> > >& deltas,
 const std::map< math::blas_real<float>, unsigned int>& delta_times,
 std::vector< whiteice::math::vertex< math::blas_real<float> > >& data,
 const std::vector< math::blas_real<float> >& correct_times);

template bool simulate_diffeq_model_nn_gradient2
(const whiteice::nnetwork< math::blas_real<double> >& diffeq,
 const whiteice::math::vertex< math::blas_real<double> >& start,
 const std::vector< whiteice::math::vertex< math::blas_real<double> > >& xdata,
 const std::vector< whiteice::math::vertex< math::blas_real<double> > >& deltas,
 const std::map< math::blas_real<double>, unsigned int>& delta_times,
 std::vector< whiteice::math::vertex< math::blas_real<double> > >& data,
 const std::vector< math::blas_real<double> >& correct_times);




template bool fit_diffeq_to_data_hmc< math::blas_real<float> >
(whiteice::nnetwork< math::blas_real<float> >& diffeq,
 const std::vector< whiteice::math::vertex< math::blas_real<float> > >& data,
 const std::vector< math::blas_real<float> >& times,
 const whiteice::math::vertex< math::blas_real<float> >& start_point,
 const unsigned int HMC_SAMPLES);


template bool fit_diffeq_to_data_hmc< math::blas_real<double> >
(whiteice::nnetwork< math::blas_real<double> >& diffeq,
 const std::vector< whiteice::math::vertex< math::blas_real<double> > >& data,
 const std::vector< math::blas_real<double> >& times,
 const whiteice::math::vertex< math::blas_real<double> >& start_point,
 const unsigned int HMC_SAMPLES);


template bool fit_diffeq_to_data_hmc2< math::blas_real<float> >
(whiteice::bayesian_nnetwork< math::blas_real<float> >& diffeq,
 const std::vector< whiteice::math::vertex< math::blas_real<float> > >& data,
 const std::vector< math::blas_real<float> >& times,
 const unsigned int HMC_SAMPLES);

template bool fit_diffeq_to_data_hmc2< math::blas_real<double> >
(whiteice::bayesian_nnetwork< math::blas_real<double> >& diffeq,
 const std::vector< whiteice::math::vertex< math::blas_real<double> > >& data,
 const std::vector< math::blas_real<double> >& times,
 const unsigned int HMC_SAMPLES);
