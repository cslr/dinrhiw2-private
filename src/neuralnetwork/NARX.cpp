
#include "NARX.h"


namespace whiteice
{
  
  template <typename T>
  NARX<T>::NARX()
  {
  }

  
  template <typename T>
  NARX<T>::~NARX()
  {
    std::unique_lock<std::mutex> lock(compute_mutex);
    
    sgd.stopComputation();
    
  }
  
  
  // computes solution directly without thread, modern computers should be fast enough
  template <typename T>
  bool NARX<T>::startOptimization(const whiteice::nnetwork<T>& net,
				  const whiteice::dataset<T>& xdata,
				  const whiteice::dataset<T>& pdata,
				  const unsigned int HISTLEN,
				  const unsigned int FUTURELEN)
  {
    std::lock_guard<std::mutex> lock(compute_mutex);

    if(xdata.size(0) < FUTURELEN || pdata.size(0) < FUTURELEN || HISTLEN == 0)
      return false;

    if(FUTURELEN == 0)
      return false;

    if(xdata.size(0) != pdata.size(0)) return false;

    if(sgd.isRunning()) return false;


    this->HISTLEN = HISTLEN;
    this->FUTURELEN = FUTURELEN;
    this->xdata = xdata;
    this->pdata = pdata;
    this->net = net;

    this->xdata.clearData(); // removes data but keeps preprocessing information
    this->pdata.clearData();
    

    // creates dataset that neural network tries to learn (predicts only the next step)

    whiteice::dataset<T> data;
    
    data.createCluster("history data", (xdata.dimension(0)+pdata.dimension(0))*HISTLEN + (FUTURELEN-1)*pdata.dimension(0));
    data.createCluster("output data", xdata.dimension(0));
    
    for(unsigned int i=0;i<xdata.size(0)-FUTURELEN;i++){

      whiteice::math::vertex<T> v;
      
      v.resize((xdata.dimension(0)+pdata.dimension(0))*HISTLEN + (FUTURELEN-1)*pdata.dimension(0));
      v.zero();
      
      for(unsigned int h=0;h<HISTLEN;h++){

	if(h <= i){
	  const auto& x = xdata.access(0,i-h);
	  
	  for(unsigned int k=0;k<xdata.dimension(0);k++){
	    v[h*(xdata.dimension(0)+pdata.dimension(0)) + k] = x[k];
	  }

	  const auto& p = pdata.access(0,i-h);

	  for(unsigned int k=0;k<pdata.dimension(0);k++){
	    v[h*(xdata.dimension(0)+pdata.dimension(0)) + xdata.dimension(0) + k] = p[k];
	  }
	}
	
      }

      for(unsigned int f=0;f<(FUTURELEN-1);f++){
	const auto& p = pdata.access(0,i+f);
	
	for(unsigned int k=0;k<pdata.dimension(0);k++){
	  v[(xdata.dimension(0)+pdata.dimension(0))*HISTLEN + f*pdata.dimension(0) + k] = p[k];
	}
      }

      data.add(0, v);
      data.add(1, xdata.access(0, i+FUTURELEN));
    }
    

    {
      sgd.setUseMinibatch(false);
      sgd.setOverfit(true);
      sgd.setMNE(true); // norm error E{|correct-predicted|}
      
      if(sgd.startOptimize(data, this->net, 1) == false){
	printf("NARX: SGD neural net optimization start FAILED.\n");
	return false;
      }
      else return true;
    }

    
  }

  
  template <typename T>
  bool NARX<T>::isRunning() const
  {
    std::lock_guard<std::mutex> lock(compute_mutex);

    whiteice::nnetwork<T> nn;
    T solution_error;
    unsigned int converged = 0;

    if(sgd.getSolution(nn, solution_error, converged) == true){
      this->net = nn;
    }
    
    return sgd.isRunning();
  }

  
  template <typename T>
  bool NARX<T>::getSolution(whiteice::math::vertex<T>& params,
			    T& solution_error)
  {
    std::lock_guard<std::mutex> lock(compute_mutex);

    whiteice::nnetwork<T> nn;

    unsigned int converged = 0;

    if(sgd.getSolution(nn, solution_error, converged) == false)
      return false;

    this->net = nn;

    nn.exportdata(params);

    return true;
  }


  template <typename T>
  bool NARX<T>::stopOptimization()
  {
    std::unique_lock<std::mutex> lock(compute_mutex);

    sgd.stopComputation();

    whiteice::nnetwork<T> nn;
    T solution_error;
    unsigned int converged = 0;

    if(sgd.getSolution(nn, solution_error, converged) == true){
      this->net = nn;
    }

    if(sgd.isRunning() == false) return true;
    else return false;
  }

  
  template <typename T>
  bool NARX<T>::predict
  (const std::vector< whiteice::math::vertex<T> >& x, // HISTLEN x(t-HISTLEN-1)..x(t) VECTOR ELEMENTS
   const std::vector< whiteice::math::vertex<T> >& p, // HISTLEN p(t-HISTLEN-1)..p(t) VECTOR ELEMENTS
   const std::vector< whiteice::math::vertex<T> >& p_future, // FUTURELEN p(t+1)..p(t+FUTURELEN) [FUTURELEN-1 elements]
   whiteice::math::vertex<T>& y) const // predicted vector y=x(t+FUTURELEN)
  {
    std::unique_lock<std::mutex> lock(compute_mutex);

    if(x.size() != HISTLEN || p.size() != HISTLEN) return false;
    if(p_future.size() != FUTURELEN-1) return false;

    // creates input vector and preprocesses input data with dataset preprocessings
    whiteice::math::vertex<T> v;

    {
      v.resize((xdata.dimension(0)+pdata.dimension(0))*HISTLEN +
	       (FUTURELEN-1)*pdata.dimension(0));
      v.zero();
      
      for(unsigned int h=0;h<HISTLEN;h++){

	{
	  auto xi = x[HISTLEN-1-h];

	  xdata.preprocess(0, xi);
	  
	  for(unsigned int k=0;k<xdata.dimension(0);k++){
	    v[h*(xdata.dimension(0)+pdata.dimension(0)) + k] = xi[k];
	  }

	  auto pi = p[HISTLEN-1-h];

	  pdata.preprocess(0, pi);

	  for(unsigned int k=0;k<pdata.dimension(0);k++){
	    v[h*(xdata.dimension(0)+pdata.dimension(0)) + xdata.dimension(0) + k] = pi[k];
	  }
	}
	
      }


      for(unsigned int f=0;f<(FUTURELEN-1);f++){
	auto pi = p_future[f];
	
	pdata.preprocess(0, pi);
	
	for(unsigned int k=0;k<pdata.dimension(0);k++){
	  v[(xdata.dimension(0)+pdata.dimension(0))*HISTLEN +
	    f*pdata.dimension(0) + k] = pi[k];
	}
      }

      
    }

    
    // predicts x(t+1)

    if(net.calculate(v, y) == false) return false;
    
    xdata.invpreprocess(0, y); // removes preprocessing from predicted x(t+1)

    return true;
  }
  
  
  template class NARX< whiteice::math::blas_real<float> >;
  template class NARX< whiteice::math::blas_real<double> >;
};
