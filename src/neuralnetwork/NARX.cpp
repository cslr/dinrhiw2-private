// FIXME: join() blocks in pca()/ica() calculations which take time so shutdown is SLOW.
// TODO: should check inside pca() and ica() and symmetric_eig() if flag to STOP has been set and
//       return immediately. (lots of stupid code)


#include "NARX.h"

#ifdef WINOS
#include <windows.h>
#endif

namespace whiteice
{
  
  template <typename T>
  NARX<T>::NARX()
  {
    optimize_running = false;
  }

  
  template <typename T>
  NARX<T>::~NARX()
  {
    std::unique_lock<std::mutex> lock(compute_mutex);
    
    optimize_running = false;
    
    if(optimize_thread){
      if(optimize_thread->joinable())
	optimize_thread->join();
      
      delete optimize_thread;
      optimize_thread = nullptr;
    }

    sgd.stopComputation();
  }
  
  
  // computes solution directly without thread, modern computers should be fast enough
  template <typename T>
  bool NARX<T>::startOptimization(const whiteice::nnetwork<T>& net,
				  const whiteice::dataset<T>& xdata,
				  const whiteice::dataset<T>& pdata,
				  const std::vector<unsigned int>& pred_indexes,
				  const unsigned int HISTLEN,
				  const unsigned int FUTURELEN,
				  const unsigned int REDUCED_DIM)
  {
    std::lock_guard<std::mutex> lock(compute_mutex);

    if(xdata.size(0) < FUTURELEN || pdata.size(0) < FUTURELEN || HISTLEN == 0)
      return false;

    if(pred_indexes.size() == 0 || pred_indexes.size() > xdata.dimension(0))
      return false;

    for(unsigned int i=0;i<pred_indexes.size();i++)
      if(pred_indexes[i] >= xdata.dimension(0))
	return false;

    if(FUTURELEN == 0)
      return false;

    if(xdata.size(0) != pdata.size(0)) return false;

    if(optimize_running) return false;

    if(sgd.isRunning()) return false;

    // dont check net compatibility with data dimensions..

    {
      std::lock_guard<std::mutex> lock(params_mutex);
      
      this->HISTLEN = HISTLEN;
      this->FUTURELEN = FUTURELEN;
      this->REDUCED_DIM = REDUCED_DIM;
      this->xdata = xdata;
      this->pdata = pdata;
      this->pred_indexes = pred_indexes;
      this->net = net;
    }

    // starts optimization thread (PCA+ICA+SGD.start())

    try{
      optimize_running = false;

      if(optimize_thread){
	if(optimize_thread->joinable())
	  optimize_thread->join();
	
	delete optimize_thread;
	optimize_thread = nullptr;
      }
      
      optimize_running = true;
      
      optimize_thread = new std::thread(&NARX<T>::optimize_function, this);
      //optimize_thread->detach();

      return true;
    }
    catch(std::exception& e){
      optimize_running = false;

      if(optimize_thread){
	if(optimize_thread->joinable())
	  optimize_thread->join();
	
	delete optimize_thread;
	optimize_thread = nullptr;
      }
      
      return false;
    }


#if 0
    // creates dataset that neural network tries to learn (predicts only the next step)

    whiteice::dataset<T> data;
    
    data.createCluster("history data", (xdata.dimension(0)+pdata.dimension(0))*HISTLEN + (FUTURELEN-1)*pdata.dimension(0));
    data.createCluster("output data", pred_indexes.size());
    
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

      whiteice::math::vertex<T> out;      
      out.resize(pred_indexes.size());

      const auto w = xdata.access(0, i+FUTURELEN);

      for(unsigned int j=0;j<pred_indexes.size();j++){
	out[j] = w[pred_indexes[j]];
      }

      
      data.add(1, out);
    }
    

    {
      sgd.setUseMinibatch(true); // was: false
      sgd.setOverfit(false); // don't overfit data (better with real data)
      sgd.setMNE(true); // norm error E{|correct-predicted|}

      // initially use NN
      if(sgd.startOptimize(data, this->net, 1, 0xFFFFFFFF, false , true , false) == false){
	printf("NARX: SGD neural net optimization start FAILED.\n");
	return false;
      }
      else return true;
    }
#endif
    
  }

  
  template <typename T>
  bool NARX<T>::isRunning() const
  {
    if(optimize_running) return true;
    
    std::lock_guard<std::mutex> lock(compute_mutex);

    whiteice::nnetwork<T> nn;
    T solution_error;
    unsigned int converged = 0;

    if(sgd.getSolution(nn, solution_error, converged) == true){
      std::lock_guard<std::mutex> lock(params_mutex);
      this->net = nn;
    }
    
    return sgd.isRunning();
  }

  
  template <typename T>
  bool NARX<T>::getSolution(whiteice::math::vertex<T>& params,
			    T& solution_error)
  {
    // if(optimize_running) return false; // THIS IS WRONG???
    
    std::lock_guard<std::mutex> lock(compute_mutex);

    whiteice::nnetwork<T> nn;

    unsigned int converged = 0;

    if(sgd.getSolution(nn, solution_error, converged) == false)
      return false;

    {
      std::lock_guard<std::mutex> lock(params_mutex);
      this->net = nn;
    }

    nn.exportdata(params);

    return true;
  }


  template <typename T>
  bool NARX<T>::stopOptimization()
  {
    std::unique_lock<std::mutex> lock(compute_mutex);

    optimize_running = false;

    if(optimize_thread){
      if(optimize_thread->joinable())
	optimize_thread->join();
      
      delete optimize_thread;
      optimize_thread = nullptr;
    }
    
    optimize_running = false;

    sgd.stopComputation();

    whiteice::nnetwork<T> nn;
    T solution_error;
    unsigned int converged = 0;

    if(sgd.getSolution(nn, solution_error, converged) == true){
      std::lock_guard<std::mutex> lock(params_mutex);
      this->net = nn;
    }

    if(sgd.isRunning() == false) return true;
    else return false;
  }



  template <typename T>
  void NARX<T>::optimize_function()
  {
    // set thread priority (non-standard)
    {
      sched_param sch_params;
      int policy = SCHED_FIFO;
      
      pthread_getschedparam(pthread_self(),
			    &policy, &sch_params);
      
#ifdef linux
      policy = SCHED_IDLE; // in linux we can set idle priority
#endif	
      sch_params.sched_priority = sched_get_priority_min(policy);
      
      if(pthread_setschedparam(pthread_self(),
			       policy, &sch_params) != 0){
	// printf("! SETTING LOW PRIORITY THREAD FAILED\n");
      }
      
#ifdef WINOS
      SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_IDLE);
      // SetPriorityClass(GetCurrentThread(), THREAD_PRIORITY_IDLE);
      
#endif	
    }
    
    

    if(optimize_running == false) return;

    // creates dataset that neural network tries to learn (predicts only the next step)
    
    whiteice::dataset<T> data;

    {
      params_mutex.lock();
      auto xdata = this->xdata;
      params_mutex.unlock();

      params_mutex.lock();
      auto pdata = this->pdata;
      params_mutex.unlock();

      params_mutex.lock();
      auto pred_indexes = this->pred_indexes;
      params_mutex.unlock();

      
      data.createCluster("history data", (xdata.dimension(0)+pdata.dimension(0))*HISTLEN + (FUTURELEN-1)*pdata.dimension(0));
      data.createCluster("output data", pred_indexes.size());
    
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
	
	whiteice::math::vertex<T> out;      
	out.resize(pred_indexes.size());
	
	const auto w = xdata.access(0, i+FUTURELEN);
	
	for(unsigned int j=0;j<pred_indexes.size();j++){
	  out[j] = w[pred_indexes[j]];
	}
	
	
	data.add(1, out);
      }
    }


    std::vector< whiteice::math::vertex<T> > reduced_data;    

    if(optimize_running == false) return;

    if(data.dimension(0) > REDUCED_DIM + (FUTURELEN-1)*pdata.dimension(0)){
      std::vector< whiteice::math::vertex<T> > listdata;

      for(unsigned int i=0;i<data.size(0);i++){
	auto w = data.access(0, i);
	w.resize((xdata.dimension(0)+pdata.dimension(0))*HISTLEN); // only dim. reduces past values (no future params)
	
	listdata.push_back(w);
      }

      // calculates PCA+ICA reduction (historical x values only, don't dimension reduce FUTURE control parameters)
      
      
      whiteice::math::matrix<T> PCA;
      whiteice::math::vertex<T> m;
      
      T orig_var, redu_var;
      
      // This calculates linear ICA from the measurements.
      // FIXME: should calculate convolutional ICA instead with different time-delays to measurement points..

      if(optimize_running == false) return;
      
      if(whiteice::math::pca(listdata, REDUCED_DIM, PCA, m, orig_var, redu_var, true, true) == false){
	return; // cannot compute PCA!!!! (error)
      }
      else{
	
	for(unsigned int i=0;i<listdata.size();i++){
	  listdata[i] = PCA*(listdata[i] - m);
	  
	  for(unsigned int k=0;k<listdata[i].size();k++){
	    if(whiteice::math::isnan(listdata[i][k]) || whiteice::math::isinf(listdata[i][k])){
	      listdata[i][k] = 0.0;
	    }
	  }
	  
	}
      }

      if(optimize_running == false) return;
      
      whiteice::math::matrix<T> ICA;
      
      T ica_tolerance = 0.01f; // converges more quickly..
      const unsigned int ICA_MAXITERS = 100; // stops always quickly [adjust to proper value] (should be 100 for proper results, 15 is useless?), was: 15, now: 50
      
      // FastICA algorithm is still slow for real-time use..
      if(whiteice::math::ica(listdata, ICA, false, ica_tolerance, ICA_MAXITERS) == false){
	// cannot compute ICA, compute PCA only instead;
	std::lock_guard<std::mutex> lock(params_mutex);
	
	ICA_reduce = PCA;
	ICA_mean_reduce = PCA*m;
      }
      else{
	std::lock_guard<std::mutex> lock(params_mutex);
	
	ICA_reduce = ICA*PCA;
	ICA_mean_reduce = ICA*PCA*m;
      }

      if(optimize_running == false) return;

      {
	params_mutex.lock();

	auto ICA = ICA_reduce;
	auto mean = ICA_mean_reduce;
	
	params_mutex.unlock();
	
	for(unsigned int i=0;i<data.size(0);i++){

	  if(optimize_running == false) return;
	  
	  whiteice::math::vertex<T> v;
	  v.resize(REDUCED_DIM + (FUTURELEN-1)*pdata.dimension(0));
	  v.zero();

	  auto w = data.access(0, i);
	  w.resize((xdata.dimension(0)+pdata.dimension(0))*HISTLEN); // only dim. reduces past values (no future params)
	  
	  auto r = ICA*w - mean;

	  // std::cout << "r = " << r << std::endl;

	  const auto& d = data.access(0, i);

	  for(unsigned int j=0;j<REDUCED_DIM;j++)
	    v[j] = r[j];

	  
	  for(unsigned int j=0;j<(FUTURELEN-1)*pdata.dimension(0);j++){
	    v[REDUCED_DIM+j] = d[((xdata.dimension(0)+pdata.dimension(0))*HISTLEN)+j];
	  }

	  reduced_data.push_back(v);
	}
      }
    }
    else{ // NO DIMENSION REDUCTION NEEDED (pads extra dimensions using zero!)
      
      // creates ICA_reduce and ICA_mean_reduce
      {
	std::lock_guard<std::mutex> lock(params_mutex);
	
	ICA_reduce.resize(REDUCED_DIM, (xdata.dimension(0)+pdata.dimension(0))*HISTLEN);
	ICA_reduce.zero();
	
	for(unsigned int k=0;k<REDUCED_DIM&&k<(xdata.dimension(0)+pdata.dimension(0))*HISTLEN;k++)
	  ICA_reduce(k,k) = 1.0f;
	
	ICA_mean_reduce.resize(REDUCED_DIM);
	ICA_mean_reduce.zero();
      }

      if(optimize_running == false) return;

      for(unsigned int i=0;i<data.size(0);i++){

	if(optimize_running == false) return;
	
	whiteice::math::vertex<T> v;
	v.resize(REDUCED_DIM + (FUTURELEN-1)*pdata.dimension(0));
	v.zero();

	const auto& d = data.access(0, i);

	for(unsigned int j=0;j<((xdata.dimension(0)+pdata.dimension(0))*HISTLEN)&&j<REDUCED_DIM;j++)
	  v[j]= d[j];

	for(unsigned int j=0;j<(FUTURELEN-1)*pdata.dimension(0);j++){
	  v[REDUCED_DIM+j] = d[((xdata.dimension(0)+pdata.dimension(0))*HISTLEN)+j];
	}

	// std::cout << "v = " << v << std::endl;

	
	reduced_data.push_back(v);
      }
    }

    if(optimize_running == false) return;

    whiteice::dataset<T> rdata;
    
    rdata.createCluster("reduced dim input", REDUCED_DIM + (FUTURELEN-1)*pdata.dimension(0));
    rdata.createCluster("output", data.dimension(1));

    rdata.add(0, reduced_data);

    for(unsigned int i=0;i<data.size(1);i++)
      rdata.add(1, data.access(1, i));

    if(optimize_running == false) return;


    // now dimension reduction (if needed) is computed
    // starts optimizer
    {
      std::lock_guard<std::mutex> lock(compute_mutex);

      params_mutex.lock();
      auto netcopy = this->net;
      params_mutex.unlock();
      
      sgd.setUseMinibatch(true); // was: false
      sgd.setOverfit(false); // don't overfit data (better with real data)
      sgd.setMNE(true); // norm error E{|correct-predicted|}

      // initially use NN
      if(sgd.startOptimize(rdata, netcopy, 1, 0xFFFFFFFF, false , true , false) == false){
	printf("NARX: SGD neural net optimization start FAILED.\n");
      }
      
      optimize_running = false; // thread execution has ended but SGD optimizer may run in background
    }
    
  }


  


  template <typename T>
  T NARX<T>::getError() const // uses predict() to calculate expected |error| per dimension
  {
    if(xdata.size(0) == 0 || pdata.size(0) == 0 || HISTLEN == 0 || FUTURELEN == 0)
      return T(-1.0f);

    T error = T(0.0f);
    unsigned int counter = 0;

    for(unsigned int i=(HISTLEN-1);i<xdata.size(0)-FUTURELEN;i++){

      std::vector< whiteice::math::vertex<T> > x;
      std::vector< whiteice::math::vertex<T> > p;
      std::vector< whiteice::math::vertex<T> > p_future;
      
      for(unsigned int h=0;h<HISTLEN;h++){	
	
	if(h <= i){
	  auto xx = xdata.access(0,i-(HISTLEN-1)+h);
	  auto pp = pdata.access(0,i-(HISTLEN-1)+h);

	  xdata.invpreprocess(0, xx);
	  pdata.invpreprocess(0, pp);

	  x.push_back(xx);
	  p.push_back(pp);
	}
	else{
	  auto xx = xdata.access(0,0);
	  auto pp = pdata.access(0,0);
	  
	  xx.zero();
	  pp.zero();

	  xdata.invpreprocess(0, xx);
	  pdata.invpreprocess(0, pp);

	  x.push_back(xx);
	  p.push_back(pp);
	}
	
      }

      for(unsigned int f=0;f<(FUTURELEN-1);f++){
	auto p = pdata.access(0,i+f);

	pdata.invpreprocess(0, p);

	p_future.push_back(p);
      }

      whiteice::math::vertex<T> y;
      
      if(this->predict(x, p, p_future, y) == false)
	return T(-1.0f);

      auto ynext = xdata.access(0, i+FUTURELEN);
      auto yfull = ynext;

      for(unsigned int i=0;i<pred_indexes.size();i++){
	yfull[pred_indexes[i]] = y[i];
      }

      xdata.preprocess(0, yfull); // adds preprocessing

      // calculates only pred_indexes term errors

      whiteice::math::vertex<T> ynext_p, y_p;

      y_p.resize(pred_indexes.size());
      ynext_p.resize(pred_indexes.size());

      for(unsigned int i=0;i<pred_indexes.size();i++){
	y_p[i] = yfull[pred_indexes[i]];
	ynext_p[i] = ynext[pred_indexes[i]];
      }
      
      error += (y_p-ynext_p).norm();
      counter += y_p.size();
    }

    if(counter)
      error /= counter;

    return error;
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
    if(REDUCED_DIM == 0) return false;

    // debug:
    //printf("predict: %d %d %d\n",
    //	   REDUCED_DIM, ICA_reduce.xsize(), (xdata.dimension(0)+pdata.dimension(0))*HISTLEN);

    if(ICA_reduce.xsize() != (xdata.dimension(0)+pdata.dimension(0))*HISTLEN)
      return false; // checks ICA/PCA dim. reduction has been computed

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

    // calculates dimension reduction in historical parameters dimensions
    // (no future parameters which are presered)
    {
      auto w = v;

      //std::cout << "v = " << v << std::endl;

      w.resize((xdata.dimension(0)+pdata.dimension(0))*HISTLEN);
      
      w = ICA_reduce*w - ICA_mean_reduce;

      w.resize(REDUCED_DIM + (FUTURELEN-1)*pdata.dimension(0));

      for(unsigned int j=0;j < (FUTURELEN-1)*pdata.dimension(0);j++)
      {
	w[REDUCED_DIM+j] = v[(xdata.dimension(0)+pdata.dimension(0))*HISTLEN + j];
      }

      //std::cout << "w = " << w << std::endl;

      v = w;
    }
    
    // predicts x(t+FUTURELEN)

    if(net.calculate(v, y) == false) return false;

    // converts output y to dim(x) long vector for invpreprocess

    auto yy = xdata.access(0,0);

    for(unsigned int k=0;k<pred_indexes.size();k++){
      yy[pred_indexes[k]] = y[k];
    }
    
    xdata.invpreprocess(0, yy); // removes preprocessing from predicted x(t+FUTURELEN)

    // converts yy back to pred_indexes low dimensional vector
    
    for(unsigned int k=0;k<pred_indexes.size();k++){
      y[k] = yy[pred_indexes[k]];
    }

    return true;
  }


  template <typename T>
  bool NARX<T>::save(const std::string& filename) const
  {
    std::unique_lock<std::mutex> lock(compute_mutex);
    std::lock_guard<std::mutex> lock2(params_mutex);
    
    if(HISTLEN == 0 || FUTURELEN == 0)
      return false;

    if(filename.size() == 0) return false;

    whiteice::dataset<T> params;

    params.createCluster("HISTLEN", 1);
    params.createCluster("FUTURELEN", 1);
    params.createCluster("REDUCED_DIM", 1);
    params.createCluster("pred_indexes", pred_indexes.size());
    params.createCluster("ICA dim(y) dim(x)", 2);
    params.createCluster("ICA matrix", ICA_reduce.ysize()*ICA_reduce.xsize());
    params.createCluster("ICA mean", ICA_mean_reduce.size());

    whiteice::math::vertex<T> v;
    v.resize(1);

    v[0] = HISTLEN;
    params.add(0, v);

    v[0] = FUTURELEN;
    params.add(1, v);

    v[0] = REDUCED_DIM;
    params.add(2, v);

    v.resize(pred_indexes.size());
    
    for(unsigned int i=0;i<v.size();i++)
      v[i] = T(pred_indexes[i]);

    params.add(3, v);

    v.resize(2);
    v[0] = ICA_reduce.ysize();
    v[1] = ICA_reduce.xsize();
    params.add(4, v);


    v.resize(ICA_reduce.size());
    for(unsigned int i=0;i<ICA_reduce.size();i++)
      v[i] = ICA_reduce[i];
    params.add(5, v);

    v = ICA_mean_reduce;
    params.add(6, v);
    

    const std::string netfile = filename + ".nnet";
    const std::string xfile = filename + ".xdata";
    const std::string pfile = filename + ".pdata";
    const std::string paramsfile = filename + ".params";
    
    if(net.save(netfile) == false ||
       xdata.save(xfile) == false ||
       pdata.save(pfile) == false ||
       params.save(paramsfile) == false)
      return false;

    return true;
  }
  

  template <typename T>
  bool NARX<T>::load(const std::string& filename)
  {
    std::unique_lock<std::mutex> lock(compute_mutex);

    if(filename.size() == 0) return false;

    const std::string netfile = filename + ".nnet";
    const std::string xfile = filename + ".xdata";
    const std::string pfile = filename + ".pdata";
    const std::string paramsfile = filename + ".params";

    whiteice::nnetwork<T> net;
    whiteice::dataset<T> xdata;
    whiteice::dataset<T> pdata;
    whiteice::dataset<T> params;
    std::vector<unsigned int> pred_indexes;

    whiteice::math::matrix<T> ICA;
    whiteice::math::vertex<T> mean;

    if(net.load(netfile) == false ||
       xdata.load(xfile) == false ||
       pdata.load(pfile) == false ||
       params.load(paramsfile) == false)
      return false;

    if(params.getNumberOfClusters() != 7) return false;
    if(params.dimension(0) != 1) return false;
    if(params.dimension(1) != 1) return false;
    if(params.dimension(2) != 1) return false;
    if(params.dimension(3) < 1) return false;
    
    if(xdata.getNumberOfClusters() != 1) return false;
    if(pdata.getNumberOfClusters() != 1) return false;

    whiteice::math::vertex<T> v;
    v = params.access(0, 0);
    const unsigned int HISTLEN = (unsigned int)v[0].c[0];
    
    v = params.access(1, 0);
    const unsigned int FUTURELEN = (unsigned int)v[0].c[0];
    
    v = params.access(2, 0);
    const unsigned int REDUCED_DIM = (unsigned int)v[0].c[0];

    if(REDUCED_DIM < 1) return false;

    v = params.access(3, 0);

    pred_indexes.resize(v.size());

    for(unsigned int i=0;i<v.size();i++)
      pred_indexes[i] = (unsigned int)v[i].c[0];

    v = params.access(4, 0);
    if(v.size() != 2) return false;
    
    const unsigned int ICA_YSIZE = (unsigned int)v[0].c[0];
    const unsigned int ICA_XSIZE = (unsigned int)v[1].c[0];

    if(ICA_YSIZE <= 0 || ICA_XSIZE <= 0) return false;

    ICA.resize(ICA_YSIZE, ICA_XSIZE);
    mean.resize(ICA_YSIZE);

    v = params.access(5, 0);
    if(v.size() != ICA.size()) return false;

    for(unsigned int i=0;i<ICA.size();i++)
      ICA[i] = v[i];

    v = params.access(6, 0);
    if(v.size() != mean.size()) return false;

    for(unsigned int i=0;i<mean.size();i++)
      mean[i] = v[i];

    if(ICA_YSIZE != REDUCED_DIM) return false;
    if(ICA_XSIZE != ((xdata.dimension(0)+pdata.dimension(0))*HISTLEN)) return false;

    if(net.input_size() != REDUCED_DIM + (FUTURELEN-1)*pdata.dimension(0))
      return false;

    if(net.output_size() != pred_indexes.size())
      return false;


    // things seems to be in order, set values to class variables

    {
      std::lock_guard<std::mutex> lock2(params_mutex);
      
      this->HISTLEN = HISTLEN;    
      this->FUTURELEN = FUTURELEN;
      this->REDUCED_DIM = REDUCED_DIM;
      this->xdata = xdata;
      this->pdata = pdata;
      this->pred_indexes = pred_indexes;
      this->net = net;
      this->ICA_reduce = ICA;
      this->ICA_mean_reduce = mean;
      
      this->optimize_running = false;
      
      if(optimize_thread){
	if(optimize_thread->joinable())
	  optimize_thread->join();
	
	delete optimize_thread;
	optimize_thread = nullptr;
      }
    }

    return true;
  }
  
  
  template class NARX< whiteice::math::blas_real<float> >;
  template class NARX< whiteice::math::blas_real<double> >;
};
