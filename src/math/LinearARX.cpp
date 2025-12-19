

#include "LinearARX.h"
#include "RNG.h"




namespace whiteice
{
  namespace math
  {

    template <typename T>
    LinearARX<T>::LinearARX()
    {
    }

    template <typename T>
    LinearARX<T>::~LinearARX()
    {
    }
    

    // computes solution directly without thread, modern computers should be fast enough
    // dataset variables should be preprocessed
    template <typename T>
    bool LinearARX<T>::computeSolution(const whiteice::dataset<T>& xdata,
				       const whiteice::dataset<T>& pdata,
				       const unsigned int HISTLEN)
    {
      if(HISTLEN < 1) return false;
      if(xdata.getNumberOfClusters() < 1) return false;
      if(pdata.getNumberOfClusters() < 1) return false;
      if(xdata.size(0) != pdata.size(0)) return false;
      if(xdata.size(0) <= 10) return false; // needs some data

      std::vector< whiteice::math::vertex<T> > x, p, y;

      for(unsigned int i=0;i<xdata.size(0)-1;i++){
	whiteice::math::vertex<T> z;

	// creates x input vector
	z.resize(xdata.dimension(0)*HISTLEN);
	z.zero();

	for(unsigned int h=0;h<HISTLEN;h++){
	  if(h <= i){
	    for(unsigned int k=0;k<xdata.dimension(0);k++){
	      z[h*xdata.dimension(0) + k] = xdata.access(0, i-h)[k];
	    }
	  }
	}
	
	x.push_back(z);

	z.resize(pdata.dimension(0)*HISTLEN);
	z.zero();

	for(unsigned int h=0;h<HISTLEN;h++){
	  if(h <= i){
	    for(unsigned int k=0;k<pdata.dimension(0);k++){
	      z[h*pdata.dimension(0) + k] = pdata.access(0, i-h)[k];
	    }
	  }
	}

	p.push_back(z);

	y.push_back(xdata.access(0, i+1));
      }


      // calculates matrices used to calculate optimal parameters A and B

      whiteice::math::matrix<T> Ryx, Rpx, Rpp, Ryp, Rxp, Rxx, Rpy, Rxy;

      Ryx.resize(y[0].size(), x[0].size());
      Rpx.resize(p[0].size(), x[0].size());
      Rpp.resize(p[0].size(), p[0].size());
      Ryp.resize(y[0].size(), p[0].size());
      Rxp.resize(x[0].size(), p[0].size());
      Rxx.resize(x[0].size(), x[0].size());
      Rpy.resize(p[0].size(), y[0].size());
      Rxy.resize(x[0].size(), y[0].size());


      Ryx.zero();
      Rpx.zero();
      Rpp.zero();
      Ryp.zero();
      Rxp.zero();
      Rxx.zero();
      Rpy.zero();
      Rxy.zero();

      for(unsigned int i=0;i<x.size();i++){
	// outer products should be optimized..
	Ryx += y[i].outerproduct(x[i]);
	Rpx += p[i].outerproduct(x[i]);
	Rpp += p[i].outerproduct();
	Ryp += y[i].outerproduct(p[i]);
	Rxp += x[i].outerproduct(p[i]);
	Rxx += x[i].outerproduct();
	Rpy += p[i].outerproduct(y[i]);
	Rxy += x[i].outerproduct(y[i]);
      }

      Ryx /= T(x.size());
      Rpx /= T(x.size());
      Rpp /= T(x.size());
      Ryp /= T(x.size());
      Rxp /= T(x.size());
      Rxx /= T(x.size());
      Rpy /= T(x.size());
      Rxy /= T(x.size());
      
      //////////////////////////////////////////////////////////////////////
      // calculates parameter matrices A and B

      // std::cout << "Rxx.det() == " << Rxx.det() << std::endl;
      
      if(Rxx.pseudoinverse() == false) return false;

      whiteice::math::matrix<T> TMP, TMP1;

      TMP1 = Rpp-Rpx*Rxx*Rxp;

      // std::cout << "TMP1.det() == " << TMP1.det() << std::endl;
      
      if(TMP1.pseudoinverse() == false) return false;
      
      TMP = Rpx*Rxx*Rxy-Rpy;

      B = TMP1*TMP;
      A = -Rxx*(Rxy+Rxp*B);

      A.transpose();
      B.transpose();

      this->xdata = xdata; // saves preprocessing information for predict()
      this->pdata = pdata;
      this->HISTLEN = HISTLEN;

      return true;      
    }



    template <typename T>
    bool LinearARX<T>::randomizeSolutionMatrices()
    {
      whiteice::RNG<T> random;
      
      for(unsigned int i=0;i<A.size();i++)
	A[i] = random.normal();

      for(unsigned int i=0;i<B.size();i++)
	B[i] = random.normal();
      
      
      return true;
    }

    

    template <typename T>
    bool LinearARX<T>::predict
    (const std::vector< whiteice::math::vertex<T> >& x, // HISTLEN x(t)..x(t-HISTLEN-1) VECTOR ELEMENTS
     const std::vector< whiteice::math::vertex<T> >& p, // HISTLEN p(t)..p(t-HISTLEN-1) VECTOR ELEMENTS
     whiteice::math::vertex<T>& y)       // predicted vector y=x(t+1)
    {
      // not perfect checks..
      if(x.size() != this->HISTLEN || p.size() != this->HISTLEN) return false;
      if(x[0].size() != xdata.dimension(0)) return false;
      if(p[0].size() != pdata.dimension(0)) return false;
      if(A.xsize() != HISTLEN*xdata.dimension(0)) return false;
      if(B.xsize() != HISTLEN*pdata.dimension(0)) return false;
      
      whiteice::math::vertex<T> xx;

      // creates x input vector
      xx.resize(xdata.dimension(0)*HISTLEN);
      xx.zero();

      for(unsigned int h=0;h<HISTLEN;h++){
	auto xi = x[h];
	xdata.preprocess(0, xi);
	
	for(unsigned int k=0;k<xdata.dimension(0);k++){
	  xx[h*xdata.dimension(0) + k] = xi[k];
	}
      }
      
      
      whiteice::math::vertex<T> pp;
      
      pp.resize(pdata.dimension(0)*HISTLEN);
      pp.zero();

      for(unsigned int h=0;h<HISTLEN;h++){
	auto pi = p[h];
	pdata.preprocess(0, pi);

	for(unsigned int k=0;k<pdata.dimension(0);k++){
	  pp[h*pdata.dimension(0) + k] = pi[k];
	}
      }
      
      y = A*xx + B*pp; // predicts next step with simple linear model

      return true;
    }


    // returns mean error using datasets (requires model is calculated)
    template <typename T>
    T LinearARX<T>::getError()
    {
      if(xdata.size(0) == 0 || pdata.size(0) == 0) return T(-1.0f); // ERROR

      std::vector< whiteice::math::vertex<T> > x, p, y;	 


      for(unsigned int i=0;i<xdata.size(0)-1;i++){
	whiteice::math::vertex<T> z;

	z = xdata.access(0, i);
	x.push_back(z);
	
	z = pdata.access(0, i);
	p.push_back(z);
	
	y.push_back(xdata.access(0, i+1));
      }

      
      T error = T(0.0f);

      for(unsigned int i=0;i<x.size()-1;i++){

	std::vector< whiteice::math::vertex<T> > xx, pp; // historical values + current one

	for(unsigned int h=0;h<HISTLEN;h++){
	  whiteice::math::vertex<T> xi;
	  xi.resize(x[0].size());
	  xi.zero();

	  if(h <= i){
	    for(unsigned int k=0;k<x[0].size();k++){
	      xi[k] = x[i-h][k];
	    }
	  }
	  
	  xx.push_back(xi);
	}


	for(unsigned int h=0;h<HISTLEN;h++){
	  whiteice::math::vertex<T> pi;
	  pi.resize(p[0].size());
	  pi.zero();

	  if(h <= i){
	    for(unsigned int k=0;k<p[0].size();k++){
	      pi[k] = p[i-h][k];
	    }
	  }
	  
	  pp.push_back(pi);
	}
	
	auto ypredict = y[i+1];

	
	if(this->predict(xx, pp, ypredict) == false){
	  return T(-1.0f); // ERROR
	}

	
	error += (ypredict - y[i+1]).norm();
      }


      if(x.size() > 1)
	error /= T((x.size()-1)*x[0].size());


      return error; // returns E{|Ax+Bp-y|} norm error
    }



    template class LinearARX< blas_real<float> >;
    template class LinearARX< blas_real<double> >;
    
  };
  
};
