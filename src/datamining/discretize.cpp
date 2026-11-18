
#include "discretize.h"

#include <set>
#include <vector>
#include <chrono>
#include <thread>

#include "dynamic_bitset.h"
#include "FrequentSetsFinder.h"
#include "list_source.h"
#include "KMeans.h"

//using namespace std::chrono_literals;


namespace whiteice
{

  //////////////////////////////////////////////////////////////////////

  bool calculate_discretize(const std::vector< std::vector<std::string> >& data,
			    std::vector<struct discretization>& disc, bool overlap) // overlap tell if discretized variables should overlap in continuous range
  {
    if(data.size() == 0) return false;

    std::vector< std::set<std::string> > elems;
    std::vector< std::vector<double> > numbers;
    std::vector<unsigned int> is_numeric;
    

    elems.resize(data[0].size());
    numbers.resize(data[0].size());
    is_numeric.resize(data[0].size());

    for(unsigned int i=0;i<data.size();i++){
      for(unsigned int j=0;j<data[i].size();j++){
	elems[j].insert(data[i][j]);

	char* p = NULL;
	double value = strtod(data[i][j].c_str(), &p);
	if(p != NULL && p != data[i][j].c_str()){
	  is_numeric[j]++;
	  numbers[j].push_back(value);
	}
      }
    }

    disc.resize(data[0].size());

    //#pragma omp parallel for
    for(unsigned int i=0;i<disc.size();i++){
      if(elems[i].size() <= 20){
	disc[i].TYPE = 1;
	disc[i].elem.resize(elems[i].size());

	unsigned int index = 0;

	for(const auto& s : elems[i]){
	  disc[i].elem[index] = s;
	  index++;
	}
      }
      else if(is_numeric[i] == data.size()){
	disc[i].TYPE = 0;

	// pairs of items has on average 10.0 rows/freq
	// [not enough of frequent item which is 20.0]


	// probability of (dim(x)/4) itemset is ROWS/data.size(), calculates single item average probability
	// and calculates number of rows of single item (DIV)
	const double ROWS = 1000.0;  // was: 50, 1000
	const double p = ROWS/data.size();
	double DIV = pow(p, 4.0f/((float)data[0].size()));
	DIV *= data.size();
	//const double DIV = 3100.0; // 3000,3100,1000

	double B = data.size()/DIV; // 4000.0;
	//B = data.size()/100;
	//B = math::pow(B, 1.0/2.0);

	std::cout << "DIV " << DIV << std::endl;

	unsigned int BINS = (unsigned int)round(B);
	//BINS = 256;
	
	
	if(BINS < 2) BINS = 2;
	//else if(BINS > 300) BINS = 300;

	//BINS = 10;
	
	std::cout << "BINS: " << BINS << std::endl;


	/*
	unsigned int BINS = data.size() / 1000; // was 200, 300, 500 don't work
	
	if(BINS < 2) BINS = 2;
	else if(BINS > 100) BINS = 100;
	*/

	disc[i].boxL.resize(BINS);
	disc[i].boxR.resize(BINS);

	whiteice::KMeans<double> km;

	std::vector< std::vector<double> > data;

	for(const auto& n : numbers[i]){
	  std::vector<double> nn;
	  nn.push_back(n);
	  data.push_back(nn);
	}

	km.startTrain(BINS, data);

	while(km.isRunning()){
	  std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}

	km.stopTrain();

	std::set<double> numset; // ordered list from smallest to largest

	for(unsigned long long k=0;k<km.size();k++){
	  numset.insert(km[k][0]);
	}
	
	//std::cout << "BINS: ";
	//unsigned long long index = 0;

	auto iter = numset.begin();
	auto prev_iter = iter;
	auto next_iter = iter;
	next_iter++;

	for(unsigned int k=0;k<numset.size();k++,iter++){
	  if(k==0){ disc[i].boxL[k] = -INFINITY; }
	  else{ disc[i].boxL[k] = (*prev_iter + *iter)/2.0f; prev_iter++; }

	  if(overlap == true){
	    if(k==numset.size()-1) disc[i].boxR[k] = INFINITY;
	    else{ disc[i].boxR[k] = (*next_iter + *iter)/2.0f; next_iter++; }
	  }
	  else{
	    if(k==numset.size()-1) disc[i].boxR[k] = INFINITY;
	    else{ disc[i].boxR[k] = *iter; next_iter++; }
	  }

	  //std::cout << "BIN " << disc[i].boxL[k] << " , " << disc[i].boxR[k] << std::endl;
	}

#if 0
	double prev = -INFINITY;
	for(const auto& n : numset){
	  if(prev > -INFINITY){
	    disc[i].boxL[index] = (n+prev)/2.0;
	    disc[i].boxR[index] = 
	    index++;
	  }
	  prev = n;
	  
	  //std::cout << n << " ";
	}

	disc[i].bins[index] = INFINITY; // last bin
#endif
		
      }
      else{
	disc[i].TYPE = 2;
	// ignore this column  
      }
    }
    
    return (disc.size() > 0);
  }

  
  //////////////////////////////////////////////////////////////////////

  
  // discretizes data and creates one-hot-encoding of discrete value in binary
  bool binarize(const std::vector< std::vector<std::string> >& data,
		const std::vector<struct discretization>& disc,
		std::vector< whiteice::dynamic_bitset >& result)
  {
    // TODO: calculate results as binary bits (dynmic_bitset of input x)
    
    if(data.size() == 0) return false;
    if(data[0].size() != disc.size()) return false;

    unsigned int binary_size = 0;

    for(const auto& d : disc){
      if(d.TYPE == 0){
	binary_size += d.boxR.size();
      }
      else if(d.TYPE == 1){
	binary_size += d.elem.size();
      }
    }

    if(binary_size == 0) return false;

    
    for(unsigned int i=0;i<data.size();i++){
      whiteice::dynamic_bitset v;
      v.resize(binary_size);
      v.reset();

      unsigned int index = 0;
      
      for(unsigned j=0;j<data[i].size();j++){
	if(disc[j].TYPE == 0){
	  char* p = NULL;
	  double value = strtod(data[i][j].c_str(), &p);

	  std::set<unsigned int> indexes;

	  for(unsigned int c=0;c<disc[j].boxR.size();c++)
	    if(disc[j].boxL[c] <= value && disc[j].boxR[c] >= value)
	      indexes.insert(c);

	  //v[index + counter] = 1.0;

	  for(const auto& c : indexes)
	    v.set(index + c, true);
	  
	  index += disc[j].boxR.size();
	}
	else if(disc[j].TYPE == 1){

	  unsigned int counter = 0;
	  for(counter=0;counter<disc[j].elem.size();counter++)
	    if(disc[j].elem[counter] == data[i][j])
	      break;

	  if(counter < disc[j].elem.size()){
	    v.set(index+counter, true);
	    //v[index+counter] = 1.0;
	  }

	  index += disc[j].elem.size();
	}
	
      }
      
      result.push_back(v);
    }


    return (result.size() > 0); 
  }

  //////////////////////////////////////////////////////////////////////

  using namespace whiteice;

  
  // creates dataset with frequent sets added as extra-variables
  bool enrich_data(const std::vector< whiteice::dynamic_bitset >& data,
		   std::vector<whiteice::dynamic_bitset>& f, // frequent sets
		   //std::vector< std::vector<double> >& result,
		   double freq_limit)
  {
    if(data.size() == 0) return false;
    if(freq_limit < 0.0 || freq_limit >= 1.0) return false;

    if(freq_limit == 0.0){
      freq_limit = 30.0/data.size(); // was: 50, 100, 30 cases for each variable minimum (30 for 10 dimensional data)
    }

    //std::vector<dynamic_bitset> fset;

    // calculates frequent itemsets
    {
      std::vector<dynamic_bitset> dbdata;
      
      for(const auto& d : data){
	dynamic_bitset x;
	x.resize(data[0].size());
	x.reset();
	
	for(unsigned int i=0;i<d.size();i++){
	  if(d[i] != 0.0) x.set(i, true);
	  else x.set(i, false);
	}
	
	dbdata.push_back(x);
      }
      
      list_source<dynamic_bitset>* source = new list_source<dynamic_bitset>(dbdata);
      
      whiteice::datamining::FrequentSetsFinder fsfinder(*source, f, freq_limit);
      
      fsfinder.find();

      delete source;
    }

    // extend datasets to all subsets of frequent sets
    // std::set<dynamic_bitset> f;

    printf("frequent sets search done: %d sets.\n", (int)f.size());
    fflush(stdout);


    std::vector< std::set<unsigned int> > sf; // sparse representation of frequent sets (dirty slow hack)
    
    {
      sf.resize(f.size());

#pragma omp parallel for schedule(dynamic, 200)
      for(unsigned int i=0;i<f.size();i++){
	std::set<unsigned int> sparse;

	for(unsigned int j=0;j<f[i].size();j++)
	  if(f[i][j]) sparse.insert(j);

	sf[i] = sparse;
      }
    }
    
#if 1
    // generates all frequent itemsets dataset
    std::vector< whiteice::dynamic_bitset > result;
    result.resize(data.size());

#pragma omp parallel
    {
      dynamic_bitset value;
      value.resize(sf.size());
      
#pragma omp for schedule(dynamic, 200)
      for(unsigned int j=0;j<data.size();j++){
	value.reset();
	unsigned int index = 0;

	for(const auto& b : sf){
	  bool fdata = true;
	  
	  for(const auto& bi : b){
	    if(data[j][bi] == false){
	      fdata = false;
	      break;
	    }
	  }

	  if(fdata) value.set(index, true);
	  index++;
	}
	
	result[j] = value;
      }
    }

    f = result;

    //printf("enrich_data() done: %d\n", (int)result.size());
    //fflush(stdout);
#endif
    

    //if(result.size() == 0) return false;
    //if(result[0].size() == 0) return false;

    if(f.size() == 0) return false;
    if(f[0].size() == 0) return false;

    return true;
  }


  // creates dataset with frequent sets added as extra-variables
  bool enrich_data_again(const std::vector< whiteice::dynamic_bitset >& data,
			 const std::vector<whiteice::dynamic_bitset>& f, // frequent sets
			 std::vector< std::vector<double> >& result)
  {

    // generates all frequent itemsets dataset
    {
      for(unsigned int j=0;j<data.size();j++){
	dynamic_bitset value;
	value.resize(f.size()*2);
	value.reset();

	unsigned int index = 0;

	for(const auto& b : f){

	  bool fdata = true;
	  bool or_data = false;

	  for(unsigned int i=0;i<b.size();i++){
	    if(b[i] && data[j][i] == true){ fdata = false; }
	    if(b[i] && data[j][i] == false){ or_data = true; }
	  }

	  if(fdata) value.set(2*index, true);
	  else value.set(2*index, false);
	  
	  if(or_data) value.set(2*index + 1, true);
	  else value.set(2*index + 1, false);

	  /*
	  if(fdata) value.set(4*index + 2, false);
	  else value.set(4*index + 2, true);
	  
	  if(or_data) value.set(4*index + 3, false);
	  else value.set(4*index + 3, true);
	  */

	  index++;
	}

	// now we have one frequent item

	std::vector<double> r;
	r.resize(value.size());

	for(unsigned int i=0;i<r.size();i++){
	  if(value[i]) r[i] = 1.0;
	  else r[i] = 0.0;
	}

	result.push_back(r);
      }
    }
    
    if(result.size() == 0) return false;
    if(result[0].size() == 0) return false;

    return true;
  } 
  
};
