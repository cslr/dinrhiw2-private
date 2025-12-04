/*
 * t-SNE dimension reduction algorithm with Barnes-Hut approximation.
 * Based on paper 
 * "Visualization Data using t-SNE. Laurens van der Maaten and Geoffrey Hinton. 11/2008".
 *
 * Mathmatical implementation notes/documentation in docs/TSNE_notes.tm (texmacs)
 * Tomas Ukkonen. 2020.
 *
 * Algorithm is loop parallelized (OpenMP) so multicore CPUs with K cores should 
 * give K-times faster results.
 *
 * NOTE:
 * This code uses Barnes-Hut approximation which means scaling is almost linear O(N*log(N)).
 * Original t-SNE code was implemented from research papers by Tomas Ukkonen.
 * Barnes-Hut approximation was added by ChatGPT prompt and testing shows it seem to work.
 * Barnes-Hut approximation is only used with dimension reduction to 2 or 3.
 *
 * Absolute value uses improved absolute value KL divergence for comparing
 * distributions which is better suited for distribution comparision.
 * abs(D_KL) = SUM p*|log(p/q)| 
 * (NOTE that minimizing KL divergence gives better absolute KL divergence but
 *  results are different and minimizing absolute value KL divergence seem to
 *  give results that are more "innovative" when used as input for neural network)
 *
 * 
 */

#ifndef TSNE_BH_h
#define TSNE_BH_h

#include "dinrhiw_blas.h"
#include "vertex.h"
#include "RNG.h"
#include "LoggingInterface.h"
#include "VisualizationInterface.h"

#include <vector>
#include <thread>
#include <atomic>
#include <functional>

namespace whiteice
{
  
  template <typename T = math::blas_real<float> >
    class TSNE_BH
    {
    public:

      // absolute value gives better results when results are fed to neural network
      TSNE_BH(const bool absolute_value = true); 
      TSNE_BH(const TSNE_BH<T>& tsne);
      
      // dimension reduces samples to DIM dimensional vectors using t-SNE algorithm
      // uses PCA to remove unnecessary input dimensions (keeps 95% of variance)
      // uses PCA+ICA to postprocess clustering results
      bool calculate(const std::vector< math::vertex<T> >& samples,
		     const unsigned int DIM,
		     std::vector< math::vertex<T> >& results,
		     const bool verbose = false,
		     LoggingInterface* const messages = NULL,
		     VisualizationInterface* const gui = NULL,
		     unsigned int* running_flag = NULL);


      // starts thread to run calculations in background
      bool calculate_start(const std::vector< math::vertex<T> >& samples,
			   const unsigned int DIM,
			   const bool verbose = false,
			   LoggingInterface* const messages = NULL,
			   VisualizationInterface* const gui = NULL);

      bool calculate_finished() const;

      bool calculate_end();

      bool calculate_get_results(std::vector< math::vertex<T> >& samples,
				 std::vector< math::vertex<T> >& results);

    private:

      void calculate_loop();

      std::thread* calculate_thread = nullptr;
      std::atomic<bool> calculate_is_running = false;
      mutable std::mutex calculate_mutex;
      
      std::vector< math::vertex<T> > calculate_samples;
      std::vector< math::vertex<T> > calculate_results;
      unsigned int calculate_DIM = 3;
      bool calculate_verbose = false;
      LoggingInterface* calculate_messages = NULL;
      VisualizationInterface* calculate_gui = NULL;

      
    private:
      
      
      // ---- Barnes–Hut acceleration (quadtree/octree) ----
      struct BHNode {
        // bounding box [min, max] in each dimension
        std::vector<T> bmin;
        std::vector<T> bmax;
        // center of mass and mass
        whiteice::math::vertex<T> com;
        T mass;
        // children indices in 'nodes' vector (-1 if no child)
        std::vector<int> child;
        // index of the point if leaf with a single point, otherwise -1
        int point_index;
        // precomputed cell size (max side length)
        T size;
        BHNode() : mass(T(0)), point_index(-1), size(T(0)) {}
      };

      // Build tree for 2D/3D (quadtree/octree). Returns index of root node.
      int bh_build_tree(const std::vector< whiteice::math::vertex<T> >& y,
                        std::vector<BHNode>& nodes) const;

      // Compute repulsive force and accumulate unnormalized q-sum for one point.
      void bh_repulsive_forces(const std::vector< whiteice::math::vertex<T> >& y,
                               const std::vector<BHNode>& nodes,
                               const int node_idx,
                               const unsigned int i,
                               const T theta,
                               whiteice::math::vertex<T>& force_i,
                               T& qsum_i) const;

      // Compute gradient using BH for repulsion and (sparse) P for attraction.
      bool bh_kl_gradient(const std::vector< std::vector<T> >& pij,
                          const std::vector< whiteice::math::vertex<T> >& y,
                          std::vector< whiteice::math::vertex<T> >& ygrad,
                          T& Z_out,
                          const T theta = T(0.5)) const;

      // Optionally sparsify P (symmetric top-K by row union), then renormalize to sum=1.
      void sparsify_pij(std::vector< std::vector<T> >& pij,
                        unsigned int topK) const;

// calculates p values for pj|i where i = index and sigma2 for index:th vector is given
      bool calculate_pvalue_given_sigma(const std::vector< math::vertex<T> >& x,
					const unsigned int index, // to x vector
					const T& sigma2,
					std::vector<T>& pj) const;
      
      // calculates distribution's perplexity
      T calculate_perplexity(const std::vector<T>& pj) const;
      
      // calculate x samples probability distribution values p
      bool calculate_pvalues(const std::vector< math::vertex<T> >& x,
			     const T perplexity,
			     std::vector< std::vector<T> >& pij) const;
      
      // calculate dimension reduced y samples probability distribution values q
      bool calculate_qvalues(const std::vector< math::vertex<T> >& y,
			     std::vector< std::vector<T> >& qij,
			     T& qsum) const;
      
      // calculates KL divergence
      bool kl_divergence(const std::vector< std::vector<T> >& pij,
			 const std::vector< std::vector<T> >& qij,
			 T& klvalue) const;
      
      // calculates gradients of KL diverence
      bool kl_gradient(const std::vector< std::vector<T> >& pij,
		       const std::vector< std::vector<T> >& qij,
		       const T& qsum,
		       const std::vector< math::vertex<T> >& y,
		       std::vector< math::vertex<T> >& ygrad) const;

      
      ////////////////////////////////////////////////////////////
      // internal variables

      bool kl_absolute_value; // whether to use absolute value in KL divergence
      bool verbose; // whether print error messages to console
    
    };


  extern template class TSNE_BH< math::blas_real<float> >;
  extern template class TSNE_BH< math::blas_real<double> >;
  
};


#endif
