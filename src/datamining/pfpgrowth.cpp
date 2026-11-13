
#include <iostream>
#include <vector>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <memory>
#include <omp.h>

using namespace std;

namespace pfpgrowth {

// Represents a node in the FP-Tree
struct FPNode {
  long long item;
  long long count;
  FPNode* parent;
  FPNode* node_link; // Links to the next node with the same item
  unordered_map<long long, unique_ptr<FPNode>> children;
  
  FPNode(long long item_id, long long cnt, FPNode* par)
    : item(item_id), count(cnt), parent(par), node_link(nullptr) {}
};


// Header table entry for fast traversal of same-item nodes
struct HeaderEntry {
  long long count = 0;
  FPNode* head = nullptr;
  FPNode* tail = nullptr;
};

class FPTree {
  
public:
  unique_ptr<FPNode> root;
  unordered_map<long long, HeaderEntry> header_table;
  
  FPTree() {
    root = make_unique<FPNode>(-1, 0, nullptr);
  }
  
  // Inserts a sorted, filtered transaction long longo the tree
  void insertTransaction(const vector<long long>& transaction, long long count = 1) {
    FPNode* current = root.get();
    for (long long item : transaction) {
      auto it = current->children.find(item);
      if (it != current->children.end()) {
	it->second->count += count;
	current = it->second.get();
      } else {
	auto new_node = make_unique<FPNode>(item, count, current);
	FPNode* raw_ptr = new_node.get();
	current->children[item] = move(new_node);
	current = raw_ptr;
	
	// Update header table linked list
	HeaderEntry& entry = header_table[item];
	if (entry.head == nullptr) {
	  entry.head = raw_ptr;
	  entry.tail = raw_ptr;
	} else {
	  entry.tail->node_link = raw_ptr;
	  entry.tail = raw_ptr;
	}
      }
      header_table[item].count += count;
    }
  }
};

// Extracts prefix paths ending at a specific item to form a Conditional Pattern Base
vector<pair<vector<long long>, long long>> getConditionalPatternBase(FPNode* head_node) {
  vector<pair<vector<long long>, long long>> cond_patterns;
  FPNode* curr = head_node;
  
  while (curr != nullptr) {
    long long path_count = curr->count;
    vector<long long> prefix_path;
    FPNode* parent = curr->parent;
    
    while (parent != nullptr && parent->item != -1) {
      prefix_path.push_back(parent->item);
      parent = parent->parent;
    }
    
    if (!prefix_path.empty()) {
      reverse(prefix_path.begin(), prefix_path.end());
      cond_patterns.push_back({prefix_path, path_count});
    }
    curr = curr->node_link;
  }
  return cond_patterns;
}

// Recursive FP-Growth mining on a conditional tree
void mineTree(const FPTree& tree, long long min_support, vector<long long>& alpha,
	      vector<pair<vector<long long>, long long>>& local_results) {
  
  // Sort header items by ascending frequency
  vector<pair<long long, long long>> sorted_items;
  for (const auto& kv : tree.header_table) {
    if (kv.second.count >= min_support) {
      sorted_items.push_back({kv.first, kv.second.count});
    }
  }
  sort(sorted_items.begin(), sorted_items.end(),
       [](const auto& a, const auto& b) { return a.second < b.second; });
  
  for (const auto& item_pair : sorted_items) {
    long long item = item_pair.first;
    long long support = item_pair.second;
    
    // 1. Emit frequent itemset: alpha U {item}
    vector<long long> beta = alpha;
    beta.push_back(item);
    local_results.push_back({beta, support});
    
    // 2. Build Conditional Pattern Base
    FPNode* head_node = tree.header_table.at(item).head;
    auto cond_patterns = getConditionalPatternBase(head_node);
    
    // 3. Count item frequencies in conditional patterns
    unordered_map<long long, long long> cond_counts;
    for (const auto& pat : cond_patterns) {
      for (long long p_item : pat.first) {
	cond_counts[p_item] += pat.second;
      }
    }
    
    // 4. Build Conditional FP-Tree
    FPTree cond_tree;
    for (const auto& pat : cond_patterns) {
      vector<long long> filtered_pat;
      for (long long p_item : pat.first) {
	if (cond_counts[p_item] >= min_support) {
	  filtered_pat.push_back(p_item);
	}
      }
      // Sort by descending frequency in the conditional base
      sort(filtered_pat.begin(), filtered_pat.end(),
	   [&](long long a, long long b) {
	     if (cond_counts[a] != cond_counts[b])
	       return cond_counts[a] > cond_counts[b];
	     return a < b;
	   });
      
      if (!filtered_pat.empty()) {
	cond_tree.insertTransaction(filtered_pat, pat.second);
      }
    }
    
    // 5. Recursively mine if conditional tree is non-empty
    if (!cond_tree.root->children.empty()) {
      mineTree(cond_tree, min_support, beta, local_results);
    }
  }
}

// Top-level Parallel FP-Growth Orchestrator
vector<pair<vector<long long>, long long>> parallelFPGrowth
(const vector<vector<long long>>& transactions, long long min_support) {
  
  // Step 1: Global frequency count (can also be parallelized via reduction)
  unordered_map<long long, long long> global_counts;
  for (const auto& tx : transactions) {
    for (long long item : tx) {
      global_counts[item]++;
    }
  }
  
  // Filter frequent 1-itemsets
  vector<pair<long long, long long>> freq_items;
  for (const auto& kv : global_counts) {
    if (kv.second >= min_support) {
      freq_items.push_back(kv);
    }
  }
  
  // Sort descending for global FP-Tree insertion
  sort(freq_items.begin(), freq_items.end(),
       [](const auto& a, const auto& b) {
	 if (a.second != b.second) return a.second > b.second;
	 return a.first < b.first;
       });
  
  // Step 2: Build the Global FP-Tree
  FPTree global_tree;
  for (const auto& tx : transactions) {
    vector<long long> filtered;
    for (long long item : tx) {
      if (global_counts[item] >= min_support) {
	filtered.push_back(item);
      }
    }
    sort(filtered.begin(), filtered.end(),
	 [&](long long a, long long b) {
	   if (global_counts[a] != global_counts[b])
	     return global_counts[a] > global_counts[b];
	   return a < b;
	 });
    
    if (!filtered.empty()) {
      global_tree.insertTransaction(filtered, 1);
    }
  }
  
  // Reverse items to mine from least frequent to most frequent
  reverse(freq_items.begin(), freq_items.end());
  
  // Step 3: Parallel Mining across OpenMP threads
  vector<pair<vector<long long>, long long>> all_frequent_itemsets;
  
#pragma omp parallel
  {
    // Thread-local storage avoids mutex contention during mining
    vector<pair<vector<long long>, long long>> thread_results;
    
    // Dynamic scheduling balances load since high-frequency items have deeper trees
#pragma omp for schedule(dynamic, 1) nowait
    for (size_t i = 0; i < freq_items.size(); ++i) {
      long long item = freq_items[i].first;
      long long support = freq_items[i].second;
      
      vector<long long> alpha = {item};
      thread_results.push_back({alpha, support});
      
      FPNode* head_node = global_tree.header_table.at(item).head;
      auto cond_patterns = getConditionalPatternBase(head_node);
      
      unordered_map<long long, long long> cond_counts;
      for (const auto& pat : cond_patterns) {
	for (long long p_item : pat.first) {
	  cond_counts[p_item] += pat.second;
	}
      }
      
      FPTree cond_tree;
      for (const auto& pat : cond_patterns) {
	vector<long long> filtered_pat;
	for (long long p_item : pat.first) {
	  if (cond_counts[p_item] >= min_support) {
	    filtered_pat.push_back(p_item);
	  }
	}
	sort(filtered_pat.begin(), filtered_pat.end(),
	     [&](long long a, long long b) {
	       if (cond_counts[a] != cond_counts[b])
		 return cond_counts[a] > cond_counts[b];
	       return a < b;
	     });
	
	if (!filtered_pat.empty()) {
	  cond_tree.insertTransaction(filtered_pat, pat.second);
	}
      }
      
      if (!cond_tree.root->children.empty()) {
	mineTree(cond_tree, min_support, alpha, thread_results);
      }
    }
    
    // Merge thread-local results safely
#pragma omp critical
    {
      all_frequent_itemsets.insert(all_frequent_itemsets.end(),
thread_results.begin(),
				   thread_results.end());
    }
  }
  
  return all_frequent_itemsets;
}

}

#if 0
int main() {
  // Sample transactional dataset (Item IDs)
  vector<vector<long long>> dataset = {
    {1, 2, 5},
    {2, 4},
    {2, 3},
    {1, 2, 4},
    {1, 3},
    {2, 3},
    {1, 3},
    {1, 2, 3, 5},
    {1, 2, 3}
  };
  
  long long min_support = 2;
  
  cout << "Running Parallel FP-Growth with " << omp_get_max_threads() << " threads...\n\n";
  auto results = parallelFPGrowth(dataset, min_support);
  
  for (const auto& res : results) {
    cout << "{ ";
    for (long long item : res.first) cout << item << " ";
    cout << "} -> Support: " << res.second << "\n";
  }
  
  return 0;
}
#endif
