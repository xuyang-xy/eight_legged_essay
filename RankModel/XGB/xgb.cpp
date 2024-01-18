#include "XgbModel.h"
#include <fstream>
#include <math.h>

#ifdef SATISFACTION_TEST_MODEL
#include <boost/algorithm/string.hpp>
#else
#include "common/Util.h"
#endif

using std::string;
using std::vector;
using std::unordered_map;

const int XgbModel::miss_value = -10;
const float XgbModel::miss_prob = 0.001;
int old_s = 0, new_s = 0;

void _split(const std::string& s, std::vector<std::string>& v, const char* c) {
#ifdef SATISFACTION_TEST_MODEL
  boost::split(v, s, boost::is_any_of(c));
#else
  sogou::util::SplitStringExp(s,v,c);
#endif
}

TreeNode::TreeNode(int _f, int _l, int _r, int _m, double _v, bool _i)
    : fid(_f), left(_l), right(_r), miss(_m), val(_v), is_leaf(_i) {}

XgbTree::XgbTree(const string& str) {
  vector<string> nodes_str;
  vector<string> tmp;
  _split(str, nodes_str, "\t");
  for (size_t i = 0; i < nodes_str.size(); ++i) {
    tmp.clear();
    _split(nodes_str[i], tmp, " ");
    if (tmp.size() == 6) {
      nid[stoi(tmp[0])] = i;
      nodes.push_back(TreeNode(stoi(tmp[1]), stoi(tmp[3]), stoi(tmp[4]), stoi(tmp[5]), stod(tmp[2]), false));
    } else if (tmp.size() == 2) {
      nid[stoi(tmp[0])] = i;
      nodes.push_back(TreeNode(-1, -1, -1, -1, stod(tmp[1]), true));
    }
  }

  unordered_map<int, XgbTreeNode*> id_to_node;
  if (nodes.size() > 0) {
    root = new XgbTreeNode(nodes[0], 0);
    id_to_node[0] = root;
    vector<XgbTreeNode*> queue;
    queue.push_back(root);
    size_t op = 0;
    while (op < queue.size()) {
      XgbTreeNode* node = queue[op++];
      TreeNode* proto_node = &nodes[node->node_id];
      id_to_node[node->node_id] = node;
      if (node->is_leaf)
        continue;
      node->leftChild = new XgbTreeNode(nodes[nid[proto_node->left]], nid[proto_node->left]);
      node->rightChild = new XgbTreeNode(nodes[nid[proto_node->right]], nid[proto_node->right]);
      queue.push_back(node->leftChild);
      queue.push_back(node->rightChild);
    }
    for (size_t i = 0; i < nodes.size(); i++) {
      if (nodes[i].is_leaf)
        continue;
      if (id_to_node.find(i) != id_to_node.end() && id_to_node.find(nid[nodes[i].miss]) != id_to_node.end()) {
        id_to_node[i]->missNode = id_to_node[nid[nodes[i].miss]];
      } else {
        fprintf(stderr, "build xgb tree mis node err!\n");
      }
    }
  }
}

XgbTree::~XgbTree() {}

double XgbTree::getVal(const bool* fea_exist, const double* fea_map) {
  XgbTreeNode* node = root;
  if (node == NULL) return 0;
  while (!node->is_leaf) {
    int fid = node->feature_id;
    if (fid < MAX_FEATURE_ID && fea_exist[fid] > 0) {
      if (fea_map[fid] < node->val) {
        node = node->leftChild;
      } else {
        node = node->rightChild;
      }
    } else {
      node = node->missNode;
    }
  }
  return node->val;
}

XgbModel::XgbModel(const string& path) {
  std::ifstream fin(path);
  if (!fin.is_open()) {
    fprintf(stderr, "Open Xgb Model file error!\n");
    exit(-1);
  }
  string line;
  while (std::getline(fin, line)) {
    trees.push_back(XgbTree(line));
  }
}

void XgbModel::predict(const std::vector<IdWeight>& fea, std::array<double, CLASS_NUM>& prob) {
  bool feature_exist[MAX_FEATURE_ID] = {0};
  double feature_map[MAX_FEATURE_ID];
  double prob_raw[CLASS_NUM] = {0.0};
  double prob_raw_sum = 0.0;
  for (size_t i = 0; i < fea.size(); ++i) {
    if (fea[i].id < MAX_FEATURE_ID) {
      feature_exist[fea[i].id] = 1;
      feature_map[fea[i].id] = fea[i].weight;
    }
  }
  for (size_t i = 0; i < trees.size(); i++) {
    int class_index = (i % CLASS_NUM);
    prob_raw[class_index] += trees[i].getVal(feature_exist, feature_map);
  }
  for (int i = 0; i < CLASS_NUM; i++) {
    prob_raw_sum += exp(prob_raw[i]);
  }
  for (int i = 0; i < CLASS_NUM; i++) {
    prob[i] = exp(prob_raw[i]) / prob_raw_sum;
  }
  return;
}
