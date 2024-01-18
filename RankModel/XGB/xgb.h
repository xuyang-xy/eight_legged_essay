#include <array>
#include <string>
#include <unordered_map>
#include <vector>

constexpr int MAX_FEATURE_ID = 999;
constexpr int CLASS_NUM = 5;

class TreeNode {
 public:
  TreeNode(int _f, int _l, int _r, int _m, double _v, bool _i);
  int fid;
  int left;
  int right;
  int miss;
  double val;
  bool is_leaf;
};

struct IdWeight {
  IdWeight(int id_, double weight_) : id(id_), weight(weight_) {}
  int id;
  double weight;
};

struct XgbTreeNode {
  XgbTreeNode(const TreeNode& node, const int node_id) {
      this->feature_id = node.fid;
      this->val = node.val;
      this->is_leaf = node.is_leaf;
      this->node_id = node_id;
  }
  int feature_id;
  double val;
  bool is_leaf;
  int node_id;
  XgbTreeNode *leftChild=NULL, *rightChild=NULL, *missNode=NULL;
};

class XgbTree {
 public:
  XgbTree(const std::string& str);
  ~XgbTree();
  double getVal(const bool* fea_exist, const double* fea_map);

 private:
  XgbTreeNode* root = NULL;
  std::vector<TreeNode> nodes;
  std::unordered_map<int, int> nid;
};
class XgbModel {
 public:
  XgbModel(){};
  XgbModel(const std::string& path);
  void predict(const std::vector<IdWeight>& fea, std::array<double, CLASS_NUM>& prob);
  static const int miss_value;
  static const float miss_prob;
  static int hasload;

 private:
  std::vector<XgbTree> trees;
};

