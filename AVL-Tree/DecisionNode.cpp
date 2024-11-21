/**
 * @file DecisionNode.cpp
 * @author John Nguyen (jvn1567@gmail.com)
 * @author Joshua Goldberg (joshgoldbergcode@gmail.com)
 * @brief Implementation of DecisionNode.h.
 * @version 0.1
 * @date 2021-09-28
 *
 * @copyright Copyright (c) 2021
 *
 */

#include "DecisionNode.h"

using namespace std;

DecisionNode::DecisionNode(double splitLoss, vector<double> values, int sampleSize,
                           int splitColumn, Generic *splitValue,
                           DecisionNode *left, DecisionNode *right) {
  this->splitColumn = splitColumn;
  this->splitValue = splitValue;
  this->splitLoss = splitLoss;
  this->count = sampleSize;
  this->values = values;
  this->left = left;
  this->right = right;
  this->height = 1;
}

bool DecisionNode::isLeaf() { return left == nullptr && right == nullptr; }

void DecisionNode::updateValues() {
  height = std::max(left != nullptr ? left->height : 0,
                    right != nullptr ? right->height : 0) +
           1;
  if (left != nullptr || right != nullptr) {
    count = (left != nullptr ? left->count : 0) + 
       (right != nullptr ? right->count : 0);
  }
}

int DecisionNode::balanceFactor() {
  return (left != nullptr ? left->height : 0) -
         (right != nullptr ? right->height : 0);
}

DecisionNode* DecisionNode::leftRotate(){
    DecisionNode* R = right;
    right = right->left;
    R->left = this;

    this->updateValues();  // the order is important
    R->updateValues();

    return R;
}

DecisionNode* DecisionNode::rightRotate(){
    DecisionNode* L = left;
    left = left->right;
    L->right = this;

    this->updateValues();  // the order is important
    L->updateValues();

    return L;
}