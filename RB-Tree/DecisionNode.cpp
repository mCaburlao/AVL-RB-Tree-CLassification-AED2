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

DecisionNode::DecisionNode(
    double splitLoss,
    vector<double> values,
    Color color,
    int splitColumn,
    Generic* splitValue,
    DecisionNode* left,
    DecisionNode* right,
    DecisionNode* parent
) {
    this->splitColumn = splitColumn;
    this->splitValue = splitValue;
    this->color = color;
    this->splitLoss = splitLoss;
    this->values = values;
    this->left = left;
    this->right = right;
    this->parent = parent;
}

bool DecisionNode::isLeaf() {
    return left == nullptr && right == nullptr;
}