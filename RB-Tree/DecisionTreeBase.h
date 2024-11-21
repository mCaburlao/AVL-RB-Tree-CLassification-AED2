/**
 * @file DecisionTreeBase.h
 * @author John Nguyen (jvn1567@gmail.com)
 * @author Joshua Goldberg (joshgoldbergcode@gmail.com)
 * @brief DecisionTreeBase is a semi-virtual parent class that contains methods
 * used in both DecisionTreeClassifier and DecisionTreeRegressor.
 * @version 0.1
 * @date 2021-10-01
 *
 * @copyright Copyright (c) 2021
 *
 */

#ifndef _DECISIONTREEBASE_H
#define _DECISIONTREEBASE_H

#include "DataFrame.h"
#include "DecisionNode.h"
#include "GenericTypeWrapper.h"
#include <string>
#include <vector>

/**
 * @brief DecisionTreeBase is a semi-virtual parent class that contains methods
 * used in both DecisionTreeClassifier and DecisionTreeRegressor.
 *
 */
struct RedBlackTree {
  DecisionNode *root;

  RedBlackTree() { this->root = nullptr; };

  void insert(DecisionNode z) {
    DecisionNode *newNode = new DecisionNode(z.splitLoss, z.values, z.color,
                                             z.splitColumn, z.splitValue);
    DecisionNode *y = nullptr;
    DecisionNode *x = root;

    while (x != nullptr) {
      y = x;
      if (newNode->splitColumn < x->splitColumn)
        x = x->left;
      else
        x = x->right;
    }

    newNode->parent = y;
    if (y == nullptr)
      root = newNode;
    else if (newNode->splitColumn < y->splitColumn)
      y->left = newNode;
    else
      y->right = newNode;

    fixInsert(newNode);
  }

  void fixInsert(DecisionNode *z) {
    while (z != root && z->parent->color == RED) {
      if (z->parent == z->parent->parent->left) {
        DecisionNode *y = z->parent->parent->right;
        if (y != nullptr && y->color == RED) {
          z->parent->color = BLACK;
          y->color = BLACK;
          z->parent->parent->color = RED;
          z = z->parent->parent;
        } else {
          if (z == z->parent->right) {
            z = z->parent;
            leftRotate(z);
          }
          z->parent->color = BLACK;
          z->parent->parent->color = RED;
          rightRotate(z->parent->parent);
        }
      } else {
        DecisionNode *y = z->parent->parent->left;
        if (y != nullptr && y->color == RED) {
          z->parent->color = BLACK;
          y->color = BLACK;
          z->parent->parent->color = RED;
          z = z->parent->parent;
        } else {
          if (z == z->parent->left) {
            z = z->parent;
            rightRotate(z);
          }
          z->parent->color = BLACK;
          z->parent->parent->color = RED;
          leftRotate(z->parent->parent);
        }
      }
    }
    root->color = BLACK;
  }

  //  left rotation
  void leftRotate(DecisionNode *x) {
    if (x == nullptr || x->right == nullptr)
      return;

    DecisionNode *y = x->right;
    x->right = y->left;
    if (y->left != nullptr)
      y->left->parent = x;
    y->parent = x->parent;
    if (x->parent == nullptr)
      root = y;
    else if (x == x->parent->left)
      x->parent->left = y;
    else
      x->parent->right = y;
    y->left = x;
    x->parent = y;
  }

  //  right rotation
  void rightRotate(DecisionNode *y) {
    if (y == nullptr || y->left == nullptr)
      return;

    DecisionNode *x = y->left;
    y->left = x->right;
    if (x->right != nullptr)
      x->right->parent = y;
    x->parent = y->parent;
    if (y->parent == nullptr)
      root = x;
    else if (y == y->parent->left)
      y->parent->left = x;
    else
      y->parent->right = x;
    x->right = y;
    y->parent = x;
  }
};

class DecisionTreeBase {
protected:
  DecisionNode *root;
  RedBlackTree *redBlackTree;

private:
  std::string lossCriterion;
  int maxDepth;
  int minSamplesSplit;
  int minSamplesLeaf;
  double maxFeatures;
  double minImpurityDecrease;
  int *featureImportance;
  void fit(DataFrame *trainData, DecisionNode *&node, bool isLeft,
           DecisionNode *&parent);
  int findSplit(DataFrame *trainData, int &bestCol, double minLoss,
                Generic *&splitValue);
  int findRowSplitString(int col, DataFrame *&trainData, int &bestCol,
                         double &minLoss, Generic *&splitValue);
  int findRowSplitDouble(int col, DataFrame *&trainData, int &bestCol,
                         double &minLoss, Generic *&splitValue);
  void printTree(DecisionNode *node, int indents);
  virtual double computeLoss(std::vector<double>) = 0;
  virtual std::vector<double> getTruthVector(DataFrame *) = 0;
  virtual void printTruthVector(std::vector<double> truthVector) = 0;
  void printSpaces(int indents);
  void keepClassificationOnLeaf(DecisionNode *&node, DecisionNode *&leaf);
  void copyToRedBlack(DecisionNode *node);
  void fillSplit(DataFrame *trainData, DecisionNode *&node);

public:
  /**
   * @brief Construct a new DecisionTreeBase object.
   *
   * @param lossCriterion type of loss calculation used
   * @param maxFeatures number of features used for splitting a node
   * @param minSamplesSplit minimun number of samples at root to consider
   * splitting
   * @param maxDepth maxiumum depth of the decision tree
   * @param minSamplesLeaf minimum number of samples in child leafs to
   * consider the split
   * @param minImpurityDecrease minimum improvement in loss calculation to
   * consider the split
   */
  DecisionTreeBase(std::string lossCriterion, double maxFeatures,
                   int minSamplesSplit, int maxDepth, int minSamplesLeaf,
                   double minImpurityDecrease);

  /**
   * @brief This method creates the decision tree, which can be used later for
   * prediction.
   *
   * @pre fit() assumes computeLoss() private method is implemented by the
   * child class. Otherwise, the user can implement their own fit() if they
   * have a different algorithm for constructing a decision tree.
   *
   * @param trainData
   */
  virtual void fit(DataFrame *trainData);
  virtual DataFrame *predict(DataFrame *testData) = 0;
  std::string getLossCriterion() const;
  void printTree();
};

#endif