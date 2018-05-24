#pragma once
#include<eigen3/Eigen/Dense>
#include<eigen3/Eigen/Sparse>
#include <iostream>
#include <fstream>
#include<string>
#include<random>
#include <ctime>
#include<stdexcept>
#include<sstream>
#include <cmath>
#define F_DEBUG false

std::string inputFile("./INPUT_FILES/benchmark/");	//Endereco pasta com arquivos
std::string outputFile("./OUTPUT_FILES/1/");  //endereço pasta saida

int setN = 2;
int NUM_CLASSES = 2;
int NUM_SAMPLES = 100;
using namespace Eigen;



typedef Matrix<double,Dynamic,Dynamic,Eigen::RowMajor> Mat;
typedef Eigen::SparseMatrix<double,Eigen::RowMajor> SMat;

typedef Eigen::Matrix<double, 1, Eigen::Dynamic> RVec;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> CVec;
