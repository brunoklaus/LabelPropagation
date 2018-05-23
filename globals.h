#pragma once
#include<eigen3/Eigen/Dense>
#include <iostream>
#include <fstream>
#include<string>
#include<random>
#include <ctime>
#include<stdexcept>


std::string inputFile("./INPUT_FILES/benchmark/");	//Endereco pasta com arquivos
std::string outputFile("./INPUT_FILES/");  //endereço pasta saida

int setN = 2;
int NUM_CLASSES = 2;
int NUM_SAMPLES = 100;
using namespace Eigen;

typedef Matrix<double,Dynamic,Dynamic,Eigen::RowMajor> Mat;
typedef Eigen::Matrix<double, 1, Eigen::Dynamic> RVec;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> CVec;
