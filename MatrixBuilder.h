#pragma once
#include "globals.h"
#include<limits>

Mat build_LG_S(Mat &X, double sigma){
		Mat W (X.rows(), X.rows());
		Mat D = Mat::Zero(X.rows(), X.rows());

		//Entradas j == i
		for (int i = 0; i < X.rows(); i++) {
			W(i,i) = 0;
		}

		//Entradas j < i
		for (int i = 0; i < X.rows(); i++) {
			for (int j = 0; j < i; j++) {
				RVec v = (X.row(i) - X.row(j));
				double d = std::exp(-v.squaredNorm() /(2*sigma*sigma) );
				D(i,i) += d;
				W(i,j) = d;
			}
		}

		//Entradas j > i
		for (int i = 0; i < X.rows(); i++) {
			for (int j = i+1; j < X.rows(); j++) {
				double d = W(j,i);
				D(i,i) += d;
				W(i,j) = d;
			}
		}

		//D <- D^{-1/2}
		for (int i = 0; i < X.rows(); i++) {
			if (D(i,i) == 0) {
				D(i,i) = 1;
			} else {
				D(i,i) = 1.0/ std::sqrt(D(i,i));
			}
		}

		return (D*W*D);
}

Mat build_LG_P(Mat &X, double sigma){
		Mat W (X.rows(), X.rows());
		Mat D = Mat::Zero(X.rows(), X.rows());

		//Entradas j == i
		for (int i = 0; i < X.rows(); i++) {
			W(i,i) = 0;
		}

		//Entradas j < i
		for (int i = 0; i < X.rows(); i++) {
			for (int j = 0; j < i; j++) {
				RVec v = (X.row(i) - X.row(j));
				double d = std::exp(-v.squaredNorm() /(2*sigma*sigma) );
				D(i,i) += d;
				W(i,j) = d;
			}
		}

		//Entradas j > i
		for (int i = 0; i < X.rows(); i++) {
			for (int j = i+1; j < X.rows(); j++) {
				double d = W(j,i);
				D(i,i) += d;
				W(i,j) = d;
			}
		}

		//D <- D^{-1/2}
		for (int i = 0; i < X.rows(); i++) {
			if (D(i,i) == 0) {
				D(i,i) = 1;
			} else {
				D(i,i) = 1.0/ (D(i,i));
			}

		}

		return (D*W);
}


Mat build_KNN_Mat(Mat &X, int k){
		Mat W (X.rows(), X.rows());
		Mat K (X.rows(), X.rows());

		for (int i = 0; i < X.rows(); i++) {
			for (int j = 0; j < X.rows(); j++) {
				RVec v = (X.row(i) - X.row(j));
				double d = v.squaredNorm();
				W(i,j) = d;
			}
		}


		for (int i = 0; i < X.rows(); i++) {
			bool* used = (bool*) calloc(X.rows(),sizeof(bool));

			for (int l = 0; l < k; l++) {
				int minIndex = -1;
				double minVal = std::numeric_limits<double>::max();

				for (int j = l; j < X.rows(); j++) {
					if (used[j] == true){
						continue;
					}
					if (W(i,j) < minVal) {
						minVal = W(i,j);
						minIndex = j;
					}
				}
				K(i,minIndex) = 1.0;
				used[minIndex] = true;

			}
			free(used);
		}
		if (F_DEBUG) {
			for (int i = 0; i < X.rows(); i++) {
				if (K.row(i).sum() != k) {
					throw std::logic_error(std::string("KNN matrix with row sum different than k - row ") +
							 std::to_string(i) + std::string(", sum ") + std::to_string(K.row(i).sum()));
				}
			}
		}
		return K;
}

