#pragma once
#include "globals.h"


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

