#pragma once
#include "globals.h"
#include<limits>

/**
 * Modifica uma matriz de afinidade W, tornando zero os valores que não correspondem
 * a arestas no grafo KNN, e o valor 1 para os demais.
 * @param W matriz de afinidade
 * @param k valor usado no grafo KNN
 * @param mutual usa KNN mutuo se verdadeiro
 */
Mat build_KNN_Mat(Mat& W, int k, bool mutual = false) {
	Mat K = Mat::Zero(W.rows(), W.rows());
	for (int i = 0; i < W.rows(); i++) {
		std::vector<std::pair<double,int> > distPairs;
		CVec vec = W.row(i);
		for (int j = 0; j < W.rows(); j++) {
			if (j == i) continue;
			//Joga o negativo da afinidade
			distPairs.push_back(std::pair<double,int>(-vec[j],j));
		}
		std::sort(distPairs.begin(),distPairs.end());
		for (int j = 0; j < k; j++) {
			K(i,distPairs[j].second) = 1;
			if(!mutual)K(distPairs[j].second,i) = 1;
		}
	}

	if (mutual){
		for (int i = 0; i < W.rows(); i++) {
			for (int j = 0; j < W.rows(); j++) {
				if (K(i,j)==0) {
					K(j,i) = 0;
				}
			}
		}
	}
	return K;
}


/**
 * Modifica uma matriz de afinidade W, tornando zero os valores que não correspondem
 * a arestas no grafo KNN, e os valores originais para os demais.
 * @param W matriz de afinidade
 * @param k valor usado no grafo KNN
 * @param mutual usa KNN mutuo se verdadeiro
 */
Mat restrict_Neighbours(Mat& W, int k, bool mutual = false) {
	Mat K = build_KNN_Mat(W, k, mutual);
	for (int i = 0; i < W.rows(); i++) {
		W.row(i) = W.row(i).cwiseProduct(K.row(i));
	}
	return W;
}

/**
 * Cria matriz de afinidade S (simetricamente normalizada)
 * @param X matriz cujas linhas correspondem a pontos
 * @param double sigma usado no heat kernel
 * @param k caso k!=-1, torna-se zero as arestas fora do grafo KNN
 *
 */
Mat build_LG_S(Mat &X, double sigma, int k = -1){
		if (k == -1) {
			k = AFFINITY_K;
		}

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
				W(i,j) = d;
			}
		}

		//Entradas j > i
		for (int i = 0; i < X.rows(); i++) {
			for (int j = i+1; j < X.rows(); j++) {
				double d = W(j,i);
				W(i,j) = d;
			}
		}
		if (k > -1) {
			W = restrict_Neighbours(W,k);
		}
		//D <- D^{-1/2}
		for (int i = 0; i < X.rows(); i++) {
			D(i,i) = W.row(i).sum();
			if (D(i,i) == 0) {
				D(i,i) = 1;
			} else {
				D(i,i) = 1.0/ std::sqrt(D(i,i));
			}
		}



		return (D*W*D);
}

/**
 * Cria matriz de afinidade W (sem normalizacao)
 * @param X matriz cujas linhas correspondem a pontos
 * @param double sigma usado no heat kernel
 * @param k caso k!=-1, torna-se zero as arestas fora do grafo KNN
 *
 */
Mat build_LG_W(Mat &X, double sigma, int k = -1){
		if (k == -1) {
			k = AFFINITY_K;
		}

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
				W(i,j) = d;
			}
		}

		//Entradas j > i
		for (int i = 0; i < X.rows(); i++) {
			for (int j = i+1; j < X.rows(); j++) {
				double d = W(j,i);
				W(i,j) = d;
			}
		}
		if (k > -1) {
			W = restrict_Neighbours(W,k);
		}
		//D <- D^{-1/2}
		for (int i = 0; i < X.rows(); i++) {
			D(i,i) = W.row(i).sum();
			if (D(i,i) == 0) {
				D(i,i) = 1;
			} else {
				D(i,i) = 1.0/ std::sqrt(D(i,i));
			}
		}



		return W;
}


/**
 * Cria matriz D (cada entrada na diagonal corresponde a soma correspondente em W)
 * @param X matriz cujas linhas correspondem a pontos
 * @param double sigma usado no heat kernel
 * @param k caso k!=-1, torna-se zero as arestas fora do grafo KNN
 *
 */
Mat build_LG_D(Mat &X, double sigma, int k = -1){
		if (k == -1) {
			k = AFFINITY_K;
		}

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
				W(i,j) = d;
			}
		}

		//Entradas j > i
		for (int i = 0; i < X.rows(); i++) {
			for (int j = i+1; j < X.rows(); j++) {
				double d = W(j,i);
				W(i,j) = d;
			}
		}
		if (k > -1) {
			W = restrict_Neighbours(W,k);
		}
		//D <- D^{-1/2}
		for (int i = 0; i < X.rows(); i++) {
			D(i,i) = W.row(i).sum();
			if (D(i,i) == 0) {
				D(i,i) = 1;
			} else {
				D(i,i) = 1.0/ std::sqrt(D(i,i));
			}
		}



		return D;
}


/**
 * Cria matriz de afinidade P (normalizada pela linha)
 * @param X matriz cujas linhas correspondem a pontos
 * @param double sigma usado no heat kernel
 * @param k caso k!=-1, torna-se zero as arestas fora do grafo KNN
 *
 */
Mat build_LG_P(Mat &X, double sigma, int k = -1){
		if (k == -1) {
			k = AFFINITY_K;
		}


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

		if (k > -1) {
			W = restrict_Neighbours(W,k);
		}

		//D <- D^{-1}
		for (int i = 0; i < X.rows(); i++) {
			if (D(i,i) == 0) {
				D(i,i) = 1;
			} else {
				D(i,i) = 1.0/ (D(i,i));
			}

		}

		Mat res = D*W;
		if (F_DEBUG) {
			for (int i = 0; i < X.rows(); i++) {
				double d = res.row(i).sum();
				if (std::abs(d-1.0) > 0.01) {
					throw std::logic_error(std::string("D-1_W matrix with row sum different than 1 - row ") +
							 std::to_string(i) + std::string(", sum ") + std::to_string(d) );
				}
			}
		}
		return (res);
}


