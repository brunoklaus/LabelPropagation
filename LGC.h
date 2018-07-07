#pragma once
#include "utils.h"
#include "MatrixBuilder.h"
#include "globals.h"




int LGC_Iter(Mat& S, Mat &X, CVec &Y, Mat &Init, double alpha, double sigma, int numIter, std::string outputFolder){

	Mat I = Mat::Identity(S.rows(), S.rows());
	Mat res = (I - alpha*S);
	res = res.inverse();
	res = res * Init;
	res = (1 - alpha) * res;
	CVec csfc = getClassification(res);

	Mat F = Mat(Init);
	CVec F_csfc =  getClassification(F);

	CVec convergence_dist = CVec::Zero(numIter + 1);
	CVec classification_dist = CVec::Zero(numIter + 1);



	int temp;

	for (int t = 0; t < numIter; t++)
	{

		if (t == 0) {
			convergence_dist(0) = (res-Init).norm();
			classification_dist(0) = getErrorRate(csfc,F_csfc);
		} else {
			F = alpha*S*F + (1-alpha)*Init;
			convergence_dist(t) = (res-F).norm();
			F_csfc =  getClassification(F);
			classification_dist(t) = getErrorRate(csfc,F_csfc);
		}

		std::ostringstream filename;
		filename << outputFolder <<  "result_iter_" << t << ".txt";
		writeToFile(F,filename.str());
		if (classification_dist(t) == 0) {
			temp = t;
			break;
		}

	}

	std::ostringstream filename1,filename2,filename3;
	filename1 << outputFolder <<  "norm_dist.txt";
	filename2 << outputFolder <<  "classification_dist.txt";
	filename3 << outputFolder <<  "final.txt";


	writeToFile(convergence_dist,filename1.str());
	writeToFile(classification_dist,filename2.str());
	writeToFile(res,filename3.str());

	return temp;

}


Mat KNN(Mat& W, Mat& Init, int k = -1) {
	if (k == -1) {
		k = AFFINITY_K;
	}

	std::ostringstream filename;
	filename << outputFolder <<  "Init.txt";
	writeToFile(Init, filename.str());
	int numLabeled = 0;
	CVec labeled(Init.rows());
	for (int i = 0; i < Init.rows(); i++) {
		if (Init.row(i).sum() > 0) {
			labeled(i) = 1;
			numLabeled++;
		} else {
			labeled(i) = 0;
		}
	}

	if (k >= numLabeled){
		k = numLabeled;
	}
	Mat res(Init);
	for (int i = 0; i < Init.rows(); i++) {
		if (labeled(i) == 1 ) continue;

		std::vector <std::pair<double, int> > vec;
		for (int j = 0; j < W.rows(); j++) {
			std::pair<double, int> p (-W(i,j),j);
			if (j != i){vec.push_back(p);}
		}
		std::sort(vec.begin(),vec.end());
		int current_k = 1;
		for (int j = 0; current_k <= k; j++) {
			if (labeled(vec[j].second) == 1) {
				res.row(i) += Init.row(vec[j].second);
				current_k++;

			}
		}

	}


	std::ostringstream filename2;
	filename2 << outputFolder <<  "Final.txt";
	writeToFile(res, filename2.str());
	return res;
}



/**
 * Calcula a matriz resultante do metodo LGC via inversa.
 */
Mat LGC(Mat& S, Mat& Init, double alpha){

	if (Init.rows() != S.rows()) {
		throw new std::invalid_argument("Wrong Init Matrix");
	}

	Mat I = Mat::Identity(S.rows(), S.rows());

	/*Mat F = Mat(Init);
	for (int t = 0; t < 100; t++)
	{
		writeToFile(F,std::string("./INPUT_FILES/result_") + std::to_string(t) + ".txt" );
		F = alpha*S*F + (1-alpha)*Init;
	}
	*/
	Mat res = (I - alpha*S);
	res = res.inverse();
	res = res * Init;
	res = (1 - alpha) * res;
	return res;
}

/**
 * Calcula a matriz resultante do metodo LGC via inversa.
 */
Mat LGC_Iter(Mat& S, Mat& Init, double alpha, int numIter = 1000){

	if (Init.rows() != S.rows()) {
		throw new std::invalid_argument("Wrong Init Matrix");
	}

	Mat I = Mat::Identity(S.rows(), S.rows());

	Mat F = Mat(Init);
	for (int t = 0; t < numIter; t++)
		{
			F = alpha*S*F + (1-alpha)*Init;
		}
	return F;
}



Mat LGC_Test_From_S(Mat &S, Mat &Init, CVec &Y, double* absError, double* probError,  double alpha, int numIter = -1){

		if (alpha < 0) alpha = 0;
		if (alpha > 1) alpha = 1;
		Mat res;
		if (numIter == -1){
			res = LGC(S, Init, alpha);
		} else {
			res = LGC_Iter(S, Init, alpha, numIter);
		}
		CVec csfc = getClassification(res);
		double error = getErrorRate(Y, csfc);
		double prob_error = getProbabilisticError(Y, res);

		if (absError != NULL) {
			*absError = error;
		}
		if (probError != NULL) {
			*probError = prob_error;
		}
		return res;
}
Mat LGC_Test(Mat &X, Mat &Init, CVec &Y, double* absError, double* probError,  double alpha, double sigma, bool printOutput = false, int k = -1) {
	Mat S = build_LG_S(X, sigma,k);
	Mat res = LGC_Test_From_S(S,Init,Y, absError,probError,alpha);
	double abs_error;
	double prob_error;
	if (absError == NULL) {
		absError = &abs_error;
	}
	if (probError == NULL) {
		probError = &prob_error;
	}

	if (printOutput) {
		std::cout << "Error for alpha=" << alpha <<",var=" <<
				sigma << " is " << *absError << " / " << *probError << std::endl;
		std::cout << getClassFreq(Y) << std::endl;
	}


}



