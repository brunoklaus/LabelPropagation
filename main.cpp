#include "globals.h"
#include "utils.h"
#include "MatrixBuilder.h"
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_vector_double.h>



typedef struct LGParams{
	Mat& X;
	Mat& Init;
	CVec& Y;
	LGParams(Mat& x, Mat& init, CVec& y) : X(x),Init(init),Y(y){};
};

typedef struct LG_GSL_Return{
	double alpha;
	double sigma;
};


std::string getFilename(int NUM_EXPERIMENT, std::string name){
	std::ostringstream filename;
			filename << outputFile <<  "set_" << setN << "_run_" << NUM_EXPERIMENT << "_samples_" << NUM_SAMPLES << "_" << name;
	return filename.str();
}


Mat* localGlobal_Iter_test(Mat X, Mat Init, double alpha, double sigma, int numIter, bool writeResult = false){

	Mat S = build_LG_S(X, sigma);
	Mat I = Mat::Identity(S.rows(), S.rows());

	Mat res = (I - alpha*S);
	res = res.inverse();
	res = res * Init;
	res = (1 - alpha) * res;

	Mat F = Mat(Init);
	CVec convergence_dist = CVec::Zero(numIter + 1);
	convergence_dist(0) = (res-Init).norm();

	for (int t = 0; t < numIter; t++)
	{
		std::ostringstream filename;
		filename << outputFile <<  "result_iter_" << t << ".txt";

		if(writeResult)
			writeToFile(F,filename.str());
		F = alpha*S*F + (1-alpha)*Init;
		convergence_dist(t+1) = (res-F).norm();
	}

	std::ostringstream filename;
	filename << outputFile <<  "convergence.txt";


	if(writeResult)
		writeToFile(convergence_dist,filename.str());
	return F;

}

Mat localGlobal(Mat S, Mat Init, double alpha){

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




Mat LGTest_From_S(Mat &S, Mat &Init, CVec &Y, double* absError, double* probError,  double alpha){

		if (alpha < 0) alpha = 0;
		if (alpha > 1) alpha = 1;

		Mat res = localGlobal(S, Init, alpha);
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
void LGTest(Mat &X, Mat &Init, CVec &Y, double* absError, double* probError,  double alpha, double sigma, bool saveToFile = false, bool printOutput = false) {
	Mat S = build_LG_S(X, sigma);
	Mat res = LGTest_From_S(S,Init,Y, absError,probError,alpha);
	double abs_error;
	double prob_error;
	if (absError == NULL) {
		absError = &abs_error;
	}
	if (probError == NULL) {
		probError = &prob_error;
	}

	if (saveToFile){
		writeToFile(res,getFilename(0, std::string("prediction.txt")) );
	}
	if (printOutput) {
		std::cout << "Error for alpha=" << alpha <<",var=" <<
				sigma << " is " << *absError << " / " << *probError << std::endl;
		std::cout << getClassFreq(Y) << std::endl;
	}


}




double
my_f (const gsl_vector *v, void *params)
{
  double alpha, sigma;
  alpha = gsl_vector_get(v,0);
  sigma = gsl_vector_get(v,1);

  LGParams *p = (LGParams *)params;
  double* error = (double*) malloc(sizeof(double));
  LGTest(p->X, p->Init, p->Y,NULL,error, alpha,sigma);

  double e = *error;
  free(error);
  return e;
}

LG_GSL_Return GSLOptimize(Mat& X, Mat& Init, CVec& Y, double initAlpha, double initSigma) {
	  gsl_vector *v = gsl_vector_alloc (2);
	  gsl_vector_set (v, 0, initAlpha);
	  gsl_vector_set (v, 1, initSigma);

	  /* Set initial step sizes to 0.1 */
	  gsl_vector *ss = gsl_vector_alloc (2);
	  gsl_vector_set (ss, 0, 0.5);
	  gsl_vector_set (ss, 1, 0.5);


	   size_t iter = 0;
	   int status;
	   double size;

	  LGParams* p = new LGParams(X,Init,Y);

	  gsl_multimin_function my_func;
	  my_func.n = 2;
	  my_func.f = my_f;
	  my_func.params = (void*) p;

	  const gsl_multimin_fminimizer_type *T =
	      gsl_multimin_fminimizer_nmsimplex2;
	  gsl_multimin_fminimizer *s = NULL;
	  s = gsl_multimin_fminimizer_alloc (T, 2);
	  gsl_multimin_fminimizer_set (s, &my_func, v, ss);

	    do
	      {
	        iter++;
	        status = gsl_multimin_fminimizer_iterate(s);

	        if (status)
	          break;

	        size = gsl_multimin_fminimizer_size (s);
	        status = gsl_multimin_test_size (size, 1e-2);

	        if (status == GSL_SUCCESS)
	          {
	            printf ("converged to minimum at\n");
	          }

	        printf ("%5d %10.3e %10.3e f() = %7.3f size = %.3f\n",
	                iter,
	                gsl_vector_get (s->x, 0),
	                gsl_vector_get (s->x, 1),
	                s->fval, size);
	      }
	    while (status == GSL_CONTINUE && iter < 100);
	    LG_GSL_Return r;
	    r.alpha = gsl_vector_get (s->x, 0);
	    r.sigma = gsl_vector_get (s->x, 1);

	    gsl_vector_free(v);
	    gsl_vector_free(ss);
	    gsl_multimin_fminimizer_free (s);

	    return r;
}

void readData(){



}
void experiment_convergence(){
	outputFile = "./OUTPUT_FILES/2/";
	std::srand(std::time(0)); //use current time as seed for random generator
	Mat X;
	CVec Y;
	getInputMatrices(inputFile,setN, X,Y);
	Mat Init = getInitialProb(Y,getSamples(Y.rows(),NUM_SAMPLES,Y));

	LG_GSL_Return best = GSLOptimize(X,Init,Y,0.1,0.1);
	localGlobal_Iter_test(X,Init, best.alpha, best.sigma, 30,true);
}
void experiment1_aux(int NUM_EXPERIMENT){

		std::srand(std::time(0)); //use current time as seed for random generator
		Mat X;
		CVec Y;
		getInputMatrices(inputFile,setN, X,Y);
		Mat Init = getInitialProb(Y,getSamples(Y.rows(),NUM_SAMPLES,Y));

		std::cout << Map<CVec> (Y.data(),4) << std::endl;
		std::cout << Map<Mat> (Init.data(),4,2) << std::endl;

		Mat errMat;
		int i = 0;
		int j = 0;

		std::vector<double> alphas({0,0.2,0.4,0.6,0.8});
		std::vector<double> sigmas({0.05,0.1,0.15});

		for (double alpha = 0.9; alpha < 1; alpha += 0.1) {
			alphas.push_back(alpha);
		}
		alphas.push_back(0.95);
		alphas.push_back(0.99);
		alphas.push_back(0.999);
		alphas.push_back(0.9999);


		for (double var = 0.2; var <= 1; var += 0.1) {
			sigmas.push_back(var);
		}
		sigmas.push_back(2);
		sigmas.push_back(4);


		errMat.resize(alphas.size() * sigmas.size(),4);

		double* abs_error = (double*) malloc(sizeof(double));
		double* prob_error = (double*) malloc(sizeof(double));

		for (int i = 0; i < sigmas.size(); i++) {
			Mat S = build_LG_S(X, sigmas[i]);
			double sigma = sigmas[i];
			for (int j = 0; j < alphas.size(); j++) {
				double alpha = alphas[j];
				std::cout << "|";
				std::cout.flush();
				errMat(i*alphas.size() + j, 0) = alpha;
				errMat(i*alphas.size() + j, 1) = sigma;
				LGTest_From_S(S, Init, Y, abs_error, prob_error, alpha);
				errMat(i*alphas.size() + j, 2) = *abs_error;
				errMat(i*alphas.size() + j, 3) = *prob_error;
			}
		}
		std::cout<<std::endl;
		free(abs_error);
		free(prob_error);
		writeToFile(errMat, getFilename(NUM_EXPERIMENT, "error.txt"));
	}

void experiment1(){
	outputFile = "./OUTPUT_FILES/1/";

	std::vector<int> sets({0,1,3,4,5,6,7});
	std::vector<int> samples({10,20,30,100});

	sets.clear();
	std::cout << "Indique qual set, bem como o id min e max para cada repeticao \n";
	int chosenSet,minRun,maxRun;
	std::cin >> chosenSet >> minRun >> maxRun;






	for (int i = minRun; i < maxRun; i++){
		std::cout << i << std::endl;
		for (int s = 0; s < sets.size(); s++) {
			setN = sets[s];
			if (setN == 6) {
				NUM_CLASSES = 6;
			} else {
				NUM_CLASSES = 2;
			}
			for (int j = 0; j < samples.size(); j++) {
				NUM_SAMPLES = samples[j];
				experiment1_aux(i);
			}
		}
	}

	}

void sometests(){
	/*		return;


		build_KNN_Mat(X,  100);
		return;

		return;
		LGTest(X, Init, Y, 0.99,10);
		LGTest(X, Init, Y, 0.99,2);
		LGTest(X, Init, Y, 0.99,1);
		LGTest(X, Init, Y, 0.99,0.5);
		LGTest(X, Init, Y, 0.99,0.2);
		LGTest(X, Init, Y, 0.99,0.1);
		LGTest(X, Init, Y, 0.99,0.05);
		LGTest(X, Init, Y, 0.99,0.01);
		LGTest(X, Init, Y, 0.99,0.005);





		LGTest(X, Init, Y, 0.9,10);
		LGTest(X, Init, Y, 0.9,2);
		LGTest(X, Init, Y, 0.9,1);
		LGTest(X, Init, Y, 0.9,0.5);
		LGTest(X, Init, Y, 0.9,0.2);
		LGTest(X, Init, Y, 0.9,0.1);
		LGTest(X, Init, Y, 0.9,0.05);
		LGTest(X, Init, Y, 0.9,0.01);
		LGTest(X, Init, Y, 0.9,0.005);

		return;
		*/


}




////////////////////////////////////////////////////////////////////////////

int main (void) {

	return 0;
}


