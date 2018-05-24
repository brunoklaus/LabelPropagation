#include "globals.h"
#include "utils.h"
#include "MatrixBuilder.h"
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_vector_double.h>

Mat localGlobal(Mat X, Mat Init, double alpha, double sigma){

	if (Init.rows() != X.rows()) {
		throw new std::invalid_argument("Wrong Init Matrix");
	}

	Mat I = Mat::Identity(X.rows(), X.rows());
	Mat S = build_LG_S(X, sigma);

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


void LGTest(Mat &X, Mat &Init, CVec &Y, double* absError, double* probError,  double alpha, double sigma){
		Mat res = localGlobal(X, Init, alpha, sigma);
		CVec csfc = getClassification(res);
		double error = getErrorRate(Y, csfc);
		double prob_error = getProbabilisticError(Y, res);
		//std::cout << "Error for alpha=" << alpha <<",var=" <<
				//sigma << " is " << error << " / " << prob_error << std::endl;
		//std::cout << getClassFreq(csfc) << std::endl;
		//writeToFile(res,outputFile +
		//		"result_" + std::to_string(alpha) + "_" + std::to_string(sigma) + ".txt");
		if (absError != NULL) {
			*absError = error;
		}
		if (probError != NULL) {
			*probError = prob_error;
		}
}



typedef struct LGParams{
	Mat& X;
	Mat& Init;
	CVec& Y;
	LGParams(Mat& x, Mat& init, CVec& y) : X(x),Init(init),Y(y){};
};

double
my_f (const gsl_vector *v, void *params)
{
  double alpha, sigma;
  alpha = gsl_vector_get(v,0);
  sigma = gsl_vector_get(v,1);

  LGParams *p = (LGParams *)params;
  double* error = (double*) malloc(sizeof(double));
  LGTest(p->X, p->Init, p->Y,NULL,error, alpha, sigma);

  double e = *error;
  free(error);
  return e;
}

int GSLOptimize(Mat& X, Mat& Init, CVec& Y, double initAlpha, double initSigma) {
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

	    gsl_vector_free(v);
	    gsl_vector_free(ss);
	    gsl_multimin_fminimizer_free (s);

	    return status;
}


void go(int NUM_EXPERIMENT){
	    std::srand(std::time(0)); //use current time as seed for random generator
		Mat X;
		CVec Y;
		getInputMatrices(inputFile,setN, X,Y);

		Mat Init = getInitialProb(Y,getSamples(Y.rows(),NUM_SAMPLES,Y));

		std::cout << Map<CVec> (Y.data(),4) << std::endl;
		std::cout << Map<Mat> (Init.data(),4,2) << std::endl;


		LGTest(X, Init, Y, NULL,NULL,0.99,0.1);


/*
		build_KNN_Mat(X,  100);
		return;


		GSLOptimize(X,Init,Y,0.1,1);
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

		for (int i = 0; i < alphas.size(); i++) {
			for (int j = 0; j < sigmas.size(); j++) {
				double alpha = alphas[i];
				double sigma = sigmas[j];
				std::cout << "|";
				std::cout.flush();
				errMat(i*sigmas.size() + j, 0) = alpha;
				errMat(i*sigmas.size() + j, 1) = sigma;
				LGTest(X, Init, Y, abs_error, prob_error, alpha, sigma);
				errMat(i*sigmas.size() + j, 2) = *abs_error;
				errMat(i*sigmas.size() + j, 3) = *prob_error;
			}
		}
		std::cout<<std::endl;
		free(abs_error);
		free(prob_error);
		std::ostringstream filename;
		filename << outputFile <<  "set_" << setN << "_run_" << NUM_EXPERIMENT << "_samples_" << NUM_SAMPLES << "_error.txt" ;

		std::ostringstream filename2;
		filename2 << outputFile <<  "set_" << setN << "_run_" << NUM_EXPERIMENT <<  "_class.txt" ;

		writeToFile(errMat, filename.str());
		//writeToFile(errMat, filename2.str());

		//std::cout << errMat << std::endl;

}

////////////////////////////////////////////////////////////////////////////

int main (void) {

		std::vector<int> sets({1,3,4,5,6,7});
	std::vector<int> samples({10,20,30,100});




	for (int i = 1; i < 15; i++){
		std::cout << i << std::endl;
		for (int s = 0; s < sets.size(); s++) {
			setN = sets[s];
			for (int j = 0; j < samples.size(); j++) {
				NUM_SAMPLES = samples[j];
				go(i);
			}
		}
	}
	return 0;
}


