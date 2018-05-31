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
	int k;
};


std::string getFilename(int NUM_EXPERIMENT, std::string name){
	std::ostringstream filename;
			filename << outputFile <<  "set_" << setN << "_run_" << NUM_EXPERIMENT << "_samples_" << NUM_SAMPLES << "_" << name;
	return filename.str();
}


Mat localGlobal_Iter_test(Mat X, Mat Y, Mat Init, double alpha, double sigma, int numIter, bool writeResult = false){

	Mat S = build_LG_S(X, sigma);
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

	convergence_dist(0) = (res-Init).norm();
	classification_dist(0) = getErrorRate(csfc,F_csfc);


	for (int t = 0; t < numIter; t++)
	{
		std::ostringstream filename;
		filename << outputFile <<  "result_iter_" << t << ".txt";

		if(writeResult)
			writeToFile(F,filename.str());
		F = alpha*S*F + (1-alpha)*Init;
		convergence_dist(t+1) = (res-F).norm();
		F_csfc =  getClassification(F);
		classification_dist(t+1) = getErrorRate(csfc,F_csfc);
	}

	std::ostringstream filename1,filename2,filename3;
	filename1 << outputFile <<  "norm_dist.txt";
	filename2 << outputFile <<  "classification_dist.txt";
	filename3 << outputFile <<  "final.txt";


	if(writeResult){
		writeToFile(convergence_dist,filename1.str());
		writeToFile(classification_dist,filename2.str());
		writeToFile(res,filename3.str());
	}
	return res;

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
void LGTest(Mat &X, Mat &Init, CVec &Y, double* absError, double* probError,  double alpha, double sigma, bool saveToFile = false, bool printOutput = false, int k = -1) {
	Mat S = build_LG_S(X, sigma,k);
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
  int k = NUM_SAMPLES * gsl_vector_get(v,2);

  if (k < 0) {
  	  return 1;
   }

  if (k > NUM_SAMPLES) {
	  return 1;
  }

  LGParams *p = (LGParams *)params;
  double* error = (double*) malloc(sizeof(double));
  LGTest(p->X, p->Init, p->Y,error,NULL, alpha,sigma,false,false,k);

  double e = *error;
  free(error);
  return e;
}

LG_GSL_Return GSLOptimize(Mat& X, Mat& Init, CVec& Y, double initAlpha, double initSigma, double initK) {
	  gsl_vector *v = gsl_vector_alloc (3);
	  gsl_vector_set (v, 0, initAlpha);
	  gsl_vector_set (v, 1, initSigma);
	  gsl_vector_set (v, 2, initK);


	  /* Set initial step sizes to 0.1 */
	  gsl_vector *ss = gsl_vector_alloc (3);
	  gsl_vector_set (ss, 0, 0.5);
	  gsl_vector_set (ss, 1, 0.5);
	  gsl_vector_set (ss, 2, 0.5);


	   size_t iter = 0;
	   int status;
	   double size;

	  LGParams* p = new LGParams(X,Init,Y);

	  gsl_multimin_function my_func;
	  my_func.n = 3;
	  my_func.f = my_f;
	  my_func.params = (void*) p;

	  const gsl_multimin_fminimizer_type *T =
	      gsl_multimin_fminimizer_nmsimplex2;
	  gsl_multimin_fminimizer *s = NULL;
	  s = gsl_multimin_fminimizer_alloc (T, 3);
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

	        printf ("%5d %10.3e %10.3e %10.3e f() = %7.3f size = %.3f\n",
	                iter,
	                gsl_vector_get (s->x, 0),
	                gsl_vector_get (s->x, 1),
	                gsl_vector_get (s->x, 2),
	                s->fval, size);
	      }
	    while (status == GSL_CONTINUE && iter < 100);
	    LG_GSL_Return r;
	    r.alpha = gsl_vector_get (s->x, 0);
	    r.sigma = gsl_vector_get (s->x, 1);
	    r.k = gsl_vector_get (s->x, 2);

	    gsl_vector_free(v);
	    gsl_vector_free(ss);
	    gsl_multimin_fminimizer_free (s);

	    return r;
}

void experiment_visualize_knn(){
	setN = 5;
	NUM_SAMPLES = 100;
	outputFile = "./OUTPUT_FILES/5/";
	std::srand(std::time(0)); //use current time as seed for random generator
	double chosenAlpha, chosenSigma;

	std::cout << "Entre Com numero do conj, #samples, alpha e sigma";
	std::cin >> setN >> NUM_SAMPLES >> chosenAlpha >> chosenSigma;
	std::ostringstream filename;
	filename << outputFile <<  "set_" << setN << "_samples_" << NUM_SAMPLES << "_ks.txt" ;

	Mat X;
	CVec Y;
	getInputMatrices(inputFile,setN, X,Y);
	//LG_GSL_Return best = GSLOptimize(X,Init,Y,0.9,0.1);
	//Mat res = localGlobal_Iter_test(X,Y,Init, best.alpha, best.sigma, 1000,true);
	std::vector<int> ks({1,3,5,7,9,10,15,20,30,40,60,70,80,90,100,500,1500,-1});

	double NUM_TIMES = 20.0;

	Mat err  = Mat::Zero(2,ks.size());
	for (int i = 0; i < ks.size();i++) { err(0,i) = (double) ks[i]; }

	for (int j = 0; j < NUM_TIMES; j++){
		Mat Init = getInitialProb(Y,getSamples(Y.rows(),NUM_SAMPLES,Y));
		for (int i = 0; i < ks.size();i++) {
			AFFINITY_K = ks[i];
			Mat res = localGlobal(build_LG_S(X, chosenSigma,ks[i]),Init,chosenAlpha);
			CVec csfc = getClassification(res);
			double e =  getErrorRate(Y,csfc) ;
			err(1,i) += e;
		}
		std::cout << j << std::endl;
	}
	for (int j = 0; j < ks.size(); j++){
		err(1,j) = err(1,j) / (double)NUM_TIMES;
	}
	writeToFile(err, filename.str());
	std::cout << err;

}

void experiment_visualize_SP(){
	setN = 20;
	NUM_SAMPLES = 2;
	outputFile = "./OUTPUT_FILES/5/";
	std::srand(std::time(0)); //use current time as seed for random generator
	Mat X;
	CVec Y;
	getInputMatrices(inputFile,setN, X,Y);
	Mat Init = getInitialProb(Y,std::vector<int>({0,2}));
	//std::cout << build_LG_W(X, 1) << std::endl<< std::endl;
	std::cout << build_LG_W(X, 0.01).row(1) << std::endl<< std::endl;
	std::cout << build_LG_S(X, 0.01).row(1) << std::endl<< std::endl;
	std::cout << std::sqrt(build_LG_D(X, 0.05)(0,0)) << std::endl<< std::endl;
	std::cout << std::sqrt(build_LG_D(X, 0.05)(2,2)) << std::endl<< std::endl;

	std::cout << build_LG_W(X, 0.1).row(1) << std::endl<< std::endl;
	std::cout << build_LG_S(X, 0.1).row(1) << std::endl<< std::endl;
	std::cout << std::sqrt(build_LG_D(X, 0.1)(0,0)) << std::endl<< std::endl;
	std::cout << std::sqrt(build_LG_D(X, 0.1)(2,2)) << std::endl<< std::endl;

	std::cout << build_LG_W(X, 1).row(1) << std::endl<< std::endl;
	std::cout << build_LG_S(X, 1).row(0) << std::endl<< std::endl;
	std::cout << std::sqrt(build_LG_D(X, 1)(0,0)) << std::endl<< std::endl;
	std::cout << std::sqrt(build_LG_D(X, 1)(2,2)) << std::endl<< std::endl;

	//std::cout << build_LG_P(X, 1) << std::endl<< std::endl;

	std::vector<double> sigmas ({0.1,0.15,0.20,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1});
	CVec conf = CVec::Zero(sigmas.size());
	for (int i = 0; i < sigmas.size();i++) {
		Mat res =  localGlobal(build_LG_P(X, sigmas[i]), Init, 0.99);
		conf[i] = res(1,1) / ((double) res.row(1).sum());
	}
	std::cout << conf;
}

void experiment_convergence(){
	outputFile = "./OUTPUT_FILES/2/";
	NUM_SAMPLES = 100;
	setN = 7;
	std::srand(std::time(0)); //use current time as seed for random generator
	Mat X;
	CVec Y;
	getInputMatrices(inputFile,setN, X,Y);
	Mat Init = getInitialProb(Y,getSamples(Y.rows(),NUM_SAMPLES,Y));

	LG_GSL_Return best = GSLOptimize(X,Init,Y,0.2,0.7,0.06);
	localGlobal_Iter_test(X,Y,Init, best.alpha, best.sigma, 100,true);
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


double experiment2_aux(int NUM_EXPERIMENT, int NUM_CHANGES, double alpha,double sigma, Mat& X, CVec& Y, Mat& S){

		std::srand(std::time(0)); //use current time as seed for random generator
		std::vector<int> samples = getSamples(Y.rows(),NUM_SAMPLES,Y);

		CVec Y2 = CVec(Y);
		for (int i = 0; i < NUM_CHANGES; i++) {
			if( Y2[samples[i]] == 0) {
				Y2[samples[i]] = 1;
			} else {
				Y2[samples[i]] = 0;
			}
		}

		Mat Init = getInitialProb(Y2,samples);

		std::cout << Map<CVec> (Y.data(),4) << std::endl;
		std::cout << Map<Mat> (Init.data(),4,2) << std::endl;

		Mat errMat;
		int i = 0;
		int j = 0;

		std::vector<double> alphas({alpha});
		std::vector<double> sigmas({sigma});

		double* abs_error = (double*) malloc(sizeof(double));
		double* prob_error = (double*) malloc(sizeof(double));

		LGTest_From_S(S, Init, Y, abs_error, prob_error, alpha);

		double error = *abs_error;
		std::cout<<std::endl;
		free(abs_error);
		free(prob_error);
		return error;

	}



void experiment1(){
	outputFile = "./OUTPUT_FILES/1/";

	std::vector<int> sets({0,1,3,4,5,6,7});
	std::vector<int> samples({10,20,30,100});

	sets.clear();
	std::cout << "Indique qual set, bem como o id min e max para cada repeticao \n";
	int chosenSet,minRun,maxRun;
	std::cin >> chosenSet >> minRun >> maxRun;
	sets.push_back(chosenSet);





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


void experiment2(){
	outputFile = "./OUTPUT_FILES/6/";
	setN = 3;
	NUM_SAMPLES = 100;
	double NUM_REPETITIONS = 1.0;

	std::vector<int> numInversions({0,1,3,5,7,10,15,20,25,30});

	std::cout << "Indique qual set\n";
	std::cin >> setN;
	std::string filenameBest = "./best.txt";
	std::ifstream filestream;
	filestream.open (filenameBest.c_str(),std::ios::in);
	std::vector<std::string> attributes;

	std::string line;
	std::getline(filestream,line);
	std::istringstream initLineStream(line);


	std::string temp;
	while (initLineStream >> temp) {
		attributes.push_back(temp.substr(1, temp.size()-2));
	}
	int index_numsamples = std::find(attributes.begin(),attributes.end(),"num_samples") - attributes.begin();
	int index_set = std::find(attributes.begin(),attributes.end(),"set") - attributes.begin();
	int index_alpha = std::find(attributes.begin(),attributes.end(),"best_alpha") - attributes.begin();
	int index_sigma = std::find(attributes.begin(),attributes.end(),"best_sigma") - attributes.begin();

	double alpha,sigma;

	while(std::getline(filestream,line)) {
		std::istringstream lineStream(line);
		std::find(attributes.begin(),attributes.end(),"num_samples") - attributes.begin();
		CVec v(attributes.size());
		for (int i = 0; i < attributes.size();i++) {
			lineStream >> v[i];
		}
		if (v[index_numsamples] == 100 && v[index_set] == setN) {
			std::cout << "Best alpha for " << v[index_set] << " is " << v[index_alpha] << std::endl;
			std::cout << "Best sigma for " << v[index_set] << " is " << v[index_sigma] << std::endl;
			alpha = v[index_alpha];
			sigma = v[index_sigma];
		}

	}


	std::srand(std::time(0)); //use current time as seed for random generator
	Mat X;
	CVec Y;
	getInputMatrices(inputFile,setN, X,Y);
	Mat S = build_LG_S(X, sigma, -1);

	Mat errorMat = Mat::Zero(numInversions.size(),3);
	for(int i = 0; i < numInversions.size();i++) {
		errorMat(i,0) = setN;
		errorMat(i,1) = numInversions[i];
	}


	for (int i = 0; i < NUM_REPETITIONS; i++){
		for (int j = 0; j < numInversions.size();j++) {
			double error;
			error = experiment2_aux(i, numInversions[j], alpha, sigma, X, Y, S);
			errorMat(j,2) += error;
		}
	}

	for(int i = 0; i < numInversions.size();i++) {
		errorMat(i,2) = errorMat(i,2) / NUM_REPETITIONS;
	}

	std::cout << errorMat << std::endl;
	std::ostringstream filename;
	filename << outputFile <<  "set_" << setN  << "_samples_" << NUM_SAMPLES << "_error.txt";
	writeToFile(errorMat, filename.str());


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
	experiment_visualize_SP(); return 0;
	experiment1();
	return 0;
	setN = 0;
	NUM_SAMPLES = 100;
	experiment_convergence();
	return 0;
}


