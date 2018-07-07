#pragma once

#include "globals.h"
#include "utils.h"
#include "LGC.h"
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
  LGC_Test(p->X, p->Init, p->Y,error,NULL, alpha,sigma,false,k);

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

