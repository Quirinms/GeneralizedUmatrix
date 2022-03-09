#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <Rcpp.h>

#include <omp.h>

using namespace Rcpp;

// [[Rcpp::plugins(openmp)]]
arma::mat OpenMPSliceDifference(arma::mat SliceOfEsomwts,
                                double DataSample){
  int n = SliceOfEsomwts.n_rows;
  int d = SliceOfEsomwts.n_cols;
  arma::mat TmpRes(n,d);
  // Define a block for openmp to indicate where the omp_set_num_threads takes
  // effect.
  {// Start of block
    int numThreads = omp_get_thread_num();
    omp_set_num_threads(numThreads);
    for(int i = 0; i < n; i++){    //pragma omp parallel for
      for(int j = 0; j < d; j++){
        TmpRes(i,j) = SliceOfEsomwts(i,j) - DataSample;
      }
    }
  }// End of block
  return TmpRes;
}


// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::cube trainstepC2(Rcpp::NumericVector vx,
                       Rcpp::NumericVector vy,
                       Rcpp::NumericMatrix DataSampled,
                       Rcpp::NumericMatrix BMUsampled,
                       double Lines,
                       double Columns,
                       double Radius,
                       bool toroid) {
  
  Rcpp::IntegerVector x_dims = vx.attr("dim");
  Rcpp::IntegerVector y_dims = vy.attr("dim");
  int NumberOfweights=x_dims[2];
  double k=Lines;
  double m=Columns;
  
  arma::cube esomwts(vx.begin(), x_dims[0], x_dims[1], x_dims[2], false);
  arma::cube aux(vy.begin(), y_dims[0], y_dims[1], y_dims[2], false);
  
  Rcpp::NumericVector DataSample(DataSampled.rows());
  Rcpp::NumericVector bmpos(BMUsampled.rows());
  arma::mat OutputDistances(k,m);
  arma::mat neighmatrix(k,m);
  
  arma::cube neigharray(k, m, NumberOfweights); 
  arma::cube inputdiff(k, m, NumberOfweights); 
  
  arma::mat kmatrix(x_dims[0], x_dims[1]);
  kmatrix.fill(k-1);
  arma::mat mmatrix(x_dims[0], x_dims[1]);
  mmatrix.fill(m-1);
  arma::mat bm1(x_dims[0], x_dims[1]);
  arma::mat bm2(x_dims[0], x_dims[1]);
  int NumberOfDataSamples=DataSampled.nrow();
  
  for(int p = 0; p < NumberOfDataSamples; p++){
    DataSample=DataSampled.row(p);
    bmpos=BMUsampled.row(p);
    bm1.fill(bmpos(0));
    bm2.fill(bmpos(1));
    if (toroid){
      OutputDistances = 0.5*sqrt(pow(kmatrix-abs(2*abs(aux.slice(0)-bm1)-kmatrix),2) + pow(mmatrix-abs(2*abs(aux.slice(1)-bm2)-mmatrix ),2));
    }else{
      OutputDistances =  sqrt(pow(aux.slice(0)-bm1,2) + pow(aux.slice(1)-bm2,2));
    }
    neighmatrix = 1 - (OutputDistances % OutputDistances)/(3.14159265*Radius*Radius);
    for(unsigned int i=0;i<neighmatrix.n_rows;i++){
      for(unsigned int j=0;j<neighmatrix.n_cols;j++){
        if(neighmatrix(i,j)<0)
          neighmatrix(i,j)=0;
      }
    }
    for(int i=0;i<NumberOfweights;i++){
      neigharray.slice(i) = neighmatrix;
    }
    // Delta3DWeightsC
    for(unsigned int i = 0; i < esomwts.n_slices; i++) {
      inputdiff.slice(i) = OpenMPSliceDifference(esomwts.slice(i), DataSample[i]);
    }
    esomwts = esomwts - neigharray % inputdiff;             // element-wise cube multiplication with %
  }
  return(esomwts);
}
