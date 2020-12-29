
#include <iostream>
#include "math.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>
using namespace Eigen;
using namespace std;
int main()
{
    int nparal=2;
    int N=3;
    int P=6;
    double sigma=1.5;
    MatrixXd K = MatrixXd::Identity(N,N);
    MatrixXd X = MatrixXd::Identity(N,P);
    MatrixXd alpha = MatrixXd::Ones(N,1);
    alpha=alpha*3;
    cout << "alpha=" << endl << alpha << endl;

    int nblock = P/nparal;
    // IN and Iparal and result is prepared for the calculation
    MatrixXd IN(N,1);
    for(int i=0;i<N;i++){
        IN(i,0)=1;
    }
    MatrixXd Iparal(nparal,1);
    for(int i=0;i<nparal;i++){
        Iparal(i,0)=1;
    }
    ArrayXd  result;
    result.resize(P);
    result.setZero();
    MatrixXd Xj;
    //    Xj.resize(N,nparal);
    MatrixXd partXI ;
    MatrixXd partIX ;
    MatrixXd partXIminusIX;
    MatrixXd  TpartXIminusIX;
            /////////////////////////////////
        //    MatrixXf M1(3,3);    // Column-major storage
        //    M1 << 1, 2, 3,
        //            4, 5, 6,
        //            7, 8, 9;
        //    MatrixXf M2=M1.transpose();
        //    Map<RowVectorXf> v1(M1.data(), M1.size());
        //    cout << "v1:" << endl << v1 << endl;
        //    Map<RowVectorXf> v2(M2.data(), M2.size());
        //    cout << "v2:" << endl << v2 << endl;
            ////////////////////////////////////
    MatrixXd partIK = kroneckerProduct(Iparal,K).eval();
    for(int j=0;j<nblock;j++){
        Xj = X.block(0,j*nparal,N,nparal);
        partXI = kroneckerProduct(Xj,IN).eval();
        partIX = kroneckerProduct(IN,Xj).eval();
        partXIminusIX = partXI - partIX;
        TpartXIminusIX = partXIminusIX.transpose();
        Map<MatrixXd> Reshape_partXIminusIX(TpartXIminusIX.data(), N*nparal,N);
        MatrixXd partRight = ((partIK.array()*(-1.0)*Reshape_partXIminusIX.array())/sigma).transpose().matrix();
        //        cout << "22222=" << endl << partIK.cols()<< endl << partIK.rows() <<endl;
        //        cout << "partXIminusIX=" << endl << Reshape_partXIminusIX.cols()<< endl << Reshape_partXIminusIX.rows() << endl;
        MatrixXd resultPre = alpha.transpose()*partRight;
        Map<MatrixXd> Reshape_resultPre(resultPre.data(), N,nparal);
//        cout << "try=" << endl << Reshape_resultPre << endl;
//        cout << "try=" << endl << Reshape_resultPre.colwise().sum() << endl;
        result.segment(j*nparal,nparal) = Reshape_resultPre.array().square().colwise().sum();
        //        cout << "partIX=" << endl << partIX.cols()<< endl << partIX.rows() << endl;
    }
    cout << "result=" << endl << result << endl;
        cout << "result=" << endl << result.sqrt() << endl;
    VectorXd rresult = (result/(N*1.0)).sqrt();
        cout << "rresult=" << endl << rresult << endl;

    //    cout << "result=" << endl << result << endl;

    //    cout << "K=" << endl << A << endl;
    return 1;
}
