#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using std::cout;
using std::endl;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0,0,0,0;

  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  if (estimations.size() != ground_truth.size()
      || estimations.size() == 0) {
    cout << "Invalid estimation or ground_truth data" << endl;
    return rmse;
  }

  // accumulate squared residuals
  for (size_t i=0; i < estimations.size(); ++i) {

    VectorXd residual = estimations[i] - ground_truth[i];

    // coefficient-wise multiplication
    residual = residual.array()*residual.array();
    rmse += residual;
  }

  // calculate the mean
  rmse = rmse/estimations.size();

  // calculate the squared root
  rmse = rmse.array().sqrt();

  // return the result
  return rmse;

}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd Hj(3,4);
  // recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  // check division by zero
  float den = pow(px,2)+pow(py,2); //denominator
  
  if (den > 0.00001){
    // compute the Jacobian matrix for h(x)
    Hj <<   px/sqrt(den), py/sqrt(den),0,0,
            -py/den, px/den, 0, 0,
            py*(vx*py-vy*px)/pow(den,1.5), px*(vy*px-vx*py)/pow(den,1.5), px/sqrt(den), py/sqrt(den);
  }
  else {
     cout << "Hj has division by zero at State: " << x_state<< endl;
  }
  return Hj;
}

VectorXd Tools::RadarCart2Polar(const VectorXd& x_state) {
  VectorXd xp_state(3);             //polar state
   // recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  // calculate and check division by zero
  float ro     = sqrt(pow(px,2)+pow(py,2));
  if (ro<0.00001){ // make ´ro´ greater again
    px += .001;    // this also prevents atan2 from blowing up
    py += .001;
    ro = sqrt(pow(px,2)+pow(py,2));
  }
  float theta  = atan2(py,px);
  float ro_dot = (px*vx+py*vy)/ro;
  
  // Feed in values
  xp_state << ro,
              theta,
              ro_dot;

  return xp_state;
}

VectorXd Tools::RadarNormalizePolar(const VectorXd& y_diff) {
  VectorXd y_norm = y_diff;             //y difference normalized
  
  // Normalize the angle between [-pi,pi]
  while (y_norm(1) > M_PI) {
    y_norm(1) -= M_PI;
  }
  while (y_norm(1) < -M_PI) {
    y_norm(1) += M_PI;
  }
  
  return y_norm;
}