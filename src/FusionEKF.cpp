#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

FusionEKF::FusionEKF() { // Constructor
  is_initialized_ = false;
  previous_timestamp_ = 0;

  // Fixed matrices are defined here
  // Laser measurement matrix
  H_laser_ = MatrixXd(2, 4);
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;
  
  // Measurement covariance matrix - laser (GIVEN)
  R_laser_ = MatrixXd(2, 2);
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  // Measurement covariance matrix - radar (GIVEN)
  R_radar_ = MatrixXd(3, 3);
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;
  
  // Set the acceleration noise components (GIVEN)
  noise_ax = 9;
  noise_ay = 9;
}

FusionEKF::~FusionEKF() {} // Destructor

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /*****************
   * Initialization
   *****************/
  if (!is_initialized_) {
    // Initialize state ekf_.x_ with the first measurement: RADAR or LIDAR
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    
    // RADAR
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) { 
      // Initialize state (polar to cartesian)
      float ro     = measurement_pack.raw_measurements_[0];
      float theta  = measurement_pack.raw_measurements_[1];
      float ro_dot = measurement_pack.raw_measurements_[2];
      ekf_.x_ << ro*cos(theta), 
                ro*sin(theta), 
                ro_dot*cos(theta), 
                ro_dot*sin(theta);
    }
    // LASER
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // Initialize state.
      ekf_.x_ << measurement_pack.raw_measurements_[0], 
              measurement_pack.raw_measurements_[1], 
              0, 
              0;
    }
    
    // Covariance matrix P (regardless of sensor)
    ekf_.P_ = MatrixXd(4, 4);
    ekf_.P_ << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1000, 0,
            0, 0, 0, 1000; 
    
    // done initializing, no need to predict or update
    is_initialized_ = true;
    previous_timestamp_ = measurement_pack.timestamp_;
    return;
  }
  
  /*********************
   * Prediction
   ********************/
   // Update the state transition matrix F accord to the new elapsed time.
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;  // dt - expressed in seconds
  previous_timestamp_ = measurement_pack.timestamp_;
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1, 0, dt, 0,
            0, 1, 0, dt,
            0, 0, 1, 0,
            0, 0, 0, 1;

  // Set the process noise covariance matrix Q
  float dt_2 = dt * dt;
  float dt_3 = dt_2 * dt;
  float dt_4 = dt_3 * dt;
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ <<  dt_4/4*noise_ax, 0, dt_3/2*noise_ax, 0,
         0, dt_4/4*noise_ay, 0, dt_3/2*noise_ay,
         dt_3/2*noise_ax, 0, dt_2*noise_ax, 0,
         0, dt_3/2*noise_ay, 0, dt_2*noise_ay;
  
  // Run predict
  ekf_.Predict();
 
  
  /*****************
   * Update
   *****************/
  //RADAR
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  }
  //LASER
  else {
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }
  
  // Print the Output Estimate
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
