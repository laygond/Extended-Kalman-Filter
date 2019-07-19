#ifndef TOOLS_H_
#define TOOLS_H_

#include <vector>
#include "Eigen/Dense"

class Tools {
 public:
  /**
   * Constructor.
   */
  Tools();

  /**
   * Destructor.
   */
  virtual ~Tools();

  /**
   * A helper method to calculate RMSE.
   */
  Eigen::VectorXd CalculateRMSE(const std::vector<Eigen::VectorXd> &estimations, 
                                const std::vector<Eigen::VectorXd> &ground_truth);

  /**
   * A helper method to calculate Jacobians.
   */
  Eigen::MatrixXd CalculateJacobian(const Eigen::VectorXd& x_state);

   /**
   * A helper method to calculate polar coord from cartessian coord.
   */
  Eigen::VectorXd RadarCart2Polar(const Eigen::VectorXd& x_state);
  
   /**
   * A helper method to normalize the theta angle between [-pi,pi]
   */
  Eigen::VectorXd RadarNormalizePolar(const Eigen::VectorXd& y_diff);
  
};

#endif  // TOOLS_H_
