package lib;

import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;

import java.util.Arrays;

public class BayesianLinearRegression {

    public static void main(String[] args) {
        // Sample data (you should replace this with your own dataset)
        double[] y = { 42, 37, 37, 28, 18, 18, 19, 20, 15, 14, 14, 13, 11, 12, 8, 7, 8, 8, 9, 15, 15 };
        double[][] x = {
                { 80, 27, 89 },
                { 80, 27, 88 },
                { 75, 25, 90 },
                { 62, 24, 87 },
                { 62, 22, 87 },
                { 62, 23, 87 },
                { 62, 24, 93 },
                { 62, 24, 93 },
                { 58, 23, 87 },
                { 58, 18, 80 },
                { 58, 18, 89 },
                { 58, 17, 88 },
                { 58, 18, 82 },
                { 58, 19, 93 },
                { 50, 18, 89 },
                { 50, 18, 86 },
                { 50, 19, 72 },
                { 50, 19, 79 },
                { 50, 20, 80 },
                { 56, 20, 82 },
                { 70, 20, 91 }
        };

        // Design matrix (X) and response vector (Y)
        RealMatrix X = MatrixUtils.createRealMatrix(x);
        RealVector Y = new ArrayRealVector(y);

        // Define the noise variance
        double noiseVariance = 1.0;

        // Compute the posterior mean and covariance using LU decomposition
        RealMatrix xTranspose = X.transpose();
        RealMatrix xTx = xTranspose.multiply(X);
        DecompositionSolver solver = new LUDecomposition(xTx).getSolver();
        RealMatrix posteriorCovariance = solver.getInverse().scalarMultiply(noiseVariance);
        RealVector posteriorMean = posteriorCovariance.multiply(xTranspose).operate(Y);

        // Perform Bayesian prediction for a new data point
        double[] newX = { 1.0, 1.0, 6.0 }; // New data point for prediction (note the 1.0 as the intercept)
        RealVector newXVector = new ArrayRealVector(newX);

        RealMatrix newXMatrix = MatrixUtils.createRowRealMatrix(newX);
        RealMatrix posteriorPredictiveCovariance = newXMatrix.multiply(posteriorCovariance)
                .multiply(newXMatrix.transpose()).scalarAdd(noiseVariance);

        RealVector posteriorPredictiveMean = newXMatrix.operate(posteriorMean);

        // Print the posterior predictive mean and covariance
        System.out.println("Posterior Predictive Mean: " + posteriorPredictiveMean);
        System.out.println("Posterior Predictive Covariance: " + posteriorPredictiveCovariance);

        // You can use MultivariateNormalDistribution to sample from the posterior
        MultivariateNormalDistribution posteriorDistribution = new MultivariateNormalDistribution(
                posteriorPredictiveMean.toArray(), posteriorPredictiveCovariance.getData());

        double[] sampledParameters = posteriorDistribution.sample();
        System.out.println("Sampled Parameters: " + Arrays.toString(sampledParameters));
    }
}
