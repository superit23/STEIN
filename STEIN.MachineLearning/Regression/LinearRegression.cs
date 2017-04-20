using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace STEIN.MachineLearning.Regression
{
    public class LinearRegression
    {
        public DenseVector Theta
        { get; set; }

        public double LearningRate
        { get; set; }

        private bool isBiased = false;

        /// <summary>
        /// Instantiates a new Linear Regression object.
        /// </summary>
        /// <param name="numVariables">Number of variables to optimize.</param>
        /// <param name="learningRate">Learning rate modifier. Setting this too high can cause divergence.</param>
        public LinearRegression(int numVariables, double learningRate = 0.05, bool useBias = true)
        {
            isBiased = useBias;
            Theta = new DenseVector(numVariables + (isBiased ? 1 : 0));
            LearningRate = learningRate;

            var rand = new Random();
            
            for(int i = 0; i < Theta.Count; i++)
            {
                Theta[i] = rand.NextDouble();
            }
        }

        /// <summary>
        /// Calculates the mean squared error.
        /// </summary>
        /// <param name="inArr">The hypothesis array.</param>
        /// <param name="y">The actual output to compare against.</param>
        /// <returns>The cost (mean squared error).</returns>
        public double ComputeCost(double[] inArr, double[] y)
        {
            //var total = 0.0;

            //for(int i = 0; i < inArr.Length; i++)
            //{
            //    var val = (inArr[i] - y[i]);
            //    total += val * val;
            //}

            //return total / (2 * inArr.Length);

            var vY = DenseVector.OfArray(y);
            var vX = DenseVector.OfArray(inArr);

            var error = vX - vY;

            return (error.PointwisePower(2)).Sum() / (vY.Count);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="y"></param>
        public void Train(double[,] inputs, double[] y)
        {
            var x = (Matrix<double>)DenseMatrix.OfArray(inputs);

            if (isBiased)
            {
                x = DenseMatrix.Create(x.RowCount, 1, 1.0).Append(x);
            }


            double[] hypotheses = Compute(inputs);
            var vectorY = DenseVector.OfArray(y);
            var error = hypotheses - vectorY;

            Vector<double> gradient = LearningRate * (x.Transpose() * error) / x.RowCount;

            Theta -= (DenseVector)gradient;

        }

        /// <summary>
        /// Computes the hypothesis given the inputs.
        /// </summary>
        /// <param name="inputs">The feature array.</param>
        /// <returns>The predicted values.</returns>
        public double[] Compute(double[,] inputs)
        {
            var X = (Matrix<double>)DenseMatrix.OfArray(inputs);

            if (isBiased)
            {
                X = DenseMatrix.Create(X.RowCount, 1, 1.0).Append(X);
            }

            return (X * Theta).ToArray();

        }

        
    }
}
