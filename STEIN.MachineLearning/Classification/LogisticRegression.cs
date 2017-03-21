using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace STEIN.MachineLearning.Classification
{
    public class LogisticRegression
    {
        public float Threshold
        { get; set; }

        public DenseVector Theta
        { get; set; }

        public double LearningRate
        { get; set; }

        public float Lambda
        { get; set; }

        private Sigmoid sigmoid;

        /// <summary>
        /// Instantiates a new Logistic Regression object.
        /// </summary>
        /// <param name="numVariables">The number of input features to develop a hypothesis with.</param>
        /// <param name="learningRate">Learning rate modifier. Setting this too high can cause divergence.</param>
        /// <param name="threshold">Anything above this threshold will be classified as positive.</param>
        public LogisticRegression(int numVariables, double learningRate = 0.05, float threshold = 0.5f, float lambda = 0.0f)
        {
            Theta = new DenseVector(numVariables);
            LearningRate = learningRate;
            Lambda = lambda;

            var rand = new Random();

            for (int i = 0; i < Theta.Count; i++)
            {
                Theta[i] = rand.NextDouble();
            }


            Threshold = threshold;
            sigmoid = new Sigmoid();
        }

        /// <summary>
        /// Computes a classification hypothesis given the feature array.
        /// </summary>
        /// <param name="input">The input matrix with training examples as rows and features as columns.</param>
        /// <returns>Vector of classification hypotheses.</returns>
        public double[] Compute(double[,] inputs)
        {
            var X = DenseMatrix.OfArray(inputs);

            //var computation = sigmoid.Function((X * Theta).ToRowMatrix()).Map(val =>
            //{
            //    if (val > Threshold)
            //    {
            //        return 1.0;
            //    }
            //    else
            //    {
            //        return 0.0;
            //    }
            //}).ToRowArrays()[0];

            var computation = sigmoid.Function((X * Theta).ToRowMatrix());

            //return computation.Select(i => Convert.ToInt32(i)).ToArray();
            return computation.ToRowArrays()[0];
        }

        /// <summary>
        /// Trains one epoch of the logistic regression classifier.
        /// </summary>
        /// <param name="inputs">The input matrix with training examples as rows and features as columns.</param>
        /// <param name="y">The actual classifications of the training examples as a row vector</param>
        public void Train(double[,] inputs, int[] y)
        {
            var X = DenseMatrix.OfArray(inputs);
            double[] hypotheses = Compute(inputs);
            var error = new double[y.Length];
            
            for(int i = 0; i < y.Length; i++)
            {
                error[i] = hypotheses[i] - y[i];
            }

            var gradient = LearningRate * (X.Transpose() * DenseVector.OfArray(error)) * (1.0 / X.RowCount) + (Theta * (Lambda / X.RowCount));

            Theta -= (DenseVector)gradient;
        }

        /// <summary>
        /// Computes the logarithmic cost of the error between the given parameters.
        /// </summary>
        public double ComputeCost(double[] h, int[] y)
        {
            var vY = DenseVector.OfArray(y.Select(i => (double)i).ToArray()).ToRowMatrix();
            var vH = DenseVector.OfArray(h).ToRowMatrix();

            double o_log = (-1 * vY * vH.PointwiseLog().Transpose())[0, 0];
            double z_log = ((1 - vY) * ((1 - vH).PointwiseLog().Transpose()))[0, 0];

            var unreg_cost = (o_log - z_log) / vY.RowCount;

            var nTheta = Theta.Clone();
            var nTheta_R = nTheta.ToRowMatrix();

            double c_reg = (Lambda * (nTheta_R.Transpose() * nTheta_R) / (2 * y.Count()))[0, 0];

            return unreg_cost + c_reg;
        }

    }
}
