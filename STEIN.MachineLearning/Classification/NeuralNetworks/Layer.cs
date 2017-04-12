using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using STEIN.MachineLearning.Classification.NeuralNetworks.ActivationFunctions;

namespace STEIN.MachineLearning.Classification.NeuralNetworks
{
    public class Layer : INeuralComponent
    {
        public Matrix<double> Theta
        { get; set; }

        public IActivationFunction AFunc
        { get; set; }

        public Matrix<double> Computations
        { get; set; }

        public Matrix<double> Input
        { get; set; }


        public Vector<double> Thresholds
        { get; set; }

        public double Lambda
        { get; set; }

        public double LearningRate
        { get; set; }

        private bool isBiased = false;


        public Layer(int numNeurons, int numInputs, IActivationFunction afunc, double lambda, double learningRate, bool useBias)
        {
            AFunc = afunc;
            Lambda = lambda;


            var tempTheta = DenseMatrix.CreateRandom(numNeurons, numInputs, new MathNet.Numerics.Distributions.ContinuousUniform());
            var thresholds = DenseVector.CreateRandom(numNeurons, new MathNet.Numerics.Distributions.ContinuousUniform());

            if (useBias)
            {
                isBiased = useBias;
                Theta = DenseMatrix.Create(numNeurons, 1, 1.0).Append(tempTheta);
                Thresholds = DenseVector.Create(numNeurons, 1).Add(thresholds - 1.0);
            }
            else
            {
                Theta = tempTheta;
                Thresholds = thresholds;
            }
            

            LearningRate = learningRate;
        }

        public virtual Matrix<double> Compute(Matrix<double> x)
        {
            //Input = x;
            Matrix<double> a = null;

            if(isBiased)
            {
               a = DenseMatrix.Create(x.RowCount, 1, 1.0).Append(x);
            }
            else
            {
                a = x;
            }

            Input = x;

            var thresholdMatrix = DenseMatrix.Create(x.RowCount, Theta.RowCount, (row, col) => Thresholds[col]);

            var z = a.Multiply(Theta.Transpose()) + thresholdMatrix;
            Computations = AFunc.Function(z);

            return Computations;
        }

        //public Matrix<double> BackPropogate(Matrix<double> y)
        //{
        //    // Calculate Delta
        //    var thetaNoBias = Theta.RemoveColumn(0);
        //    var delta = y * thetaNoBias;
        //    delta = delta.PointwiseMultiply(AFunc.Derivative(Input));

        //    // Calculate the regularized gradient
        //    var gradient = delta.Transpose().Multiply(Computations);
        //    gradient = gradient.Divide(Input.RowCount);
        //    gradient = LearningRate * gradient.Transpose() + thetaNoBias.Multiply(Lambda / Input.RowCount);

        //    Theta -= DenseMatrix.Create(gradient.RowCount, 1, 0).Append(gradient);

        //    return delta;
        //}

        public Matrix<double> BackPropogate(Matrix<double> delta)
        {
            
            Matrix<double> thetaNoBias = null;
            if (isBiased)
            {
                thetaNoBias = Theta.RemoveColumn(0);
            }
            else
            {
                thetaNoBias = Theta;
            }

            // Calculate Delta
            //var delta = y * Theta;
            var l_delta = (delta * thetaNoBias).PointwiseMultiply(AFunc.Derivative(Input));

            // Calculate the regularized gradient
            var gradient = Computations.Transpose() * l_delta;
            
            //if(isBiased)
            //{
            //    gradient = gradient.RemoveColumn(0);
            //}
            
            if(isBiased)
            {
                thetaNoBias = DenseMatrix.Create(Theta.RowCount, 1, 0).Append(thetaNoBias);
                gradient = DenseMatrix.Create(Theta.RowCount, 1, 0).Append(gradient);
            }

            gradient = gradient.Divide(Input.RowCount);
            gradient = LearningRate * (gradient + thetaNoBias.Multiply(Lambda / Input.RowCount));

            Theta += gradient;

            return l_delta;
        }

        /// <summary>
        /// Computes the logarithmic cost of the error between the given parameters.
        /// </summary>
        public double ComputeCost(double[,] h, double[,] y)
        {
            //var nY = new double[y.GetLength(0), y.GetLength(1)];

            //for(var i = 0; i < y.GetLength(0); i++)
            //{
            //    for(var j = 0; j < y.GetLength(1); i++)
            //    {
            //        nY[i, j] = y[i, j];
            //    }
            //}

            var vY = DenseMatrix.OfArray(y);
            var vH = DenseMatrix.OfArray(h);

            double o_log = (-1 * vY * vH.PointwiseLog().Transpose())[0, 0];
            double z_log = ((1 - vY) * ((1 - vH).PointwiseLog().Transpose()))[0, 0];

            var unreg_cost = (o_log - z_log) / vY.RowCount;

            //var nTheta = Theta.Clone();
            //var nTheta_R = nTheta;

            double c_reg = (Lambda * (Theta.Transpose() * Theta) / (2 * vY.RowCount))[0, 0];

            return unreg_cost + c_reg;
        }

    }
}
