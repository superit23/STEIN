using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace STEIN.MachineLearning
{
    public abstract class SupervisedLearning
    {
        public abstract double Run(double[] input, double[] output);

        public double RunEpoch(double[,] input, double[,] output)
        {
            return RunEpoch(DenseMatrix.OfArray(input), DenseMatrix.OfArray(output));
        }

        double RunEpoch(double[][] input, double[][] output)
        {
            var total = 0.0;
            for(var i = 0; i < input.Length; i++)
            {
                total += Run(input[i], output[i]);
            }

            return total;
            
        }

        public abstract double RunEpoch(Matrix<Double> input, Matrix<Double> output);
    }
}
