using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace STEIN.MachineLearning.CostFunctions
{
    public class MultinomialCrossEntropy : ICostFunction
    {
        public double Calculate(Matrix<double> actual, Matrix<double> expected)
        {
            return (expected * actual.PointwiseLog().Transpose())[0, 0] / -actual.RowCount;

        }
    }
}
