using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace STEIN.MachineLearning.CostFunctions
{
    public class SquaredError : ICostFunction
    {
        public double Calculate(Matrix<double> actual, Matrix<double> expected)
        {
            return (actual - expected).PointwisePower(2).RowSums().Sum();
        }
    }
}
