using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace STEIN.MachineLearning
{
    public interface ICostFunction
    {
        double Calculate(Matrix<double> actual, Matrix<double> expected);
    }
}
