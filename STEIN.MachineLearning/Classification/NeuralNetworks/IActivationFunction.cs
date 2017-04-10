using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace STEIN.MachineLearning.Classification.NeuralNetworks
{
    public interface IActivationFunction
    {
        double Function(double x);

        Matrix<double> Function(Matrix<double> x);

        double Derivative(double x);

        Matrix<double> Derivative(Matrix<double> x);

    }
}
