using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace STEIN.MachineLearning
{
    public class Sigmoid : Classification.NeuralNetworks.ActivationFunction
    {
        public Matrix<double> Derivative(Matrix<double> x)
        {
            var g = Function(x);
            return g.PointwiseMultiply((1 - g));
        }

        public double Derivative(double x)
        {
            var g = Function(x);
            return g * (1 - g);
        }

        public Matrix<double> Function(Matrix<double> x)
        {
            return 1 / (1.0 + x.PointwiseExp());
        }

        public double Function(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }
    }
}
