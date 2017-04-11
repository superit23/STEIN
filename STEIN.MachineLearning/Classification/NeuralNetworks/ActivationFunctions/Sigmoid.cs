using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace STEIN.MachineLearning.Classification.NeuralNetworks.ActivationFunctions
{
    public class Sigmoid : IActivationFunction
    {
        public double Alpha
        { get; set; }

        public Sigmoid()
        {
            Alpha = 2.0;
        }

        public Matrix<double> Derivative(Matrix<double> x)
        {
            var g = Function(x);
            return Alpha * g.PointwiseMultiply((1 - g));
        }

        public double Derivative(double x)
        {
            var g = Function(x);
            //return g * (1 - g);
            return Alpha * g * (1 - g);
        }

        public Matrix<double> Function(Matrix<double> x)
        {
            //return 1 / (1.0 + x.PointwiseExp());
            return 1 / (1.0 + x.Multiply(-Alpha).PointwiseExp());
        }

        public double Function(double x)
        {
            //return 1.0 / (1.0 + Math.Exp(-x));
            return 1.0 / (1.0 + Math.Exp(-Alpha * x));
        }
    }
}
