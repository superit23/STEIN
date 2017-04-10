using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace STEIN.MachineLearning.Classification.NeuralNetworks
{
    public class SoftmaxLayer : Layer
    {
        public SoftmaxLayer(int numNeurons, int numInputs, IActivationFunction afunc, double lambda, double learningRate, bool useBias) : base(numNeurons, numInputs, afunc, lambda, learningRate, useBias)
        {
        }

        public override Matrix<double> Compute(Matrix<double> x)
        {
            return MathFunctions.Softmax(base.Compute(x));
        }
    }
}

