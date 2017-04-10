using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace STEIN.MachineLearning.Classification.NeuralNetworks
{
    public class NoiseLayer : Layer
    {
        private int _minRand;
        public int MinRand
        {
            get
            {
                return _minRand;
            }

            set
            {
                _minRand = value;
            }
        }

        private int _maxRand;
        public int MaxRand
        {
            get
            {
                return _maxRand;
            }

            set
            {
                _maxRand = value;
            }
        }

        private Random random = new Random();
        public NoiseLayer(int minRand, int maxRand, int numNeurons, int numInputs, IActivationFunction afunc, double lambda, double learningRate, bool useBias) : base(numNeurons, numInputs, afunc, lambda, learningRate, useBias)
        {
            MinRand = minRand;
            MaxRand = maxRand;
        }

        public override Matrix<double> Compute(Matrix<double> x)
        {
            for(var j = 0; j < x.ColumnCount; j++)
            {
                for (var i = 0; i < x.ColumnCount; i++)
                {
                    x[i, j] += random.NextDouble() * MathFunctions.GenerateInt(_minRand, _maxRand);
                }
            }
            return x;
        }
    }
}
