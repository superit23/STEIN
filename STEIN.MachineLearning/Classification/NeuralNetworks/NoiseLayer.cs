using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using STEIN.MachineLearning.Classification.NeuralNetworks.ActivationFunctions;

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
                diff = _maxRand - _minRand;
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
                diff = _maxRand - _minRand;
            }
        }

        private int diff = 0;
        private Random random = new Random();
        public NoiseLayer(int minRand, int maxRand, int numNeurons, int numInputs, IActivationFunction afunc, double lambda, double learningRate, bool useBias) : base(numNeurons, numInputs, afunc, lambda, learningRate, useBias)
        {
            // Call setter on MaxRand to compute diff
            _minRand = minRand;
            MaxRand = maxRand;
        }

        public override Matrix<double> Compute(Matrix<double> x)
        {
            for(var i = 0; i < x.RowCount; i++)
            {
                for (var j = 0; j < x.ColumnCount; j++)
                {
                    x[i, j] += random.NextDouble() * MathFunctions.GenerateInt(0, diff) + _minRand;
                }
            }
            return x;
        }
    }
}
