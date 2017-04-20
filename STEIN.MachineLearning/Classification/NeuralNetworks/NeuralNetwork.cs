using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using STEIN.MachineLearning.Classification.NeuralNetworks.ActivationFunctions;

namespace STEIN.MachineLearning.Classification.NeuralNetworks
{
    public class NeuralNetwork : INeuralComponent
    {
        public List<Layer> Layers
        { get; set; }


        public NeuralNetwork(int[] layers, IActivationFunction afunc, double lambda, double learningRate = 1, bool useBias = true)
        {
            Layers = new List<Layer>();

            for (int i = 1; i < layers.Length; i++)
            {
                Layers.Add(new Layer(layers[i], layers[i - 1], afunc, lambda, learningRate, useBias));
            }
        }

        public NeuralNetwork(IEnumerable<Layer> layers)
        {
            Layers = layers.ToList();
        }


        public double[,] Compute(double[,] x)
        {
            return Compute(DenseMatrix.OfArray(x)).ToArray();
        }


        public Matrix<double> Compute(Matrix<double> x)
        {
            var lastLayer = x;

            for (var i = 0; i < Layers.Count; i++)
            {
                lastLayer = Layers[i].Compute(lastLayer);
            }

            return lastLayer;
        }
    }
}
