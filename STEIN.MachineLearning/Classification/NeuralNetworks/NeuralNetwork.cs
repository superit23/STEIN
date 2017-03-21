using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace STEIN.MachineLearning.Classification.NeuralNetworks
{
    public class NeuralNetwork
    {
        public List<Layer> Layers
        { get; set; }


        public NeuralNetwork(int[] layers, ActivationFunction afunc, double lambda, double learningRate = 1, bool useBias = true)
        {
            Layers = new List<Layer>();

            for(int i = 1; i < layers.Length; i++)
            {
                Layers.Add(new Layer(layers[i], layers[i - 1], afunc, lambda, learningRate, useBias));
            }
        }

        public double[,] Compute(double[,] x)
        {
            var lastLayer = (Matrix<double>)DenseMatrix.OfArray(x);

            for(var i =0; i < Layers.Count; i++)
            {
                lastLayer = Layers[i].Compute(lastLayer);
            }

            return lastLayer.ToArray();
        }

        private void BackPropogate(Matrix<double> a, Matrix<double> y)
        {
            var lastLayer = y - a;

            for(int i = Layers.Count - 1; i > -1; i--)
            {
                //Console.WriteLine(i);
                lastLayer = Layers[i].BackPropogate(lastLayer);
            }
        }

        public void Train(double[,] x, double[,] y)
        {  
            BackPropogate(DenseMatrix.OfArray(Compute(x)), DenseMatrix.OfArray(y));
            
        }


    }
}
