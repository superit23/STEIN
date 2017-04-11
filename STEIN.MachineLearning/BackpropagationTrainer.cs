using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using STEIN.MachineLearning.Classification.NeuralNetworks;
using MathNet.Numerics.LinearAlgebra.Double;
using STEIN.MachineLearning.CostFunctions;

namespace STEIN.MachineLearning
{
    /// <summary>
    /// Responsible for training models by fitting parameters and hyperparameters.
    /// </summary>
    public class BackpropagationTrainer : SupervisedLearning
    {
        public NeuralNetwork Network
        { get; set; }

        public ICostFunction CostFunc
        { get; set; }


        public BackpropagationTrainer(NeuralNetwork network)
        {
            Network = network;
            CostFunc = new SquaredError();
        }

        public override double Run(double[] input, double[] output)
        {
            return RunEpoch(DenseVector.OfArray(input).ToRowMatrix(), DenseVector.OfArray(output).ToRowMatrix());
        }

        public override double RunEpoch(Matrix<double> input, Matrix<double> output)
        {
            // Compute error and cost
            var activationFunc = Network.Layers[0].AFunc;
            var result = Network.Compute(input);
            var error = (output - result).PointwiseMultiply(activationFunc.Derivative(output));
            var cost = CostFunc.Calculate(result, output);

            var layerErrors = new Vector<double>[Network.Layers.Count];
            //var totalError = 0.0;

            // For each sample
            for (int rowIdx = 0; rowIdx < error.RowCount; rowIdx++)
            {
                for (var i = 0; i < Network.Layers.Count; i++)
                {
                    layerErrors[i] = DenseVector.Create(Network.Layers[i].Theta.RowCount, 0);
                }

                // Set last layer error to output cost
                layerErrors[Network.Layers.Count - 1] = error.EnumerateRows().ElementAt(rowIdx);

                // Get error per layer
                for (int i = Network.Layers.Count - 2; i >= 0; i--)
                {
                    var currLayer = Network.Layers[i];
                    var nextLayer = Network.Layers[i + 1];

                    var currErrors = layerErrors[i];
                    var nextErrors = layerErrors[i + 1];

                    // Sum total error per neuron with respect the connections in the next layer
                    var sum = nextErrors * nextLayer.Theta;
                    currErrors = sum * activationFunc.Derivative(currLayer.Computations).Transpose();
                    //totalError += currErrors.Sum();
                }
            }

            //return totalError;
            return cost;
        }
    }
}
