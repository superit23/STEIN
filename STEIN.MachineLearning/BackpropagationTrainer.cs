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
            // We calculate errors and updates for the network as a whole
            // before modifying it to make it easy not to dirty the state
            var result = Network.Compute(input);
            var cost = CostFunc.Calculate(result, output);
            var error = CalculateError(result, output);
            var updates = CalculateUpdates(input, error);
            UpdateNetwork(updates);


            return cost;
        }

        private Matrix<double>[] CalculateError(Matrix<double> result, Matrix<double> output)
        {
            var numLayers = Network.Layers.Count;

            // Compute error
            var activationFunc = Network.Layers[numLayers - 1].AFunc;
            var error = (output - result).PointwiseMultiply(activationFunc.Derivative(output));

            //var layerErrors = new Vector<double>[numLayers];
            var layerErrors = new Matrix<double>[numLayers];

            // For each sample
            for (int rowIdx = 0; rowIdx < error.RowCount; rowIdx++)
            {
                for (var i = 0; i < numLayers; i++)
                {
                    //layerErrors[i] = DenseVector.Create(Network.Layers[i].Theta.RowCount, 0);
                    layerErrors[i] = DenseMatrix.Create(error.RowCount, Network.Layers[i].Theta.RowCount, 0);
                }

                // Set last layer error to output error
                layerErrors[numLayers - 1].SetRow(rowIdx, error.EnumerateRows().ElementAt(rowIdx));
                //layerErrors[Network.Layers.Count - 1] = error.EnumerateRows().ElementAt(rowIdx);

                // Get error per layer
                for (int i = numLayers - 2; i >= 0; i--)
                {
                    var currLayer = Network.Layers[i];
                    var nextLayer = Network.Layers[i + 1];

                    var currErrors = layerErrors[i];
                    var nextErrors = layerErrors[i + 1];

                    // Sum total error per neuron with respect the connections in the next layer
                    var sum = nextErrors * nextLayer.Theta;
                    currErrors = sum * activationFunc.Derivative(currLayer.Computations).Transpose();
                }
            }

            return layerErrors;
        }


        private Dictionary<Layer, NeuralUpdateInfo> CalculateUpdates(Matrix<double> input, Matrix<double>[] error)
        {
            var numLayers = Network.Layers.Count;
            var updateInfo = new Dictionary<Layer, NeuralUpdateInfo>();

            var weightsUpdates = new Matrix<double>[numLayers];
            var thresholdUpdates = new Vector<double>[numLayers];

            var layerInput = input;

            for (int i = 0; i < numLayers; i++)
            {
                var currLayer = Network.Layers[i];
                var layerWeightsUpdates = weightsUpdates[i];
                var layerThresholdUpdates = thresholdUpdates[i];
                var errorMatrix = error[i];

                //var errorMatrix = DenseMatrix.Create(layerInput.RowCount, errorVector.Count, (row, col) => errorVector[col]);
                layerWeightsUpdates = errorMatrix.Transpose().Multiply(layerInput);
                layerThresholdUpdates = errorMatrix.ColumnSums();
                //layerThresholdUpdates = errorVector;

                updateInfo.Add(currLayer, new NeuralUpdateInfo(layerWeightsUpdates, layerThresholdUpdates));

                layerInput = Network.Layers[i].Computations;

            }

            return updateInfo;
        }


        private void UpdateNetwork(Dictionary<Layer, NeuralUpdateInfo> updateInfo)
        {
            for(var i = 0; i < Network.Layers.Count; i++)
            {
                var currLayer = Network.Layers[i];
                currLayer.Theta += updateInfo[currLayer].Weights;
                currLayer.Thresholds += updateInfo[currLayer].Thresholds;
            }
        }
    }
}
