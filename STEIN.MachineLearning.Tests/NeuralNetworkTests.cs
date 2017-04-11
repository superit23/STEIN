using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using STEIN.MachineLearning.Classification.NeuralNetworks;
using Accord.Neuro;
using Accord.Neuro.Learning;
using System.Linq;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.LinearAlgebra;
using STEIN.MachineLearning.Classification.NeuralNetworks.ActivationFunctions;

namespace STEIN.MachineLearning.Tests
{
    [TestClass]
    public class NeuralNetworkTests
    {
        [TestMethod]
        public void TestForwardPropogation()
        {
            var y = new double[,] {
                { 0,0,0,0,1 },
                { 0,0,0,1,0 },
                { 0,1,0,0,0 }
            };

            var x = new double[,] {
                { 1,1,1,0,0,0,0,0,0,1 },
                { 0,0,0,1,1,1,0,0,0,1 },
                { 0,0,0,0,0,0,1,1,1,1 }
            };

            var accordY = new double[][] {
                new double[]{ 0,0,0,0,1 },
                new double[]{ 0,0,0,1,0 },
                new double[]{ 0,1,0,0,0 }
            };

            var accordX = new double[][] {
                new double[]{ 1,1,1,0,0,0,0,0,0,1 },
                new double[]{ 0,0,0,1,1,1,0,0,0,1 },
                new double[]{ 0,0,0,0,0,0,1,1,1,1 }
            };


            var nn = new NeuralNetwork(new int[] { 10, 10, 5 }, new Sigmoid(), 0, 0.05, false);
            var accordNetwork = new ActivationNetwork(new SigmoidFunction(), 10, new int[] { 10, 5 });
            accordNetwork.Randomize();

            // Make models equivalent
            for (var i = 0; i < nn.Layers.Count; i++)
            {
                nn.Layers[i].Theta = DenseMatrix.OfRowArrays(accordNetwork.Layers[i].Neurons.Select(neuron => neuron.Weights).ToArray());
                nn.Layers[i].Thresholds = DenseVector.OfEnumerable(accordNetwork.Layers[i].Neurons.Select(neuron => (neuron as ActivationNeuron).Threshold));
            }

            // Assert networks have same parameters (in case the matrices distribute data in nonintuitive way)
            for (var i = 0; i < nn.Layers.Count; i++)
            {
                var accordNeurons = accordNetwork.Layers[i].Neurons;
                for (var j = 0; j < accordNeurons.Count(); j++)
                {
                    Assert.AreEqual(nn.Layers[i].Thresholds[j], (accordNeurons[j] as ActivationNeuron).Threshold);
                    for (var k = 0; k < accordNeurons[j].Weights.Count(); k++)
                    {
                        Assert.AreEqual(accordNeurons[j].Weights[k], nn.Layers[i].Theta[j, k]);
                    }
                }
            }

            var backprop = new BackpropagationTrainer(nn);
            var teacher = new BackPropagationLearning(accordNetwork);


            // Assert outputs are equal
            var steinOutput = nn.Compute(x);
            for (var i = 0; i < accordX.Length; i++)
            {
                var accordOutput = accordNetwork.Compute(accordX[i]);
                for (var j = 0; j < accordOutput.Length; j++)
                {
                    Assert.AreEqual(accordOutput[j], steinOutput[i, j]);
                }
            }

            // Assert costs are equal
            var steinError = backprop.RunEpoch(x, y);
            var accordError = teacher.RunEpoch(accordX, accordY);

            Assert.IsTrue(steinError / 2 - accordError < 0.0001);

        }

        [TestMethod]
        public void NeuralNetwork()
        {
            //var nn = new NeuralNetwork(new int[] { 10, 10, 5 }, new Sigmoid(), 0, 0.05, false);
            var nn = new NeuralNetwork(new int[] { 10, 10 }, new Sigmoid(), 0, 0.05, false);
            nn.Layers.Add(new Classification.NeuralNetworks.Layer(5, 10, new Sigmoid(), 0, 0.05, false));

            //foreach(var layer in nn.Layers)
            //{
            //    for(var i = 0; i <  layer.Theta.RowCount; i++)
            //    {
            //        for(var j = 0; j < layer.Theta.ColumnCount; j++)
            //        {
            //            Console.Write(layer.Theta[i, j] + " ");
            //        }
            //        Console.WriteLine();
            //    }
            //    Console.WriteLine();
            //}

            //testNN.Train(null, null);
            var y = new double[,] {
                { 0,0,0,0,1 },
                { 0,0,0,1,0 },
                { 0,1,0,0,0 }
            };

            //var y = new double[,] {
            //    { 1,1,0,1,1,1,1,0,1,0 },
            //    { 1,1,1,1,0,1,0,0,1,0 },
            //    { 0,1,0,1,1,0,1,0,1,1 }
            //};

            var x = new double[,] {
                { 1,1,1,0,0,0,0,0,0,1 },
                { 0,0,0,1,1,1,0,0,0,1 },
                { 0,0,0,0,0,0,1,1,1,1 }
            };

            for (int i = 0; i < 20000; i++)
            {
                nn.Train(x, y);

                //if (i % 5 == 0)
                //{
                //    //Console.WriteLine(nn.Layers[nn.Layers.Count - 1].ComputeCost(nn.Compute(x), y));
                //}

            }

            var output = nn.Compute(x);

            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 5; j++)
                {
                    Console.Write(output[i, j]);
                    //Console.WriteLine(y[i, j] - output[i, j]);
                }
                Console.WriteLine();

            }

        }
    }

}
