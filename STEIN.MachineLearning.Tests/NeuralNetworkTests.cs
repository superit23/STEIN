using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using STEIN.MachineLearning.Classification.NeuralNetworks;

namespace STEIN.MachineLearning.Tests
{
    [TestClass]
    public class NeuralNetworkTests
    {
        [TestMethod]
        public void NeuralNetwork()
        {
            //var nn = new NeuralNetwork(new int[] { 10, 10, 5 }, new Sigmoid(), 0, 0.05, false);
            var nn = new NeuralNetwork(new int[] { 10, 10 }, new Sigmoid(), 0, 0.05, false);
            nn.Layers.Add(new Layer(5, 10, new Sigmoid(), 0, 0.05, false));

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
