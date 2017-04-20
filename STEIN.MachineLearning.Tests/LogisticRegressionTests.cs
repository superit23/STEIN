using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using STEIN.MachineLearning.Classification;

namespace STEIN.MachineLearning.Tests
{
    [TestClass]
    public class LogisticRegressionTests
    {

        [TestMethod]
        public void LogisticRegression()
        {
            var lg = new LogisticRegression(5, 0.05);

            var x = new double[,] {
                { 0,1,0,0,1 },
                { 0,0,0,1,1 },
                { 0,1,0,0,0 }
            };

            var y = new int[] {
                1,
                1,
                0
            };

            for (int i = 0; i < 10000; i++)
            {
                lg.Train(x, y);

                if (i % 5 == 0)
                {
                    Console.WriteLine(lg.ComputeCost(lg.Compute(x), y));
                }
            }

            var output = lg.Compute(x);
            for (var i = 0; i < output.Length; i++)
            {
                //Console.WriteLine(val);
                Assert.AreEqual(y[i], Math.Round(output[i], 1));
            }

            
        }
    }
}
