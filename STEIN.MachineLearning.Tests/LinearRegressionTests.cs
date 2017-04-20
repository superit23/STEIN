using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using STEIN.MachineLearning;
using STEIN.MachineLearning.Regression;

namespace STEIN.MachineLearning.Tests
{
    [TestClass]
    public class LinearRegressionTests
    {
        [TestMethod]
        public void LinearRegression()
        {
            var numTraining = 100;
            var numTheta = 1;
            var trainIter = 100;

            var linReg = new LinearRegression(numTheta, 0.0005, false);

            var training = new double[numTraining, numTheta];
            var y = new double[numTraining];
            for (int j = 0; j < numTraining; j++)
            {

                for (int i = 0; i < numTheta; i++)
                {
                    training[j, i] = j;
                }

                y[j] = j * 3;
            }

            for (int i = 0; i < trainIter; i++)
            {
                linReg.Train(training, y);

                if (i % 5 == 0)
                {
                    Console.WriteLine(linReg.ComputeCost(linReg.Compute(training), y));
                }
            }


            var validation = new double[1, numTheta];
            var m = numTraining + 1;
            for (int i = 0; i < numTheta; i++)
            {
                validation[0, i] = m;
            }

            var valError = (m * 3) - linReg.Compute(validation)[0];
            //Console.WriteLine(valError);
            Assert.IsTrue(valError < 0.0000001);
        }


    }

}
