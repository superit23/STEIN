using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace STEIN.MachineLearning.Classification.NeuralNetworks
{
    public class NeuralUpdateInfo
    {
        public Matrix<double> Weights
        { get; set; }

        public Vector<double> Thresholds
        { get; set; }

        public NeuralUpdateInfo(Matrix<double> weights, Vector<double> thresholds)
        {
            Weights = weights;
            Thresholds = thresholds;
        }
    }
}
