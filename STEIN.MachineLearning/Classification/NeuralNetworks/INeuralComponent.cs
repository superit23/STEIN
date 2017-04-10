using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace STEIN.MachineLearning.Classification.NeuralNetworks
{
    public interface INeuralComponent
    {
        Matrix<double> Compute(Matrix<double> x);

    }
}
