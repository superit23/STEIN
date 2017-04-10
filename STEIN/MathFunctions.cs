using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;

namespace STEIN
{
    public class MathFunctions
    {
        public static double[] Softmax(double[] input)
        {
            var length = input.Length;
            var result = new double[length];
            var total = 0.0;

            for (var i = 0; i < length; i++)
            {
                var exp = Math.Exp(input[i]);
                result[i] = exp;
                total += exp;
            }

            if (total != 0)
            {
                for (var i = 0; i < length; i++)
                {
                    result[i] /= total;
                }
            }

            return result;
        }

        public static Matrix<Double> Softmax(Matrix<Double> input)
        {
            var softMaxedVecs = new double[input.RowCount][];
            var i = 0;
            foreach(var vector in input.ToRowArrays())
            {
                softMaxedVecs[i] = Softmax(vector);
                i++;
            }

            return DenseMatrix.OfRowArrays(softMaxedVecs);
        }

        static RandomNumberGenerator rng = new RNGCryptoServiceProvider();

        public static int GenerateInt(int min, int max)
        {
            var randBytes = new byte[sizeof(int)];
            rng.GetBytes(randBytes);

            var val = BitConverter.ToInt32(randBytes, 0);
            val &= 0x7fffffff;

            val = val % (max - min + 1);

            return val + min;
        }
    }
}
